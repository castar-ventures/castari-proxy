import { resolveConfig, Env } from './config';
import {
  errorResponse,
  authenticationError,
  invalidRequest,
  upstreamError,
} from './errors';
import { normalizeCastariHeaders, readJsonBody, isJsonMime } from './utils';
import { categorizeServerTools, detectServerTools, resolveProvider } from './provider';
import { buildOpenRouterRequest, mapOpenRouterResponse } from './translator';
import { streamOpenRouterToAnthropic } from './stream';
import {
  AnthropicRequest,
  AnthropicResponse,
  CastariMetadata,
  CastariReasoningConfig,
  OpenRouterResponse,
  WebSearchOptions,
  WorkerConfig,
} from './types';

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    try {
      if (new URL(request.url).pathname !== '/v1/messages' || request.method !== 'POST') {
        return new Response('Not found', { status: 404 });
      }

      const config = resolveConfig(env);
      const headers = normalizeCastariHeaders(request.headers);
      const body = await readJsonBody<AnthropicRequest>(request.clone());
      const authHeader = extractApiKey(request.headers);
      const metadata = normalizeMetadata(body.metadata);
      const reasoning = metadata?.castari?.reasoning as CastariReasoningConfig | undefined;
      let webSearch = metadata?.castari?.web_search_options as WebSearchOptions | undefined;

      let { provider, wireModel, originalModel } = resolveProvider(headers, body, config);

      const serverToolEntries = categorizeServerTools(body.tools);
      const webSearchTools = serverToolEntries.filter((entry) => entry.kind === 'websearch');
      const otherServerTools = serverToolEntries.filter((entry) => entry.kind === 'other');

      if (provider === 'openrouter' && otherServerTools.length) {
        if (config.serverToolsMode === 'error') {
          throw invalidRequest('Server tools require Anthropic provider', {
            tools: otherServerTools.map((entry) => entry.label),
          });
        }
        if (config.serverToolsMode === 'enforceAnthropic') {
          provider = 'anthropic';
          wireModel = originalModel;
        }
        // emulate mode would be implemented when server backends are available
      }

      if (provider === 'openrouter') {
        const wantsWebSearch = webSearchTools.length > 0;
        if (wantsWebSearch && !webSearch) {
          webSearch = {};
        }
      }

      if (body.mcp_servers?.length && provider === 'openrouter' && env.MCP_BRIDGE_MODE !== 'http-sse') {
        throw invalidRequest('MCP servers require Anthropic routing or http-sse bridge', { mode: env.MCP_BRIDGE_MODE ?? 'off' });
      }

      if (provider === 'anthropic') {
        return await proxyAnthropic(body, request, authHeader.value, config.anthropicBaseUrl);
      }

      return await handleOpenRouter({
        body,
        wireModel,
        originalModel,
        apiKey: authHeader.value,
        config,
        reasoning,
        webSearch,
      });
    } catch (error) {
      return errorResponse(error);
    }
  },
};

function normalizeMetadata(metadata: AnthropicRequest['metadata']): CastariMetadata | undefined {
  if (!metadata || typeof metadata !== 'object') return undefined;
  return metadata as CastariMetadata;
}

function extractApiKey(headers: Headers): { value: string; type: 'x-api-key' | 'bearer' } {
  const auth = headers.get('authorization');
  if (auth && auth.toLowerCase().startsWith('bearer ')) {
    const token = auth.slice(7).trim();
    if (token) return { value: token, type: 'bearer' };
  }
  const key = headers.get('x-api-key');
  if (key) return { value: key, type: 'x-api-key' };
  throw authenticationError('Missing API key');
}

async function proxyAnthropic(
  body: AnthropicRequest,
  request: Request,
  apiKey: string,
  upstreamUrl: string,
): Promise<Response> {
  const upstreamResp = await fetch(upstreamUrl, {
    method: 'POST',
    headers: buildAnthropicHeaders(request.headers, apiKey),
    body: JSON.stringify(body),
  });
  if (!upstreamResp.ok) {
    const text = await upstreamResp.text();
    return new Response(text || JSON.stringify({ error: 'Anthropic upstream error' }), {
      status: upstreamResp.status,
      headers: {
        'content-type': upstreamResp.headers.get('content-type') ?? 'application/json',
      },
    });
  }
  return upstreamResp;
}

function buildAnthropicHeaders(original: Headers, apiKey: string): HeadersInit {
  const headers = new Headers();
  headers.set('content-type', 'application/json');
  headers.set('x-api-key', apiKey);
  const anthropicVersion = original.get('anthropic-version');
  if (anthropicVersion) headers.set('anthropic-version', anthropicVersion);
  return headers;
}

interface OpenRouterContext {
  body: AnthropicRequest;
  wireModel: string;
  originalModel: string;
  apiKey: string;
  config: WorkerConfig;
  reasoning?: CastariReasoningConfig;
  webSearch?: WebSearchOptions;
}

async function handleOpenRouter(ctx: OpenRouterContext): Promise<Response> {
  const openRouterRequest = buildOpenRouterRequest(ctx.body, {
    wireModel: ctx.wireModel,
    reasoning: ctx.reasoning,
    webSearch: ctx.webSearch,
  });

  const upstreamResp = await fetch(ctx.config.openRouterBaseUrl, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      authorization: `Bearer ${ctx.apiKey}`,
    },
    body: JSON.stringify(openRouterRequest),
  });

  if (ctx.body.stream) {
    if (!upstreamResp.ok) {
      throw upstreamError(
        upstreamResp.status,
        'OpenRouter streaming error',
        await readUpstreamErrorDetails(upstreamResp),
      );
    }
    return streamOpenRouterToAnthropic(upstreamResp, { originalModel: ctx.originalModel });
  }

  if (!upstreamResp.ok) {
    throw upstreamError(
      upstreamResp.status,
      'OpenRouter error',
      await readUpstreamErrorDetails(upstreamResp),
    );
  }
  const json = (await upstreamResp.json()) as OpenRouterResponse;
  const responseBody: AnthropicResponse = mapOpenRouterResponse(json, ctx.originalModel);
  return new Response(JSON.stringify(responseBody), {
    status: 200,
    headers: {
      'content-type': 'application/json',
      'cache-control': 'no-store',
    },
  });
}

async function readUpstreamErrorDetails(
  response: Response,
): Promise<Record<string, unknown>> {
  const text = await response.text();
  let body: unknown = text;
  if (text && isJsonMime(response.headers.get('content-type'))) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }

  const details: Record<string, unknown> = {
    status: response.status,
    body,
  };
  const retryAfter = response.headers.get('retry-after');
  if (retryAfter) details.retryAfter = retryAfter;
  return details;
}
