import {
  AnthropicContent,
  AnthropicContentImage,
  AnthropicContentText,
  AnthropicContentToolResult,
  AnthropicContentToolUse,
  AnthropicMessage,
  AnthropicRequest,
  AnthropicResponse,
  AnthropicToolDefinition,
  CastariReasoningConfig,
  OpenRouterContentImage,
  OpenRouterContentText,
  OpenRouterMessage,
  OpenRouterMessageContent,
  OpenRouterRequest,
  OpenRouterResponse,
  OpenRouterToolCall,
  OpenRouterToolDefinition,
  WebSearchOptions,
} from './types';
import { invalidRequest } from './errors';
import { randomId } from './utils';
import { isServerTool } from './provider';

export interface OpenRouterBuildOptions {
  wireModel: string;
  reasoning?: CastariReasoningConfig;
  webSearch?: WebSearchOptions;
}

export function buildOpenRouterRequest(
  body: AnthropicRequest,
  options: OpenRouterBuildOptions,
): OpenRouterRequest {
  const messages = convertMessages(body);
  const request: OpenRouterRequest = {
    model: options.wireModel,
    messages,
    max_tokens: body.max_tokens,
    temperature: body.temperature,
    top_p: body.top_p,
    stop: body.stop_sequences,
    stream: body.stream ?? false,
  };

  const clientTools = convertTools(body.tools);
  if (clientTools.length) request.tools = clientTools;
  const toolChoice = convertToolChoice(body.tool_choice);
  if (toolChoice) request.tool_choice = toolChoice;

  if (options.reasoning) request.reasoning = options.reasoning;

  if (options.webSearch) {
    request.plugins = [{ id: 'web', engine: options.webSearch.engine, max_results: options.webSearch.max_results }];
    request.web_search_options = options.webSearch;
  }

  return request;
}

function convertMessages(body: AnthropicRequest): OpenRouterMessage[] {
  const output: OpenRouterMessage[] = [];
  if (body.system) {
    output.push({ role: 'system', content: stringifySystem(body.system) });
  }
  for (const message of body.messages) {
    output.push(...convertMessage(message));
  }
  return output;
}

function stringifySystem(system: string | AnthropicContentText[]): string {
  if (typeof system === 'string') return system;
  return system.map((block) => block.text).join('\n');
}

function convertMessage(message: AnthropicMessage): OpenRouterMessage[] {
  const segments = Array.isArray(message.content)
    ? message.content
    : [{ type: 'text', text: message.content } as AnthropicContentText];
  const textSegments: AnthropicContent[] = [];
  const toolResults: AnthropicContentToolResult[] = [];
  const toolUses: AnthropicContentToolUse[] = [];

  for (const segment of segments) {
    if (!segment || typeof segment !== 'object') continue;
    if (segment.type === 'tool_result') toolResults.push(segment);
    else if (segment.type === 'tool_use') toolUses.push(segment);
    else textSegments.push(segment);
  }

  const resolved: OpenRouterMessage[] = [];

  if (textSegments.length) {
    resolved.push({ role: message.role, content: convertContentParts(textSegments) });
  }

  if (message.role === 'assistant' && toolUses.length) {
    resolved.push({
      role: 'assistant',
      content: '',
      tool_calls: toolUses.map((item) => ({
        id: item.type === 'tool_use' ? item.id : randomId('tool'),
        type: 'function',
        function: {
          name: item.type === 'tool_use' ? item.name : 'unknown_tool',
          arguments: safeStringify(item.type === 'tool_use' ? item.input : {}),
        },
      })),
    });
  }

  if (toolResults.length) {
    for (const result of toolResults) {
      resolved.push({
        role: 'tool',
        tool_call_id: result.tool_use_id,
        content: deriveToolResultContent(result),
      });
    }
  }

  return resolved.length ? resolved : [{ role: message.role, content: '' }];
}

function convertContentParts(parts: AnthropicContent[]): OpenRouterMessageContent {
  const results: (OpenRouterContentText | OpenRouterContentImage)[] = [];
  let hasImage = false;
  for (const block of parts) {
    if (block.type === 'text') {
      results.push({ type: 'text', text: block.text });
    } else if (block.type === 'image') {
      hasImage = true;
      if (block.source.type === 'url') {
        results.push({ type: 'image_url', image_url: { url: block.source.url } });
      } else {
        const dataUri = `data:${block.source.media_type};base64,${block.source.data}`;
        results.push({ type: 'image_url', image_url: { url: dataUri } });
      }
    }
  }

  if (!hasImage && results.every((item) => item.type === 'text')) {
    return (results as OpenRouterContentText[]).map((item) => item.text).join('');
  }
  return results;
}

function convertTools(tools?: AnthropicToolDefinition[]): OpenRouterToolDefinition[] {
  if (!Array.isArray(tools)) return [];
  const converted: OpenRouterToolDefinition[] = [];
  for (const tool of tools) {
    if (isServerTool(tool)) continue;
    if (!tool?.name || !tool.input_schema) continue;
    converted.push({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.input_schema,
      },
    });
  }
  return converted;
}

function convertToolChoice(choice: AnthropicRequest['tool_choice']): OpenRouterRequest['tool_choice'] | undefined {
  if (!choice) return undefined;
  if (choice === 'auto') return 'auto';
  if (choice === 'none') return 'none';
  if (typeof choice === 'object' && 'name' in choice) {
    return { type: 'function', function: { name: choice.name } };
  }
  return 'auto';
}

function deriveToolResultContent(result: AnthropicContent): string {
  if (result.type !== 'tool_result') return '';
  if (typeof result.content === 'string') return result.content;
  if (Array.isArray(result.content)) {
    return result.content.map((block) => (block.type === 'text' ? block.text : '')).join('\n');
  }
  return '';
}

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value ?? {});
  } catch {
    return '{}';
  }
}

export function mapOpenRouterResponse(
  providerResponse: OpenRouterResponse,
  originalModel: string,
): AnthropicResponse {
  const choice = providerResponse.choices[0];
  if (!choice) throw invalidRequest('OpenRouter response missing choices');
  const content: AnthropicContent[] = [];
  if (choice.message?.content) {
    content.push(...convertOpenRouterContent(choice.message.content));
  }
  if (choice.message?.tool_calls) {
    for (const call of choice.message.tool_calls) {
      content.push({
        type: 'tool_use',
        id: call.id,
        name: call.function.name,
        input: parseToolArguments(call.function.arguments),
      });
    }
  }

  const usage = providerResponse.usage
    ? {
        input_tokens: providerResponse.usage.prompt_tokens,
        output_tokens: providerResponse.usage.completion_tokens,
        reasoning_tokens: providerResponse.usage.reasoning_tokens,
      }
    : undefined;

  return {
    id: providerResponse.id ?? randomId('msg'),
    type: 'message',
    role: 'assistant',
    model: originalModel,
    stop_reason: mapStopReason(choice.finish_reason),
    stop_sequence: null,
    content,
    usage,
  };
}

function convertOpenRouterContent(content: string | null | OpenRouterMessageContent): AnthropicContent[] {
  if (!content) return [];
  if (typeof content === 'string') return [{ type: 'text', text: content }];
  return content.map((part) =>
    part.type === 'text'
      ? { type: 'text', text: part.text }
      : ({ type: 'image', source: { type: 'url', url: part.image_url.url } } as AnthropicContentImage),
  );
}

function parseToolArguments(value?: string): unknown {
  if (!value) return {};
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

export function mapStopReason(reason: string | null): string | null {
  if (!reason) return null;
  switch (reason) {
    case 'stop':
      return 'end_turn';
    case 'tool_calls':
      return 'tool_use';
    case 'length':
      return 'max_tokens';
    case 'content_filter':
      return 'content_filter';
    default:
      return reason;
  }
}
