import { afterEach, describe, expect, it, vi } from 'vitest';
import worker from '../src/index';
import { Env } from '../src/config';

const env: Env = {
  UPSTREAM_OPENROUTER_BASE_URL: 'https://openrouter.test/api',
  OPENROUTER_DEFAULT_VENDOR: 'openai',
};

function openRouterRequest(stream: boolean): Request {
  return new Request('https://worker.test/v1/messages', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': 'test-key',
      'x-castari-provider': 'openrouter',
    },
    body: JSON.stringify({
      model: 'or:gpt-5-mini',
      max_tokens: 32,
      stream,
      messages: [{ role: 'user', content: 'hello' }],
    }),
  });
}

describe('OpenRouter upstream errors', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('preserves non-stream upstream status for non-JSON errors', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response('<html>temporarily unavailable</html>', {
          status: 503,
          headers: {
            'content-type': 'text/html',
            'retry-after': '7',
          },
        }),
      ),
    );

    const response = await worker.fetch(openRouterRequest(false), env);
    const body = await response.json() as any;

    expect(response.status).toBe(503);
    expect(body).toMatchObject({
      type: 'api_error',
      message: 'OpenRouter error',
      retryable: true,
      details: {
        status: 503,
        body: '<html>temporarily unavailable</html>',
        retryAfter: '7',
      },
    });
  });

  it('preserves streaming upstream status and JSON details', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: { message: 'rate limited' } }), {
          status: 429,
          headers: {
            'content-type': 'application/json',
            'retry-after': '30',
          },
        }),
      ),
    );

    const response = await worker.fetch(openRouterRequest(true), env);
    const body = await response.json() as any;

    expect(response.status).toBe(429);
    expect(body).toMatchObject({
      type: 'invalid_request_error',
      message: 'OpenRouter streaming error',
      retryable: false,
      details: {
        status: 429,
        body: { error: { message: 'rate limited' } },
        retryAfter: '30',
      },
    });
  });
});
