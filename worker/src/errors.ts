export class CastariError extends Error {
  public readonly status: number;
  public readonly type: string;
  public readonly retryable: boolean;
  public readonly details?: Record<string, unknown>;

  constructor(
    status: number,
    type: string,
    message: string,
    options?: { retryable?: boolean; details?: Record<string, unknown> },
  ) {
    super(message);
    this.status = status;
    this.type = type;
    this.retryable = options?.retryable ?? false;
    this.details = options?.details;
  }
}

export function invalidRequest(message: string, details?: Record<string, unknown>): CastariError {
  return new CastariError(400, 'invalid_request_error', message, { details });
}

export function authenticationError(message = 'Authentication failed'): CastariError {
  return new CastariError(401, 'authentication_error', message);
}

export function permissionError(message: string): CastariError {
  return new CastariError(403, 'permission_error', message);
}

export function rateLimitError(message: string, details?: Record<string, unknown>): CastariError {
  return new CastariError(429, 'rate_limit_error', message, { retryable: true, details });
}

export function upstreamError(
  status: number,
  message: string,
  details?: Record<string, unknown>,
): CastariError {
  const type = status >= 500 ? 'api_error' : 'invalid_request_error';
  return new CastariError(status, type, message, { retryable: status >= 500, details });
}

export function errorResponse(error: unknown): Response {
  const encoder = JSON.stringify;
  if (error instanceof CastariError) {
    return new Response(
      encoder({
        type: error.type,
        message: error.message,
        retryable: error.retryable,
        details: error.details,
      }),
      {
        status: error.status,
        headers: jsonHeaders(),
      },
    );
  }

  const message = error instanceof Error ? error.message : 'Unknown error';
  return new Response(
    encoder({ type: 'api_error', message, retryable: false }),
    { status: 500, headers: jsonHeaders() },
  );
}

export function jsonHeaders(): HeadersInit {
  return {
    'content-type': 'application/json',
    'cache-control': 'no-store',
  };
}
