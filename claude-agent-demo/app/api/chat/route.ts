import type { SDKMessage, SDKUserMessage } from '@anthropic-ai/claude-agent-sdk';
import type { APIUserMessage } from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs';
import { NextRequest } from 'next/server';
import { z } from 'zod';
import { mapMessageToUIEvents } from '@/lib/agent/events';
import { buildUserMessageContent } from '@/lib/agent/messages';
import { startQuery } from '@/lib/agent/session';
import type { ToolMode } from '@/lib/policy/permission';
import { transcriptStore } from '@/lib/store/transcripts';
import type { UIEvent } from '@/lib/types/events';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const MAX_IMAGES_PER_MESSAGE = 4;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024; // 5MB each
const MAX_TOTAL_IMAGE_BYTES = 15 * 1024 * 1024; // 15MB combined
const THINKING_MIN_BUDGET = 1_024;
const THINKING_MAX_BUDGET = 64_000;

/**
 * Calculate the actual decoded size of a base64 string.
 * Base64 encodes 3 bytes into 4 characters, so decoded size ≈ length * 3/4.
 * Padding characters (=) don't contribute to decoded bytes.
 */
function getBase64DecodedSize(base64: string): number {
  const padding = (base64.match(/=+$/) || [''])[0].length;
  return Math.floor((base64.length * 3) / 4) - padding;
}

const imageSchema = z.object({
  id: z.string().min(1),
  mimeType: z
    .string()
    .regex(/^image\/[\w.+-]+$/i, 'Invalid image mime type'),
  base64: z
    .string()
    .regex(/^[a-zA-Z0-9+/=]+$/, 'Image payload must be base64 encoded'),
  size: z
    .number()
    .int()
    .positive()
    .optional()
}).refine(
  (img) => {
    const actualSize = getBase64DecodedSize(img.base64);
    return actualSize <= MAX_IMAGE_BYTES;
  },
  {
    message: `Images must be ${MAX_IMAGE_BYTES / (1024 * 1024)}MB or smaller`,
  }
);

const thinkingSchema = z.object({
  enabled: z.boolean(),
  budgetTokens: z
    .number()
    .int()
    .min(THINKING_MIN_BUDGET, `Thinking budget must be at least ${THINKING_MIN_BUDGET}`)
    .max(THINKING_MAX_BUDGET, `Thinking budget must be <= ${THINKING_MAX_BUDGET}`)
});

const requestSchema = z.object({
  message: z.string().optional().default(''),
  model: z.string().optional(),
  sessionId: z.string().optional(),
  useProjectInstructions: z.boolean().optional(),
  toolMode: z.enum(['safe', 'full']).optional(),
  images: z.array(imageSchema).max(MAX_IMAGES_PER_MESSAGE).optional(),
  thinking: thinkingSchema.optional()
});

export async function POST(req: NextRequest) {
  const payload = await parseRequest(req);
  if ('error' in payload) {
    return new Response(JSON.stringify({ error: payload.error }), {
      status: payload.status,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  const { message, model, useProjectInstructions, sessionId, thinking } = payload.body;
  const images = payload.body.images ?? [];
  const mode: ToolMode = payload.body.toolMode ?? 'safe';
  const trimmed = message.trim();
  // Calculate actual payload size server-side instead of trusting client-reported size
  const totalImageBytes = images.reduce((sum, img) => sum + getBase64DecodedSize(img.base64), 0);
  const thinkingConfig = thinking?.enabled ? thinking : undefined;

  if (!trimmed && images.length === 0) {
    return new Response(JSON.stringify({ error: 'Provide a message or at least one image' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  if (totalImageBytes > MAX_TOTAL_IMAGE_BYTES) {
    return new Response(
      JSON.stringify({
        error: `Images exceed ${Math.floor(MAX_TOTAL_IMAGE_BYTES / (1024 * 1024))}MB combined`
      }),
      {
        status: 400,
        headers: { 'Content-Type': 'application/json' }
      }
    );
  }

  let userMessage: APIUserMessage;
  try {
    userMessage = buildUserMessageContent(trimmed, images);
  } catch (error) {
    return new Response(JSON.stringify({ error: formatError(error) }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' }
    });
  }

  const promptStream = buildPromptStream(userMessage);

  const query = startQuery(promptStream, {
    mode,
    model: model || undefined,
    useProjectInstructions: Boolean(useProjectInstructions),
    sessionId: sessionId || undefined,
    thinking: thinkingConfig
  });

  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const message of query) {
          const events = mapMessageToUIEvents(message);
          if (!events.length) continue;
          for (const event of events) {
            await recordTranscriptEntry(message, event);
            controller.enqueue(encodeEvent(event, encoder));
          }
        }
      } catch (error) {
        controller.enqueue(
          encodeEvent(
            {
              type: 'error',
              data: { message: formatError(error) }
            },
            encoder
          )
        );
      } finally {
        controller.close();
      }
    },
    cancel() {
      query.interrupt?.().catch(() => {});
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Cache-Control': 'no-store'
    }
  });
}

async function parseRequest(
  req: NextRequest
): Promise<
  | { body: z.infer<typeof requestSchema> }
  | { error: string; status: number }
> {
  try {
    const json = await req.json();
    const result = requestSchema.safeParse(json);
    if (!result.success) {
      return { error: result.error.errors.map((err) => err.message).join(', '), status: 400 };
    }
    return { body: result.data };
  } catch {
    return { error: 'Invalid JSON body', status: 400 };
  }
}

function encodeEvent(event: UIEvent, encoder: TextEncoder) {
  return encoder.encode(JSON.stringify(event) + '\n');
}

function formatError(error: unknown) {
  if (error instanceof Error) return error.message;
  return 'Unexpected error';
}

async function recordTranscriptEntry(message: SDKMessage, event: UIEvent) {
  try {
    const sessionId = (message as { session_id?: string }).session_id ?? 'session';
    await transcriptStore.append(sessionId, {
      ts: Date.now(),
      kind: `sdk:${message.type}`,
      payload: event
    });
  } catch (err) {
    console.error('Failed to append transcript entry', err);
  }
}

function buildPromptStream(
  message: APIUserMessage
): AsyncIterable<SDKUserMessage> {
  return {
    async *[Symbol.asyncIterator]() {
      yield {
        type: 'user',
        session_id: '',
        message,
        parent_tool_use_id: null
      };
    }
  };
}
