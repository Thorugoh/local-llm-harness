const BASE_URL = "http://192.168.1.17:8083";
const MODEL = "qwen3.6-35b-a3b";

export type Role = "system" | "user" | "assistant" | "tool";

export interface Message {
    role: Role;
    content: string | null;
    [key: string]: unknown;
}

export interface SamplingOptions {
    temperature?: number;
    top_p?: number;
    top_k?: number;
    min_p?: number;
    repeat_penalty?: number;
    max_tokens?: number;
}

const DEFAULT_SAMPLING: Required<Pick<SamplingOptions, "temperature" | "max_tokens">> =
{
    temperature: 0.2,
    max_tokens: 2048,
}

export interface ChatResult {
    content: string;
    finishReason: string;
    usage: {
        promptTokens: number;
        completionTokens: number;
        totalTokens: number;
    };
    timing: {
        totalMs: number;
        ttftMs: number | null;
        tokensPerSecond: number | null;
    };
    raw: Message;
}

export class LLMError extends Error {
    constructor(
        message: string,
        public readonly kind: "connection" | "timeout" | "http" | "parse",
        public readonly status: number,
    ) {
        super(message);
        this.name = "LLMError"
    }
}

// Non-streaming call
export async function chat(
    messages: Message[],
    options: SamplingOptions = {},
    timeoutMs = 120_000,
): Promise<ChatResult> {
    const started = performance.now();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    let response: Response;

    try {
        response = await fetch(`${BASE_URL}/v1/chat/completions`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            signal: controller.signal,
            body: JSON.stringify({
                model: MODEL,
                messages,
                stream: false,
                temperature: options.temperature ?? DEFAULT_SAMPLING.temperature,
                max_tokens: options.max_tokens ?? DEFAULT_SAMPLING.max_tokens,
                top_p: options.top_p,
                top_k: options.top_k,
                min_p: options.min_p,
                repeat_penalty: options.repeat_penalty,
            }),
        });
    } catch (err) {
        if (err instanceof Error && err.name === "AbortError") {
            throw new LLMError(`Request timed out after ${timeoutMs}ms`, "timeout", 1);
        }
        throw new LLMError(
            `Could not react the model server at ${BASE_URL}. Is llama server running?`,
            "connection",
            2,
        )
    } finally {
        clearTimeout(timer);
    }

    if (!response.ok) {
        const body = await response.text().catch(() => "");
        throw new LLMError(
            `Server returned ${response.status}: ${body.slice(0, 200)}`,
            "http",
            response.status,
        );
    }

    let data: any;
    try {
        data = await response.json();
    } catch {
        throw new LLMError("Server returned malformed JSON", "parse", 3);
    }

    const totalMs = performance.now() - started;
    const choice = data.choices?.[0];
    const usage = data.usage ?? {};
    const completionTokens = usage.completion_tokens ?? 0;

    return {
        content: choice?.message?.content ?? "",
        finishReason: choice?.finish_reason ?? "unknown",
        usage: {
            promptTokens: usage.prompt_tokens ?? 0,
            completionTokens,
            totalTokens: usage.total_tokens ?? 0,
        },
        timing: {
            totalMs,
            ttftMs: null, // not measurable without streaming
            tokensPerSecond: completionTokens > 0 ? completionTokens / (totalMs / 1000) : null,
        },
        raw: choice?.message ?? { role: "assistant", content: "" },
    };
}

export async function chatStream(
    messages: Message[],
    onDelta: (text: string) => void,
    options: SamplingOptions,
    timeoutMs: 120_000,
): Promise<ChatResult> {
    const started = performance.now();
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs)

    let response: Response;
    try {
        response = await fetch(`${BASE_URL}/v1/chat/completions`, {
            method: "POST",
            headers: { "Content-Type": "applications/json" },
            signal: controller.signal,
            body: JSON.stringify({
                model: MODEL,
                messages,
                stream: true,
                stream_options: { include_usage: true },
                temperature: options.temperature ?? DEFAULT_SAMPLING.temperature,
                max_tokens: options.max_tokens ?? DEFAULT_SAMPLING.max_tokens,
                top_p: options.top_p,
                top_k: options.top_k,
                repeat_penalty: options.repeat_penalty,
            })
        });
    } catch (err) {
        clearTimeout(timer);
        if (err instanceof Error && err.name === "AbortError") {
            throw new LLMError(`Request timed out after ${timeoutMs}ms`, "timeout", 0);
        }
        throw new LLMError(
            `Could not reach the model server at ${BASE_URL}. Is llama-server running?`,
            "connection",
            1
        );
    }

    if (!response.ok || !response.body) {
        clearTimeout(timer);
        const body = await response.text().catch(() => "");
        throw new LLMError(`Server returned ${response.status}: ${body.slice(0, 200)}`, "http", response.status);
    }

    let content = "";
    let finishReason = "unknown";
    let ttftMs: number | null = null;
    let completionTokens = 0;
    let promptTokens = 0;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // SSE frames are separated by double newlines; each line starts with "data: ".
            const lines = buffer.split("\n");
            buffer = lines.pop() ?? ""; // keep the last (possibly partial) line

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed.startsWith("data:")) continue;

                const payload = trimmed.slice(5).trim();
                if (payload === "[DONE]") continue;

                let chunk: any;
                try {
                    chunk = JSON.parse(payload);
                } catch {
                    continue; // ignore malformed frame
                }

                const delta = chunk.choices?.[0]?.delta?.content;
                if (delta) {
                    if (ttftMs === null) ttftMs = performance.now() - started; // first token!
                    content += delta;
                    onDelta(delta);
                }

                if (chunk.choices?.[0]?.finish_reason) {
                    finishReason = chunk.choices[0].finish_reason;
                }
                if (chunk.usage) {
                    completionTokens = chunk.usage.completion_tokens ?? completionTokens;
                    promptTokens = chunk.usage.prompt_tokens ?? promptTokens;
                }
            }
        }
    } finally {
        clearTimeout(timer);
    }

    const totalMs = performance.now() - started;
    // Decode speed excludes prefill: measure tokens over the post-TTFT window.
    const decodeMs = ttftMs !== null ? totalMs - ttftMs : totalMs;
    const tokensPerSecond =
        completionTokens > 0 && decodeMs > 0 ? completionTokens / (decodeMs / 1000) : null;

    return {
        content,
        finishReason,
        usage: {
            promptTokens,
            completionTokens,
            totalTokens: promptTokens + completionTokens,
        },
        timing: { totalMs, ttftMs, tokensPerSecond },
        raw: { role: "assistant", content },
    };
}
