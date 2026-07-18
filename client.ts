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
        ttftMs: number;
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

