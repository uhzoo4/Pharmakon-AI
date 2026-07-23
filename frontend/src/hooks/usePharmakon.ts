import { useState, useCallback } from "react";

export interface AltToken {
  char: string;
  prob: number;
}

export interface Token {
  char: string;
  alts?: AltToken[];
}

export interface Message {
  role: "user" | "model" | "system";
  content: string; // for user
  tokens?: Token[]; // for model
}

export interface GenerationOptions {
  temperature?: number;
  topP?: number;
  topK?: number;
  maxTokens?: number;
  blacklistStr?: string;
}

export interface Session {
  id: string;
  personalityId: string;
  timestamp: number;
  title: string;
  messages: Message[];
}

const SESSIONS_STORAGE_KEY = "pharmakon_sessions_v1";

export function usePharmakon() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Lazy initial state for sessions to prevent React 19 effect lint error
  const [sessions, setSessions] = useState<Session[]>(() => {
    if (typeof window === "undefined") return [];
    try {
      const saved = localStorage.getItem(SESSIONS_STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch (e) {
      console.error("Failed to load saved sessions:", e);
      return [];
    }
  });

  const saveCurrentSession = useCallback((personalityId: string, msgs: Message[]) => {
    if (msgs.length === 0) return;
    try {
      const firstUserMsg = msgs.find((m) => m.role === "user")?.content || "Untitled Session";
      const title = firstUserMsg.slice(0, 30) + (firstUserMsg.length > 30 ? "..." : "");
      
      setSessions((prev) => {
        const existingIdx = prev.findIndex((s) => s.personalityId === personalityId && s.messages.length > 0);
        let updated: Session[];
        if (existingIdx >= 0) {
          updated = [...prev];
          updated[existingIdx] = {
            ...updated[existingIdx],
            timestamp: Date.now(),
            messages: msgs,
          };
        } else {
          const newSession: Session = {
            id: `session_${Date.now()}`,
            personalityId,
            timestamp: Date.now(),
            title,
            messages: msgs,
          };
          updated = [newSession, ...prev];
        }
        if (typeof window !== "undefined") {
          localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(updated.slice(0, 20))); // Keep last 20 sessions
        }
        return updated;
      });
    } catch (e) {
      console.error("Failed to save session:", e);
    }
  }, []);

  const deleteSession = useCallback((sessionId: string) => {
    setSessions((prev) => {
      const updated = prev.filter((s) => s.id !== sessionId);
      if (typeof window !== "undefined") {
        localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(updated));
      }
      return updated;
    });
  }, []);

  const generate = useCallback(
    async (
      personality: string,
      prompt: string,
      options: GenerationOptions = {}
    ) => {
      const {
        temperature = 0.8,
        topP = 0.9,
        topK = 50,
        maxTokens = 250,
        blacklistStr = "",
      } = options;

      // Append user message
      const userMsg: Message = { role: "user", content: prompt };
      setMessages((prev) => [...prev, userMsg]);
      setIsGenerating(true);
      setError(null);

      // Create an empty model message placeholder
      setMessages((prev) => [...prev, { role: "model", content: "", tokens: [] }]);

      // Parse blacklist (CSV to array of characters)
      const blacklist = blacklistStr
        .split(",")
        .map((c) => c.trim())
        .filter((c) => c.length === 1);

      try {
        const response = await fetch("http://127.0.0.1:8000/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            personality,
            prompt,
            temperature,
            top_p: topP,
            top_k: topK,
            max_tokens: maxTokens,
            blacklist,
          }),
        });

        if (!response.ok) {
          let errMsg = `HTTP Error ${response.status}`;
          try {
            const errData = await response.json();
            if (errData.error) errMsg = errData.error;
          } catch {}
          throw new Error(errMsg);
        }

        if (!response.body) throw new Error("No response body returned.");

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        let currentText = "";
        let finalMessages: Message[] = [];

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const events = chunk.split("\n\n");

          for (const event of events) {
            if (!event.startsWith("data: ")) continue;

            const dataStr = event.slice(6).trim();
            if (!dataStr) continue;

            try {
              const data = JSON.parse(dataStr);

              if (data.error) {
                throw new Error(data.error);
              }
              if (data.done) {
                break;
              }
              if (data.text) {
                currentText += data.text;
                const rawAlts = Array.isArray(data.alts) ? data.alts : [];
                const alts: AltToken[] = rawAlts.map((alt: { char?: unknown; prob?: unknown }) => ({
                  char: String(alt?.char ?? ""),
                  prob: typeof alt?.prob === "number" ? alt.prob : 0,
                }));
                const newToken: Token = { char: String(data.text), alts };

                setMessages((prev) => {
                  if (prev.length === 0) return prev;
                  const next = [...prev];
                  const lastIndex = next.length - 1;
                  const targetMsg = next[lastIndex];
                  if (!targetMsg) return prev;

                  const lastMsg: Message = {
                    ...targetMsg,
                    content: currentText,
                    tokens: [...(targetMsg.tokens || []), newToken],
                  };
                  next[lastIndex] = lastMsg;
                  finalMessages = next;
                  return next;
                });
              }
            } catch (e) {
              console.error("Failed to parse SSE JSON:", e);
            }
          }
        }

        if (finalMessages.length > 0) {
          saveCurrentSession(personality, finalMessages);
        }
      } catch (err: unknown) {
        const errMsg = err instanceof Error ? err.message : String(err);
        setError(errMsg);
        setMessages((prev) => {
          const next = [...prev];
          const lastMsg = next[next.length - 1];
          if (lastMsg && lastMsg.role === "model" && lastMsg.content === "") {
            next.pop();
          }
          next.push({ role: "system", content: `[SYSTEM FAILURE]: ${errMsg}` });
          return next;
        });
      } finally {
        setIsGenerating(false);
      }
    },
    [saveCurrentSession]
  );

  const clearMessages = useCallback(() => setMessages([]), []);

  return {
    messages,
    setMessages,
    isGenerating,
    error,
    generate,
    clearMessages,
    sessions,
    deleteSession,
  };
}
