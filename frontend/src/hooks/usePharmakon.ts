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

export function usePharmakon() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const generate = useCallback(async (
    personality: string,
    prompt: string,
    temperature: number,
    blacklistStr: string
  ) => {
    // Append user message
    setMessages((prev) => [...prev, { role: "user", content: prompt }]);
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
          blacklist,
          max_tokens: 300,
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
              const newToken = { char: data.text, alts: data.alts || [] };
              setMessages((prev) => {
                const next = [...prev];
                const lastMsg = { ...next[next.length - 1] };
                lastMsg.content = currentText;
                lastMsg.tokens = [...(lastMsg.tokens || []), newToken];
                next[next.length - 1] = lastMsg;
                return next;
              });
            }
          } catch (e) {
            console.error("Failed to parse SSE JSON:", e);
          }
        }
      }
    } catch (err: any) {
      setError(err.message);
      setMessages((prev) => {
        // If the model message is empty, remove it and add the error
        let next = [...prev];
        if (next[next.length - 1].role === "model" && next[next.length - 1].content === "") {
          next.pop();
        }
        next.push({ role: "system", content: `[SYSTEM FAILURE]: ${err.message}` });
        return next;
      });
    } finally {
      setIsGenerating(false);
    }
  }, []);

  const clearMessages = useCallback(() => setMessages([]), []);

  return { messages, isGenerating, error, generate, clearMessages };
}
