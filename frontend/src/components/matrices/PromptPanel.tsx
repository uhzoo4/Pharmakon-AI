"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { SendHorizontal, Sparkles, X } from "lucide-react";
import { Button } from "../scalars/Button";

interface PromptPanelProps {
  onGenerate: (prompt: string) => void;
  isGenerating: boolean;
  activePersonalityId?: string | null;
}

const SAMPLE_PROMPTS: Record<string, string[]> = {
  kafkaesque: [
    "The official notification arrived before sunrise...",
    "The inspector demanded proof of authorization for...",
    "To appeal the decree, one must navigate...",
  ],
  camus_stranger: [
    "The sunlight hit the beach with unbearable intensity...",
    "It made no difference whether I stayed or left...",
    "Mother died today. Or maybe yesterday, I don't know...",
  ],
  dark_romance: [
    "Under the shadow of the cathedral ruins...",
    "Her portrait hung in the dimly lit hall...",
    "The silence was broken only by the tempest outside...",
  ],
  default: [
    "The sun rises, yet...",
    "Explain the nature of...",
    "In the quiet hours of night...",
  ],
};

export function PromptPanel({ onGenerate, isGenerating, activePersonalityId }: PromptPanelProps) {
  const [prompt, setPrompt] = useState("");

  const handleSubmit = () => {
    if (prompt.trim() && !isGenerating) {
      onGenerate(prompt);
      setPrompt("");
    }
  };

  const suggestions = (activePersonalityId && SAMPLE_PROMPTS[activePersonalityId]) || SAMPLE_PROMPTS.default;

  return (
    <div className="w-full max-w-3xl flex flex-col gap-3 relative">
      {/* Quick Suggestion Chips */}
      {prompt === "" && !isGenerating && (
        <div className="flex items-center gap-2 overflow-x-auto pb-1 max-w-full text-xs font-mono">
          <span className="text-np-text-tertiary flex items-center gap-1 shrink-0">
            <Sparkles className="w-3 h-3 text-np-semantic-amber" />
            Prompt hints:
          </span>
          {suggestions.map((s, idx) => (
            <button
              key={idx}
              onClick={() => setPrompt(s)}
              className="px-2.5 py-1 bg-np-surface-2 hover:bg-np-surface-3 border border-np-border-hairline hover:border-np-border-visible text-np-text-secondary rounded-np-sm transition-colors whitespace-nowrap shrink-0"
            >
              {s.slice(0, 32)}...
            </button>
          ))}
        </div>
      )}

      {/* Main Textarea Container */}
      <motion.div
        animate={
          isGenerating
            ? {
                boxShadow: [
                  "0px 0px 0px rgba(142,202,230,0)",
                  "0px 0px 30px rgba(142,202,230,0.15)",
                  "0px 0px 0px rgba(142,202,230,0)",
                ],
              }
            : { boxShadow: "0px 0px 0px rgba(142,202,230,0)" }
        }
        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        className="relative w-full rounded-np-md"
      >
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value.slice(0, 500))}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          }}
          disabled={isGenerating}
          placeholder="The sun rises, yet..."
          className="w-full min-h-[130px] max-h-[50vh] overflow-y-auto bg-np-surface-2/90 border border-np-border-visible rounded-np-md p-5 pb-12 text-np-text-primary text-base font-serif resize-none focus:outline-none focus:border-np-accent-steel-bright transition-all disabled:opacity-50"
        />

        {/* Footer controls inside textarea container */}
        <div className="absolute bottom-3 left-4 right-4 flex justify-between items-center z-10 pointer-events-none">
          <div className="flex items-center gap-3 pointer-events-auto">
            <span className="text-[11px] font-mono text-np-text-tertiary">
              {prompt.length}/500 chars
            </span>
            {prompt.length > 0 && !isGenerating && (
              <button
                onClick={() => setPrompt("")}
                className="text-np-text-tertiary hover:text-np-semantic-red text-xs font-mono flex items-center gap-0.5 transition-colors"
              >
                <X className="w-3 h-3" /> Clear
              </button>
            )}
          </div>

          <div className="pointer-events-auto">
            <Button
              onClick={handleSubmit}
              disabled={!prompt.trim() || isGenerating}
              variant="primary"
              className="h-9 w-9 p-0 rounded-full flex items-center justify-center shadow-lg"
            >
              <SendHorizontal className="w-4.5 h-4.5" />
            </Button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
