"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { SendHorizontal } from "lucide-react";
import { Button } from "../scalars/Button";

interface PromptPanelProps {
  onGenerate: (prompt: string) => void;
  isGenerating: boolean;
}

export function PromptPanel({ onGenerate, isGenerating }: PromptPanelProps) {
  const [prompt, setPrompt] = useState("");

  const handleSubmit = () => {
    if (prompt.trim() && !isGenerating) {
      onGenerate(prompt);
      setPrompt("");
    }
  };

  return (
    <div className="w-full max-w-3xl flex flex-col gap-2 relative">
      <motion.div
        animate={isGenerating ? { boxShadow: ["0px 0px 0px rgba(142,202,230,0)", "0px 0px 40px rgba(142,202,230,0.15)", "0px 0px 0px rgba(142,202,230,0)"] } : { boxShadow: "0px 0px 0px rgba(142,202,230,0)" }}
        transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        className="relative w-full rounded-np-md"
      >
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSubmit();
            }
          }}
          disabled={isGenerating}
          placeholder="The sun rises, yet..."
          className="w-full min-h-[140px] bg-np-surface-1 border border-np-border-visible rounded-np-md p-5 text-np-text-primary text-lg resize-none focus:outline-none focus:border-np-accent-steel-bright transition-colors disabled:opacity-50"
        />
      </motion.div>
      <div className="absolute bottom-4 right-4 z-10">
        <Button
          onClick={handleSubmit}
          disabled={!prompt.trim() || isGenerating}
          variant="primary"
          className="h-10 w-10 p-0 rounded-full"
        >
          <SendHorizontal className="w-5 h-5" />
        </Button>
      </div>
    </div>
  );
}
