"use client";

import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Settings2, Download, Copy, Check, FileText, Trash2, RotateCcw } from "lucide-react";
import { Carousel, PERSONALITIES } from "@/components/vectors/Carousel";
import { PromptPanel } from "@/components/matrices/PromptPanel";
import { SettingsDrawer } from "@/components/matrices/SettingsDrawer";
import { Button } from "@/components/scalars/Button";
import { Gnomon } from "@/components/scalars/Gnomon";
import { cn } from "@/lib/utils";
import { usePharmakon, Token } from "@/hooks/usePharmakon";

const TokenStream = ({ tokens }: { tokens: Token[] }) => {
  return (
    <span className="break-words leading-loose">
      {tokens.map((t, i) => (
        <span
          key={i}
          className="group relative cursor-crosshair hover:bg-np-accent-ice/20 transition-colors inline-block whitespace-pre-wrap rounded-px px-0.5"
        >
          {t.char}
          {t.alts && t.alts.length > 0 && (
            <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:flex flex-col bg-np-surface-3/95 backdrop-blur-md border border-np-border-hairline p-3 rounded-np-sm shadow-2xl text-xs z-50 min-w-[14rem] pointer-events-none font-mono">
              <span className="text-np-text-tertiary mb-2 uppercase tracking-widest text-[10px] flex justify-between border-b border-np-border-hairline pb-1">
                <span>Top-K Predictions</span>
                <span className="text-np-accent-ice">Prob %</span>
              </span>
              {t.alts.slice(0, 5).map((a, j) => {
                const charLabel = a.char === " " ? "␣ (space)" : a.char === "\n" ? "↵ (newline)" : a.char;
                const percent = (a.prob * 100).toFixed(1);
                return (
                  <div key={j} className="flex flex-col gap-0.5 mb-1.5 last:mb-0">
                    <div className="flex justify-between items-center text-[11px]">
                      <span className="text-np-semantic-amber font-bold">&apos;{charLabel}&apos;</span>
                      <span className="text-np-text-secondary">{percent}%</span>
                    </div>
                    <div className="w-full bg-np-surface-1 h-1 rounded-full overflow-hidden">
                      <div
                        className="bg-np-accent-steel-bright h-full rounded-full transition-all"
                        style={{ width: `${Math.min(100, Math.max(2, parseFloat(percent)))}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </span>
          )}
        </span>
      ))}
    </span>
  );
};

export default function Home() {
  const [activePersonality, setActivePersonality] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [temperature, setTemperature] = useState(0.5);
  const [topP, setTopP] = useState(0.9);
  const [topK, setTopK] = useState(50);
  const [maxTokens, setMaxTokens] = useState(250);
  const [blacklist, setBlacklist] = useState("");
  const [language, setLanguage] = useState("EN");
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  const chatContainerRef = useRef<HTMLDivElement>(null);

  const {
    messages,
    setMessages,
    isGenerating,
    generate,
    clearMessages,
    sessions,
    deleteSession,
  } = usePharmakon();

  // Auto-scroll chat to bottom on new tokens
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, isGenerating]);

  const handleGenerate = (prompt: string) => {
    if (!activePersonality) return;
    generate(activePersonality, prompt, {
      temperature,
      topP,
      topK,
      maxTokens,
      blacklistStr: blacklist,
    });
  };

  const handleCopyMessage = (content: string, index: number) => {
    navigator.clipboard.writeText(content);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const handleRegenerate = () => {
    if (messages.length < 2 || isGenerating || !activePersonality) return;
    // Find last user prompt
    const lastUserIndex = [...messages].reverse().findIndex((m) => m.role === "user");
    if (lastUserIndex === -1) return;
    const actualIndex = messages.length - 1 - lastUserIndex;
    const lastPrompt = messages[actualIndex].content;

    // Truncate messages up to that user prompt
    setMessages((prev) => prev.slice(0, actualIndex));
    handleGenerate(lastPrompt);
  };

  const handleExportMarkdown = () => {
    if (messages.length === 0) return;
    const activeTitle = PERSONALITIES.find((p) => p.id === activePersonality)?.title || "Pharmakon AI";
    let md = `# Pharmakon AI Transcript - ${activeTitle}\n\n`;
    messages.forEach((m) => {
      const roleName = m.role === "user" ? "Seeker" : m.role === "model" ? activeTitle : "System Kernel";
      md += `### ${roleName}\n${m.content}\n\n`;
    });

    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pharmakon_${activePersonality}_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExtractPDF = () => {
    window.print();
  };

  const activePersonaObj = PERSONALITIES.find((p) => p.id === activePersonality);

  return (
    <div className="flex flex-1 flex-row h-screen w-full overflow-hidden grimoire-bg-mesh print:h-auto print:overflow-visible">
      {/* Left Rail (Navigation / Session Log) */}
      <aside className="w-80 bg-np-surface-2/95 backdrop-blur-md border-r border-np-border-hairline flex flex-col p-6 z-20 shadow-2xl relative print:hidden">
        <div className="flex items-center justify-between">
          <h1 className="text-np-text-primary font-serif font-bold tracking-widest text-xl flex items-center gap-2">
            <span className="text-np-accent-ice">φάρμακον</span>
          </h1>
          <span className="text-[10px] font-mono uppercase tracking-widest px-2 py-0.5 rounded bg-np-surface-3 border border-np-border-hairline text-np-text-tertiary">
            v1.0
          </span>
        </div>

        <p className="text-np-text-tertiary text-xs mt-1 font-mono uppercase tracking-wider">
          {activePersonaObj ? activePersonaObj.title : "The Grimoire"}
        </p>

        {activePersonality && (
          <div className="mt-6 flex flex-col gap-4 flex-1 overflow-hidden">
            <div className="flex justify-between items-center">
              <h3 className="text-[10px] uppercase tracking-widest text-np-text-tertiary font-mono">
                Saved Threads ({sessions.length})
              </h3>
            </div>

            <div className="flex flex-col gap-2 overflow-y-auto pr-1 flex-1">
              <div className="px-3 py-2 bg-np-surface-3/80 rounded-np-sm border border-np-accent-steel/40 text-np-accent-ice text-xs font-mono flex items-center justify-between shadow-sm">
                <span className="flex items-center gap-2 truncate">
                  <Gnomon active={isGenerating} className="w-3 h-3 shrink-0 text-np-accent-ice" />
                  <span className="truncate">Current Active Thread</span>
                </span>
                <span className="w-2 h-2 rounded-full bg-np-semantic-amber animate-pulse shrink-0" />
              </div>

              {sessions.map((s) => (
                <div
                  key={s.id}
                  onClick={() => {
                    setActivePersonality(s.personalityId);
                    setMessages(s.messages);
                  }}
                  className="group px-3 py-2 bg-np-surface-1 hover:bg-np-surface-3 rounded-np-sm border border-np-border-hairline text-np-text-secondary hover:text-np-text-primary text-xs font-sans cursor-pointer transition-all flex items-center justify-between"
                >
                  <div className="flex flex-col truncate">
                    <span className="truncate font-medium">{s.title}</span>
                    <span className="text-[9px] font-mono text-np-text-tertiary">
                      {new Date(s.timestamp).toLocaleDateString()}
                    </span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteSession(s.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 text-np-text-tertiary hover:text-np-semantic-red p-1 transition-opacity"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              ))}
            </div>

            <div className="mt-auto pt-4 border-t border-np-border-hairline flex flex-col gap-2">
              <Button
                variant="ghost"
                className="w-full justify-start text-xs text-np-semantic-red hover:text-np-semantic-red/80 hover:bg-np-semantic-red/10"
                onClick={() => {
                  setActivePersonality(null);
                  clearMessages();
                }}
              >
                Sever Connection
              </Button>
            </div>
          </div>
        )}
      </aside>

      {/* Main Workspace (The Canvas) */}
      <main className="flex-1 bg-np-surface-1/70 flex flex-col relative print:h-auto print:overflow-visible">
        <AnimatePresence mode="wait">
          {!activePersonality ? (
            <motion.div
              key="landing"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.04, filter: "blur(10px)" }}
              transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
              className="flex-1 flex flex-col items-center justify-center relative print:hidden overflow-y-auto"
            >
              <div className="text-center mb-6">
                <span className="text-np-accent-ice text-xs font-mono tracking-[0.3em] uppercase block mb-2">
                  Pharmakon Neural Engine
                </span>
                <h2 className="text-np-text-primary text-3xl font-serif">Select Personality Construct</h2>
              </div>
              <Carousel onSelect={setActivePersonality} />
            </motion.div>
          ) : (
            <motion.div
              key="workspace"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              className="flex-1 flex flex-col relative h-full print:h-auto overflow-hidden"
            >
              {/* Header Bar */}
              <header className="h-16 border-b border-np-border-hairline flex items-center justify-between px-8 bg-np-surface-2/80 backdrop-blur-md z-10 absolute top-0 left-0 right-0 print:hidden">
                <div className="flex items-center gap-3">
                  <span className="text-np-text-primary font-serif font-medium text-lg">
                    {activePersonaObj?.title}
                  </span>
                  <div className="flex gap-1.5 items-center">
                    <span className="text-np-text-secondary font-mono text-[11px] border border-np-border-hairline rounded-np-sm px-2 py-0.5 bg-np-surface-3">
                      T={temperature.toFixed(2)}
                    </span>
                    <span className="text-np-text-secondary font-mono text-[11px] border border-np-border-hairline rounded-np-sm px-2 py-0.5 bg-np-surface-3">
                      p={topP.toFixed(2)}
                    </span>
                    {blacklist && (
                      <span className="text-np-semantic-amber font-mono text-[11px] border border-np-border-hairline rounded-np-sm px-2 py-0.5 bg-np-surface-3">
                        BL=[{blacklist}]
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Button variant="ghost" className="h-8 text-xs font-mono" onClick={handleExportMarkdown}>
                    <FileText className="w-3.5 h-3.5 mr-1" /> Export .MD
                  </Button>
                  <Button variant="secondary" className="h-8 text-xs font-mono" onClick={handleExtractPDF}>
                    <Download className="w-3.5 h-3.5 mr-1" /> PDF
                  </Button>
                  <Button
                    variant="ghost"
                    onClick={() => setIsSettingsOpen(true)}
                    className="h-8 w-8 p-0 rounded-full hover:bg-np-surface-3"
                  >
                    <Settings2 className="w-4 h-4 text-np-accent-ice" />
                  </Button>
                </div>
              </header>

              {/* Chat History Messages */}
              <div
                ref={chatContainerRef}
                className="flex-1 overflow-y-auto px-8 pt-24 pb-12 flex flex-col gap-8 max-w-4xl mx-auto w-full font-serif print:pt-0 print:overflow-visible"
              >
                {messages.length === 0 && (
                  <div className="m-auto text-np-text-tertiary font-mono text-sm uppercase tracking-widest flex items-center gap-3 print:hidden py-16">
                    <Gnomon active={true} /> System ready for input
                  </div>
                )}
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={cn(
                      "flex flex-col gap-2 max-w-[85%] relative group",
                      msg.role === "user" ? "ml-auto text-right" : "mr-auto text-left",
                      msg.role === "system" ? "mx-auto text-center max-w-full" : "",
                      "print:max-w-full print:mb-8"
                    )}
                  >
                    <div className="flex items-center justify-between text-xs font-mono uppercase tracking-widest text-np-text-tertiary">
                      <span>
                        {msg.role === "user"
                          ? "Seeker"
                          : msg.role === "system"
                          ? "Kernel"
                          : activePersonaObj?.title}
                      </span>
                      {msg.role !== "system" && (
                        <div className="opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-2 print:hidden">
                          <button
                            onClick={() => handleCopyMessage(msg.content, i)}
                            className="text-np-text-tertiary hover:text-np-accent-ice transition-colors p-1"
                            title="Copy text"
                          >
                            {copiedIndex === i ? (
                              <Check className="w-3.5 h-3.5 text-np-accent-ice" />
                            ) : (
                              <Copy className="w-3.5 h-3.5" />
                            )}
                          </button>
                          {msg.role === "model" && i === messages.length - 1 && !isGenerating && (
                            <button
                              onClick={handleRegenerate}
                              className="text-np-text-tertiary hover:text-np-accent-ice transition-colors p-1"
                              title="Regenerate response"
                            >
                              <RotateCcw className="w-3.5 h-3.5" />
                            </button>
                          )}
                        </div>
                      )}
                    </div>

                    <div
                      className={cn(
                        "text-lg leading-relaxed whitespace-pre-wrap p-4 rounded-np-md border shadow-sm",
                        msg.role === "user"
                          ? "bg-np-surface-2 border-np-border-hairline text-np-text-secondary"
                          : msg.role === "system"
                          ? "bg-np-semantic-red/10 border-np-semantic-red/30 text-np-semantic-red font-mono text-sm"
                          : "bg-np-surface-3/60 border-np-border-hairline text-np-text-primary",
                        msg.role === "model" && temperature > 1.5 ? "glitch-text" : ""
                      )}
                    >
                      {msg.tokens && msg.tokens.length > 0 ? (
                        <TokenStream tokens={msg.tokens} />
                      ) : (
                        msg.content
                      )}
                      {msg.role === "model" && isGenerating && i === messages.length - 1 && (
                        <Gnomon className="ml-2 -mb-0.5 inline-block print:hidden text-np-accent-ice" />
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Input Area */}
              <div className="p-6 pb-10 w-full flex justify-center bg-gradient-to-t from-np-surface-1 via-np-surface-1/90 to-transparent relative z-10 print:hidden">
                <PromptPanel
                  onGenerate={handleGenerate}
                  isGenerating={isGenerating}
                  activePersonalityId={activePersonality}
                />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <SettingsDrawer
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        temperature={temperature}
        setTemperature={setTemperature}
        topP={topP}
        setTopP={setTopP}
        topK={topK}
        setTopK={setTopK}
        maxTokens={maxTokens}
        setMaxTokens={setMaxTokens}
        blacklist={blacklist}
        setBlacklist={setBlacklist}
        language={language}
        setLanguage={setLanguage}
      />
    </div>
  );
}
