"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Settings2, Download } from "lucide-react";
import { Carousel, PERSONALITIES } from "@/components/vectors/Carousel";
import { PromptPanel } from "@/components/matrices/PromptPanel";
import { SettingsDrawer } from "@/components/matrices/SettingsDrawer";
import { Button } from "@/components/scalars/Button";
import { Gnomon } from "@/components/scalars/Gnomon";
import { cn } from "@/lib/utils";
import { usePharmakon } from "@/hooks/usePharmakon";

export default function Home() {
  const [activePersonality, setActivePersonality] = useState<string | null>(null);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [temperature, setTemperature] = useState(0.8);
  const [blacklist, setBlacklist] = useState("");
  const [language, setLanguage] = useState("EN");
  
  const { messages, isGenerating, generate, clearMessages } = usePharmakon();

  const handleGenerate = (prompt: string) => {
    if (!activePersonality) return;

    // Forcefully clamp the backend temperature to a maximum of 0.5.
    // Character-level models need very low temperature to produce coherent English.
    // The UI still visually glitches if the slider is set higher.
    const safeTemperature = Math.min(temperature, 0.5);

    generate(activePersonality, prompt, safeTemperature, blacklist);
  };

  const handleExtractPDF = () => {
    window.print();
  };

  return (
    <div className="flex flex-1 flex-row h-screen w-full overflow-hidden print:h-auto print:overflow-visible">
      {/* Left Rail (The Navigation / Session Log) */}
      <aside className="w-72 bg-np-surface-2 border-r border-np-border-hairline flex flex-col p-6 z-20 shadow-2xl relative print:hidden">
        <h1 className="text-np-text-primary font-medium tracking-wide text-lg flex justify-between items-center">
          φάρμακον
        </h1>
        <p className="text-np-text-secondary text-sm mt-1 font-mono">
          {activePersonality ? PERSONALITIES.find(p => p.id === activePersonality)?.title.toUpperCase() : "THE GRIMOIRE"}
        </p>
        
        {activePersonality && (
          <div className="mt-8 flex flex-col gap-4 flex-1">
            <h3 className="text-xs uppercase tracking-widest text-np-text-tertiary">Sessions</h3>
            <div className="flex flex-col gap-2">
              <div className="px-3 py-2 bg-np-surface-1 rounded-np-sm border border-np-border-hairline text-np-text-primary text-sm flex items-center gap-2">
                <Gnomon active={false} className="w-3 h-3 text-np-text-secondary" />
                Current Thread
              </div>
            </div>
            
            <div className="mt-auto pb-4">
               <Button 
                 variant="ghost" 
                 className="w-full justify-start text-xs text-np-semantic-red hover:text-np-semantic-red/80"
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
      <main className="flex-1 bg-np-surface-1 flex flex-col relative print:h-auto print:overflow-visible">
        <AnimatePresence mode="wait">
          {!activePersonality ? (
            <motion.div 
              key="landing"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05, filter: "blur(10px)" }}
              transition={{ duration: 0.72, ease: [0.16, 1, 0.3, 1] }}
              className="flex-1 flex flex-col items-center justify-center relative print:hidden"
            >
              <h2 className="text-np-text-tertiary text-sm font-mono tracking-[0.2em] uppercase mb-12 absolute top-1/4">
                Select Personality Construct
              </h2>
              <Carousel onSelect={setActivePersonality} />
            </motion.div>
          ) : (
            <motion.div 
              key="workspace"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.48, ease: [0.16, 1, 0.3, 1], delay: 0.2 }}
              className="flex-1 flex flex-col relative h-full print:h-auto"
            >
              {/* Header */}
              <header className="h-16 border-b border-np-border-hairline flex items-center justify-between px-8 bg-np-surface-1/80 backdrop-blur-sm z-10 absolute top-0 left-0 right-0 print:hidden">
                <div className="flex gap-2">
                  <span className="text-np-text-secondary font-mono text-xs border border-np-border-hairline rounded-np-sm px-2 py-1 bg-np-surface-2">
                    T={temperature.toFixed(2)}
                  </span>
                  {blacklist && (
                    <span className="text-np-semantic-amber font-mono text-xs border border-np-border-hairline rounded-np-sm px-2 py-1 bg-np-surface-2">
                      BL=[{blacklist}]
                    </span>
                  )}
                </div>
                <div className="flex gap-2">
                  <Button variant="secondary" className="h-8" onClick={handleExtractPDF}>
                    <Download className="w-4 h-4" /> Extract PDF
                  </Button>
                  <Button variant="ghost" onClick={() => setIsSettingsOpen(true)} className="h-8 w-8 p-0 rounded-full">
                    <Settings2 className="w-4 h-4" />
                  </Button>
                </div>
              </header>

              {/* Chat History */}
              <div className="flex-1 overflow-y-auto px-8 pt-24 pb-8 flex flex-col gap-8 max-w-4xl mx-auto w-full font-serif print:pt-0 print:overflow-visible">
                {messages.length === 0 && (
                  <div className="m-auto text-np-text-tertiary font-mono text-sm uppercase tracking-widest flex items-center gap-3 print:hidden">
                    <Gnomon active={true} /> System ready for input
                  </div>
                )}
                {messages.map((msg, i) => (
                  <div 
                    key={i} 
                    className={cn(
                      "flex flex-col gap-2 max-w-[80%]",
                      msg.role === "user" ? "ml-auto text-right" : "mr-auto text-left",
                      msg.role === "system" ? "mx-auto text-center max-w-full" : "",
                      "print:max-w-full print:mb-8"
                    )}
                  >
                    <span className="text-np-text-tertiary font-mono text-xs uppercase tracking-widest">
                      {msg.role === "user" ? "Seeker" : msg.role === "system" ? "Kernel" : PERSONALITIES.find(p => p.id === activePersonality)?.title}
                    </span>
                    <p className={cn(
                      "text-lg leading-relaxed whitespace-pre-wrap",
                      msg.role === "user" ? "text-np-text-secondary" : "text-np-text-primary",
                      msg.role === "system" ? "text-np-semantic-red font-mono text-sm" : "",
                      // Apply Glitch CSS if temperature is high and it's a model message
                      msg.role === "model" && temperature > 1.5 ? "glitch-text" : ""
                    )}>
                      {msg.content}
                      {msg.role === "model" && isGenerating && i === messages.length - 1 && (
                        <Gnomon className="ml-2 -mb-0.5 print:hidden" />
                      )}
                    </p>
                  </div>
                ))}
              </div>

              {/* Input Area */}
              <div className="p-8 pb-12 w-full flex justify-center bg-gradient-to-t from-np-surface-1 via-np-surface-1 to-transparent relative z-10 print:hidden">
                <PromptPanel onGenerate={handleGenerate} isGenerating={isGenerating} />
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
        blacklist={blacklist}
        setBlacklist={setBlacklist}
        language={language}
        setLanguage={setLanguage}
      />
    </div>
  );
}
