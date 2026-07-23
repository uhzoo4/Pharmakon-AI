"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X, Sliders, Ban, Globe, Gauge, Cpu, Sparkles } from "lucide-react";
import { Button } from "../scalars/Button";

interface SettingsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  temperature: number;
  setTemperature: (v: number) => void;
  topP: number;
  setTopP: (v: number) => void;
  topK: number;
  setTopK: (v: number) => void;
  maxTokens: number;
  setMaxTokens: (v: number) => void;
  blacklist: string;
  setBlacklist: (v: string) => void;
  language?: string;
  setLanguage?: (v: string) => void;
}

const LANGUAGES = [
  { id: "EN", label: "English" },
  { id: "FR", label: "Français" },
  { id: "DE", label: "Deutsch" },
  { id: "LA", label: "Latin" },
  { id: "GR", label: "Ancient Greek" },
];

const PRESETS = [
  { label: "Analytical", temp: 0.2, topP: 0.8, topK: 20 },
  { label: "Balanced", temp: 0.5, topP: 0.9, topK: 50 },
  { label: "Creative", temp: 0.8, topP: 0.95, topK: 80 },
  { label: "Chaos Lipogram", temp: 1.2, topP: 1.0, topK: 100 },
];

const QUICK_BLACKLIST_CHARS = ["e", "a", "t", "s", "i", "o", "n", "r"];

export function SettingsDrawer({
  isOpen,
  onClose,
  temperature,
  setTemperature,
  topP,
  setTopP,
  topK,
  setTopK,
  maxTokens,
  setMaxTokens,
  blacklist,
  setBlacklist,
  language = "EN",
  setLanguage,
}: SettingsDrawerProps) {
  const blacklistChars = blacklist
    .split(",")
    .map((c) => c.trim())
    .filter(Boolean);

  const toggleBlacklistChar = (char: string) => {
    if (blacklistChars.includes(char)) {
      const updated = blacklistChars.filter((c) => c !== char).join(", ");
      setBlacklist(updated);
    } else {
      const updated = [...blacklistChars, char].join(", ");
      setBlacklist(updated);
    }
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-np-surface-0/70 backdrop-blur-md z-40"
          />
          <motion.aside
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 26, stiffness: 220 }}
            className="fixed top-0 right-0 bottom-0 w-[26rem] bg-np-surface-2/95 border-l border-np-border-hairline z-50 p-6 flex flex-col shadow-2xl overflow-hidden"
          >
            <div className="flex justify-between items-center mb-6 pb-4 border-b border-np-border-hairline">
              <div className="flex items-center gap-2">
                <Sliders className="w-5 h-5 text-np-accent-ice" />
                <h2 className="text-np-text-primary font-medium tracking-wide text-lg">System Sampler Config</h2>
              </div>
              <Button variant="ghost" onClick={onClose} className="p-2 h-auto text-np-text-tertiary hover:text-np-text-primary">
                <X className="w-5 h-5" />
              </Button>
            </div>

            <div className="flex flex-col gap-8 overflow-y-auto pr-2 pb-12">
              {/* Presets */}
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-2 text-np-text-tertiary">
                  <Sparkles className="w-4 h-4 text-np-semantic-amber" />
                  <h3 className="text-xs uppercase tracking-widest font-mono">Sampling Presets</h3>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {PRESETS.map((p) => (
                    <button
                      key={p.label}
                      onClick={() => {
                        setTemperature(p.temp);
                        setTopP(p.topP);
                        setTopK(p.topK);
                      }}
                      className="px-3 py-2 rounded-np-sm bg-np-surface-1 border border-np-border-hairline hover:border-np-accent-steel text-xs font-mono text-left transition-all hover:bg-np-surface-3"
                    >
                      <span className="block text-np-text-primary font-sans font-medium">{p.label}</span>
                      <span className="text-[10px] text-np-text-tertiary">T={p.temp} | p={p.topP}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Temperature Settings */}
              <div className="flex flex-col gap-3">
                <div className="flex justify-between items-center text-np-text-tertiary">
                  <div className="flex items-center gap-2">
                    <Gauge className="w-4 h-4 text-np-accent-steel" />
                    <h3 className="text-xs uppercase tracking-widest font-mono">Temperature (T)</h3>
                  </div>
                  <span className="text-np-accent-ice font-mono text-sm font-semibold">{temperature.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.05"
                  max="2.0"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-np-accent-ice cursor-pointer"
                />
                <div className="flex justify-between text-np-text-tertiary font-mono text-[10px]">
                  <span>0.05 (Deterministic)</span>
                  <span className="text-np-semantic-amber">0.5 (Balanced)</span>
                  <span className="text-np-semantic-red">2.0 (Chaos)</span>
                </div>
              </div>

              {/* Top-P (Nucleus) & Top-K */}
              <div className="grid grid-cols-2 gap-4">
                <div className="flex flex-col gap-2">
                  <div className="flex justify-between items-center text-np-text-tertiary">
                    <h3 className="text-xs uppercase tracking-widest font-mono">Top-P</h3>
                    <span className="text-np-text-primary font-mono text-xs">{topP.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={topP}
                    onChange={(e) => setTopP(parseFloat(e.target.value))}
                    className="w-full accent-np-accent-steel cursor-pointer"
                  />
                </div>

                <div className="flex flex-col gap-2">
                  <div className="flex justify-between items-center text-np-text-tertiary">
                    <h3 className="text-xs uppercase tracking-widest font-mono">Top-K</h3>
                    <span className="text-np-text-primary font-mono text-xs">{topK}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="100"
                    step="1"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value))}
                    className="w-full accent-np-accent-steel cursor-pointer"
                  />
                </div>
              </div>

              {/* Max Tokens */}
              <div className="flex flex-col gap-3">
                <div className="flex justify-between items-center text-np-text-tertiary">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 text-np-accent-steel" />
                    <h3 className="text-xs uppercase tracking-widest font-mono">Max Tokens</h3>
                  </div>
                  <span className="text-np-text-primary font-mono text-xs font-semibold">{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="20"
                  max="500"
                  step="10"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                  className="w-full accent-np-accent-steel cursor-pointer"
                />
              </div>

              {/* Token Blacklist with Interactive Chips */}
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-2 text-np-text-tertiary">
                  <Ban className="w-4 h-4 text-np-semantic-red" />
                  <h3 className="text-xs uppercase tracking-widest font-mono">Constrained Blacklist</h3>
                </div>

                <input
                  type="text"
                  value={blacklist}
                  onChange={(e) => setBlacklist(e.target.value)}
                  placeholder="e, a, t..."
                  className="w-full bg-np-surface-1 border border-np-border-hairline rounded-np-sm p-3 text-np-text-primary font-mono text-sm focus:outline-none focus:border-np-accent-steel transition-colors"
                />

                <div className="flex flex-wrap gap-1.5 pt-1">
                  <span className="text-[11px] font-mono text-np-text-tertiary w-full">Quick Lipogram Toggles:</span>
                  {QUICK_BLACKLIST_CHARS.map((char) => {
                    const isSelected = blacklistChars.includes(char);
                    return (
                      <button
                        key={char}
                        onClick={() => toggleBlacklistChar(char)}
                        className={`w-7 h-7 rounded-np-sm border font-mono text-xs transition-all flex items-center justify-center ${
                          isSelected
                            ? "bg-np-semantic-red/20 border-np-semantic-red text-np-semantic-red font-bold shadow-sm"
                            : "bg-np-surface-1 border-np-border-hairline text-np-text-tertiary hover:border-np-border-visible"
                        }`}
                      >
                        {char}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Interface Language */}
              <div className="flex flex-col gap-3">
                <div className="flex items-center gap-2 text-np-text-tertiary">
                  <Globe className="w-4 h-4" />
                  <h3 className="text-xs uppercase tracking-widest font-mono">Interface Tongue</h3>
                </div>
                <div className="flex flex-wrap gap-2">
                  {LANGUAGES.map((lang) => (
                    <button
                      key={lang.id}
                      onClick={() => setLanguage && setLanguage(lang.id)}
                      className={`px-3 py-1.5 rounded-np-sm border text-xs font-mono transition-colors ${
                        language === lang.id
                          ? "bg-np-accent-steel/20 border-np-accent-steel text-np-accent-ice font-semibold"
                          : "bg-np-surface-1 border-np-border-hairline text-np-text-secondary hover:border-np-border-visible"
                      }`}
                    >
                      {lang.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
