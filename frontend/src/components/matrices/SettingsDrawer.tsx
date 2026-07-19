"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X, Sliders, Ban, Globe } from "lucide-react";
import { Button } from "../scalars/Button";

interface SettingsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  temperature: number;
  setTemperature: (v: number) => void;
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

export function SettingsDrawer({ 
  isOpen, 
  onClose, 
  temperature, 
  setTemperature, 
  blacklist, 
  setBlacklist,
  language = "EN",
  setLanguage
}: SettingsDrawerProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-np-surface-0/60 backdrop-blur-sm z-40"
          />
          <motion.aside
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="fixed top-0 right-0 bottom-0 w-96 bg-np-surface-2 border-l border-np-border-hairline z-50 p-6 flex flex-col shadow-2xl"
          >
            <div className="flex justify-between items-center mb-8">
              <h2 className="text-np-text-primary font-medium tracking-wide text-lg">System Config</h2>
              <Button variant="ghost" onClick={onClose} className="p-2 h-auto">
                <X className="w-5 h-5" />
              </Button>
            </div>

            <div className="flex flex-col gap-10 overflow-y-auto pr-2 pb-12">
              {/* Language Settings */}
              <div className="flex flex-col gap-4">
                <div className="flex items-center gap-2 text-np-text-tertiary">
                  <Globe className="w-4 h-4" />
                  <h3 className="text-xs uppercase tracking-widest">Interface Tongue</h3>
                </div>
                <div className="flex flex-wrap gap-2">
                  {LANGUAGES.map((lang) => (
                    <button
                      key={lang.id}
                      onClick={() => setLanguage && setLanguage(lang.id)}
                      className={`px-3 py-1.5 rounded-np-sm border text-sm font-mono transition-colors ${
                        language === lang.id 
                          ? "bg-np-accent-steel/10 border-np-accent-steel text-np-accent-steel-bright" 
                          : "bg-np-surface-1 border-np-border-hairline text-np-text-secondary hover:border-np-border-visible"
                      }`}
                    >
                      {lang.label}
                    </button>
                  ))}
                </div>
                <p className="text-np-text-tertiary text-xs leading-relaxed font-sans">
                  Sets the linguistic context for the Seeker interface and model prompting hints.
                </p>
              </div>

              {/* Temperature Settings */}
              <div className="flex flex-col gap-4">
                <div className="flex items-center gap-2 text-np-text-tertiary">
                  <Sliders className="w-4 h-4" />
                  <h3 className="text-xs uppercase tracking-widest">Temperature (T)</h3>
                </div>
                
                <div className="flex flex-col gap-2">
                  <input 
                    type="range" 
                    min="0.1" 
                    max="2.0" 
                    step="0.1"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full accent-np-accent-steel"
                  />
                  <div className="flex justify-between text-np-text-tertiary font-mono text-xs">
                    <span>0.1 (Rigid)</span>
                    <span className="text-np-text-primary">{temperature.toFixed(2)}</span>
                    <span className="text-np-semantic-red">2.0 (Chaos)</span>
                  </div>
                </div>
                <p className="text-np-text-tertiary text-xs leading-relaxed font-sans">
                  Controls the thermodynamic probability of the sampler. High temperature causes linguistic dissolution.
                </p>
              </div>

              {/* Blacklist Settings */}
              <div className="flex flex-col gap-4">
                <div className="flex items-center gap-2 text-np-text-tertiary">
                  <Ban className="w-4 h-4" />
                  <h3 className="text-xs uppercase tracking-widest">Token Blacklist</h3>
                </div>
                
                <input 
                  type="text" 
                  value={blacklist}
                  onChange={(e) => setBlacklist(e.target.value)}
                  placeholder="e, a, t..."
                  className="w-full bg-np-surface-1 border border-np-border-hairline rounded-np-sm p-3 text-np-text-primary font-mono text-sm focus:outline-none focus:border-np-accent-steel transition-colors"
                />
                <p className="text-np-text-tertiary text-xs leading-relaxed font-sans">
                  Comma-separated characters to ban from generation. Useful for forced constrained writing (e.g., lipograms).
                </p>
              </div>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
