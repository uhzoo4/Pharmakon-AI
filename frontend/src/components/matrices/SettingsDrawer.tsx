"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X, Settings2 } from "lucide-react";
import { Button } from "../scalars/Button";

interface SettingsDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  temperature: number;
  setTemperature: (v: number) => void;
  blacklist: string;
  setBlacklist: (v: string) => void;
}

export function SettingsDrawer({
  isOpen, onClose, temperature, setTemperature, blacklist, setBlacklist
}: SettingsDrawerProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
            className="fixed inset-0 bg-np-surface-0/60 backdrop-blur-md z-40"
            onClick={onClose}
          />

          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ duration: 0.48, ease: [0.16, 1, 0.3, 1] }}
            className="fixed right-0 top-0 bottom-0 w-96 bg-np-surface-3 border-l border-np-border-hairline z-50 p-6 flex flex-col gap-8 shadow-2xl"
          >
            <div className="flex justify-between items-center">
              <h2 className="text-np-text-primary text-lg font-medium flex items-center gap-2">
                <Settings2 className="w-5 h-5 text-np-accent-steel" />
                Parameters
              </h2>
              <Button variant="ghost" onClick={onClose} className="p-2 rounded-full h-8 w-8">
                <X className="w-4 h-4" />
              </Button>
            </div>

            <div className="flex flex-col gap-6">
              <div className="flex flex-col gap-2">
                <label className="text-np-text-secondary text-sm font-mono flex justify-between">
                  <span>Temperature</span>
                  <span className="text-np-accent-ice">{temperature.toFixed(2)}</span>
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="2.0"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-np-accent-steel"
                />
                <p className="text-np-text-tertiary text-xs">
                  High temperature causes linguistic dissolution.
                </p>
              </div>

              <div className="flex flex-col gap-2">
                <label className="text-np-text-secondary text-sm font-mono">
                  Blacklist (CSV)
                </label>
                <input
                  type="text"
                  value={blacklist}
                  onChange={(e) => setBlacklist(e.target.value)}
                  placeholder="e, t, s"
                  className="w-full bg-np-surface-2 border border-np-border-visible rounded-np-sm p-3 text-np-text-primary font-mono text-sm focus:outline-none focus:border-np-accent-ice transition-colors"
                />
                <p className="text-np-text-tertiary text-xs">
                  Lipogram constraints. Characters separated by commas.
                </p>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
