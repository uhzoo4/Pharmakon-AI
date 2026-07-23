"use client";

import { motion, useMotionTemplate, useMotionValue } from "framer-motion";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { Sparkles, BookOpen, Layers, Cpu } from "lucide-react";

export interface Personality {
  id: string;
  title: string;
  description: string;
  category: "philosophical" | "construct" | "monolith";
  badge?: string;
  specs?: string;
}

export const PERSONALITIES: Personality[] = [
  {
    id: "kafkaesque",
    title: "The Bureaucrat",
    description: "A labyrinthine logic engine. Highly constrained, analytical, and circular.",
    category: "philosophical",
    badge: "Kafka",
    specs: "3-Layer Transformer",
  },
  {
    id: "camus_stranger",
    title: "The Absurdist",
    description: "Coldly detached and direct. Existential, stark, and uncompromising.",
    category: "philosophical",
    badge: "Camus",
    specs: "3-Layer Transformer",
  },
  {
    id: "dark_romance",
    title: "The Doomed",
    description: "Feverish, emotionally volatile, and deeply existential Gothic prose.",
    category: "philosophical",
    badge: "Romanticism",
    specs: "3-Layer Transformer",
  },
  {
    id: "modern_conversation",
    title: "The Casual Talker",
    description: "Speaks in modern, conversational, everyday English with normal pacing.",
    category: "construct",
    badge: "Modern SFT",
    specs: "4-Layer Transformer",
  },
  {
    id: "the_assistant",
    title: "The AI Assistant",
    description: "A helpful instruction-tuned AI model trained to answer questions clearly.",
    category: "construct",
    badge: "Instruction Tuned",
    specs: "4-Layer Transformer",
  },
  {
    id: "the_pinnacle",
    title: "The Pinnacle",
    description: "Hyper-scaled 4-layer architecture trained on massive internet conversational data.",
    category: "monolith",
    badge: "High Capacity",
    specs: "4-Layer Monolith",
  },
  {
    id: "the_leviathan",
    title: "The Leviathan",
    description: "Multi-gigabyte 12-layer monolith. Stresses hardware limits with 10M+ characters of corpus data.",
    category: "monolith",
    badge: "12-Layer Titan",
    specs: "12-Layer Monolith",
  },
];

interface CarouselCardProps {
  person: Personality;
  index: number;
  hoveredIndex: number | null;
  setHoveredIndex: (index: number | null) => void;
  onSelect: (id: string) => void;
}

function CarouselCard({ person, index, hoveredIndex, setHoveredIndex, onSelect }: CarouselCardProps) {
  const isHovered = hoveredIndex === index;
  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  function handleMouseMove({ currentTarget, clientX, clientY }: React.MouseEvent) {
    const { left, top } = currentTarget.getBoundingClientRect();
    mouseX.set(clientX - left);
    mouseY.set(clientY - top);
  }

  return (
    <motion.div
      onMouseMove={handleMouseMove}
      onHoverStart={() => setHoveredIndex(index)}
      onHoverEnd={() => setHoveredIndex(null)}
      onClick={() => onSelect(person.id)}
      className={cn(
        "relative flex flex-col p-7 cursor-pointer overflow-hidden group select-none",
        "w-80 h-[27rem] bg-np-surface-2/90 backdrop-blur-md rounded-np-md border transition-all duration-300 ease-np-approach",
        isHovered
          ? "border-np-accent-ice shadow-[0_0_35px_rgba(142,202,230,0.18)]"
          : "border-np-border-hairline hover:border-np-border-visible"
      )}
      initial={{ rotateY: 0, z: 0 }}
      animate={{
        rotateY: isHovered ? 0 : hoveredIndex !== null ? (index < hoveredIndex ? 10 : -10) : 0,
        z: isHovered ? 40 : 0,
        scale: isHovered ? 1.04 : 1,
        opacity: hoveredIndex !== null && !isHovered ? 0.6 : 1,
      }}
      transition={{ duration: 0.45, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Shared Layout Ink Bleed Background */}
      {isHovered && (
        <motion.div
          layoutId="ink-bleed"
          className="absolute inset-0 bg-np-accent-steel/20 mix-blend-overlay rounded-np-md pointer-events-none"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.25 }}
          exit={{ opacity: 0 }}
        />
      )}

      {/* Interactive cursor glow (Ink Bleed Tracking) */}
      <motion.div
        className="pointer-events-none absolute -inset-px rounded-np-md opacity-0 transition duration-500 group-hover:opacity-100"
        style={{
          background: useMotionTemplate`
            radial-gradient(
              350px circle at ${mouseX}px ${mouseY}px,
              rgba(142, 202, 230, 0.2),
              transparent 80%
            )
          `,
        }}
      />

      {/* Card Header & Badges */}
      <div className="relative z-10 flex justify-between items-center w-full mb-auto">
        <span className="text-[10px] uppercase font-mono tracking-widest px-2.5 py-1 rounded-np-sm bg-np-surface-3 border border-np-border-hairline text-np-text-tertiary group-hover:text-np-accent-ice transition-colors">
          {person.badge || person.category}
        </span>
        {person.specs && (
          <span className="text-[10px] font-mono text-np-text-tertiary flex items-center gap-1">
            <Cpu className="w-3 h-3 text-np-accent-steel" />
            {person.specs}
          </span>
        )}
      </div>

      {/* Dark Gradient to ensure text readability */}
      <div className="absolute inset-0 bg-gradient-to-t from-np-surface-0 via-np-surface-0/60 to-transparent opacity-95 pointer-events-none" />

      <div className="relative z-10 flex flex-col h-full justify-end pointer-events-none pt-12">
        <div className="flex flex-col gap-3">
          <h3 className="text-np-text-primary font-serif text-3xl tracking-wide leading-tight group-hover:text-np-accent-ice transition-colors duration-300">
            {person.title}
          </h3>
          <p className="text-np-text-secondary text-sm font-sans leading-relaxed">
            {person.description}
          </p>

          <div className="mt-2 pt-3 border-t border-np-border-hairline flex items-center justify-between text-xs font-mono text-np-text-tertiary group-hover:text-np-text-secondary">
            <span className="flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-np-semantic-amber" />
              Summon Personality
            </span>
            <span>&rarr;</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export function Carousel({ onSelect }: { onSelect: (id: string) => void }) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [activeCategory, setActiveCategory] = useState<"all" | "philosophical" | "construct" | "monolith">("all");

  const filteredPersonalities = PERSONALITIES.filter((p) =>
    activeCategory === "all" ? true : p.category === activeCategory
  );

  return (
    <div className="relative flex flex-col items-center justify-center w-full h-full max-w-7xl px-4 py-8">
      {/* Category Tabs */}
      <div className="flex gap-2 mb-8 bg-np-surface-2 p-1.5 rounded-np-md border border-np-border-hairline z-20">
        {[
          { id: "all", label: "All Constructs", icon: BookOpen },
          { id: "philosophical", label: "Philosophers", icon: Sparkles },
          { id: "construct", label: "Assistants", icon: Layers },
          { id: "monolith", label: "Monoliths", icon: Cpu },
        ].map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveCategory(tab.id as typeof activeCategory)}
              className={cn(
                "px-4 py-2 rounded-np-sm text-xs font-mono tracking-wider transition-all flex items-center gap-2",
                activeCategory === tab.id
                  ? "bg-np-surface-3 text-np-accent-ice border border-np-border-hairline shadow-sm"
                  : "text-np-text-tertiary hover:text-np-text-secondary hover:bg-np-surface-1/50"
              )}
            >
              <Icon className="w-3.5 h-3.5" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Cards Grid / Carousel */}
      <div className="flex gap-6 items-center flex-wrap justify-center z-10" style={{ perspective: "1200px" }}>
        {filteredPersonalities.map((person, index) => (
          <CarouselCard
            key={person.id}
            person={person}
            index={index}
            hoveredIndex={hoveredIndex}
            setHoveredIndex={setHoveredIndex}
            onSelect={onSelect}
          />
        ))}
      </div>

      {/* Corpus Legends Footer */}
      <div className="mt-12 text-center text-[10px] uppercase font-mono tracking-widest text-np-text-tertiary opacity-50 select-none">
        <p className="mb-2 opacity-60">Architectural Foundations & Corpus Weights</p>
        <div className="flex gap-4 justify-center flex-wrap max-w-4xl opacity-80 leading-loose">
          Dostoevsky / Nietzsche / Kafka / Camus / Schopenhauer / Brontë / Ovid / Homer / Zola
        </div>
      </div>
    </div>
  );
}
