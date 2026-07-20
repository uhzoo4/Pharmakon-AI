"use client";

import { motion, useMotionTemplate, useMotionValue } from "framer-motion";
import { useState } from "react";
import { cn } from "@/lib/utils";

export const PERSONALITIES = [
  { 
    id: "kafkaesque", 
    title: "The Bureaucrat", 
    description: "A labyrinthine logic engine. Highly constrained, analytical, and circular.",
  },
  { 
    id: "camus_stranger", 
    title: "The Absurdist", 
    description: "Coldly detached and direct. (Limited corpus — output quality is degraded due to insufficient public-domain training data.)",
  },
  { 
    id: "dark_romance", 
    title: "The Doomed", 
    description: "Feverish, emotionally volatile, and deeply existential.",
  },
  {
    id: "modern_conversation",
    title: "The Casual Talker",
    description: "Speaks in modern, conversational, everyday English with normal pacing and slang.",
  },
  {
    id: "the_assistant",
    title: "The AI Assistant",
    description: "A helpful instruction-tuned AI model trained to answer questions (Claude-style SFT).",
  },
  {
    id: "the_pinnacle",
    title: "The Pinnacle",
    description: "Hyper-scaled 4-layer architecture trained on massive internet conversational data.",
  },
  {
    id: "the_leviathan",
    title: "The Leviathan",
    description: "Multi-gigabyte 12-layer monolith. Stresses hardware limits with 10M+ characters of internet data.",
  }
];

function CarouselCard({ person, index, hoveredIndex, setHoveredIndex, onSelect }: any) {
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
        "relative flex flex-col p-8 cursor-pointer overflow-hidden group",
        "w-72 h-[26rem] bg-np-surface-2 rounded-np-md border transition-all duration-300 ease-np-approach",
        isHovered ? "border-np-accent-ice shadow-[0_0_30px_rgba(92,122,148,0.15)]" : "border-np-border-hairline"
      )}
      initial={{ rotateY: 0, z: 0 }}
      animate={{
        rotateY: isHovered ? 0 : hoveredIndex !== null ? (index < hoveredIndex ? 15 : -15) : 0,
        z: isHovered ? 50 : 0,
        scale: isHovered ? 1.05 : 1,
        opacity: hoveredIndex !== null && !isHovered ? 0.5 : 1,
      }}
      transition={{ duration: 0.48, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Shared Layout Ink Bleed Background */}
      {isHovered && (
        <motion.div
          layoutId="ink-bleed"
          className="absolute inset-0 bg-np-accent-steel mix-blend-overlay rounded-np-md pointer-events-none"
          initial={{ opacity: 0 }}
          animate={{ opacity: 0.2 }}
          exit={{ opacity: 0 }}
        />
      )}

      {/* Interactive cursor glow (Ink Bleed Tracking) */}
      <motion.div
        className="pointer-events-none absolute -inset-px rounded-np-md opacity-0 transition duration-500 group-hover:opacity-100"
        style={{
          background: useMotionTemplate`
            radial-gradient(
              400px circle at ${mouseX}px ${mouseY}px,
              rgba(92, 122, 148, 0.25),
              transparent 80%
            )
          `,
        }}
      />
      
      {/* Dark Gradient to ensure text readability */}
      <div className="absolute inset-0 bg-gradient-to-t from-np-surface-0 via-transparent to-transparent opacity-90 pointer-events-none" />
      
      <div className="relative z-10 flex flex-col h-full pointer-events-none">
        <div className="mt-auto flex flex-col gap-3">
          <h3 className="text-np-text-primary font-serif text-3xl tracking-wide leading-tight group-hover:text-np-accent-ice transition-colors duration-500">
            {person.title}
          </h3>
          <p className="text-np-text-secondary text-sm font-sans leading-relaxed">
            {person.description}
          </p>
        </div>
      </div>
    </motion.div>
  );
}

export function Carousel({ onSelect }: { onSelect: (id: string) => void }) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <div className="relative flex flex-col items-center justify-center w-full h-full">
      {/* 3-Card Carousel */}
      <div className="flex gap-8 items-center" style={{ perspective: "1000px" }}>
        {PERSONALITIES.map((person, index) => (
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
      <div className="absolute bottom-8 text-center text-[10px] uppercase font-mono tracking-widest text-np-text-tertiary opacity-40 select-none">
        <p className="mb-2 opacity-50">Weights Trained On</p>
        <div className="flex gap-4 justify-center flex-wrap max-w-4xl opacity-80 leading-loose">
          Dostoevsky / Nietzsche / Kafka / Camus / Schopenhauer / Brontë / Ovid / Homer / Zola
        </div>
      </div>
    </div>
  );
}
