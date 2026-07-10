import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

export function Gnomon({ className, active = true }: { className?: string; active?: boolean }) {
  return (
    <motion.svg
      width="12"
      height="12"
      viewBox="0 0 12 12"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={cn("text-np-accent-ice inline-block", className)}
      animate={active ? { opacity: [1, 0.3, 1] } : { opacity: 0 }}
      transition={active ? { duration: 1.5, repeat: Infinity, ease: "easeInOut" } : {}}
    >
      <path
        d="M2 10L10 2M10 2V8M10 2H4"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="square"
        strokeLinejoin="miter"
      />
    </motion.svg>
  );
}
