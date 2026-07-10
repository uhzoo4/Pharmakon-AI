"use client";

import { cn } from "@/lib/utils";
import { motion, HTMLMotionProps } from "framer-motion";
import { forwardRef } from "react";

interface ButtonProps extends HTMLMotionProps<"button"> {
  variant?: "primary" | "secondary" | "ghost";
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "secondary", children, ...props }, ref) => {
    return (
      <motion.button
        ref={ref}
        whileHover={{ backgroundColor: variant === "ghost" ? "var(--color-np-surface-2)" : undefined }}
        whileTap={{ scale: 0.99, transition: { duration: 0.1 } }}
        className={cn(
          "inline-flex items-center justify-center gap-2 whitespace-nowrap",
          "rounded-np-sm px-4 py-2 text-sm font-medium transition-colors duration-np-fast",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-np-accent-ice focus-visible:ring-offset-2 focus-visible:ring-offset-np-surface-0",
          "disabled:pointer-events-none disabled:opacity-50",
          variant === "primary" && "bg-np-accent-steel text-np-surface-0 hover:bg-np-accent-steel-bright",
          variant === "secondary" && "bg-np-surface-1 text-np-text-primary border border-np-border-visible hover:bg-np-surface-2 hover:border-np-border-hairline",
          variant === "ghost" && "bg-transparent text-np-text-secondary hover:text-np-text-primary",
          className
        )}
        {...props}
      >
        {children}
      </motion.button>
    );
  }
);
Button.displayName = "Button";
