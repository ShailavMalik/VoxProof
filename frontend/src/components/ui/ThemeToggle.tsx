import { motion } from "framer-motion";
import { Sun, Moon } from "lucide-react";
import { useTheme } from "@/components/providers/ThemeProvider";

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <motion.button
      onClick={toggleTheme}
      className="relative p-3 rounded-xl glass-card-hover"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
      <motion.div
        initial={false}
        animate={{
          rotate: theme === "dark" ? 0 : 180,
        }}
        transition={{ duration: 0.5, ease: "easeInOut" }}>
        {theme === "dark" ?
          <Moon className="w-5 h-5 text-neon-cyan" />
        : <Sun className="w-5 h-5 text-amber-500" />}
      </motion.div>

      {/* Glow effect */}
      <motion.div
        className="absolute inset-0 rounded-xl opacity-0"
        animate={{
          opacity: [0, 0.5, 0],
          scale: [1, 1.2, 1],
        }}
        transition={{
          duration: 2,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        style={{
          background:
            theme === "dark" ?
              "radial-gradient(circle, rgba(0, 245, 255, 0.3) 0%, transparent 70%)"
            : "radial-gradient(circle, rgba(251, 191, 36, 0.3) 0%, transparent 70%)",
        }}
      />
    </motion.button>
  );
}
