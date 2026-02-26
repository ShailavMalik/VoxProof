import { motion } from "framer-motion";
import { Shield } from "lucide-react";

export function PageLoader() {
  return (
    <motion.div
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-light-50 dark:bg-dark-900">
      <div className="flex flex-col items-center gap-4">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className="relative">
          <Shield className="w-16 h-16 text-neon-cyan" />
          <div className="absolute inset-0 blur-xl bg-neon-cyan/30" />
        </motion.div>
        <motion.p
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="text-lg font-medium neon-text">
          Loading...
        </motion.p>
      </div>
    </motion.div>
  );
}

export function Spinner({ className = "" }: { className?: string }) {
  return (
    <motion.div
      animate={{ rotate: 360 }}
      transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
      className={`w-6 h-6 border-2 border-neon-cyan border-t-transparent rounded-full ${className}`}
    />
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <motion.div
      animate={{ opacity: [0.5, 1, 0.5] }}
      transition={{ duration: 1.5, repeat: Infinity }}
      className={`bg-light-200 dark:bg-dark-600 rounded ${className}`}
    />
  );
}
