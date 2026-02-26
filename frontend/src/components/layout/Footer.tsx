import { motion } from "framer-motion";
import { Shield } from "lucide-react";

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="relative mt-auto">
      {/* Gradient Divider */}
      <div className="h-px bg-gradient-to-r from-transparent via-neon-cyan/30 to-transparent" />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        viewport={{ once: true }}
        className="max-w-7xl mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-neon-cyan" />
            <span className="text-sm font-medium text-dark-600 dark:text-light-300">
              VoxProof
            </span>
          </div>

          {/* Copyright */}
          <p className="text-sm text-dark-500 dark:text-light-400">
            VoxProof © {currentYear}
          </p>

          {/* Links */}
          <div className="flex items-center gap-6">
            <a
              href="#"
              className="text-sm text-dark-500 dark:text-light-400 hover:text-neon-cyan transition-colors">
              Privacy
            </a>
            <a
              href="#"
              className="text-sm text-dark-500 dark:text-light-400 hover:text-neon-cyan transition-colors">
              Terms
            </a>
          </div>
        </div>
      </motion.div>
    </footer>
  );
}
