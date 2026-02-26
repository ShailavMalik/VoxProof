import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Dark theme colors
        dark: {
          900: "#0a0a0f",
          800: "#111118",
          700: "#1a1a24",
          600: "#242432",
          500: "#2d2d3d",
        },
        // Light theme colors
        light: {
          50: "#fafbff",
          100: "#f0f2f9",
          200: "#e4e7f2",
          300: "#d1d5e4",
          400: "#b8bdd1",
        },
        // Accent colors
        neon: {
          cyan: "#00f5ff",
          purple: "#bf00ff",
          pink: "#ff00aa",
          blue: "#0066ff",
        },
        // Status colors
        verdict: {
          ai: {
            primary: "#ff4444",
            secondary: "#ff8800",
          },
          human: {
            primary: "#00cc88",
            secondary: "#00aaff",
          },
        },
      },
      backgroundImage: {
        // Dark gradients
        "gradient-dark-radial":
          "radial-gradient(ellipse at center, rgba(99, 102, 241, 0.15) 0%, transparent 70%)",
        "gradient-dark-mesh":
          "linear-gradient(135deg, #0a0a0f 0%, #111118 50%, #1a1a24 100%)",
        "gradient-neon-glow":
          "linear-gradient(135deg, rgba(0, 245, 255, 0.1) 0%, rgba(191, 0, 255, 0.1) 100%)",
        // Light gradients
        "gradient-light-radial":
          "radial-gradient(ellipse at center, rgba(99, 102, 241, 0.08) 0%, transparent 70%)",
        "gradient-light-mesh":
          "linear-gradient(135deg, #fafbff 0%, #f0f2f9 50%, #e4e7f2 100%)",
        // Glassmorphism
        "glass-dark":
          "linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%)",
        "glass-light":
          "linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%)",
      },
      boxShadow: {
        "neon-cyan": "0 0 20px rgba(0, 245, 255, 0.3)",
        "neon-purple": "0 0 20px rgba(191, 0, 255, 0.3)",
        "neon-glow": "0 0 40px rgba(99, 102, 241, 0.2)",
        glass:
          "0 8px 32px 0 rgba(0, 0, 0, 0.37), inset 0 0 0 1px rgba(255, 255, 255, 0.05)",
        "glass-light":
          "0 8px 32px 0 rgba(0, 0, 0, 0.08), inset 0 0 0 1px rgba(255, 255, 255, 0.5)",
        verdict: "0 10px 50px -10px",
      },
      animation: {
        "gradient-shift": "gradient-shift 8s ease infinite",
        "pulse-slow": "pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        float: "float 6s ease-in-out infinite",
        glow: "glow 2s ease-in-out infinite alternate",
        "spin-slow": "spin 8s linear infinite",
      },
      keyframes: {
        "gradient-shift": {
          "0%, 100%": { backgroundPosition: "0% 50%" },
          "50%": { backgroundPosition: "100% 50%" },
        },
        float: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(-10px)" },
        },
        glow: {
          "0%": { opacity: "0.5" },
          "100%": { opacity: "1" },
        },
      },
      backdropBlur: {
        xs: "2px",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "system-ui", "sans-serif"],
        display: ["var(--font-inter)", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
