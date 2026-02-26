import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Shield,
  Mic,
  Brain,
  CheckCircle,
  AlertTriangle,
  ChevronRight,
  Waves,
  Fingerprint,
  Lock,
} from "lucide-react";

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0 },
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
    },
  },
};

export default function HomePage() {
  return (
    <div className="relative">
      {/* Hero Section */}
      <section className="min-h-[90vh] flex items-center justify-center px-4 relative">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={staggerContainer}
          className="max-w-5xl mx-auto text-center">
          {/* Badge */}
          <motion.div
            variants={fadeInUp}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-8">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-neon-cyan opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-neon-cyan"></span>
            </span>
            <span className="text-sm text-dark-600 dark:text-light-300">
              AI Voice Detection Platform
            </span>
          </motion.div>

          {/* Main Title */}
          <motion.h1
            variants={fadeInUp}
            className="text-5xl md:text-7xl lg:text-8xl font-bold mb-6">
            <span className="neon-text">VoxProof</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            variants={fadeInUp}
            className="text-xl md:text-2xl text-dark-600 dark:text-light-300 mb-4">
            AI Voice Authenticity,{" "}
            <span className="text-neon-cyan">Proven.</span>
          </motion.p>

          {/* Description */}
          <motion.p
            variants={fadeInUp}
            className="text-base md:text-lg text-dark-500 dark:text-light-400 max-w-2xl mx-auto mb-12">
            Advanced neural network analysis to distinguish between human voices
            and AI-generated audio. Protect against voice fraud, deepfakes, and
            synthetic audio manipulation.
          </motion.p>

          {/* CTA Buttons */}
          <motion.div
            variants={fadeInUp}
            className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link to="/dashboard">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="btn-primary flex items-center gap-2">
                Launch App
                <ChevronRight className="w-5 h-5" />
              </motion.button>
            </Link>
            <a href="#how-it-works">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="btn-secondary">
                Learn More
              </motion.button>
            </a>
          </motion.div>

          {/* Floating icons */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1, duration: 1 }}
            className="absolute inset-0 pointer-events-none">
            <motion.div
              animate={{ y: [0, -15, 0] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
              className="absolute top-20 left-10 md:left-20">
              <Mic className="w-8 h-8 text-neon-cyan/30" />
            </motion.div>
            <motion.div
              animate={{ y: [0, 15, 0] }}
              transition={{
                duration: 5,
                repeat: Infinity,
                ease: "easeInOut",
                delay: 0.5,
              }}
              className="absolute top-40 right-10 md:right-20">
              <Waves className="w-10 h-10 text-neon-purple/30" />
            </motion.div>
            <motion.div
              animate={{ y: [0, -10, 0] }}
              transition={{
                duration: 3.5,
                repeat: Infinity,
                ease: "easeInOut",
                delay: 1,
              }}
              className="absolute bottom-40 left-20 md:left-40">
              <Brain className="w-6 h-6 text-neon-pink/30" />
            </motion.div>
          </motion.div>
        </motion.div>
      </section>

      {/* What is VoxProof */}
      <section className="py-24 px-4">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={staggerContainer}
          className="max-w-6xl mx-auto">
          <motion.div variants={fadeInUp} className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              What is <span className="neon-text">VoxProof</span>?
            </h2>
            <p className="text-dark-500 dark:text-light-400 max-w-2xl mx-auto">
              A cutting-edge AI-powered platform that analyzes audio to detect
              synthetic voices with high accuracy.
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                icon: Shield,
                title: "Voice Authentication",
                description:
                  "Verify the authenticity of audio recordings with advanced neural network analysis.",
              },
              {
                icon: Fingerprint,
                title: "Voice Forensics",
                description:
                  "Deep acoustic analysis detects subtle artifacts unique to AI-generated speech.",
              },
              {
                icon: Lock,
                title: "Fraud Prevention",
                description:
                  "Protect your organization from voice phishing and identity theft attacks.",
              },
            ].map((item) => (
              <motion.div
                key={item.title}
                variants={fadeInUp}
                className="glass-card-hover p-8 text-center">
                <div className="inline-flex p-4 rounded-2xl bg-neon-cyan/10 mb-6">
                  <item.icon className="w-8 h-8 text-neon-cyan" />
                </div>
                <h3 className="text-xl font-semibold mb-3">{item.title}</h3>
                <p className="text-dark-500 dark:text-light-400">
                  {item.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* How it Works */}
      <section id="how-it-works" className="py-24 px-4 relative">
        <div className="absolute inset-0 bg-gradient-neon-glow opacity-50" />

        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={staggerContainer}
          className="max-w-6xl mx-auto relative">
          <motion.div variants={fadeInUp} className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              How it <span className="neon-text">Works</span>
            </h2>
            <p className="text-dark-500 dark:text-light-400 max-w-2xl mx-auto">
              Three simple steps to verify voice authenticity
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                step: "01",
                icon: Mic,
                title: "Upload Audio",
                description:
                  "Upload any audio file (MP3, WAV) up to 15 seconds for analysis.",
              },
              {
                step: "02",
                icon: Brain,
                title: "AI Analysis",
                description:
                  "Our neural network extracts 798 acoustic features and analyzes voice patterns.",
              },
              {
                step: "03",
                icon: CheckCircle,
                title: "Get Verdict",
                description:
                  "Receive instant classification with confidence score and detailed explanation.",
              },
            ].map((item, index) => (
              <motion.div
                key={item.step}
                variants={fadeInUp}
                className="relative">
                <div className="glass-card p-8 relative overflow-hidden">
                  {/* Step number */}
                  <span className="absolute -top-4 -right-4 text-[120px] font-bold text-neon-cyan/5">
                    {item.step}
                  </span>

                  <div className="relative">
                    <div className="inline-flex p-3 rounded-xl bg-gradient-to-r from-neon-cyan/20 to-neon-purple/20 mb-6">
                      <item.icon className="w-6 h-6 text-neon-cyan" />
                    </div>
                    <h3 className="text-xl font-semibold mb-3">{item.title}</h3>
                    <p className="text-dark-500 dark:text-light-400">
                      {item.description}
                    </p>
                  </div>
                </div>

                {/* Connector line */}
                {index < 2 && (
                  <div className="hidden md:block absolute top-1/2 -right-4 w-8 h-px bg-gradient-to-r from-neon-cyan/50 to-transparent" />
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>
      </section>

      {/* Why it Matters */}
      <section className="py-24 px-4">
        <motion.div
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          variants={staggerContainer}
          className="max-w-6xl mx-auto">
          <motion.div variants={fadeInUp} className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Why it <span className="neon-text">Matters</span>
            </h2>
            <p className="text-dark-500 dark:text-light-400 max-w-2xl mx-auto">
              AI voice cloning poses real threats to individuals and
              organizations
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 gap-8">
            {[
              {
                icon: AlertTriangle,
                title: "Voice Phishing Scams",
                description:
                  "Criminals clone family members' voices to extract money or sensitive information from unsuspecting victims.",
                color: "text-verdict-ai-primary",
              },
              {
                icon: AlertTriangle,
                title: "Identity Fraud",
                description:
                  "AI voices can bypass voice authentication systems used by banks and financial institutions.",
                color: "text-verdict-ai-secondary",
              },
              {
                icon: AlertTriangle,
                title: "Audio Misinformation",
                description:
                  "Fake audio of public figures can spread rapidly, damaging reputations and spreading false information.",
                color: "text-verdict-ai-primary",
              },
              {
                icon: AlertTriangle,
                title: "Corporate Espionage",
                description:
                  "Synthetic voices impersonating executives can authorize fraudulent transactions or access confidential data.",
                color: "text-verdict-ai-secondary",
              },
            ].map((item) => (
              <motion.div
                key={item.title}
                variants={fadeInUp}
                className="glass-card p-6 flex gap-4">
                <div className={`flex-shrink-0 ${item.color}`}>
                  <item.icon className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold mb-2">{item.title}</h3>
                  <p className="text-dark-500 dark:text-light-400 text-sm">
                    {item.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>

          {/* CTA */}
          <motion.div variants={fadeInUp} className="text-center mt-16">
            <Link to="/dashboard">
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="btn-primary">
                Start Detecting Now
              </motion.button>
            </Link>
          </motion.div>
        </motion.div>
      </section>
    </div>
  );
}
