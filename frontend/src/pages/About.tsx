import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import {
  Shield,
  Linkedin,
  ExternalLink,
  Brain,
  Waves,
  Fingerprint,
  Code,
  Award,
  Radio,
  TrendingUp,
  Database,
  AlertTriangle,
  CheckCircle,
  Sparkles,
  BookOpen,
  ChevronRight,
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

const teamMembers = [
  {
    name: "Ritika Sharma",
    linkedin: "https://www.linkedin.com/in/ritika-sharma-012979398/",
    gradient: "from-neon-purple to-neon-pink",
    image: "/img/ritika.png",
  },
  {
    name: "Shailav Malik",
    linkedin: "https://linkedin.com/in/shailavmalik",
    gradient: "from-neon-cyan to-neon-blue",
    image: "/img/shailav.png",
  },
  {
    name: "Sarthak Vats",
    linkedin: "https://www.linkedin.com/in/sarthak-vats-301a3a358/",
    gradient: "from-neon-cyan to-neon-purple",
    image: "/img/sarthak.png",
  },
  {
    name: "Tarun Kumar",
    linkedin: "https://www.linkedin.com/in/tarun-kumar-7238b1367/",
    gradient: "from-neon-pink to-neon-cyan",
    image: "/img/tarun.png",
  },
];

const features = [
  {
    icon: Brain,
    title: "Neural Network Analysis",
    description:
      "ResNet-style architecture with Wav2Vec2 embeddings for deep audio understanding.",
  },
  {
    icon: Waves,
    title: "798 Acoustic Features",
    description:
      "MFCCs, pitch analysis, spectral features, and deep speech representations.",
  },
  {
    icon: Fingerprint,
    title: "Voice Forensics",
    description:
      "Detects subtle artifacts unique to AI-generated speech like pitch jitter and smoothness.",
  },
  {
    icon: Code,
    title: "Production Ready",
    description:
      "FastAPI backend with async processing, deployed on scalable cloud infrastructure.",
  },
];

export default function AboutPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 pt-4 pb-8">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={staggerContainer}>
        {/* Header */}
        <motion.div variants={fadeInUp} className="text-center mb-16">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200 }}
            className="inline-flex p-4 rounded-2xl bg-neon-cyan/10 mb-6">
            <Shield className="w-12 h-12 text-neon-cyan" />
          </motion.div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            About <span className="neon-text">VoxProof</span>
          </h1>
          <p className="text-lg text-dark-500 dark:text-light-400 max-w-2xl mx-auto">
            An advanced AI voice forensics system designed to combat audio fraud
            and deepfake technology.
          </p>
        </motion.div>

        {/* Team Section */}
        <motion.div variants={fadeInUp} className="mb-16">
          <div className="text-center mb-12">
            <h2 className="text-2xl md:text-3xl font-bold mb-4">
              Meet <span className="neon-text">Meerut Coders</span>
            </h2>
            <p className="text-dark-500 dark:text-light-400">
              The talented team behind VoxProof
            </p>
          </div>

          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {teamMembers.map((member) => (
              <motion.div
                key={member.name}
                variants={fadeInUp}
                whileHover={{ y: -5 }}
                className="glass-card-hover p-6 text-center group">
                <div className="relative mb-6">
                  <div className="relative w-24 h-24 mx-auto rounded-full overflow-hidden ring-2 ring-transparent group-hover:ring-neon-cyan/50 shadow-lg group-hover:shadow-neon-glow transition-all duration-300">
                    <div
                      className={`absolute -inset-0.5 bg-gradient-to-br ${member.gradient} rounded-full opacity-0 group-hover:opacity-100 transition-opacity blur-sm`}
                    />
                    <img
                      src={member.image}
                      alt={member.name}
                      className="relative w-full h-full object-cover rounded-full"
                    />
                  </div>
                  <motion.div
                    className={`absolute inset-0 rounded-full bg-gradient-to-br ${member.gradient} blur-xl opacity-0 group-hover:opacity-30 transition-opacity`}
                  />
                </div>

                <h3 className="font-semibold text-lg mb-4">{member.name}</h3>

                <motion.a
                  href={member.linkedin}
                  target="_blank"
                  rel="noopener noreferrer"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="inline-flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-gradient-to-r from-neon-cyan/20 to-neon-cyan/10 border-2 border-neon-cyan hover:from-neon-cyan/30 hover:to-neon-cyan/20 hover:border-neon-purple transition-all duration-300 group">
                  <Linkedin className="w-5 h-5 text-neon-cyan group-hover:text-neon-purple transition-colors" />
                  <span className="text-sm font-semibold text-neon-cyan group-hover:text-neon-purple transition-colors">
                    LinkedIn
                  </span>
                  <ExternalLink className="w-4 h-4 text-neon-cyan group-hover:text-neon-purple transition-colors opacity-70 group-hover:opacity-100" />
                </motion.a>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Project Info */}
        <motion.div variants={fadeInUp} className="mb-16">
          <div className="glass-card p-8 md:p-12">
            <div className="flex flex-col md:flex-row items-center gap-8">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-4">
                  <Award className="w-5 h-5 text-neon-cyan" />
                  <span className="text-sm font-medium text-neon-cyan">
                    AI Impact Buildathon 2026
                  </span>
                </div>
                <h2 className="text-2xl md:text-3xl font-bold mb-4">
                  Detecting AI Voices with AI
                </h2>
                <p className="text-dark-500 dark:text-light-400 mb-6">
                  VoxProof is a production-ready AI voice detection platform
                  that analyzes audio recordings to determine whether they were
                  produced by a human or generated by AI systems like
                  ElevenLabs, OpenAI, or Coqui TTS.
                </p>
                <p className="text-dark-500 dark:text-light-400">
                  Our system extracts 798 acoustic features from audio samples
                  and uses a ResNet-style neural network with Wav2Vec2
                  embeddings to achieve high-accuracy classification with
                  detailed explanations.
                </p>
              </div>
              <div className="w-full md:w-auto">
                <div className="grid grid-cols-2 gap-4">
                  <StatCard label="Features" value="798" />
                  <StatCard label="Accuracy" value="95%+" />
                  <StatCard label="Languages" value="5" />
                  <StatCard label="Latency" value="<8s" />
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* How Detection Works - Futuristic Section */}
        <motion.div variants={fadeInUp} className="mb-16 relative">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              How <span className="neon-text">Detection</span> Works
            </h2>
            <p className="text-dark-500 dark:text-light-400 max-w-2xl mx-auto">
              Our AI analyzes 798 acoustic features to detect synthetic voices
              with 95%+ accuracy
            </p>
          </div>

          {/* Animated Visualization Section */}
          <div className="mb-12 glass-card p-8 overflow-hidden">
            <div className="grid md:grid-cols-2 gap-8 items-center">
              {/* Audio Waveform Visualization */}
              <motion.div>
                <h3 className="font-semibold mb-4 text-neon-cyan">
                  Real-Time Analysis
                </h3>
                <div className="h-40 bg-dark-800 dark:bg-dark-900 rounded-lg p-4 flex items-center justify-center gap-1 border border-neon-cyan/20 overflow-hidden">
                  {Array.from({ length: 40 }).map((_, i) => (
                    <motion.div
                      key={i}
                      className="flex-1 bg-gradient-to-t from-neon-cyan to-neon-purple rounded-sm"
                      animate={{
                        height: [
                          `${Math.random() * 100}%`,
                          `${Math.random() * 100}%`,
                          `${Math.random() * 100}%`,
                        ],
                      }}
                      transition={{
                        duration: 0.8,
                        repeat: Infinity,
                        delay: i * 0.02,
                      }}
                    />
                  ))}
                </div>
                <p className="text-sm text-dark-400 dark:text-light-500 mt-3">
                  Audio signal decomposition into frequency components
                </p>
              </motion.div>

              {/* Feature Space Visualization */}
              <motion.div>
                <h3 className="font-semibold mb-4 text-neon-purple">
                  Feature Space (798D)
                </h3>
                <div className="grid grid-cols-8 gap-2 p-4 bg-dark-800 dark:bg-dark-900 rounded-lg border border-neon-purple/20">
                  {Array.from({ length: 64 }).map((_, i) => (
                    <motion.div
                      key={i}
                      className="aspect-square rounded-md bg-gradient-to-br from-neon-cyan/30 to-neon-purple/30 border border-neon-purple/40"
                      animate={{
                        opacity: [0.3, 1, 0.3],
                        backgroundColor: [
                          "rgba(0, 255, 255, 0.1)",
                          "rgba(168, 85, 247, 0.2)",
                          "rgba(0, 255, 255, 0.1)",
                        ],
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        delay: Math.random() * 2,
                      }}
                    />
                  ))}
                </div>
                <p className="text-sm text-dark-400 dark:text-light-500 mt-3">
                  Extracted acoustic and neural embeddings
                </p>
              </motion.div>
            </div>

            {/* Classification Result Display */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-8 p-6 bg-gradient-to-r from-neon-cyan/10 to-neon-purple/10 rounded-lg border border-neon-cyan/30">
              <div className="grid md:grid-cols-2 gap-6">
                <div className="text-center">
                  <div className="flex justify-center mb-3">
                    <motion.div
                      className="w-12 h-12 rounded-full bg-gradient-to-br from-verdict-ai-primary to-red-500 flex items-center justify-center text-white font-bold"
                      animate={{
                        boxShadow: [
                          "0 0 0px rgba(255,0,0,0.5)",
                          "0 0 20px rgba(255,0,0,0.3)",
                        ],
                      }}
                      transition={{ duration: 1.5, repeat: Infinity }}>
                      AI
                    </motion.div>
                  </div>
                  <p className="text-dark-500 dark:text-light-400 text-sm">
                    If AI is detected
                  </p>
                  <div className="mt-3 p-2 bg-verdict-ai-primary/10 rounded text-xs text-verdict-ai-primary border border-verdict-ai-primary/30">
                    High confidence score with explainable features
                  </div>
                </div>
                <div className="text-center">
                  <div className="flex justify-center mb-3">
                    <motion.div
                      className="w-12 h-12 rounded-full bg-gradient-to-br from-verdict-human-primary to-green-500 flex items-center justify-center text-white font-bold"
                      animate={{
                        boxShadow: [
                          "0 0 0px rgba(34,197,94,0.5)",
                          "0 0 20px rgba(34,197,94,0.3)",
                        ],
                      }}
                      transition={{ duration: 1.5, repeat: Infinity }}>
                      Human
                    </motion.div>
                  </div>
                  <p className="text-dark-500 dark:text-light-400 text-sm">
                    If Human is detected
                  </p>
                  <div className="mt-3 p-2 bg-verdict-human-primary/10 rounded text-xs text-verdict-human-primary border border-verdict-human-primary/30">
                    Natural voice markers confirmed
                  </div>
                </div>
              </div>
            </motion.div>
          </div>

          {/* Detection Pipeline */}
          <div className="grid md:grid-cols-4 gap-4 mb-12">
            {[
              {
                icon: Radio,
                title: "Audio Input",
                desc: "MP3, WAV, FLAC",
                color: "from-neon-cyan",
              },
              {
                icon: Waves,
                title: "Feature Extract",
                desc: "798 dimensions",
                color: "from-neon-purple",
              },
              {
                icon: Brain,
                title: "Neural Network",
                desc: "ResNet + Wav2Vec2",
                color: "from-neon-pink",
              },
              {
                icon: CheckCircle,
                title: "Classification",
                desc: "AI or Human",
                color: "from-verdict-human-primary",
              },
            ].map((step, i) => (
              <motion.div
                key={step.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="relative group">
                {/* Connecting line */}
                {i < 3 && (
                  <div className="absolute -right-2 top-1/2 w-4 h-0.5 bg-gradient-to-r from-neon-cyan/50 to-transparent hidden md:block" />
                )}

                <div className="glass-card p-6 text-center relative overflow-hidden">
                  {/* Glow background */}
                  <motion.div
                    className={`absolute inset-0 bg-gradient-to-br ${step.color} via-transparent opacity-0 group-hover:opacity-10 transition-opacity`}
                  />

                  {/* Icon with animation */}
                  <motion.div
                    animate={{ y: [0, -8, 0] }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      delay: i * 0.2,
                    }}
                    className="relative mb-4 flex justify-center">
                    <div
                      className={`p-3 rounded-xl bg-gradient-to-br ${step.color} to-transparent text-white`}>
                      <step.icon className="w-8 h-8" />
                    </div>
                  </motion.div>

                  <h3 className="font-bold mb-2">{step.title}</h3>
                  <p className="text-xs text-dark-400 dark:text-light-500">
                    {step.desc}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>

          {/* AI Detection Signals */}
          <div className="grid md:grid-cols-2 gap-6 mb-12">
            {/* AI Indicators */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="glass-card p-8 border-l-4 border-verdict-ai-primary">
              <div className="flex items-center gap-3 mb-6">
                <AlertTriangle className="w-6 h-6 text-verdict-ai-primary" />
                <h3 className="text-xl font-bold text-verdict-ai-primary">
                  AI Voice Signals
                </h3>
              </div>

              <div className="space-y-4">
                {[
                  {
                    pattern: "2x Lower Pitch Jitter",
                    desc: "AI voices lack natural micro-variations",
                  },
                  {
                    pattern: "95%+ Smooth Transitions",
                    desc: "No articulatory gaps between phonemes",
                  },
                  {
                    pattern: "Flat Energy Envelope",
                    desc: "Unnaturally consistent volume",
                  },
                  {
                    pattern: "Low MFCC Variance",
                    desc: "Timbre too consistent over time",
                  },
                  {
                    pattern: "Perfect Timing",
                    desc: "No natural speech hesitations or fillers",
                  },
                  {
                    pattern: "Robotic Spectral Profile",
                    desc: "Missing natural formant patterns",
                  },
                ].map((signal, i) => (
                  <motion.div
                    key={signal.pattern}
                    initial={{ opacity: 0, x: -10 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    viewport={{ once: true }}
                    className="flex gap-3">
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        delay: i * 0.1,
                      }}
                      className="w-2 h-2 rounded-full bg-verdict-ai-primary flex-shrink-0 mt-1.5"
                    />
                    <div>
                      <p className="font-semibold text-sm text-light-300">
                        {signal.pattern}
                      </p>
                      <p className="text-xs text-dark-400 dark:text-light-500">
                        {signal.desc}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>

            {/* Human Indicators */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="glass-card p-8 border-l-4 border-verdict-human-primary">
              <div className="flex items-center gap-3 mb-6">
                <CheckCircle className="w-6 h-6 text-verdict-human-primary" />
                <h3 className="text-xl font-bold text-verdict-human-primary">
                  Human Voice Signals
                </h3>
              </div>

              <div className="space-y-4">
                {[
                  {
                    pattern: "5-8% Pitch Jitter",
                    desc: "Natural vocal cord vibrations",
                  },
                  {
                    pattern: "Variable Transitions",
                    desc: "Natural speech discontinuities",
                  },
                  {
                    pattern: "Dynamic Energy",
                    desc: "Breathing, emphasis, emotion",
                  },
                  { pattern: "High MFCC Delta", desc: "Rich timbre evolution" },
                  {
                    pattern: "Natural Hesitations",
                    desc: "Ums, ahs, pauses, breathing",
                  },
                  {
                    pattern: "Complex Spectral Profile",
                    desc: "Formants, harmonics, noise",
                  },
                ].map((signal, i) => (
                  <motion.div
                    key={signal.pattern}
                    initial={{ opacity: 0, x: 10 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                    viewport={{ once: true }}
                    className="flex gap-3">
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        delay: i * 0.1,
                      }}
                      className="w-2 h-2 rounded-full bg-verdict-human-primary flex-shrink-0 mt-1.5"
                    />
                    <div>
                      <p className="font-semibold text-sm text-light-300">
                        {signal.pattern}
                      </p>
                      <p className="text-xs text-dark-400 dark:text-light-500">
                        {signal.desc}
                      </p>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </div>
        </motion.div>

        {/* Training Data Statistics */}
        <motion.div variants={fadeInUp} className="mb-16 relative">
          <motion.div
            className="absolute -inset-0.5 bg-gradient-to-r from-neon-cyan/20 via-neon-purple/20 to-neon-cyan/20 rounded-2xl blur-xl opacity-50"
            animate={{ opacity: [0.3, 0.5, 0.3] }}
            transition={{ duration: 3, repeat: Infinity }}
          />

          <div className="relative glass-card p-8 md:p-12 overflow-hidden">
            {/* Corner decorations */}
            <div className="absolute top-2 left-2 w-6 h-6 border-l-2 border-t-2 border-neon-cyan/40 rounded-tl-lg" />
            <div className="absolute top-2 right-2 w-6 h-6 border-r-2 border-t-2 border-neon-purple/40 rounded-tr-lg" />

            <div className="relative z-10">
              <div className="flex items-center gap-3 mb-8">
                <Database className="w-8 h-8 text-neon-cyan" />
                <h2 className="text-2xl md:text-3xl font-bold">
                  Trained on <span className="neon-text">Real Data</span>
                </h2>
              </div>

              <div className="grid md:grid-cols-3 gap-8 mb-12">
                {/* AI Samples */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  className="text-center">
                  <motion.div
                    animate={{ scale: [1, 1.05, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="text-5xl font-bold mb-2 bg-gradient-to-r from-verdict-ai-primary to-red-500 bg-clip-text text-transparent">
                    650+
                  </motion.div>
                  <p className="text-dark-400 dark:text-light-500 mb-3">
                    AI Generated Samples
                  </p>
                  <p className="text-sm text-dark-500 dark:text-light-600">
                    ElevenLabs, Coqui, pyttsx3, Google TTS and more
                  </p>
                </motion.div>

                {/* Human Samples */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  viewport={{ once: true }}
                  className="text-center">
                  <motion.div
                    animate={{ scale: [1, 1.05, 1] }}
                    transition={{ duration: 2, repeat: Infinity, delay: 0.2 }}
                    className="text-5xl font-bold mb-2 bg-gradient-to-r from-verdict-human-primary to-green-500 bg-clip-text text-transparent">
                    450+
                  </motion.div>
                  <p className="text-dark-400 dark:text-light-500 mb-3">
                    Human Voice Samples
                  </p>
                  <p className="text-sm text-dark-500 dark:text-light-600">
                    Common Voice (Mozilla), diverse speakers & languages
                  </p>
                </motion.div>

                {/* Languages */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  viewport={{ once: true }}
                  className="text-center">
                  <motion.div
                    animate={{ scale: [1, 1.05, 1] }}
                    transition={{ duration: 2, repeat: Infinity, delay: 0.4 }}
                    className="text-5xl font-bold mb-2 bg-gradient-to-r from-neon-cyan to-neon-purple bg-clip-text text-transparent">
                    5
                  </motion.div>
                  <p className="text-dark-400 dark:text-light-500 mb-3">
                    Supported Languages
                  </p>
                  <p className="text-sm text-dark-500 dark:text-light-600">
                    English, Hindi, Tamil, Telugu, Malayalam
                  </p>
                </motion.div>
              </div>

              {/* Training Approach */}
              <div className="grid md:grid-cols-2 gap-6">
                {[
                  {
                    title: "Data Augmentation",
                    items: [
                      "Noise injection",
                      "Time masking",
                      "Speed/pitch shift",
                      "Volume normalization",
                    ],
                    icon: Sparkles,
                  },
                  {
                    title: "Model Optimization",
                    items: [
                      "Focal Loss training",
                      "Mixup regularization",
                      "Cosine annealing",
                      "Early stopping",
                    ],
                    icon: TrendingUp,
                  },
                ].map((section, i) => (
                  <motion.div
                    key={section.title}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.15 }}
                    viewport={{ once: true }}
                    className="p-6 rounded-xl bg-white/5 border border-white/10 backdrop-blur">
                    <div className="flex items-center gap-2 mb-4">
                      <section.icon className="w-5 h-5 text-neon-cyan" />
                      <h3 className="font-semibold">{section.title}</h3>
                    </div>
                    <ul className="space-y-2">
                      {section.items.map((item) => (
                        <li
                          key={item}
                          className="text-sm text-dark-400 dark:text-light-500 pl-4 border-l-2 border-neon-cyan/30">
                          {item}
                        </li>
                      ))}
                    </ul>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Bottom corner decorations */}
            <div className="absolute bottom-2 left-2 w-6 h-6 border-l-2 border-b-2 border-neon-purple/40 rounded-bl-lg" />
            <div className="absolute bottom-2 right-2 w-6 h-6 border-r-2 border-b-2 border-neon-cyan/40 rounded-br-lg" />
          </div>
        </motion.div>

        {/* Technical Features */}
        <motion.div variants={fadeInUp} className="mb-16">
          <h2 className="text-2xl md:text-3xl font-bold text-center mb-12">
            Technical <span className="neon-text">Features</span>
          </h2>
          <div className="grid md:grid-cols-2 gap-6">
            {features.map((feature) => (
              <motion.div
                key={feature.title}
                variants={fadeInUp}
                whileHover={{ scale: 1.02 }}
                className="glass-card-hover p-6 flex gap-4">
                <div className="flex-shrink-0 p-3 rounded-xl bg-neon-cyan/10">
                  <feature.icon className="w-6 h-6 text-neon-cyan" />
                </div>
                <div>
                  <h3 className="font-semibold mb-2">{feature.title}</h3>
                  <p className="text-sm text-dark-500 dark:text-light-400">
                    {feature.description}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Tech Stack */}
        <motion.div variants={fadeInUp} className="mt-16">
          <div className="glass-card p-8">
            <h3 className="text-xl font-semibold mb-6 text-center">
              Built With
            </h3>
            <div className="flex flex-wrap justify-center gap-4">
              {[
                "React",
                "Vite",
                "FastAPI",
                "PyTorch",
                "Wav2Vec2",
                "Tailwind CSS",
                "Framer Motion",
                "Python",
              ].map((tech) => (
                <span
                  key={tech}
                  className="px-4 py-2 rounded-full bg-dark-800 dark:bg-neon-cyan/10 text-neon-cyan text-sm font-medium border border-neon-cyan/30 shadow-sm">
                  {tech}
                </span>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Technical Documentation CTA */}
        <motion.div variants={fadeInUp} className="mt-20">
          <div className="relative overflow-hidden">
            {/* Background gradient animation */}
            <motion.div
              className="absolute inset-0 bg-gradient-to-r from-neon-cyan/20 via-neon-purple/20 to-neon-pink/20 rounded-2xl blur-xl"
              animate={{ opacity: [0.3, 0.6, 0.3] }}
              transition={{ duration: 4, repeat: Infinity }}
            />

            <div className="relative glass-card p-8 md:p-12 border-2 border-neon-cyan/50 hover:border-neon-purple/50 transition-colors">
              <div className="grid md:grid-cols-2 gap-8 items-center">
                <motion.div
                  initial={{ opacity: 0, x: -30 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}>
                  <div className="flex items-center gap-3 mb-4">
                    <BookOpen className="w-8 h-8 text-neon-cyan" />
                    <h2 className="text-2xl md:text-3xl font-bold">
                      Want the <span className="neon-text">Technical</span>{" "}
                      Details?
                    </h2>
                  </div>
                  <p className="text-dark-500 dark:text-light-400 mb-6">
                    Explore comprehensive documentation covering system
                    architecture, feature extraction, model internals, API
                    endpoints, and performance metrics. Perfect for developers,
                    researchers, and anyone interested in AI voice detection
                    technology.
                  </p>
                  <Link to="/technical">
                    <motion.button
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                      className="inline-flex items-center gap-2 px-8 py-3 bg-gradient-to-r from-neon-cyan/20 to-neon-purple/20 border-2 border-neon-cyan rounded-lg hover:from-neon-cyan/30 hover:to-neon-purple/30 hover:border-neon-purple transition-all duration-300 font-semibold">
                      Read Technical Docs
                      <ChevronRight className="w-5 h-5" />
                    </motion.button>
                  </Link>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, x: 30 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  className="grid grid-cols-2 gap-4">
                  {[
                    { title: "Architecture", icon: "🏗️" },
                    { title: "Model Details", icon: "🧠" },
                    { title: "Feature Extract", icon: "📊" },
                    { title: "API Reference", icon: "🔌" },
                  ].map((item, i) => (
                    <motion.div
                      key={item.title}
                      initial={{ opacity: 0, y: 10 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      transition={{ delay: i * 0.1 }}
                      viewport={{ once: true }}
                      className="p-4 rounded-lg bg-dark-800/50 border border-neon-cyan/20 hover:border-neon-purple/50 transition-colors text-center">
                      <div className="text-2xl mb-2">{item.icon}</div>
                      <p className="text-sm font-medium text-dark-300 dark:text-light-300">
                        {item.title}
                      </p>
                    </motion.div>
                  ))}
                </motion.div>
              </div>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-4 rounded-xl bg-white/50 dark:bg-dark-700/50 text-center">
      <p className="text-2xl font-bold neon-text">{value}</p>
      <p className="text-xs text-dark-500 dark:text-light-400">{label}</p>
    </div>
  );
}
