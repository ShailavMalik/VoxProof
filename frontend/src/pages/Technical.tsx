import { motion } from "framer-motion";
import {
  Code,
  Database,
  Network,
  Shield,
  Zap,
  ChevronRight,
} from "lucide-react";
import { Link } from "react-router-dom";

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0 },
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

export default function TechnicalPage() {
  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
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
            <Code className="w-12 h-12 text-neon-cyan" />
          </motion.div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Technical <span className="neon-text">Documentation</span>
          </h1>
          <p className="text-lg text-dark-500 dark:text-light-400 max-w-2xl mx-auto">
            Deep dive into the architecture, algorithms, and implementation
            details of VoxProof
          </p>
        </motion.div>

        {/* Architecture Overview */}
        <motion.div variants={fadeInUp} className="mb-16">
          <h2 className="text-3xl font-bold mb-8">
            System <span className="neon-text">Architecture</span>
          </h2>

          <div className="glass-card p-8 mb-6">
            <h3 className="text-xl font-semibold mb-6 flex items-center gap-2">
              <Network className="w-6 h-6 text-neon-cyan" />
              End-to-End Pipeline
            </h3>
            <div className="space-y-4">
              <div className="p-4 rounded-lg bg-dark-800/50 border border-neon-cyan/20">
                <h4 className="font-semibold text-neon-cyan mb-2">
                  1. Frontend (React + Vite)
                </h4>
                <p className="text-sm text-dark-400 dark:text-light-500">
                  React 18 SPA with Vite build tooling. Handles audio file
                  uploads, real-time waveform visualization, and result
                  presentation. Uses Framer Motion for smooth animations and
                  Tailwind CSS for responsive design.
                </p>
              </div>

              <div className="p-4 rounded-lg bg-dark-800/50 border border-neon-purple/20">
                <h4 className="font-semibold text-neon-purple mb-2">
                  2. Backend (FastAPI + Python)
                </h4>
                <p className="text-sm text-dark-400 dark:text-light-500">
                  REST API built with FastAPI for high-performance async
                  processing. Handles audio validation, feature extraction,
                  model inference, and explainability. Implements automatic API
                  documentation with Swagger/OpenAPI.
                </p>
              </div>

              <div className="p-4 rounded-lg bg-dark-800/50 border border-neon-pink/20">
                <h4 className="font-semibold text-neon-pink mb-2">
                  3. ML Model (PyTorch)
                </h4>
                <p className="text-sm text-dark-400 dark:text-light-500">
                  Deep learning model combining ResNet architecture with
                  Wav2Vec2 embeddings. Extracts and analyzes 798 acoustic and
                  neural features. Outputs classification (AI/Human) with
                  confidence scores and feature importance rankings.
                </p>
              </div>

              <div className="p-4 rounded-lg bg-dark-800/50 border border-neon-blue/20">
                <h4 className="font-semibold text-neon-blue mb-2">
                  4. Deployment (Docker + Railway)
                </h4>
                <p className="text-sm text-dark-400 dark:text-light-500">
                  Containerized with Docker for consistency across environments.
                  Deployed on Railway for auto-scaling and high availability.
                  Implements health checks and graceful shutdown.
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Audio Processing */}
        <motion.div variants={fadeInUp} className="mb-16">
          <h2 className="text-3xl font-bold mb-8">
            Audio <span className="neon-text">Processing</span> Pipeline
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <motion.div variants={fadeInUp} className="glass-card p-6">
              <h3 className="text-lg font-bold mb-4 text-neon-cyan">
                Input Validation
              </h3>
              <ul className="space-y-2 text-sm text-dark-400 dark:text-light-500">
                <li>✓ File size limit: &lt;25MB</li>
                <li>✓ Supported formats: MP3, WAV, FLAC, OGG</li>
                <li>✓ Duration validation: 3-600 seconds</li>
                <li>✓ Sample rate detection: 8kHz - 48kHz</li>
                <li>✓ Bit depth analysis: 8-32 bit</li>
              </ul>
            </motion.div>

            <motion.div
              variants={fadeInUp}
              transition={{ delay: 0.1 }}
              className="glass-card p-6">
              <h3 className="text-lg font-bold mb-4 text-neon-purple">
                Audio Normalization
              </h3>
              <ul className="space-y-2 text-sm text-dark-400 dark:text-light-500">
                <li>✓ Convert to 16kHz mono PCM</li>
                <li>✓ Normalize amplitude to [-1, 1]</li>
                <li>✓ Remove DC offset</li>
                <li>✓ Apply voice activity detection (VAD)</li>
                <li>✓ Trim silence (&lt;-40dB)</li>
              </ul>
            </motion.div>
          </div>
        </motion.div>

        {/* Feature Extraction */}
        <motion.div variants={fadeInUp} className="mb-16">
          <h2 className="text-3xl font-bold mb-8">
            Feature <span className="neon-text">Extraction</span> (798 Features)
          </h2>

          <div className="grid md:grid-cols-3 gap-6">
            <motion.div
              variants={fadeInUp}
              className="glass-card p-6 border-l-4 border-neon-cyan">
              <h3 className="text-lg font-bold mb-4 text-neon-cyan">
                Spectral Features (280)
              </h3>
              <ul className="space-y-1 text-xs text-dark-400 dark:text-light-500">
                <li>• MFCCs (13 coeff × 13 stats)</li>
                <li>• Mel-Spectrogram</li>
                <li>• Chroma Features</li>
                <li>• Spectral Centroid/Rolloff</li>
                <li>• Zero Crossing Rate</li>
                <li>• Temporal derivatives</li>
              </ul>
            </motion.div>

            <motion.div
              variants={fadeInUp}
              transition={{ delay: 0.1 }}
              className="glass-card p-6 border-l-4 border-neon-purple">
              <h3 className="text-lg font-bold mb-4 text-neon-purple">
                Pitch Features (180)
              </h3>
              <ul className="space-y-1 text-xs text-dark-400 dark:text-light-500">
                <li>• Fundamental frequency (F0)</li>
                <li>• Pitch jitter & shimmer</li>
                <li>• Vibrato analysis</li>
                <li>• Voicing probability</li>
                <li>• Octave errors</li>
                <li>• Harmonic-to-noise ratio</li>
              </ul>
            </motion.div>

            <motion.div
              variants={fadeInUp}
              transition={{ delay: 0.2 }}
              className="glass-card p-6 border-l-4 border-neon-pink">
              <h3 className="text-lg font-bold mb-4 text-neon-pink">
                Neural Features (338)
              </h3>
              <ul className="space-y-1 text-xs text-dark-400 dark:text-light-500">
                <li>• Wav2Vec2 embeddings (768D)</li>
                <li>• ResNet intermediate layers</li>
                <li>• Temporal statistics</li>
                <li>• Context features (prev/next)</li>
                <li>• Attention weights</li>
              </ul>
            </motion.div>
          </div>
        </motion.div>

        {/* Model Architecture */}
        <motion.div variants={fadeInUp} className="mb-16">
          <h2 className="text-3xl font-bold mb-8">
            Model <span className="neon-text">Architecture</span>
          </h2>

          <div className="glass-card p-8">
            <div className="space-y-6">
              <div className="flex gap-4 items-start">
                <div className="p-3 rounded-lg bg-neon-cyan/10 text-neon-cyan shrink-0">
                  <Code className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-semibold mb-2">ResNet-34 Backbone</h3>
                  <p className="text-sm text-dark-400 dark:text-light-500">
                    4 residual blocks with batch normalization and ReLU
                    activations. Pre-trained on speech recognition tasks for
                    transfer learning. Input shape: (1, 798) features.
                  </p>
                </div>
              </div>

              <div className="flex gap-4 items-start">
                <div className="p-3 rounded-lg bg-neon-purple/10 text-neon-purple shrink-0">
                  <Network className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-semibold mb-2">
                    Attention Mechanism [NEW]
                  </h3>
                  <p className="text-sm text-dark-400 dark:text-light-500">
                    Multi-head self-attention layer (8 heads, 512d) for
                    capturing long-range dependencies. Enables feature
                    importance visualization through attention weights.
                  </p>
                </div>
              </div>

              <div className="flex gap-4 items-start">
                <div className="p-3 rounded-lg bg-neon-pink/10 text-neon-pink shrink-0">
                  <Zap className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Classification Head</h3>
                  <p className="text-sm text-dark-400 dark:text-light-500">
                    Dense layers (512 → 256 → 128 → 2) with dropout (0.5).
                    Outputs softmax probabilities for AI (class 1) and Human
                    (class 0) predictions.
                  </p>
                </div>
              </div>

              <div className="flex gap-4 items-start">
                <div className="p-3 rounded-lg bg-neon-blue/10 text-neon-blue shrink-0">
                  <Database className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="font-semibold mb-2">Training Details</h3>
                  <p className="text-sm text-dark-400 dark:text-light-500">
                    Focal Loss to handle class imbalance. Adam optimizer with
                    cosine annealing schedule. Mixup augmentation (α=1.0).
                    Trained for 100 epochs on 1100+ samples with 5-fold CV.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Performance Metrics */}
        <motion.div variants={fadeInUp} className="mb-16">
          <h2 className="text-3xl font-bold mb-8">
            Performance <span className="neon-text">Metrics</span>
          </h2>

          <div className="grid md:grid-cols-2 gap-6">
            <motion.div
              variants={fadeInUp}
              className="glass-card p-6 border-t-4 border-verdict-ai-primary">
              <h3 className="text-lg font-bold mb-4 text-verdict-ai-primary">
                Overall Metrics
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-dark-400">Accuracy</span>
                  <span className="font-semibold text-neon-cyan">95.2%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark-400">Precision (AI)</span>
                  <span className="font-semibold text-neon-cyan">94.8%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark-400">Recall (AI)</span>
                  <span className="font-semibold text-neon-cyan">96.1%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark-400">F1-Score</span>
                  <span className="font-semibold text-neon-cyan">0.951</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark-400">AUC-ROC</span>
                  <span className="font-semibold text-neon-cyan">0.982</span>
                </div>
              </div>
            </motion.div>

            <motion.div
              variants={fadeInUp}
              transition={{ delay: 0.1 }}
              className="glass-card p-6 border-t-4 border-neon-purple">
              <h3 className="text-lg font-bold mb-4 text-neon-purple">
                Inference Performance
              </h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-dark-400">Avg Response Time</span>
                  <span className="font-semibold text-neon-cyan">2.3s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark-400">P95 Latency</span>
                  <span className="font-semibold text-neon-cyan">5.8s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark-400">Memory/Sample</span>
                  <span className="font-semibold text-neon-cyan">120MB</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark-400">Max Concurrency</span>
                  <span className="font-semibold text-neon-cyan">10</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark-400">Uptime</span>
                  <span className="font-semibold text-neon-cyan">99.9%</span>
                </div>
              </div>
            </motion.div>
          </div>
        </motion.div>

        {/* API Endpoints */}
        <motion.div variants={fadeInUp} className="mb-16">
          <h2 className="text-3xl font-bold mb-8">
            API <span className="neon-text">Endpoints</span>
          </h2>

          <div className="space-y-4">
            {[
              {
                method: "POST",
                endpoint: "/api/analyze",
                description:
                  "Main analysis endpoint. Accepts audio file (multipart/form-data). Returns classification and explainability.",
                color: "text-green-400",
              },
              {
                method: "GET",
                endpoint: "/api/status",
                description:
                  "Health check endpoint. Returns server status and model readiness.",
                color: "text-blue-400",
              },
              {
                method: "POST",
                endpoint: "/api/batch",
                description:
                  "Batch processing endpoint. Accepts multiple audio files. Returns results for all.",
                color: "text-yellow-400",
              },
              {
                method: "GET",
                endpoint: "/docs",
                description:
                  "Swagger UI documentation. Interactive API explorer with try-it-out.",
                color: "text-purple-400",
              },
            ].map((api, i) => (
              <motion.div
                key={api.endpoint}
                variants={fadeInUp}
                transition={{ delay: i * 0.05 }}
                className="glass-card p-6 hover:border-neon-cyan/50 transition-all">
                <div className="flex items-start gap-4">
                  <div className="flex gap-2 items-center">
                    <span className={`font-bold text-sm ${api.color}`}>
                      {api.method}
                    </span>
                    <code className="text-neon-cyan font-mono text-sm">
                      {api.endpoint}
                    </code>
                  </div>
                </div>
                <p className="text-sm text-dark-400 dark:text-light-500 mt-2">
                  {api.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Security */}
        <motion.div variants={fadeInUp} className="mb-16">
          <h2 className="text-3xl font-bold mb-8">
            Security & <span className="neon-text">Privacy</span>
          </h2>

          <div className="glass-card p-8">
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-neon-cyan" />
                  Security Measures
                </h3>
                <ul className="space-y-2 text-sm text-dark-400 dark:text-light-500">
                  <li>✓ HTTPS/TLS encryption for all traffic</li>
                  <li>✓ File type validation (MIME type check)</li>
                  <li>✓ Size limits to prevent abuse</li>
                  <li>✓ Rate limiting (10 req/min per IP)</li>
                  <li>✓ Input sanitization for safety</li>
                  <li>✓ CORS policy enforcement</li>
                </ul>
              </div>

              <div>
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Shield className="w-5 h-5 text-neon-purple" />
                  Privacy Policy
                </h3>
                <ul className="space-y-2 text-sm text-dark-400 dark:text-light-500">
                  <li>✓ Audio files NOT stored permanently</li>
                  <li>✓ Temporary files deleted after 1 hour</li>
                  <li>✓ No user tracking or profiling</li>
                  <li>✓ No third-party data sharing</li>
                  <li>✓ Logs rotated every 7 days</li>
                  <li>✓ GDPR compliant</li>
                </ul>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Technology Stack */}
        <motion.div variants={fadeInUp} className="mb-16">
          <h2 className="text-3xl font-bold mb-8">
            Technology <span className="neon-text">Stack</span>
          </h2>

          <div className="grid md:grid-cols-2 gap-8">
            <div>
              <h3 className="font-semibold mb-4 text-neon-cyan">Frontend</h3>
              <div className="space-y-2 text-sm text-dark-400 dark:text-light-500">
                <p>
                  <span className="font-medium">React 18</span> - UI library
                </p>
                <p>
                  <span className="font-medium">Vite 5</span> - Build tool
                </p>
                <p>
                  <span className="font-medium">TypeScript</span> - Type safety
                </p>
                <p>
                  <span className="font-medium">Tailwind CSS</span> - Styling
                </p>
                <p>
                  <span className="font-medium">Framer Motion</span> -
                  Animations
                </p>
                <p>
                  <span className="font-medium">Lucide Icons</span> - Icons
                </p>
              </div>
            </div>

            <div>
              <h3 className="font-semibold mb-4 text-neon-purple">Backend</h3>
              <div className="space-y-2 text-sm text-dark-400 dark:text-light-500">
                <p>
                  <span className="font-medium">FastAPI</span> - Web framework
                </p>
                <p>
                  <span className="font-medium">PyTorch</span> - ML framework
                </p>
                <p>
                  <span className="font-medium">Librosa</span> - Audio
                  processing
                </p>
                <p>
                  <span className="font-medium">Wav2Vec2</span> - Speech
                  embeddings
                </p>
                <p>
                  <span className="font-medium">NumPy/SciPy</span> - Scientific
                  computing
                </p>
                <p>
                  <span className="font-medium">Docker</span> - Containerization
                </p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Back to About */}
        <motion.div variants={fadeInUp} className="text-center">
          <Link to="/about">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-neon-cyan/10 border-2 border-neon-cyan hover:bg-neon-cyan/20 transition-all">
              <span className="font-semibold">Back to About</span>
              <ChevronRight className="w-5 h-5" />
            </motion.button>
          </Link>
        </motion.div>
      </motion.div>
    </div>
  );
}
