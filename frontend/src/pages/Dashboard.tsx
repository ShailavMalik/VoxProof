import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileAudio,
  X,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
  Info,
  Waves,
  Brain,
  Cpu,
  Shield,
  Zap,
  AudioWaveform,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

// Analysis steps for the futuristic loading animation
const analysisSteps: { icon: LucideIcon; text: string; color: string }[] = [
  { icon: AudioWaveform, text: "Decoding audio stream...", color: "#00f5ff" },
  { icon: Waves, text: "Extracting waveform data...", color: "#00f5ff" },
  { icon: Cpu, text: "Processing 798 acoustic features...", color: "#bf00ff" },
  { icon: Brain, text: "Running neural network analysis...", color: "#bf00ff" },
  { icon: Zap, text: "Analyzing Wav2Vec2 embeddings...", color: "#ff00aa" },
  {
    icon: Shield,
    text: "Generating voice authenticity verdict...",
    color: "#00f5ff",
  },
];

interface AnalysisResult {
  status: string;
  language: string;
  classification: "AI_GENERATED" | "HUMAN";
  confidenceScore: number;
  explanation: string;
}

interface FileDetails {
  name: string;
  size: number;
  type: string;
}

const fadeInUp = {
  hidden: { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
};

export default function DashboardPage() {
  const [file, setFile] = useState<File | null>(null);
  const [fileDetails, setFileDetails] = useState<FileDetails | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Progress through analysis steps during loading
  useEffect(() => {
    if (!isAnalyzing) {
      setCurrentStep(0);
      return;
    }

    const interval = setInterval(() => {
      setCurrentStep((prev) => (prev + 1) % analysisSteps.length);
    }, 1500);

    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const validateFile = (file: File): boolean => {
    const validTypes = ["audio/mpeg", "audio/wav", "audio/mp3", "audio/x-wav"];
    const validExtensions = [".mp3", ".wav"];
    const maxSize = 10 * 1024 * 1024; // 10MB

    const extension = file.name.toLowerCase().slice(file.name.lastIndexOf("."));

    if (
      !validTypes.includes(file.type) &&
      !validExtensions.includes(extension)
    ) {
      setError("Please upload an MP3 or WAV file");
      return false;
    }

    if (file.size > maxSize) {
      setError("File size must be less than 10MB");
      return false;
    }

    return true;
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    setError(null);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && validateFile(droppedFile)) {
      setFile(droppedFile);
      setFileDetails({
        name: droppedFile.name,
        size: droppedFile.size,
        type: droppedFile.type,
      });
      setResult(null);
    }
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null);
    const selectedFile = e.target.files?.[0];
    if (selectedFile && validateFile(selectedFile)) {
      setFile(selectedFile);
      setFileDetails({
        name: selectedFile.name,
        size: selectedFile.size,
        type: selectedFile.type,
      });
      setResult(null);
    }
  };

  const clearFile = () => {
    setFile(null);
    setFileDetails(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const analyzeAudio = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    try {
      // Convert file to base64
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const result = reader.result as string;
          // Remove data URL prefix
          const base64Data = result.split(",")[1];
          resolve(base64Data);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });

      // Get format from file extension
      const extension = file.name
        .toLowerCase()
        .slice(file.name.lastIndexOf(".") + 1);
      const audioFormat = extension === "wav" ? "wav" : "mp3";

      // Call the API (Vite uses import.meta.env)
      const apiBaseUrl =
        import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

      const response = await fetch(`${apiBaseUrl}/api/voice-detection`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": import.meta.env.VITE_API_KEY || "",
        },
        body: JSON.stringify({
          language: "English",
          audioFormat: audioFormat,
          audioBase64: base64,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `API error: ${response.status}`);
      }

      const data: AnalysisResult = await response.json();
      setResult(data);
    } catch (err) {
      console.error("Analysis error:", err);
      setError(
        err instanceof Error ?
          err.message
        : "Failed to analyze audio. Please try again.",
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={{
          hidden: { opacity: 0 },
          visible: { opacity: 1, transition: { staggerChildren: 0.1 } },
        }}>
        {/* Header */}
        <motion.div variants={fadeInUp} className="text-center mb-12 relative">
          {/* Decorative lines */}
          <div className="absolute left-0 right-0 top-1/2 flex items-center justify-center gap-4 -z-10">
            <motion.div
              className="h-px w-24 bg-gradient-to-r from-transparent to-neon-cyan/50"
              animate={{ opacity: [0.3, 0.7, 0.3] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <div className="w-2 h-2 rounded-full bg-neon-cyan/30" />
            <motion.div
              className="h-px w-24 bg-gradient-to-l from-transparent to-neon-purple/50"
              animate={{ opacity: [0.3, 0.7, 0.3] }}
              transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
            />
          </div>

          <h1 className="text-3xl md:text-4xl font-bold mb-4">
            Voice <span className="neon-text">Analysis</span>
          </h1>
          <p className="text-dark-500 dark:text-light-400">
            Upload an audio file to detect if it's AI-generated or human
          </p>

          {/* Status indicator */}
          <div className="flex items-center justify-center gap-2 mt-4">
            <motion.div
              className="w-2 h-2 rounded-full bg-neon-cyan"
              animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            />
            <span className="text-xs text-neon-cyan font-medium tracking-wider uppercase">
              System Ready
            </span>
          </div>
        </motion.div>

        {/* Futuristic Loading Animation Overlay */}
        <AnimatePresence>
          {isAnalyzing && <FuturisticLoader currentStep={currentStep} />}
        </AnimatePresence>

        {/* Upload Area */}
        <motion.div variants={fadeInUp} className="mb-8 relative">
          {/* Subtle outer glow when dragging */}
          {isDragging && (
            <motion.div
              className="absolute -inset-1 bg-gradient-to-r from-neon-cyan via-neon-purple to-neon-cyan rounded-2xl blur-md"
              initial={{ opacity: 0 }}
              animate={{ opacity: 0.4 }}
            />
          )}

          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => !file && fileInputRef.current?.click()}
            className={`
              relative glass-card p-12 text-center cursor-pointer transition-all duration-300 overflow-hidden
              ${
                isDragging ?
                  "border-neon-cyan border-2 bg-neon-cyan/5"
                : "border border-dashed border-light-300 dark:border-dark-500 hover:border-neon-cyan/50"
              }
              ${file ? "cursor-default" : ""}
            `}>
            {/* Background grid pattern */}
            <div
              className="absolute inset-0 opacity-[0.03]"
              style={{
                backgroundImage: `
                linear-gradient(rgba(0,245,255,0.5) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,245,255,0.5) 1px, transparent 1px)
              `,
                backgroundSize: "30px 30px",
              }}
            />

            {/* Corner tech decorations */}
            <div className="absolute top-2 left-2 w-6 h-6 border-l-2 border-t-2 border-neon-cyan/30 rounded-tl-lg" />
            <div className="absolute top-2 right-2 w-6 h-6 border-r-2 border-t-2 border-neon-purple/30 rounded-tr-lg" />
            <div className="absolute bottom-2 left-2 w-6 h-6 border-l-2 border-b-2 border-neon-purple/30 rounded-bl-lg" />
            <div className="absolute bottom-2 right-2 w-6 h-6 border-r-2 border-b-2 border-neon-cyan/30 rounded-br-lg" />

            <input
              ref={fileInputRef}
              type="file"
              accept=".mp3,.wav,audio/mpeg,audio/wav"
              onChange={handleFileSelect}
              className="hidden"
              disabled={isAnalyzing}
            />

            <AnimatePresence mode="wait">
              {!file ?
                <motion.div
                  key="upload"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="flex flex-col items-center relative z-10">
                  <motion.div
                    animate={{ y: isDragging ? -10 : 0 }}
                    className="relative p-4 rounded-2xl bg-gradient-to-br from-neon-cyan/20 to-neon-purple/10 mb-6">
                    {/* Glowing effect */}
                    <motion.div
                      className="absolute inset-0 rounded-2xl bg-neon-cyan/20 blur-xl"
                      animate={{ opacity: [0.3, 0.6, 0.3] }}
                      transition={{ duration: 2, repeat: Infinity }}
                    />
                    <Upload className="w-10 h-10 text-neon-cyan relative z-10" />
                  </motion.div>
                  <p className="text-lg font-semibold mb-2 bg-gradient-to-r from-neon-cyan to-neon-purple bg-clip-text text-transparent">
                    {isDragging ?
                      "Drop your file here"
                    : "Drag & drop audio file"}
                  </p>
                  <p className="text-sm text-dark-500 dark:text-light-400 mb-4">
                    or click to browse
                  </p>
                  <div className="flex items-center gap-2 text-xs text-dark-400 dark:text-light-500">
                    <div className="w-1 h-1 rounded-full bg-neon-cyan" />
                    <span>Supported: MP3, WAV</span>
                    <div className="w-1 h-1 rounded-full bg-neon-purple" />
                    <span>Max 10MB</span>
                  </div>
                </motion.div>
              : <motion.div
                  key="file"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  className="flex items-center justify-between relative z-10">
                  <div className="flex items-center gap-4">
                    <div className="p-3 rounded-xl bg-neon-cyan/10">
                      <FileAudio className="w-8 h-8 text-neon-cyan" />
                    </div>
                    <div className="text-left">
                      <p className="font-medium truncate max-w-xs">
                        {fileDetails?.name}
                      </p>
                      <p className="text-sm text-dark-500 dark:text-light-400">
                        {fileDetails && formatFileSize(fileDetails.size)}
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      clearFile();
                    }}
                    disabled={isAnalyzing}
                    className="p-2 rounded-lg hover:bg-red-500/10 text-red-500 transition-colors disabled:opacity-50">
                    <X className="w-5 h-5" />
                  </button>
                </motion.div>
              }
            </AnimatePresence>
          </div>
        </motion.div>

        {/* Error Message */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center gap-3">
              <AlertTriangle className="w-5 h-5 text-red-500 flex-shrink-0" />
              <p className="text-red-500 text-sm">{error}</p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Analyze Button */}
        <motion.div variants={fadeInUp} className="mb-8 relative">
          {/* Button glow effect when enabled */}
          {file && !isAnalyzing && (
            <motion.div
              className="absolute -inset-0.5 bg-gradient-to-r from-neon-cyan via-neon-purple to-neon-pink rounded-xl blur-sm"
              animate={{ opacity: [0.4, 0.7, 0.4] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          )}

          <motion.button
            onClick={analyzeAudio}
            disabled={!file || isAnalyzing}
            whileHover={{ scale: file && !isAnalyzing ? 1.02 : 1 }}
            whileTap={{ scale: file && !isAnalyzing ? 0.98 : 1 }}
            className={`
              relative w-full py-4 rounded-xl font-semibold text-white overflow-hidden
              ${
                file && !isAnalyzing ?
                  "bg-gradient-to-r from-neon-cyan via-neon-purple to-neon-pink"
                : "bg-dark-600/80 cursor-not-allowed"
              }
              transition-all duration-300
            `}>
            {/* Animated shine effect */}
            {file && !isAnalyzing && (
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -skew-x-12"
                initial={{ x: "-100%" }}
                animate={{ x: "200%" }}
                transition={{ duration: 2, repeat: Infinity, repeatDelay: 1 }}
              />
            )}

            {/* Inner border */}
            <div className="absolute inset-0.5 rounded-xl border border-white/10" />

            <span className="relative z-10 flex items-center justify-center gap-2">
              {isAnalyzing ?
                <>
                  <motion.div
                    className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{
                      duration: 1,
                      repeat: Infinity,
                      ease: "linear",
                    }}
                  />
                  Processing...
                </>
              : <>
                  <Zap className="w-5 h-5" />
                  Analyze Audio
                </>
              }
            </span>
          </motion.button>
        </motion.div>

        {/* Result Display */}
        <AnimatePresence>
          {result && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.95 }}
              transition={{ duration: 0.5, ease: "easeOut" }}>
              <ResultCard result={result} onReset={clearFile} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Info Section - Futuristic */}
        <motion.div variants={fadeInUp} className="mt-12 relative">
          {/* Subtle glow */}
          <div className="absolute -inset-0.5 bg-gradient-to-r from-neon-cyan/20 via-transparent to-neon-purple/20 rounded-xl blur-sm" />

          <div className="relative glass-card p-6 overflow-hidden">
            {/* Corner decorations */}
            <div className="absolute top-1 left-1 w-3 h-3 border-l border-t border-neon-cyan/50 rounded-tl" />
            <div className="absolute top-1 right-1 w-3 h-3 border-r border-t border-neon-purple/50 rounded-tr" />
            <div className="absolute bottom-1 left-1 w-3 h-3 border-l border-b border-neon-purple/50 rounded-bl" />
            <div className="absolute bottom-1 right-1 w-3 h-3 border-r border-b border-neon-cyan/50 rounded-br" />

            <div className="flex items-start gap-4">
              <div className="p-2 rounded-lg bg-gradient-to-br from-neon-cyan/20 to-neon-purple/10">
                <Info className="w-5 h-5 text-neon-cyan" />
              </div>
              <div className="text-sm text-dark-500 dark:text-light-400 flex-1">
                <p className="mb-3">
                  <strong className="text-neon-cyan">How it works:</strong> Our
                  AI extracts 798 acoustic features including MFCCs, pitch
                  analysis, and Wav2Vec2 embeddings to detect synthetic voices.
                </p>
                <p>
                  <strong className="text-neon-purple">Best results:</strong>{" "}
                  Use clear audio recordings up to 15 seconds for optimal
                  accuracy.
                </p>
              </div>
            </div>

            {/* Animated bottom line */}
            <motion.div
              className="absolute bottom-0 left-0 right-0 h-px"
              style={{
                background:
                  "linear-gradient(90deg, transparent, #00f5ff, #bf00ff, transparent)",
              }}
              animate={{ opacity: [0.3, 0.7, 0.3] }}
              transition={{ duration: 3, repeat: Infinity }}
            />
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}

// Result Card Component
function ResultCard({
  result,
  onReset,
}: {
  result: AnalysisResult;
  onReset: () => void;
}) {
  const isAI = result.classification === "AI_GENERATED";
  const confidence = Math.round(result.confidenceScore * 100);

  return (
    <div className="relative">
      {/* Outer glow effect */}
      <motion.div
        className={`absolute -inset-1 rounded-2xl opacity-30 blur-xl ${isAI ? "bg-verdict-ai-primary" : "bg-verdict-human-primary"}`}
        animate={{ opacity: [0.2, 0.4, 0.2] }}
        transition={{ duration: 3, repeat: Infinity }}
      />

      <div className="relative glass-card p-8 overflow-hidden">
        {/* Animated background grid */}
        <div className="absolute inset-0 opacity-5">
          <div
            className="absolute inset-0"
            style={{
              backgroundImage: `
              linear-gradient(rgba(0,245,255,0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(0,245,255,0.1) 1px, transparent 1px)
            `,
              backgroundSize: "20px 20px",
            }}
          />
        </div>

        {/* Corner tech decorations */}
        <div className="absolute top-2 left-2 w-8 h-8 border-l-2 border-t-2 border-neon-cyan/40 rounded-tl-lg" />
        <div className="absolute top-2 right-2 w-8 h-8 border-r-2 border-t-2 border-neon-purple/40 rounded-tr-lg" />
        <div className="absolute bottom-2 left-2 w-8 h-8 border-l-2 border-b-2 border-neon-purple/40 rounded-bl-lg" />
        <div className="absolute bottom-2 right-2 w-8 h-8 border-r-2 border-b-2 border-neon-cyan/40 rounded-br-lg" />

        {/* Top scanning line */}
        <motion.div
          className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-neon-cyan to-transparent"
          animate={{ opacity: [0, 1, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        />
        {/* Verdict */}
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200, delay: 0.2 }}
            className={`
            inline-flex p-6 rounded-full mb-6
            ${isAI ? "bg-verdict-ai-primary/10" : "bg-verdict-human-primary/10"}
          `}>
            {isAI ?
              <AlertTriangle className="w-12 h-12 text-verdict-ai-primary" />
            : <CheckCircle className="w-12 h-12 text-verdict-human-primary" />}
          </motion.div>

          <motion.h2
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className={`text-2xl md:text-3xl font-bold mb-2 ${
              isAI ? "text-verdict-ai-primary" : "text-verdict-human-primary"
            }`}>
            {isAI ? "AI Generated" : "Human Voice"}
          </motion.h2>

          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="text-dark-500 dark:text-light-400">
            {isAI ?
              "This audio appears to be synthetically generated"
            : "This audio appears to be from a real human voice"}
          </motion.p>
        </div>

        {/* Confidence Ring */}
        <div className="flex justify-center mb-8">
          <ConfidenceRing confidence={confidence} isAI={isAI} />
        </div>

        {/* Explanation - Futuristic Box */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="relative mb-6">
          {/* Gradient border glow */}
          <div className="absolute -inset-0.5 bg-gradient-to-r from-neon-cyan via-neon-purple to-neon-pink rounded-xl opacity-50 blur-sm" />
          <div className="absolute -inset-0.5 bg-gradient-to-r from-neon-cyan via-neon-purple to-neon-pink rounded-xl opacity-30" />

          {/* Main content box */}
          <div className="relative p-6 rounded-xl bg-dark-900/90 dark:bg-dark-800/95 backdrop-blur-xl border border-white/10">
            {/* Corner decorations */}
            <div className="absolute top-0 left-0 w-4 h-4 border-l-2 border-t-2 border-neon-cyan rounded-tl-lg" />
            <div className="absolute top-0 right-0 w-4 h-4 border-r-2 border-t-2 border-neon-purple rounded-tr-lg" />
            <div className="absolute bottom-0 left-0 w-4 h-4 border-l-2 border-b-2 border-neon-purple rounded-bl-lg" />
            <div className="absolute bottom-0 right-0 w-4 h-4 border-r-2 border-b-2 border-neon-cyan rounded-br-lg" />

            {/* Header with icon */}
            <div className="flex items-center gap-3 mb-3">
              <div className="p-2 rounded-lg bg-gradient-to-br from-neon-cyan/20 to-neon-purple/20">
                <Brain className="w-4 h-4 text-neon-cyan" />
              </div>
              <h3 className="font-semibold text-sm bg-gradient-to-r from-neon-cyan to-neon-purple bg-clip-text text-transparent">
                Analysis Details
              </h3>
              {/* Animated scan line */}
              <motion.div
                className="flex-1 h-px bg-gradient-to-r from-transparent via-neon-cyan/50 to-transparent"
                animate={{ opacity: [0.3, 0.8, 0.3] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            </div>

            {/* Content with subtle glow */}
            <p className="text-sm text-light-300 leading-relaxed pl-11">
              {result.explanation}
            </p>

            {/* Bottom tech line decoration */}
            <div className="absolute bottom-2 left-6 right-6 flex items-center gap-2">
              <div className="flex-1 h-px bg-gradient-to-r from-neon-cyan/30 via-transparent to-neon-purple/30" />
              <div className="flex gap-1">
                {[...Array(3)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="w-1 h-1 rounded-full bg-neon-cyan"
                    animate={{ opacity: [0.3, 1, 0.3] }}
                    transition={{
                      duration: 1,
                      repeat: Infinity,
                      delay: i * 0.2,
                    }}
                  />
                ))}
              </div>
              <div className="flex-1 h-px bg-gradient-to-r from-neon-purple/30 via-transparent to-neon-cyan/30" />
            </div>
          </div>
        </motion.div>

        {/* Actions */}
        <motion.button
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7 }}
          onClick={onReset}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          className="relative w-full py-4 rounded-xl overflow-hidden group">
          {/* Button gradient background */}
          <div className="absolute inset-0 bg-gradient-to-r from-neon-cyan/20 via-neon-purple/20 to-neon-cyan/20 group-hover:from-neon-cyan/30 group-hover:via-neon-purple/30 group-hover:to-neon-cyan/30 transition-all duration-300" />
          <div className="absolute inset-0 border border-neon-cyan/50 rounded-xl group-hover:border-neon-cyan/80 transition-colors" />

          {/* Animated shine effect */}
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -skew-x-12"
            initial={{ x: "-100%" }}
            whileHover={{ x: "100%" }}
            transition={{ duration: 0.6 }}
          />

          <span className="relative flex items-center justify-center gap-2 font-semibold text-neon-cyan">
            <RefreshCw className="w-4 h-4" />
            Analyze Another File
          </span>
        </motion.button>
      </div>
    </div>
  );
}

// Futuristic Loading Animation Component
function FuturisticLoader({ currentStep }: { currentStep: number }) {
  const step = analysisSteps[currentStep];
  const Icon = step.icon;

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-dark-900/90 dark:bg-dark-900/95 backdrop-blur-xl">
      <div className="relative flex flex-col items-center">
        {/* Outer rotating rings */}
        <div className="relative w-64 h-64">
          {/* Ring 1 - Outer */}
          <motion.div
            className="absolute inset-0 rounded-full border-2 border-neon-cyan/30"
            animate={{ rotate: 360 }}
            transition={{ duration: 8, repeat: Infinity, ease: "linear" }}>
            {/* Glowing dot on ring */}
            <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-3 h-3 bg-neon-cyan rounded-full shadow-[0_0_10px_#00f5ff,0_0_20px_#00f5ff]" />
          </motion.div>

          {/* Ring 2 */}
          <motion.div
            className="absolute inset-6 rounded-full border-2 border-neon-purple/40"
            animate={{ rotate: -360 }}
            transition={{ duration: 6, repeat: Infinity, ease: "linear" }}>
            <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-3 h-3 bg-neon-purple rounded-full shadow-[0_0_10px_#bf00ff,0_0_20px_#bf00ff]" />
          </motion.div>

          {/* Ring 3 */}
          <motion.div
            className="absolute inset-12 rounded-full border-2 border-neon-pink/40"
            animate={{ rotate: 360 }}
            transition={{ duration: 4, repeat: Infinity, ease: "linear" }}>
            <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 w-2 h-2 bg-neon-pink rounded-full shadow-[0_0_10px_#ff00aa,0_0_20px_#ff00aa]" />
          </motion.div>

          {/* Pulsing core glow */}
          <motion.div
            className="absolute inset-16 rounded-full bg-gradient-to-br from-neon-cyan/20 via-neon-purple/10 to-neon-pink/20"
            animate={{
              scale: [1, 1.15, 1],
              opacity: [0.4, 0.7, 0.4],
            }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          />

          {/* Center icon container */}
          <div className="absolute inset-0 flex items-center justify-center">
            <motion.div
              className="relative"
              animate={{ scale: [1, 1.05, 1] }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: "easeInOut",
              }}>
              {/* Glowing background behind icon */}
              <motion.div
                className="absolute inset-0 blur-2xl rounded-full -z-10"
                style={{
                  backgroundColor: step.color,
                  width: 100,
                  height: 100,
                  marginLeft: -25,
                  marginTop: -25,
                }}
                animate={{ opacity: [0.3, 0.6, 0.3] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              />

              {/* Icon with animated entrance */}
              <AnimatePresence mode="wait">
                <motion.div
                  key={currentStep}
                  initial={{ opacity: 0, scale: 0.5, rotate: -180 }}
                  animate={{ opacity: 1, scale: 1, rotate: 0 }}
                  exit={{ opacity: 0, scale: 0.5, rotate: 180 }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  className="relative p-5 rounded-2xl bg-dark-800/90 border border-white/10"
                  style={{
                    boxShadow: `0 0 40px ${step.color}40, 0 0 80px ${step.color}20`,
                  }}>
                  <Icon className="w-10 h-10" style={{ color: step.color }} />
                </motion.div>
              </AnimatePresence>
            </motion.div>
          </div>

          {/* Orbiting particles */}
          {[...Array(8)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 rounded-full"
              style={{
                backgroundColor:
                  i % 3 === 0 ? "#00f5ff"
                  : i % 3 === 1 ? "#bf00ff"
                  : "#ff00aa",
                left: "50%",
                top: "50%",
                marginLeft: "-4px",
                marginTop: "-4px",
                boxShadow: `0 0 6px ${
                  i % 3 === 0 ? "#00f5ff"
                  : i % 3 === 1 ? "#bf00ff"
                  : "#ff00aa"
                }`,
              }}
              animate={{
                x: [
                  Math.cos((i * Math.PI * 2) / 8) * 90,
                  Math.cos((i * Math.PI * 2) / 8 + Math.PI) * 90,
                  Math.cos((i * Math.PI * 2) / 8) * 90,
                ],
                y: [
                  Math.sin((i * Math.PI * 2) / 8) * 90,
                  Math.sin((i * Math.PI * 2) / 8 + Math.PI) * 90,
                  Math.sin((i * Math.PI * 2) / 8) * 90,
                ],
                scale: [1, 1.5, 1],
                opacity: [0.6, 1, 0.6],
              }}
              transition={{
                duration: 4,
                repeat: Infinity,
                delay: i * 0.15,
                ease: "easeInOut",
              }}
            />
          ))}
        </div>

        {/* Text display */}
        <div className="mt-10 text-center">
          <AnimatePresence mode="wait">
            <motion.p
              key={currentStep}
              initial={{ opacity: 0, y: 20, filter: "blur(10px)" }}
              animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
              exit={{ opacity: 0, y: -20, filter: "blur(10px)" }}
              transition={{ duration: 0.4 }}
              className="text-xl font-medium text-white mb-3"
              style={{ textShadow: `0 0 30px ${step.color}` }}>
              {step.text}
            </motion.p>
          </AnimatePresence>

          {/* Progress dots */}
          <div className="flex items-center justify-center gap-2 mt-4">
            {analysisSteps.map((_, i) => (
              <motion.div
                key={i}
                className="w-2 h-2 rounded-full transition-colors duration-300"
                style={{
                  backgroundColor: i === currentStep ? step.color : "#333",
                  boxShadow:
                    i === currentStep ? `0 0 8px ${step.color}` : "none",
                }}
                animate={
                  i === currentStep ? { scale: [1, 1.4, 1] } : { scale: 1 }
                }
                transition={{
                  duration: 0.6,
                  repeat: i === currentStep ? Infinity : 0,
                }}
              />
            ))}
          </div>

          {/* Scanning line effect */}
          <motion.div className="mt-6 h-1 w-64 rounded-full overflow-hidden bg-dark-700">
            <motion.div
              className="h-full w-20 rounded-full"
              style={{
                background: `linear-gradient(90deg, transparent, ${step.color}, transparent)`,
              }}
              animate={{ x: ["-80px", "320px"] }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                ease: "easeInOut",
              }}
            />
          </motion.div>
        </div>

        {/* Corner decorations */}
        <div className="absolute -top-8 -left-8 w-12 h-12 border-l-2 border-t-2 border-neon-cyan/50" />
        <div className="absolute -top-8 -right-8 w-12 h-12 border-r-2 border-t-2 border-neon-purple/50" />
        <div className="absolute -bottom-8 -left-8 w-12 h-12 border-l-2 border-b-2 border-neon-purple/50" />
        <div className="absolute -bottom-8 -right-8 w-12 h-12 border-r-2 border-b-2 border-neon-cyan/50" />
      </div>
    </motion.div>
  );
}

// Confidence Ring Component
function ConfidenceRing({
  confidence,
  isAI,
}: {
  confidence: number;
  isAI: boolean;
}) {
  const radius = 60;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (confidence / 100) * circumference;
  const color = isAI ? "#ff4444" : "#00cc88";
  const secondaryColor = isAI ? "#ff6b6b" : "#00ffaa";

  return (
    <div className="relative">
      {/* Outer glow */}
      <motion.div
        className="absolute inset-0 rounded-full blur-xl"
        style={{ backgroundColor: color }}
        animate={{ opacity: [0.1, 0.25, 0.1] }}
        transition={{ duration: 2, repeat: Infinity }}
      />

      {/* Decorative outer ring */}
      <motion.div
        className="absolute -inset-3 rounded-full border border-dashed"
        style={{ borderColor: `${color}30` }}
        animate={{ rotate: 360 }}
        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
      />

      <svg width="150" height="150" className="transform -rotate-90 relative">
        {/* Background circle */}
        <circle
          cx="75"
          cy="75"
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth="8"
          className="text-light-200 dark:text-dark-600"
        />
        {/* Outer glow circle */}
        <motion.circle
          cx="75"
          cy="75"
          r={radius + 4}
          fill="none"
          stroke={color}
          strokeWidth="1"
          strokeDasharray="8 4"
          initial={{ opacity: 0 }}
          animate={{ opacity: [0.3, 0.6, 0.3], rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity }}
          style={{ transformOrigin: "center" }}
        />
        {/* Progress circle */}
        <motion.circle
          cx="75"
          cy="75"
          r={radius}
          fill="none"
          stroke={`url(#confidenceGradient)`}
          strokeWidth="10"
          strokeLinecap="round"
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset }}
          transition={{ duration: 1, delay: 0.5, ease: "easeOut" }}
          style={{
            strokeDasharray: circumference,
            filter: `drop-shadow(0 0 15px ${color}80)`,
          }}
        />
        {/* Gradient definition */}
        <defs>
          <linearGradient
            id="confidenceGradient"
            x1="0%"
            y1="0%"
            x2="100%"
            y2="100%">
            <stop offset="0%" stopColor={color} />
            <stop offset="100%" stopColor={secondaryColor} />
          </linearGradient>
        </defs>
      </svg>

      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {/* Glowing background */}
        <motion.div
          className="absolute w-16 h-16 rounded-full blur-lg"
          style={{ backgroundColor: color }}
          animate={{ opacity: [0.1, 0.2, 0.1] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        />
        <motion.span
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.8 }}
          className="text-4xl font-bold relative"
          style={{ color, textShadow: `0 0 20px ${color}60` }}>
          {confidence}%
        </motion.span>
        <span className="text-xs text-dark-500 dark:text-light-400 font-medium tracking-wider uppercase">
          Confidence
        </span>
      </div>

      {/* Orbiting dot */}
      <motion.div
        className="absolute w-2 h-2 rounded-full"
        style={{
          backgroundColor: color,
          boxShadow: `0 0 10px ${color}`,
          left: "50%",
          top: "50%",
          marginLeft: "-4px",
          marginTop: "-4px",
        }}
        animate={{
          x: [
            0,
            radius * Math.cos(0),
            radius * Math.cos(Math.PI / 2),
            radius * Math.cos(Math.PI),
            radius * Math.cos((3 * Math.PI) / 2),
            0,
          ],
          y: [
            0,
            radius * Math.sin(0),
            radius * Math.sin(Math.PI / 2),
            radius * Math.sin(Math.PI),
            radius * Math.sin((3 * Math.PI) / 2),
            0,
          ],
        }}
        transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
      />
    </div>
  );
}
