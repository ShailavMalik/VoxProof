/**
 * Demo Audio Samples Configuration
 *
 * Files in public/demo-samples/:
 * - common_voice_* files are HUMAN recordings (Mozilla Common Voice dataset)
 * - All others are AI-generated (ElevenLabs, TTS engines)
 */

export interface DemoSample {
  id: string;
  name: string;
  description: string;
  type: "ai" | "human";
  source: string;
  language: string;
  fileUrl: string;
}

export const demoSamples: DemoSample[] = [
  // AI-Generated Samples
  {
    id: "ai-1",
    name: "Brim",
    description: "Confident & smooth voice clone",
    type: "ai",
    source: "ElevenLabs",
    language: "English",
    fileUrl: "/demo-samples/ai-elevenlabs-brim.mp3",
  },
  {
    id: "ai-2",
    name: "Brittney",
    description: "Fun, youthful social media voice",
    type: "ai",
    source: "AI TTS",
    language: "English",
    fileUrl: "/demo-samples/ai-brittney-social.mp3",
  },
  {
    id: "ai-3",
    name: "Raghav",
    description: "Calm, confident & engaging",
    type: "ai",
    source: "AI TTS",
    language: "Hindi",
    fileUrl: "/demo-samples/ai-raghav-calm.mp3",
  },
  {
    id: "ai-4",
    name: "Riya Rao",
    description: "Confident & professional tone",
    type: "ai",
    source: "AI TTS",
    language: "Hindi",
    fileUrl: "/demo-samples/ai-riya-professional.mp3",
  },
  {
    id: "ai-5",
    name: "Shardul K",
    description: "Deep horror storyteller",
    type: "ai",
    source: "AI TTS",
    language: "Hindi",
    fileUrl: "/demo-samples/ai-shardul-deep.mp3",
  },

  // Human Voice Samples (Common Voice dataset)
  {
    id: "human-1",
    name: "Speaker — EN",
    description: "Natural English speech",
    type: "human",
    source: "Common Voice",
    language: "English",
    fileUrl: "/demo-samples/human-english-1.mp3",
  },
  {
    id: "human-2",
    name: "Speaker — HI #1",
    description: "Natural Hindi speech",
    type: "human",
    source: "Common Voice",
    language: "Hindi",
    fileUrl: "/demo-samples/human-hindi-1.mp3",
  },
  {
    id: "human-3",
    name: "Speaker — HI #2",
    description: "Natural Hindi speech",
    type: "human",
    source: "Common Voice",
    language: "Hindi",
    fileUrl: "/demo-samples/human-hindi-2.mp3",
  },
  {
    id: "human-4",
    name: "Speaker — HI #3",
    description: "Natural Hindi speech",
    type: "human",
    source: "Common Voice",
    language: "Hindi",
    fileUrl: "/demo-samples/human-hindi-3.mp3",
  },
];
