# VoxProof - AI Voice Authenticity Detection Platform

> **Detect AI-generated voices in real-time** - Built for the AI Impact Buildathon 2026

[![Live Frontend](https://img.shields.io/badge/Frontend-Vercel-black)](https://voxproof.vercel.app)
[![Live API](https://img.shields.io/badge/API-HuggingFace%20Spaces-orange)](https://shailavmalik-voxproof.hf.space)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Vite](https://img.shields.io/badge/Vite-5-646CFF)](https://vitejs.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red)](https://pytorch.org)

---

## 🌐 Live Demo

| Resource              | URL                                                                                      |
| --------------------- | ---------------------------------------------------------------------------------------- |
| **Frontend App**      | [https://voxproof.vercel.app](https://voxproof.vercel.app)                               |
| **API Base**          | [https://shailavmalik-voxproof.hf.space](https://shailavmalik-voxproof.hf.space)                 |
| **API Documentation** | [https://shailavmalik-voxproof.hf.space/docs](https://shailavmalik-voxproof.hf.space/docs)     |
| **Health Check**      | [https://shailavmalik-voxproof.hf.space/health](https://shailavmalik-voxproof.hf.space/health) |

> **Try it out:** Don't have audio? The dashboard includes **sample audio clips** (both AI-generated and real human voices) you can play and analyze instantly.

---

## 🎯 The Problem

AI voice cloning tools (ElevenLabs, OpenAI, etc.) make it trivially easy to impersonate anyone. This enables:

- 📞 **Voice phishing scams** - Criminals clone family members' voices
- 🏦 **Identity fraud** - Bypass voice authentication systems
- 📰 **Audio misinformation** - Fake audio of public figures

**VoxProof provides a production-ready platform to detect synthetic voices.**

---

## ✨ Features

### Frontend

- 🌓 **Dark/Light Theme** with smooth animated transitions
- 🎨 **Glassmorphism Design** with neon accents
- 🎬 **Cinematic Animations** using Framer Motion
- 📱 **Fully Responsive** for all devices
- 🔊 **Drag & Drop Upload** for audio files
- 📊 **Animated Results** with confidence visualization
- 🎧 **Demo Sample Library** - Play and analyze pre-loaded AI & human voice clips

### Backend

- 🧠 **Neural Network Analysis** with 798 acoustic features
- 🎙️ **Wav2Vec2 Embeddings** for deep speech understanding
- ⚡ **Fast Inference** (2-8 seconds per file)
- 🔐 **API Key Authentication**
- 📝 **Detailed Explanations** for each verdict
- 🚀 **Optimized Cold Start** - Models pre-loaded at startup

---

## 🔬 How It Works

```
Audio → Preprocessing → Feature Extraction → Neural Network → AI or Human?
         (16kHz mono)   (30 acoustic + 768 deep)   (ResNet-style)
```

### Feature Extraction (798 dimensions)

| Category           | Features | Description                                    |
| ------------------ | -------- | ---------------------------------------------- |
| **MFCCs**          | 13       | Timbre characteristics                         |
| **MFCC Dynamics**  | 5        | Temporal smoothness (AI voices are too smooth) |
| **Pitch Analysis** | 4        | Mean, std, range, jitter (AI has low jitter)   |
| **Spectral**       | 4        | Centroid, rolloff, bandwidth, contrast         |
| **Energy**         | 2        | RMS mean and variance (AI is too consistent)   |
| **ZCR**            | 2        | Zero-crossing rate statistics                  |
| **Wav2Vec2**       | 768      | Deep speech representations                    |

### Model Architecture

```
Input (798) → Linear(512) → ResBlock(512) → ResBlock(256) → ResBlock(128) → Output(1)
```

- **ResidualBlocks** with skip connections for stable training
- **GELU activation** for smooth gradients
- **BatchNorm + Dropout** for regularization
- **Focal Loss** training for hard example mining
- **Mixup + Label Smoothing** for robustness

### Key AI Detection Signals

| Signal              | Human               | AI (TTS)         |
| ------------------- | ------------------- | ---------------- |
| Pitch Jitter        | >5%                 | <2%              |
| MFCC Delta Variance | High                | Low (too smooth) |
| Energy Variance     | Natural dynamics    | Flat             |
| Signal Cleanliness  | Room noise, breaths | Too clean        |

---

## 📁 Project Structure

```
VoxProof/
├── frontend/                       # React + Vite Frontend
│   ├── src/
│   │   ├── main.tsx                # Entry point
│   │   ├── App.tsx                 # Root component with routes
│   │   ├── pages/
│   │   │   ├── Home.tsx            # Landing page
│   │   │   ├── Dashboard.tsx       # Upload, demo samples & analysis
│   │   │   └── About.tsx           # Team & project info
│   │   ├── components/
│   │   │   ├── layout/             # Navbar, Footer, Background
│   │   │   ├── providers/          # ThemeProvider
│   │   │   └── ui/                 # ThemeToggle, Loading
│   │   └── lib/
│   │       ├── utils.ts            # Utility functions
│   │       └── demoSamples.ts      # Demo audio sample configuration
│   ├── public/
│   │   └── demo-samples/           # Pre-loaded audio clips (AI + Human)
│   ├── tailwind.config.ts
│   ├── vite.config.ts
│   └── package.json
│
├── app.py                          # FastAPI server
├── audio/
│   └── processing.py               # Audio preprocessing & feature extraction
├── model/
│   ├── model.py                    # ResNet classifier + Wav2Vec2 embedder
│   ├── classifier.pth              # Trained weights
│   └── classifier_best.pth         # Best validation checkpoint
├── utils/
│   └── explain.py                  # Human-readable explanation generator
├── dataset/
│   ├── human/                      # Real voice training samples
│   └── ai/                         # AI-generated training samples
├── Dockerfile                      # Backend container (multi-stage)
├── requirements.txt                # Python dependencies
└── HUGGINGFACE_SPACES_BACKEND_SETUP.md  # HF Spaces deployment guide
```

---

## 🚀 Quick Start

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local

# Edit .env.local
# VITE_API_BASE_URL=http://localhost:8000

# Run development server
npm run dev
```

**Frontend:** http://localhost:5173

### Backend Setup

```bash
# From project root
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg
# Windows: winget install Gyan.FFmpeg
# Linux:   sudo apt install ffmpeg
# Mac:     brew install ffmpeg

# Set environment variables
echo "API_KEY=your-secret-key-here" > .env

# Run the API
uvicorn app:app --reload --port 8000
```

**API docs:** http://localhost:8000/docs

---

## 📡 API Usage

### Endpoint: POST /api/voice-detection

**Headers:**

```
x-api-key: your-api-key
Content-Type: application/json
```

**Request Body:**

```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64-encoded-audio>"
}
```

**Supported Languages:** English, Tamil, Hindi, Malayalam, Telugu  
_Flexible input:_ `en`, `eng`, `english`, `English` (and similar for other languages)

**Response (Success):**

```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92,
  "explanation": "High confidence: Pitch is unnaturally stable - lacks the micro-variations present in human vocal cords. Additionally, voice timbre transitions are too smooth - lacking natural articulatory variation."
}
```

**Response (Human Detected):**

```json
{
  "status": "success",
  "language": "English",
  "classification": "HUMAN",
  "confidenceScore": 0.87,
  "explanation": "High confidence: Natural pitch variation detected, consistent with biological voice production. Additionally, natural volume dynamics with breathing patterns."
}
```

### Audio Limits

| Constraint    | Value                    | Reason                         |
| ------------- | ------------------------ | ------------------------------ |
| Max duration  | 15 seconds               | Ensures fast processing on CPU |
| Max file size | ~2MB (base64)            | Railway request limits         |
| Sample rate   | Any (resampled to 16kHz) | Wav2Vec2 requirement           |
| Format        | MP3, WAV, FLAC, OGG      | Pydub supported formats        |

---

## 🧪 Testing

### Quick Test with cURL

```bash
# Encode audio
BASE64=$(base64 -w 0 audio.mp3)

# Test API
curl -X POST "https://shailavmalik-voxproof.hf.space/api/voice-detection" \
  -H "x-api-key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"language\":\"English\",\"audioFormat\":\"mp3\",\"audioBase64\":\"$BASE64\"}"
```

### Quick Test with Python

```python
import base64
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Read and encode audio
with open("test_audio.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "https://shailavmalik-voxproof.hf.space/api/voice-detection",
    headers={
        "x-api-key": os.getenv("API_KEY"),
        "Content-Type": "application/json"
    },
    json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    },
    timeout=120
)

result = response.json()
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidenceScore']:.1%}")
print(f"Explanation: {result['explanation']}")
```

---

## 🏋️ Training

### Prepare Dataset

```
dataset/
├── human/     # Real voice samples (.mp3, .wav, .flac)
└── ai/        # AI-generated samples (.mp3, .wav, .flac)
```

Recommended: 50+ samples per class, diverse speakers and content.

### Train Model

```bash
# Full training with augmentation
python train_improved_v2.py

# Quick training (fewer augmentations)
python train_fast.py
```

**Training Features:**

- Data augmentation (noise, time masking, speed/pitch shift)
- Focal Loss for hard example mining
- Mixup regularization
- Cosine annealing with warmup
- Early stopping + best checkpoint saving

**Output:**

- `model/classifier.pth` - Final model
- `model/classifier_best.pth` - Best validation checkpoint

---

## ⚙️ Environment Variables

| Variable        | Default                       | Description             |
| --------------- | ----------------------------- | ----------------------- |
| `API_KEY`       | (required)                    | API authentication key  |
| `PORT`          | `8000`                        | Server port             |
| `MODEL_PATH`    | `model/classifier.pth`        | Path to trained weights |
| `WAV2VEC_MODEL` | `facebook/wav2vec2-base-960h` | Wav2Vec2 model          |
| `SAMPLE_RATE`   | `16000`                       | Audio sample rate (Hz)  |
| `PRODUCTION`    | `false`                       | Enable production mode  |

---

## 🛠️ Tech Stack

| Component            | Technology                     |
| -------------------- | ------------------------------ |
| **Frontend**         | React 18, Vite 5, Tailwind CSS |
| **Animations**       | Framer Motion                  |
| **API Framework**    | FastAPI + Uvicorn              |
| **ML Framework**     | PyTorch 2.2 (CPU)              |
| **Speech Model**     | Wav2Vec2 (HuggingFace)         |
| **Audio Processing** | librosa + pydub + FFmpeg       |
| **Frontend Hosting** | Vercel                         |
| **Backend Hosting**  | **Hugging Face Spaces (Docker)**|

---

## 🚀 Deployment

### Frontend (Vercel)

```bash
cd frontend
npm i -g vercel
vercel
```

Set environment variables in Vercel Dashboard:

- `VITE_API_BASE_URL` = your API URL
- `VITE_API_KEY` = your API key

Or connect your GitHub repo for automatic deployments.

### Backend (Hugging Face Spaces - Docker)

**Live:** [https://shailavmalik-voxproof.hf.space](https://shailavmalik-voxproof.hf.space)

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space) with **Docker SDK**
2. Clone the Space repo and push your backend code:
   ```bash
   git clone https://huggingface.co/spaces/<username>/<space-name>
   cd <space-name>
   # Copy app.py, Dockerfile, requirements.txt, audio/, model/, utils/
   git add . && git commit -m "Deploy VoxProof" && git push
   ```
3. Set Space Secrets (Settings → Variables and secrets):
   - `API_KEY`: Your secret API key
   - `PRODUCTION`: `true`
4. Space auto-builds Docker image and deploys

**Benefits over Railway/Render:**
- More free RAM (HF allocates generously for open projects)
- No cold-start penalty
- Built-in git integration
- Perfect for ML model serving

See [HUGGINGFACE_SPACES_BACKEND_SETUP.md](HUGGINGFACE_SPACES_BACKEND_SETUP.md) for detailed instructions.

### Docker (Local)

```bash
docker build -t voxproof-api .
docker run -p 8000:8000 -e API_KEY=your-key -e PRODUCTION=true voxproof-api
```

---

## 📊 Performance

| Metric         | Value                           |
| -------------- | ------------------------------- |
| Inference time | 2-8 seconds (15s audio, CPU)    |
| Cold start     | ~30 seconds (models pre-loaded) |
| Accuracy       | 95%+ (depends on training data) |
| Supported TTS  | ElevenLabs, OpenAI, Coqui, etc. |

---

## 🔒 Security

- API key authentication required for all requests
- Base64 validation to prevent injection
- Request timeout (120s) to prevent resource exhaustion
- No audio storage (processed in memory only)
- CORS configured for frontend domain
- Pure ASGI middleware for minimal overhead

---

## 👥 Team - Meerut Coders

Built for the **AI Impact Buildathon 2026**

| Name          | Role                    |
| ------------- | ----------------------- |
| Shailav Malik | Team Lead & ML Engineer |
| Ritika Sharma | Frontend Developer      |
| Sarthak Vats  | Architecture Design     |
| Tarun Kumar   | ML & Data Engineer      |

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.
