# VoxProof - AI Voice Detection API

Detects whether a voice sample is **AI-generated** or spoken by a **real human**.

Built for the AI for Fraud Detection Hackathon.

---

## The Problem

AI voice cloning tools make it easy to impersonate anyone. This enables:

- Voice phishing scams
- Identity fraud
- Fake audio misinformation

VoxProof provides an API to detect synthetic voices.

---

## How It Works

```
Audio → Preprocessing → Feature Extraction → Neural Network → AI or Human?
```

**Features used:**

- Acoustic features (MFCCs, pitch, spectral rolloff, zero-crossing rate)
- Deep embeddings from Wav2Vec2 (pretrained speech model)

**Model:**

- Custom neural network classifier (786 → 512 → 256 → 128 → 64 → 1)
- Binary classification with sigmoid output

---

## Project Structure

```
VoxProof/
├── app.py                 # FastAPI server
├── train_classifier.py    # Training script
├── audio/
│   └── processing.py      # Audio preprocessing
├── model/
│   ├── model.py           # Neural network
│   └── classifier.pth     # Trained weights
├── utils/
│   └── explain.py         # Explanation generator
└── requirements.txt
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required for audio)
# Windows: winget install Gyan.FFmpeg
# Linux: sudo apt install ffmpeg
# Mac: brew install ffmpeg

# Run the API
uvicorn app:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

## API Usage

### POST /api/voice-detection

**Headers:**

```
x-api-key: your-api-key
Content-Type: application/json
```

**Request:**

```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64-encoded-audio>"
}
```

**Response:**

```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.92,
  "explanation": "Unnaturally stable pitch detected"
}
```

---

## Training

```bash
# Prepare dataset
dataset/
  human/  # Real voice samples (.mp3)
  ai/     # AI-generated samples (.mp3)

# Train
python train_classifier.py
```

Output: `model/classifier.pth`

---

## Environment Variables

| Variable  | Default                    | Description        |
| --------- | -------------------------- | ------------------ |
| `API_KEY` | `voxproof-secret-key-2024` | API authentication |
| `PORT`    | `8000`                     | Server port        |

---

## Tech Stack

- FastAPI + Uvicorn
- PyTorch
- HuggingFace Transformers (Wav2Vec2)
- librosa + pydub

---

## License

MIT
