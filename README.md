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

## Testing the API

### 1. Convert Audio to Base64

First, you need to encode your MP3 file to Base64:

**Windows (PowerShell):**

```powershell
$base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("path/to/audio.mp3"))
$base64 | Out-File -Encoding utf8 audio_base64.txt
```

**Linux/Mac:**

```bash
base64 audio.mp3 > audio_base64.txt
# Remove newlines (optional, but recommended)
base64 -w 0 audio.mp3 > audio_base64.txt
```

This creates a text file with the Base64-encoded audio. Use the content for API requests.

### 2. Test with cURL

```bash
# Load Base64 from file and API key from .env
$BASE64 = Get-Content audio_base64.txt -Raw
$APIKEY = (Get-Content .env | Select-String 'API_KEY=' | ForEach-Object { $_ -replace 'API_KEY=', '' })

# Make request
curl -X POST "http://localhost:8000/api/voice-detection" `
  -H "x-api-key: $APIKEY" `
  -H "Content-Type: application/json" `
  -d "{\"language\":\"English\",\"audioFormat\":\"mp3\",\"audioBase64\":\"$BASE64\"}"
```

**Linux/Mac:**

```bash
# Load BASE64 and API key from .env
BASE64=$(cat audio_base64.txt)
API_KEY=$(grep '^API_KEY=' .env | cut -d'=' -f2)

curl -X POST "http://localhost:8000/api/voice-detection" \
  -H "x-api-key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"language\":\"English\",\"audioFormat\":\"mp3\",\"audioBase64\":\"$BASE64\"}"
```

### 3. Test with Python

```python
import base64
import requests

# Read and encode audio file
with open("path/to/audio.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Load API key from environment
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

# Send request
response = requests.post(
    "http://localhost:8000/api/voice-detection",
    headers={
        "x-api-key": api_key,
        "Content-Type": "application/json"
    },
    json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    }
)

# Print response
print(response.json())
```

### 4. Test with Swagger UI

The easiest way - just visit the interactive API docs:

1. Start the server: `uvicorn app:app --reload --port 8000`
2. Open browser: http://localhost:8000/docs
3. Click "Try it out" on the `/api/voice-detection` endpoint
4. Paste your Base64-encoded audio into the `audioBase64` field
5. Click "Execute"

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

| Variable  | Default                            | Description                                     |
| --------- | ---------------------------------- | ----------------------------------------------- |
| `API_KEY` | `99d8f7fefa2c12ce971e4b320ee3af70` | 128-bit random API key (REQUIRED - set in .env) |
| `PORT`    | `8000`                             | Server port                                     |

---

## Tech Stack

- FastAPI + Uvicorn
- PyTorch
- HuggingFace Transformers (Wav2Vec2)
- librosa + pydub

---

## License

MIT
