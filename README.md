# VoxProof - AI Voice Detection API

A production-ready FastAPI system for detecting whether a voice sample is AI-generated or spoken by a human. Built for the AI Security Hackathon.

## 🏗️ Architecture

```
VoxProof/
├── app.py                  # FastAPI entry point
├── train.py                # Training script for classifier
├── test_client.py          # CLI tool for testing the API
├── audio/
│   ├── __init__.py
│   └── processing.py       # Audio decoding & feature extraction
├── model/
│   ├── __init__.py
│   ├── model.py            # Classifier model + wav2vec2 embeddings
│   ├── classifier.pth      # Trained model weights
│   └── classifier_best.pth # Best checkpoint from training
├── utils/
│   ├── __init__.py
│   └── explain.py          # Explanation logic
├── .env                    # Configuration
├── .env.example            # Example configuration
├── requirements.txt        # Dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install FFmpeg (Required for audio processing)

**Windows:**

```bash
# Using winget (recommended)
winget install Gyan.FFmpeg

# Or using Chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
```

**Linux:**

```bash
sudo apt-get install ffmpeg
```

**Mac:**

```bash
brew install ffmpeg
```

### 3. Configure Environment

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

Edit `.env`:

```env
API_KEY=your-secure-api-key-here
MODEL_PATH=model/classifier.pth
WAV2VEC_MODEL=facebook/wav2vec2-base-960h
SAMPLE_RATE=16000
```

### 4. Run the Server

```bash
# Development mode with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python app.py
```

The API will be available at: `http://localhost:8000`

## 📖 API Documentation

Once running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔌 API Endpoints

### Health Check

```
GET /
GET /health
```

### Voice Detection

```
POST /api/voice-detection
```

**Headers:**

```
x-api-key: <YOUR_API_KEY>
Content-Type: application/json
```

**Request Body:**

```json
{
  "language": "Tamil | English | Hindi | Malayalam | Telugu",
  "audioFormat": "mp3",
  "audioBase64": "<Base64 encoded MP3>"
}
```

**Response:**

```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED | HUMAN",
  "confidenceScore": 0.85,
  "explanation": "Unnaturally stable pitch pattern detected - synthetic voices often lack natural pitch fluctuations"
}
```

## 🧪 Testing the API

### Using cURL

```bash
# Encode your audio file to Base64
$audioBase64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("path/to/audio.mp3"))

# Make API request
curl -X POST "http://localhost:8000/api/voice-detection" \
  -H "Content-Type: application/json" \
  -H "x-api-key: voxproof-secret-key-2024" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "'"$audioBase64"'"
  }'
```

### Using Python

```python
import base64
import requests

# Read and encode audio file
with open("path/to/audio.mp3", "rb") as f:
    audio_base64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    "http://localhost:8000/api/voice-detection",
    headers={
        "x-api-key": "voxproof-secret-key-2024",
        "Content-Type": "application/json"
    },
    json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_base64
    }
)

print(response.json())
```

### Using PowerShell

```powershell
# Encode audio file
$audioBytes = [System.IO.File]::ReadAllBytes("C:\path\to\audio.mp3")
$audioBase64 = [System.Convert]::ToBase64String($audioBytes)

# Create request body
$body = @{
    language = "English"
    audioFormat = "mp3"
    audioBase64 = $audioBase64
} | ConvertTo-Json

# Make request
$response = Invoke-RestMethod -Uri "http://localhost:8000/api/voice-detection" `
    -Method Post `
    -Headers @{"x-api-key" = "voxproof-secret-key-2024"} `
    -ContentType "application/json" `
    -Body $body

$response | ConvertTo-Json
```

## 🧠 How It Works

### Audio Processing Pipeline

1. **Base64 Decoding**: Decode the Base64 encoded MP3 data
2. **MP3 Conversion**: Convert MP3 to waveform using pydub
3. **Resampling**: Resample to 16kHz mono
4. **Normalization**: Normalize audio to [-1, 1] range

### Feature Extraction

- **MFCCs**: 13 Mel-frequency cepstral coefficients (mean values)
- **Pitch (F0)**: Fundamental frequency statistics (mean, std)
- **Spectral Rolloff**: Frequency below which 85% of energy is contained
- **Zero Crossing Rate**: Measure of signal noisiness

### Deep Embeddings

Uses `facebook/wav2vec2-base-960h` from HuggingFace Transformers:

- Pre-trained speech representation model
- Extracts 768-dimensional embeddings
- Mean-pooled across time dimension

### Classifier

Neural network architecture:

- Input: 786 dimensions (18 acoustic + 768 wav2vec2)
- 4 hidden layers with BatchNorm, ReLU, Dropout
- Output: Sigmoid for binary classification

### Explanation System

Rule-based explanations analyzing:

- Pitch stability/variability
- Signal clarity (zero crossing rate)
- Spectral characteristics
- Audio duration

## 🔒 Security

- API key authentication required
- CORS middleware configured
- Input validation with Pydantic

## ⚡ Performance Optimizations

- Models loaded once at startup
- Singleton pattern for processors
- `torch.no_grad()` for inference
- Async request handling

## 📝 Training Your Own Model

The included weights are randomly initialized for demo purposes. To train:

1. Collect labeled audio samples (AI-generated vs human)
2. Extract features using the `AudioProcessor` and `Wav2VecEmbedder`
3. Train the `VoiceClassifier` using your dataset
4. Save weights to `model/classifier.pth`:

```python
import torch
from model.model import VoiceClassifier

# After training
classifier = VoiceClassifier()
# ... training code ...
torch.save(classifier.state_dict(), "model/classifier.pth")
```

## 🐛 Troubleshooting

### FFmpeg not found

Install FFmpeg and ensure it's in your PATH.

### CUDA out of memory

The system will automatically use CPU if CUDA is unavailable.

### Model download slow

The wav2vec2 model (~360MB) is downloaded on first run. Subsequent runs use cached model.

### Invalid audio format

Ensure audio is valid MP3 format before encoding to Base64.

## 📄 License

MIT License - Built for AI Security Hackathon

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

Built with ❤️ for the AI Security Hackathon
