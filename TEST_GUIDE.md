# VoxProof API Test Guide

## 📊 Test Samples Generated

### Human Voice Samples (Expected Result: HUMAN)

| #   | File                              | Size    | Sample Source        |
| --- | --------------------------------- | ------- | -------------------- |
| 1   | `test_samples/human_sample_1.txt` | 20.8 KB | Common Voice Dataset |
| 2   | `test_samples/human_sample_2.txt` | 35.6 KB | Common Voice Dataset |
| 3   | `test_samples/human_sample_3.txt` | 36.1 KB | Common Voice Dataset |

### AI-Generated Voice Samples (Expected Result: AI)

| #   | File                           | Size    | Sample Source     |
| --- | ------------------------------ | ------- | ----------------- |
| 1   | `test_samples/ai_sample_1.txt` | 29.7 KB | Synthetic Dataset |
| 2   | `test_samples/ai_sample_2.txt` | 29.4 KB | Synthetic Dataset |
| 3   | `test_samples/ai_sample_3.txt` | 24.6 KB | Synthetic Dataset |

---

## 🚀 Quick Test Commands

### 1. Test with cURL (Easiest)

**Test Human Sample:**

```bash
curl -X POST "https://voxproof-api.onrender.com/api/voice-detection" \
  -H "x-api-key: 99d8f7fefa2c12ce971e4b320ee3af70" \
  -H "Content-Type: application/json" \
  -d @test_curl_human.json
```

**Test AI Sample:**

```bash
curl -X POST "https://voxproof-api.onrender.com/api/voice-detection" \
  -H "x-api-key: 99d8f7fefa2c12ce971e4b320ee3af70" \
  -H "Content-Type: application/json" \
  -d @test_curl_ai.json
```

### 2. Test with Python

```python
import requests
import json

API_KEY = "99d8f7fefa2c12ce971e4b320ee3af70"
API_URL = "https://voxproof-api.onrender.com/api/voice-detection"

# Read base64 sample
with open("test_samples/human_sample_1.txt", "r") as f:
    audio_base64 = f.read()

# Test human sample
payload = {"audio_base64": audio_base64}
headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

response = requests.post(API_URL, json=payload, headers=headers)
result = response.json()

print("Result for Human Sample:")
print(f"  Classification: {result.get('classification')}")
print(f"  Confidence: {result.get('confidence'):.2%}")
print(f"  Explanation: {result.get('explanation')}")
```

### 3. Test with Postman (GUI)

1. **Create New Request:**
   - Method: `POST`
   - URL: `https://voxproof-api.onrender.com/api/voice-detection`

2. **Headers Tab:**

   ```
   x-api-key: 99d8f7fefa2c12ce971e4b320ee3af70
   Content-Type: application/json
   ```

3. **Body Tab (raw JSON):**

   ```json
   {
     "audio_base64": "paste-entire-contents-of-test_samples/human_sample_1.txt-here"
   }
   ```

4. **Click Send**

---

## 📋 Expected Results

### For Human Samples

```json
{
  "classification": "HUMAN",
  "confidence": 0.92,
  "is_ai": false,
  "explanation": "This voice sample exhibits natural variations...",
  "processing_time_ms": 4200
}
```

### For AI Samples

```json
{
  "classification": "AI",
  "confidence": 0.88,
  "is_ai": true,
  "explanation": "This voice sample shows characteristics of AI-generated speech...",
  "processing_time_ms": 3800
}
```

---

## 🧪 Testing All 6 Samples

Create this Python script to test all samples:

```python
import requests
import json
from pathlib import Path

API_KEY = "99d8f7fefa2c12ce971e4b320ee3af70"
API_URL = "https://voxproof-api.onrender.com/api/voice-detection"

headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

print("=" * 80)
print("TESTING ALL 6 VOXPROOF SAMPLES")
print("=" * 80)

# Test human samples
print("\n📍 HUMAN VOICE SAMPLES:")
print("-" * 80)
for i in range(1, 4):
    with open(f"test_samples/human_sample_{i}.txt", "r") as f:
        audio_base64 = f.read()

    response = requests.post(API_URL, json={"audio_base64": audio_base64}, headers=headers)
    result = response.json()

    print(f"[Human {i}] Classification: {result['classification']} | Confidence: {result['confidence']:.2%}")

# Test AI samples
print("\n🤖 AI-GENERATED SAMPLES:")
print("-" * 80)
for i in range(1, 4):
    with open(f"test_samples/ai_sample_{i}.txt", "r") as f:
        audio_base64 = f.read()

    response = requests.post(API_URL, json={"audio_base64": audio_base64}, headers=headers)
    result = response.json()

    print(f"[AI {i}] Classification: {result['classification']} | Confidence: {result['confidence']:.2%}")

print("\n" + "=" * 80)
```

---

## ⚠️ Important Notes

1. **First Request Slow (~30 sec):** Free Render tier has cold starts
2. **API Key:** Always include `x-api-key: 99d8f7fefa2c12ce971e4b320ee3af70` header
3. **Base64 Format:** Audio must be base64-encoded MP3/WAV
4. **Render URL:** Replace `https://voxproof-api.onrender.com` with your actual Render URL

---

## 🔗 API Endpoints

| Endpoint               | Method | Purpose                                 |
| ---------------------- | ------ | --------------------------------------- |
| `/health`              | GET    | Health check                            |
| `/docs`                | GET    | Interactive API documentation (Swagger) |
| `/api/voice-detection` | POST   | Detect AI vs Human voice                |

---

## 📞 Troubleshooting

**Error: 401 Unauthorized**

- Check API key is correct
- Ensure `x-api-key` header is included

**Error: 422 Invalid audio**

- Audio must be valid base64
- Check file wasn't corrupted

**Timeout after 30+ seconds**

- Normal for first request (model loading)
- Subsequent requests are faster

**Error: Backend not responding**

- Check if Render deployment succeeded
- Visit health endpoint: `https://your-url.onrender.com/health`
