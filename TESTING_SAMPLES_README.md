# 🎤 VoxProof Test Audio Samples - Summary

## ✅ What's Generated

### Sample Files Created

```
test_samples/
├── human_sample_1.txt    (20.8 KB)  ← Human voice - should detect as HUMAN
├── human_sample_2.txt    (35.6 KB)  ← Human voice - should detect as HUMAN
├── human_sample_3.txt    (36.1 KB)  ← Human voice - should detect as HUMAN
├── ai_sample_1.txt       (29.7 KB)  ← AI-generated - should detect as AI
├── ai_sample_2.txt       (29.4 KB)  ← AI-generated - should detect as AI
└── ai_sample_3.txt       (24.6 KB)  ← AI-generated - should detect as AI
```

### Test Scripts Created

- `test_api_samples.py` - Automated test runner (recommended)
- `test_curl_human.json` - Quick curl test for human sample
- `test_curl_ai.json` - Quick curl test for AI sample
- `TEST_GUIDE.md` - Comprehensive testing documentation

---

## 🚀 How to Test Your API

### Option 1: Automated Test (Python) - Quickest

```bash
python test_api_samples.py
```

**What it does:**

- Tests all 6 samples automatically
- Shows pass/fail for each
- Displays success rate
- Shows confidence scores

### Option 2: Manual cURL Test

```bash
# Test human sample
curl -X POST "https://your-render-url.onrender.com/api/voice-detection" \
  -H "x-api-key: 99d8f7fefa2c12ce971e4b320ee3af70" \
  -H "Content-Type: application/json" \
  -d @test_curl_human.json

# Test AI sample
curl -X POST "https://your-render-url.onrender.com/api/voice-detection" \
  -H "x-api-key: 99d8f7fefa2c12ce971e4b320ee3af70" \
  -H "Content-Type: application/json" \
  -d @test_curl_ai.json
```

### Option 3: Postman (GUI)

1. Create new POST request
2. URL: `https://your-render-url.onrender.com/api/voice-detection`
3. Headers:
   - `x-api-key: 99d8f7fefa2c12ce971e4b320ee3af70`
   - `Content-Type: application/json`
4. Body (raw): Copy contents of `test_curl_human.json`
5. Click Send

### Option 4: Frontend Dashboard

1. Go to `https://voxproof.vercel.app`
2. Update `frontend/.env.local`:
   ```
   VITE_API_BASE_URL=https://your-render-url.onrender.com
   ```
3. Upload `test_samples/human_sample_1.txt` or `test_samples/ai_sample_1.txt`

---

## 📊 Sample Details

### Human Samples (from Common Voice Dataset)

- **Source:** Mozilla Common Voice project
- **Languages:** English, Hindi
- **Characteristics:** Natural speech patterns, background noise variations, human imperfections
- **Expected Result:** `"classification": "HUMAN"` with high confidence

### AI Samples (from Synthetic Dataset)

- **Generator:** Mix of ElevenLabs, PyTTSX3, Coqui TTS
- **Characteristics:** Too-smooth prosody, unnatural pitch patterns, no background noise
- **Expected Result:** `"classification": "AI"` with high confidence

---

## ⚠️ Before You Test

1. **Update Render URL**
   - In `test_api_samples.py`, change:
     ```python
     API_URL = "https://your-actual-render-url.onrender.com/api/voice-detection"
     ```

2. **Check API is Running**
   - Visit: `https://your-render-url.onrender.com/health`
   - Should return: `{"status": "ok"}`

3. **First Request is Slow**
   - First test takes 30+ seconds (model loading)
   - Subsequent tests are 2-8 seconds

---

## 🎯 Expected Results Example

### Successful Test Output

```
==========================================
 VOXPROOF API TEST SUITE
==========================================
Timestamp: 2026-02-27T10:30:00.123456

📍 TESTING HUMAN SAMPLES:
------------------------------------------
Testing Human Voice 1... ✅ HUMAN (Confidence: 94.23%)
Testing Human Voice 2... ✅ HUMAN (Confidence: 91.56%)
Testing Human Voice 3... ✅ HUMAN (Confidence: 89.12%)

🤖 TESTING AI-GENERATED SAMPLES:
------------------------------------------
Testing AI-Generated 1... ✅ AI (Confidence: 87.34%)
Testing AI-Generated 2... ✅ AI (Confidence: 92.01%)
Testing AI-Generated 3... ✅ AI (Confidence: 88.56%)

==========================================
TEST SUMMARY
==========================================
Total Tests: 6
✅ Passed: 6
❌ Failed: 0
Success Rate: 100.0%
==========================================
```

---

## 📝 JSON Request Format

Each test uses this JSON format:

```json
{
  "audio_base64": "SUQzBAAAAAAAI1NUVVQAAAAPAADDTmFtZTogUGF0ryBvZiBMZWFydmVzCkdQUm2Ym8P..."
}
```

Where `audio_base64` contains the entire MP3 encoded as base64.

---

## 🔍 Reading Result Details

```json
{
  "classification": "HUMAN", // Either "HUMAN" or "AI"
  "confidence": 0.9234, // Confidence score 0-1
  "is_ai": false, // True if AI-generated
  "explanation": "This voice sample exhibits natural variations in pitch jitter, energy dynamics, and spectral features characteristic of genuine human speech. The detected pitch jitter of 5.2% and MFCC delta variance of 12.1 are within natural ranges.",
  "processing_time_ms": 4234 // How long analysis took
}
```

---

## 📚 Files Reference

| File                       | Purpose                         |
| -------------------------- | ------------------------------- |
| `test_api_samples.py`      | Main test script (run this!)    |
| `test_curl_human.json`     | Raw curl payload for human test |
| `test_curl_ai.json`        | Raw curl payload for AI test    |
| `TEST_GUIDE.md`            | Detailed testing guide          |
| `generate_test_samples.py` | Script that generated samples   |

---

## 💡 Tips

1. **Save Logs:** `python test_api_samples.py > test_results.txt`
2. **Test Repeatedly:** Backend gets faster after first request
3. **Check Frontend:** Update `frontend/.env.local` with your Render URL
4. **Monitor API:** Visit `/docs` for live API documentation

---

## 🐛 Troubleshooting

| Error                   | Fix                                                 |
| ----------------------- | --------------------------------------------------- |
| `Cannot connect to API` | Check Render URL is correct                         |
| `401 Unauthorized`      | Verify API key: `99d8f7fefa2c12ce971e4b320ee3af70`  |
| `Request timeout`       | Normal for first request, wait up to 60s            |
| `File not found`        | Run from project root: `python test_api_samples.py` |

Happy Testing! 🚀
