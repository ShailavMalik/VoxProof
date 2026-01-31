# VoxProof - Railway Deployment Guide

## Quick Deploy to Railway

### 1. Prerequisites

- Railway account (sign up at https://railway.app)
- GitHub repository (optional but recommended)

### 2. Deploy Steps

**Option A: Deploy from GitHub (Recommended)**

1. Push your code to GitHub
2. Go to https://railway.app
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your VoxProof repository
5. Railway will auto-detect Python and deploy

**Option B: Deploy from CLI**

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up
```

### 3. Configure Environment Variables

In Railway Dashboard → Your Project → Variables, add:

```
API_KEY=your-secure-production-key-here
MODEL_PATH=model/classifier.pth
WAV2VEC_MODEL=facebook/wav2vec2-base-960h
SAMPLE_RATE=16000
```

**Important**: Change `API_KEY` to a secure random string for production!

### 4. Important Notes

#### Model Weights

- The `model/classifier.pth` file should be in your repository or uploaded separately
- If the file doesn't exist, the API will create dummy weights (for testing only)
- For production, train a real model first: `python train.py`

#### First Deployment

- First deployment may take 3-5 minutes (downloading wav2vec2 model ~360MB)
- Subsequent deployments are faster (model is cached)
- Check logs in Railway dashboard for progress

#### Memory Requirements

- Minimum: 2GB RAM (CPU inference)
- Recommended: 4GB RAM for stable operation
- If you hit memory limits, consider using a smaller model or upgrading your Railway plan

#### Cold Starts

- Railway may sleep your app after inactivity (on free tier)
- First request after sleep will be slow (~30-60 seconds)
- Keep-alive pings can prevent sleeping (not recommended for free tier)

### 5. Testing Your Deployment

Once deployed, Railway will give you a URL like: `https://your-app.railway.app`

**Test the health endpoint:**

```bash
curl https://your-app.railway.app/health
```

**Test voice detection:**

```bash
# PowerShell
$audioBase64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("audio.mp3"))
$body = @{language="English"; audioFormat="mp3"; audioBase64=$audioBase64} | ConvertTo-Json
Invoke-RestMethod -Uri "https://your-app.railway.app/api/voice-detection" `
  -Method Post `
  -Headers @{"x-api-key"="your-api-key"} `
  -ContentType "application/json" `
  -Body $body
```

**View API Documentation:**

- Swagger UI: `https://your-app.railway.app/docs`
- ReDoc: `https://your-app.railway.app/redoc`

### 6. Monitoring

**View Logs:**

- Railway Dashboard → Your Project → Deployments → Click latest deployment → View Logs

**Watch for:**

- ✅ "VoxProof API Ready!" - successful startup
- ⏱️ Model loading time (should be ~5-10 seconds after first run)
- 🎯 Request logs with timings
- ❌ Any error messages

### 7. Troubleshooting

**Problem: Memory exceeded**

- Solution: Upgrade Railway plan or use CPU-only inference (default)

**Problem: Timeout during startup**

- Solution: Increase healthcheck timeout in `railway.json`
- First run downloads wav2vec2 model (~360MB), allow 5 minutes

**Problem: Model predictions are random**

- Solution: Train the model properly with real data: `python train.py`
- Current weights are dummy/random initialization

**Problem: Slow requests**

- Solution: First request after cold start is slow (~30s)
- Subsequent requests are fast (~2-5 seconds)

### 8. Scaling (Production)

For production workloads:

1. **Use GPU**: Consider GPU-enabled hosting for faster inference
2. **Cache Models**: Models are already cached in memory (singleton pattern)
3. **Load Balancing**: Railway handles this automatically on higher tiers
4. **Custom Domain**: Add in Railway dashboard → Settings → Domain

### 9. Cost Estimates

**Railway Pricing (as of 2024):**

- Hobby Plan: $5/month - 512MB RAM, $0.000231/minute beyond quota
- Pro Plan: $20/month - Higher limits

**Estimated Usage:**

- Startup: ~5-10 seconds (one-time per deployment)
- Per Request: ~2-5 seconds CPU inference
- Memory: ~1-2GB with wav2vec2 model loaded

### 10. Security Checklist

Before deploying to production:

- [ ] Change API_KEY to a strong random value
- [ ] Don't commit `.env` file (use Railway env vars)
- [ ] Enable HTTPS (automatic on Railway)
- [ ] Configure CORS appropriately in `app.py`
- [ ] Rate limit requests (add middleware)
- [ ] Monitor logs for suspicious activity

## Need Help?

- Railway Docs: https://docs.railway.app
- VoxProof Issues: Check your repository issues
- Discord: Join Railway Discord for community support
