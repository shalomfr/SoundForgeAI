# 🚀 Deploy SoundForge AI to Render

## Quick Deploy (5 Minutes)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit - SoundForge AI"
git branch -M main
git remote add origin https://github.com/shalomfr/SoundForgeAI.git
git push -u origin main
```

### Step 2: Connect to Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account
4. Select the `SoundForgeAI` repository

### Step 3: Configure Build

Render will auto-detect the `render.yaml` file! Just verify:

- **Name**: `soundforge-ai`
- **Runtime**: Python
- **Build Command**: 
  ```bash
  pip install -r backend/requirements.txt && cd frontend && npm ci && npm run build && mkdir -p ../backend/static && cp -r dist/* ../backend/static/
  ```
- **Start Command**: 
  ```bash
  cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
  ```
- **Plan**: Free or Starter

### Step 4: Deploy! 🎉

Click **"Create Web Service"** and wait 5-10 minutes.

Your app will be live at: `https://soundforge-ai.onrender.com`

---

## Environment Variables (Optional)

Add these in Render dashboard under "Environment":

```env
PYTHON_VERSION=3.11.0
NODE_VERSION=20
MAX_UPLOAD_SIZE=100MB
```

---

## Troubleshooting

### Build fails?
- Check that `render.yaml` is in root directory
- Verify Python 3.11 is specified
- Check all paths are correct

### App crashes?
- Check logs in Render dashboard
- Verify all dependencies are in `requirements.txt`
- Make sure static files are built

### Audio processing too slow?
- Upgrade to Starter plan for better CPU
- Consider adding Redis for job queue
- Enable caching for repeated files

---

## Upgrade to Production

For production use:

1. **Custom Domain**: Add your domain in Render settings
2. **HTTPS**: Automatic with Render
3. **Scaling**: Enable auto-scaling
4. **CDN**: Add CloudFlare for static assets
5. **Storage**: Add S3 for uploaded files
6. **Database**: Add PostgreSQL for user accounts

---

Made with ❤️ by SoundForge AI

