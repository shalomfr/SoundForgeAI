# SoundForge AI 🎙️

**Professional AI-Powered Audio Enhancement**

Transform your audio with 12 professional-grade processing engines, all powered by AI. One click to enhance podcasts, interviews, voiceovers, and more.

![SoundForge AI](https://raw.githubusercontent.com/shalomfr/SoundForgeAI/main/.github/banner.png)

## ✨ Features

### Auto-Magic Processing
- **AI Analysis**: Automatically detects audio problems
- **Smart Processing**: Applies the right fixes in the right amounts
- **One-Click Enhancement**: No manual tweaking required

### 12 Professional Engines
| Engine | Description |
|--------|-------------|
| AI Analyzer | Detects noise, reverb, sibilance, and more |
| Noise Gate | Removes noise during silence |
| Spectral Denoiser | AI-powered background noise removal |
| De-Reverb | Reduces room echo and reverb |
| De-Esser | Tames harsh S and T sounds |
| Breath Remover | Automatically reduces breath sounds |
| Voice Enhancer | Adds clarity and presence |
| Smart EQ | Intelligent frequency balancing |
| Compressor | Smooth, consistent dynamics |
| Stereo Widener | Enhanced stereo image |
| Limiter | Prevents clipping and distortion |
| LUFS Normalizer | Broadcast-standard loudness |

### Presets
- **Auto Magic**: AI decides everything
- **Podcast**: Optimized for podcast episodes
- **Interview**: Clear dialogue for conversations
- **Audiobook**: Warm, intimate narration
- **Voiceover**: Broadcast-quality voice

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- FFmpeg (for audio processing)

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/shalomfr/SoundForgeAI.git
cd SoundForgeAI
```

2. **Setup Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Setup Frontend**
```bash
cd frontend
npm install
```

4. **Run Development Servers**

Backend (Terminal 1):
```bash
cd backend
uvicorn main:app --reload --port 8000
```

Frontend (Terminal 2):
```bash
cd frontend
npm run dev
```

5. **Open http://localhost:3000**

## 🌐 Deployment to Render

### Option 1: Using render.yaml (Recommended)

1. Push to GitHub
2. Connect your repo to Render
3. Render will auto-detect `render.yaml`
4. Deploy!

### Option 2: Manual Setup

1. Create a new **Web Service** on Render
2. Connect your GitHub repo
3. Configure:
   - **Build Command**: `cd backend && pip install -r requirements.txt && cd ../frontend && npm install && npm run build`
   - **Start Command**: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.11

### Option 3: Docker

```bash
docker build -t soundforge-ai .
docker run -p 8000:8000 soundforge-ai
```

## 📡 API Reference

### `POST /analyze`
Analyze audio file for problems.

**Request**: `multipart/form-data` with `file` field

**Response**:
```json
{
  "audio_info": {
    "duration_seconds": 30.5,
    "sample_rate": 44100,
    "channels": 1
  },
  "problems": {
    "noise_level": {"value": 0.6, "severity": "medium"},
    "reverb": {"value": 0.3, "severity": "low"}
  },
  "recommendations": {...}
}
```

### `POST /process`
Process audio with enhancement pipeline.

**Query Parameters**:
- `auto_mode`: boolean (default: true)
- `preset`: string (auto, podcast, interview, audiobook, voiceover)
- `noise_reduction`: float 0-1
- `de_reverb`: float 0-1
- `de_esser`: float 0-1
- `compression_ratio`: float 1-8
- `target_lufs`: float -24 to -6
- `output_format`: string (wav, mp3, flac)

**Request**: `multipart/form-data` with `file` field

**Response**: Audio file stream

### `GET /presets`
Get available processing presets.

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **librosa** - Audio analysis
- **pedalboard** - Spotify's audio effects library
- **noisereduce** - AI noise reduction
- **pyloudnorm** - LUFS normalization

### Frontend
- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **WaveSurfer.js** - Waveform visualization
- **Lucide React** - Icons

## 📝 License

MIT License - feel free to use for any project!

## 🙏 Credits

Built with love using:
- [Spotify Pedalboard](https://github.com/spotify/pedalboard)
- [noisereduce](https://github.com/timsainb/noisereduce)
- [librosa](https://librosa.org/)
- [WaveSurfer.js](https://wavesurfer-js.org/)

---

Made with ❤️ for audio creators everywhere

