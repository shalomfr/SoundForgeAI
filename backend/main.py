"""
SoundForge AI - Professional Audio Enhancement API
FastAPI backend with AI-powered audio processing
"""

import os
import uuid
import asyncio
from typing import Optional
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

from processors.pipeline import AudioPipeline, ProcessingSettings
from utils.audio_utils import load_audio, save_audio, get_audio_info

# Initialize FastAPI app
app = FastAPI(
    title="SoundForge AI API",
    description="Professional AI-powered audio enhancement with 12 processing engines",
    version="1.0.0"
)

# Serve static files in production
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize audio pipeline
pipeline = AudioPipeline(sample_rate=44100)

# In-memory storage for processed files (use Redis/S3 in production)
processed_files = {}


class ProcessingRequest(BaseModel):
    """Request body for processing settings."""
    auto_mode: bool = True
    preset: str = "auto"  # auto, podcast, interview, audiobook, voiceover
    
    # Individual toggles
    enable_noise_gate: bool = True
    enable_denoiser: bool = True
    enable_de_reverb: bool = True
    enable_de_esser: bool = True
    enable_breath_removal: bool = True
    enable_eq: bool = True
    enable_compression: bool = True
    enable_voice_enhance: bool = True
    enable_stereo_enhance: bool = False
    enable_limiter: bool = True
    enable_normalize: bool = True
    
    # Manual settings
    noise_reduction_strength: float = 0.5
    de_reverb_strength: float = 0.3
    de_esser_strength: float = 0.4
    compression_ratio: float = 3.0
    eq_preset: str = "podcast"
    target_lufs: float = -16.0


class AnalysisResponse(BaseModel):
    """Response for audio analysis."""
    problems: dict
    recommendations: dict
    audio_info: dict


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": "SoundForge AI API",
        "status": "running",
        "version": "1.0.0",
        "engines": 12
    }


@app.get("/presets")
async def get_presets():
    """Get available processing presets."""
    return {
        "presets": [
            {
                "id": "auto",
                "name": "Auto Magic",
                "description": "AI analyzes and fixes everything automatically",
                "icon": "sparkles"
            },
            {
                "id": "podcast",
                "name": "Podcast",
                "description": "Optimized for podcast episodes",
                "icon": "microphone"
            },
            {
                "id": "interview",
                "name": "Interview",
                "description": "Clear dialogue for interviews",
                "icon": "users"
            },
            {
                "id": "audiobook",
                "name": "Audiobook",
                "description": "Warm, intimate voice for narration",
                "icon": "book"
            },
            {
                "id": "voiceover",
                "name": "Voiceover",
                "description": "Broadcast-quality voice",
                "icon": "megaphone"
            }
        ]
    }


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze audio file and detect problems.
    Returns detected issues and recommended settings.
    """
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Load audio
        audio, sr = load_audio(contents)
        
        # Get audio info
        info = get_audio_info(audio, sr)
        
        # Analyze
        report = pipeline.analyze_only(audio)
        
        return {
            "audio_info": info,
            "problems": report["problems"],
            "recommendations": report["recommendations"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing audio: {str(e)}")


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    auto_mode: bool = True,
    preset: str = "auto",
    noise_reduction: float = 0.5,
    de_reverb: float = 0.3,
    de_esser: float = 0.4,
    compression_ratio: float = 3.0,
    target_lufs: float = -16.0,
    output_format: str = "wav"
):
    """
    Process audio file with enhancement pipeline.
    Returns processed audio file.
    """
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Load audio
        audio, sr = load_audio(contents)
        
        # Create settings
        if preset != "auto" and preset in pipeline.get_presets():
            settings = pipeline.get_presets()[preset]
        else:
            settings = ProcessingSettings(
                auto_mode=auto_mode,
                preset=preset,
                noise_reduction_strength=noise_reduction,
                de_reverb_strength=de_reverb,
                de_esser_strength=de_esser,
                compression_ratio=compression_ratio,
                target_lufs=target_lufs
            )
        
        # Process audio
        result = pipeline.process(audio, settings)
        
        # Save to bytes
        output_bytes = save_audio(result.audio, sr, format=output_format)
        
        # Generate filename
        original_name = file.filename or "audio"
        base_name = os.path.splitext(original_name)[0]
        output_filename = f"{base_name}_enhanced.{output_format}"
        
        # Return as streaming response
        return StreamingResponse(
            BytesIO(output_bytes),
            media_type=f"audio/{output_format}",
            headers={
                "Content-Disposition": f'attachment; filename="{output_filename}"',
                "X-Processing-Time-Ms": str(round(result.processing_time_ms, 2)),
                "X-Problems-Detected": str(len([k for k, v in result.problems_detected.items() if v > 0.3])),
                "X-Processors-Applied": str(len(result.processing_applied))
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")


@app.post("/process-with-details")
async def process_audio_with_details(
    file: UploadFile = File(...),
    auto_mode: bool = True,
    preset: str = "auto"
):
    """
    Process audio and return detailed results including analysis.
    """
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Load audio
        audio, sr = load_audio(contents)
        
        # Get original info
        original_info = get_audio_info(audio, sr)
        
        # Create settings
        if preset != "auto" and preset in pipeline.get_presets():
            settings = pipeline.get_presets()[preset]
        else:
            settings = ProcessingSettings(auto_mode=auto_mode, preset=preset)
        
        # Process audio
        result = pipeline.process(audio, settings)
        
        # Get processed info
        processed_info = get_audio_info(result.audio, sr)
        
        # Save to bytes
        output_bytes = save_audio(result.audio, sr, format="wav")
        
        # Store temporarily
        file_id = str(uuid.uuid4())
        processed_files[file_id] = {
            "audio": output_bytes,
            "format": "wav"
        }
        
        return {
            "file_id": file_id,
            "original_info": original_info,
            "processed_info": processed_info,
            "problems_detected": result.problems_detected,
            "processing_applied": result.processing_applied,
            "settings_used": result.settings_used,
            "processing_time_ms": round(result.processing_time_ms, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")


@app.get("/download/{file_id}")
async def download_processed(file_id: str):
    """Download a processed audio file by ID."""
    if file_id not in processed_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_data = processed_files[file_id]
    
    return StreamingResponse(
        BytesIO(file_data["audio"]),
        media_type=f"audio/{file_data['format']}",
        headers={
            "Content-Disposition": f'attachment; filename="enhanced.{file_data["format"]}"'
        }
    )


@app.post("/process-realtime")
async def process_realtime_chunk(
    audio_data: bytes,
    sample_rate: int = 44100,
    auto_mode: bool = True
):
    """
    Process a real-time audio chunk.
    For WebSocket streaming in production.
    """
    try:
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_data, dtype=np.float32)
        audio = audio.reshape(1, -1)
        
        # Quick processing (subset of full pipeline for latency)
        settings = ProcessingSettings(
            auto_mode=auto_mode,
            enable_stereo_enhance=False,  # Skip for mono realtime
            enable_breath_removal=False   # Skip for latency
        )
        
        result = pipeline.process(audio, settings)
        
        return {
            "audio": result.audio.tobytes(),
            "processing_time_ms": result.processing_time_ms
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


# Cleanup old files periodically
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("SoundForge AI API starting...")
    print(f"Loaded {len(pipeline.get_presets())} presets")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    processed_files.clear()
    print("SoundForge AI API shutting down...")


# Serve frontend in production
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve frontend static files."""
    if STATIC_DIR.exists():
        # Try to serve the exact file
        file_path = STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # Otherwise serve index.html for SPA routing
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Not found")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

