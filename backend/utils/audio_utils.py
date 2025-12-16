"""
Audio utility functions for loading, saving, and manipulating audio files.
"""

import numpy as np
import soundfile as sf
import librosa
from io import BytesIO
from typing import Tuple, Optional
import tempfile
import os


def load_audio(file_bytes: bytes, target_sr: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Load audio from bytes into numpy array.
    
    Args:
        file_bytes: Raw audio file bytes
        target_sr: Target sample rate (default 44100)
    
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    # Create a temporary file to handle various formats
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    try:
        # Load with librosa for format compatibility
        audio, sr = librosa.load(tmp_path, sr=target_sr, mono=False)
        
        # Ensure we have the right shape (samples,) or (channels, samples)
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)  # Mono to (1, samples)
        
        return audio, sr
    finally:
        os.unlink(tmp_path)


def save_audio(audio: np.ndarray, sr: int, format: str = 'wav') -> bytes:
    """
    Save audio array to bytes.
    
    Args:
        audio: Audio data as numpy array (channels, samples) or (samples,)
        sr: Sample rate
        format: Output format ('wav', 'mp3', 'flac')
    
    Returns:
        Audio file as bytes
    """
    # Ensure correct shape for soundfile
    if audio.ndim == 2:
        audio = audio.T  # (channels, samples) -> (samples, channels)
    
    buffer = BytesIO()
    sf.write(buffer, audio, sr, format=format.upper())
    buffer.seek(0)
    return buffer.read()


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Audio data
        target_db: Target peak level in dB
    
    Returns:
        Normalized audio
    """
    target_amplitude = 10 ** (target_db / 20)
    current_peak = np.max(np.abs(audio))
    
    if current_peak > 0:
        gain = target_amplitude / current_peak
        return audio * gain
    return audio


def get_audio_info(audio: np.ndarray, sr: int) -> dict:
    """
    Get information about audio file.
    
    Args:
        audio: Audio data (channels, samples)
        sr: Sample rate
    
    Returns:
        Dictionary with audio information
    """
    if audio.ndim == 1:
        channels = 1
        samples = len(audio)
    else:
        channels, samples = audio.shape
    
    duration = samples / sr
    peak_db = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
    rms_db = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-10)
    
    return {
        'sample_rate': sr,
        'channels': channels,
        'samples': samples,
        'duration_seconds': round(duration, 2),
        'peak_db': round(peak_db, 2),
        'rms_db': round(rms_db, 2)
    }


def stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    """Convert stereo to mono by averaging channels."""
    if audio.ndim == 2 and audio.shape[0] == 2:
        return np.mean(audio, axis=0, keepdims=True)
    return audio


def mono_to_stereo(audio: np.ndarray) -> np.ndarray:
    """Convert mono to stereo by duplicating channel."""
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)
    if audio.shape[0] == 1:
        return np.vstack([audio, audio])
    return audio


def apply_fade(audio: np.ndarray, sr: int, fade_in_ms: int = 10, fade_out_ms: int = 10) -> np.ndarray:
    """
    Apply fade in/out to avoid clicks.
    
    Args:
        audio: Audio data (channels, samples)
        sr: Sample rate
        fade_in_ms: Fade in duration in milliseconds
        fade_out_ms: Fade out duration in milliseconds
    
    Returns:
        Audio with fades applied
    """
    fade_in_samples = int(sr * fade_in_ms / 1000)
    fade_out_samples = int(sr * fade_out_ms / 1000)
    
    result = audio.copy()
    
    # Fade in
    if fade_in_samples > 0:
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        if result.ndim == 2:
            result[:, :fade_in_samples] *= fade_in_curve
        else:
            result[:fade_in_samples] *= fade_in_curve
    
    # Fade out
    if fade_out_samples > 0:
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        if result.ndim == 2:
            result[:, -fade_out_samples:] *= fade_out_curve
        else:
            result[-fade_out_samples:] *= fade_out_curve
    
    return result

