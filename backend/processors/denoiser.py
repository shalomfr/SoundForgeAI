"""
Advanced Denoiser - Combines multiple noise reduction techniques.
Includes spectral gating, AI-powered noise reduction, and adaptive filtering.
"""

import numpy as np
import noisereduce as nr
from scipy import signal
from scipy.ndimage import median_filter
import librosa
from typing import Optional, Tuple
from pedalboard import Pedalboard, NoiseGate


class Denoiser:
    """
    Professional-grade denoiser combining multiple techniques:
    1. Noise Gate - Removes noise during silence
    2. Spectral Denoising - AI-powered spectral subtraction
    3. Adaptive Filtering - Real-time noise tracking
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def process(self, audio: np.ndarray, 
                strength: float = 0.5,
                gate_threshold_db: float = -50,
                use_noise_gate: bool = True,
                use_spectral_denoise: bool = True,
                noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply comprehensive noise reduction.
        
        Args:
            audio: Input audio (channels, samples)
            strength: Overall denoising strength (0-1)
            gate_threshold_db: Noise gate threshold in dB
            use_noise_gate: Whether to apply noise gate
            use_spectral_denoise: Whether to apply spectral denoising
            noise_profile: Optional noise sample for better reduction
        
        Returns:
            Denoised audio
        """
        if strength <= 0:
            return audio
        
        result = audio.copy()
        
        # Ensure correct shape
        if result.ndim == 1:
            result = result.reshape(1, -1)
        
        # Step 1: Noise Gate (removes noise during silence)
        if use_noise_gate:
            result = self._apply_noise_gate(result, gate_threshold_db)
        
        # Step 2: Spectral Denoising (AI-powered)
        if use_spectral_denoise:
            result = self._apply_spectral_denoise(result, strength, noise_profile)
        
        # Step 3: High-pass filter to remove rumble
        result = self._apply_highpass(result, cutoff=80)
        
        return result
    
    def _apply_noise_gate(self, audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """Apply noise gate using pedalboard."""
        result = audio.copy()
        
        # Process each channel
        for ch in range(audio.shape[0]):
            # Pedalboard expects (samples,) or (samples, channels)
            channel_audio = audio[ch].astype(np.float32)
            
            board = Pedalboard([
                NoiseGate(
                    threshold_db=threshold_db,
                    attack_ms=5.0,
                    release_ms=50.0,
                    ratio=10.0
                )
            ])
            
            result[ch] = board(channel_audio, self.sample_rate)
        
        return result
    
    def _apply_spectral_denoise(self, audio: np.ndarray, 
                                 strength: float,
                                 noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply AI-powered spectral noise reduction."""
        result = audio.copy()
        
        # Map strength to prop_decrease (0.5 = moderate, 1.0 = aggressive)
        prop_decrease = strength * 0.8 + 0.2  # Range: 0.2 to 1.0
        
        for ch in range(audio.shape[0]):
            channel_audio = audio[ch]
            
            # Use noisereduce library with stationary noise model
            if noise_profile is not None and noise_profile.ndim >= 1:
                # Use provided noise profile
                noise_sample = noise_profile if noise_profile.ndim == 1 else noise_profile[0]
                reduced = nr.reduce_noise(
                    y=channel_audio,
                    sr=self.sample_rate,
                    y_noise=noise_sample,
                    prop_decrease=prop_decrease,
                    stationary=True,
                    n_fft=2048,
                    hop_length=512
                )
            else:
                # Auto-detect noise (non-stationary mode for better results)
                reduced = nr.reduce_noise(
                    y=channel_audio,
                    sr=self.sample_rate,
                    prop_decrease=prop_decrease,
                    stationary=False,
                    n_fft=2048,
                    hop_length=512,
                    thresh_n_mult_nonstationary=2.0,
                    n_std_thresh_stationary=1.5
                )
            
            result[ch] = reduced
        
        return result
    
    def _apply_highpass(self, audio: np.ndarray, cutoff: float = 80) -> np.ndarray:
        """Apply high-pass filter to remove low frequency rumble."""
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        # Design butterworth high-pass filter
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        
        result = audio.copy()
        for ch in range(audio.shape[0]):
            result[ch] = signal.filtfilt(b, a, audio[ch])
        
        return result
    
    def extract_noise_profile(self, audio: np.ndarray, 
                              duration_ms: int = 500) -> np.ndarray:
        """
        Extract noise profile from quietest part of audio.
        
        Args:
            audio: Input audio
            duration_ms: Duration of noise sample to extract
        
        Returns:
            Noise profile array
        """
        if audio.ndim == 2:
            mono = np.mean(audio, axis=0)
        else:
            mono = audio
        
        # Find quietest section
        frame_length = int(self.sample_rate * duration_ms / 1000)
        hop = frame_length // 4
        
        min_rms = float('inf')
        min_start = 0
        
        for start in range(0, len(mono) - frame_length, hop):
            frame = mono[start:start + frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            if rms < min_rms and rms > 1e-10:  # Avoid completely silent sections
                min_rms = rms
                min_start = start
        
        return mono[min_start:min_start + frame_length]
    
    def remove_breath_sounds(self, audio: np.ndarray, 
                             sensitivity: float = 0.5) -> np.ndarray:
        """
        Detect and attenuate breath sounds.
        
        Args:
            audio: Input audio (channels, samples)
            sensitivity: Detection sensitivity (0-1)
        
        Returns:
            Audio with reduced breath sounds
        """
        if audio.ndim == 2:
            mono = np.mean(audio, axis=0)
        else:
            mono = audio
            audio = audio.reshape(1, -1)
        
        # Detect breath sounds using spectral characteristics
        frame_length = int(0.03 * self.sample_rate)  # 30ms frames
        hop = frame_length // 2
        
        # Get spectral features
        flatness = librosa.feature.spectral_flatness(y=mono, hop_length=hop)[0]
        rms = librosa.feature.rms(y=mono, frame_length=frame_length, hop_length=hop)[0]
        centroid = librosa.feature.spectral_centroid(y=mono, sr=self.sample_rate, hop_length=hop)[0]
        
        # Normalize features
        rms_norm = rms / (np.max(rms) + 1e-10)
        centroid_norm = centroid / (self.sample_rate / 2)
        
        # Breath detection: high flatness, moderate RMS, lower centroid
        avg_rms = np.mean(rms_norm)
        breath_mask = (
            (flatness > 0.05 * (1 - sensitivity * 0.5)) & 
            (rms_norm > avg_rms * 0.05) & 
            (rms_norm < avg_rms * (0.8 - sensitivity * 0.3)) &
            (centroid_norm < 0.3)
        )
        
        # Smooth the mask
        breath_mask = median_filter(breath_mask.astype(float), size=5)
        
        # Create gain envelope
        gain = 1 - (breath_mask * 0.9 * sensitivity)  # Reduce breath by up to 90%
        
        # Interpolate gain to sample length
        gain_samples = np.interp(
            np.linspace(0, len(gain), audio.shape[1]),
            np.arange(len(gain)),
            gain
        )
        
        # Apply gain with smoothing to avoid clicks
        window_size = int(0.01 * self.sample_rate)  # 10ms smoothing
        if window_size > 1:
            gain_samples = np.convolve(gain_samples, np.ones(window_size)/window_size, mode='same')
        
        result = audio * gain_samples
        
        return result

