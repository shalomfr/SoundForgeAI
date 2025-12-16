"""
Advanced Denoiser - Noise reduction using spectral gating.
Lightweight implementation without heavy dependencies.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, gaussian_filter1d
import librosa
from typing import Optional
from pedalboard import Pedalboard, NoiseGate


class Denoiser:
    """
    Professional-grade denoiser combining multiple techniques:
    1. Noise Gate - Removes noise during silence
    2. Spectral Gating - Frequency-based noise reduction
    3. High-pass filtering - Removes rumble
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.n_fft = 2048
        self.hop_length = 512
    
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
        
        # Step 2: Spectral Gating (lightweight noise reduction)
        if use_spectral_denoise:
            result = self._apply_spectral_gating(result, strength, noise_profile)
        
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
    
    def _apply_spectral_gating(self, audio: np.ndarray, 
                                strength: float,
                                noise_profile: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply spectral gating for noise reduction.
        This is a lightweight alternative to noisereduce.
        """
        result = audio.copy()
        
        for ch in range(audio.shape[0]):
            channel = audio[ch]
            
            # STFT
            stft = librosa.stft(channel, n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor
            if noise_profile is not None and len(noise_profile) > 0:
                noise_stft = librosa.stft(noise_profile, n_fft=self.n_fft, hop_length=self.hop_length)
                noise_mag = np.mean(np.abs(noise_stft), axis=1, keepdims=True)
            else:
                # Auto-estimate noise from quietest frames
                frame_energy = np.sum(magnitude ** 2, axis=0)
                quiet_frames = frame_energy < np.percentile(frame_energy, 10)
                if np.any(quiet_frames):
                    noise_mag = np.mean(magnitude[:, quiet_frames], axis=1, keepdims=True)
                else:
                    noise_mag = np.percentile(magnitude, 5, axis=1, keepdims=True)
            
            # Spectral subtraction with soft mask
            threshold = noise_mag * (1 + strength * 2)  # Scale threshold by strength
            
            # Create soft mask
            mask = np.clip((magnitude - threshold) / (magnitude + 1e-10), 0, 1)
            mask = gaussian_filter1d(mask, sigma=2, axis=1)  # Smooth temporally
            
            # Apply mask
            magnitude_cleaned = magnitude * mask
            
            # Ensure we don't remove too much
            min_magnitude = magnitude * (0.1 * (1 - strength))
            magnitude_cleaned = np.maximum(magnitude_cleaned, min_magnitude)
            
            # Reconstruct
            stft_cleaned = magnitude_cleaned * np.exp(1j * phase)
            result[ch] = librosa.istft(stft_cleaned, hop_length=self.hop_length, length=len(channel))
        
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
