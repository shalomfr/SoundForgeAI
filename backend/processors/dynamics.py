"""
Dynamics Processor - Compressor, Limiter, and Expander.
Professional-grade dynamics processing for voice.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import pyloudnorm as pyln
from pedalboard import Pedalboard, Compressor, Limiter, Gain
from typing import Tuple


class DynamicsProcessor:
    """
    Professional dynamics processing:
    1. Compressor - Reduces dynamic range
    2. Limiter - Prevents clipping
    3. LUFS Normalization - Broadcast standard loudness
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(sample_rate)
    
    def compress(self, audio: np.ndarray,
                 threshold_db: float = -20,
                 ratio: float = 4.0,
                 attack_ms: float = 10,
                 release_ms: float = 100,
                 makeup_gain_db: float = 0) -> np.ndarray:
        """
        Apply compression to reduce dynamic range.
        
        Args:
            audio: Input audio (channels, samples)
            threshold_db: Compression threshold in dB
            ratio: Compression ratio (e.g., 4 = 4:1)
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds
            makeup_gain_db: Makeup gain in dB
        
        Returns:
            Compressed audio
        """
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        result = np.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            channel = audio[ch].astype(np.float32)
            
            board = Pedalboard([
                Compressor(
                    threshold_db=threshold_db,
                    ratio=ratio,
                    attack_ms=attack_ms,
                    release_ms=release_ms
                ),
                Gain(gain_db=makeup_gain_db)
            ])
            
            result[ch] = board(channel, self.sample_rate)
        
        return result
    
    def multiband_compress(self, audio: np.ndarray,
                           low_threshold_db: float = -24,
                           mid_threshold_db: float = -20,
                           high_threshold_db: float = -18,
                           ratio: float = 3.0) -> np.ndarray:
        """
        Apply multiband compression for more transparent dynamics control.
        Splits audio into 3 bands: low (<300Hz), mid (300-3000Hz), high (>3000Hz)
        
        Args:
            audio: Input audio
            low_threshold_db: Threshold for low band
            mid_threshold_db: Threshold for mid band
            high_threshold_db: Threshold for high band
            ratio: Compression ratio for all bands
        
        Returns:
            Multiband compressed audio
        """
        from scipy import signal
        
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        nyquist = self.sample_rate / 2
        
        # Design crossover filters
        low_cutoff = 300 / nyquist
        high_cutoff = 3000 / nyquist
        
        # Butterworth filters
        b_low, a_low = signal.butter(4, low_cutoff, btype='low')
        b_mid, a_mid = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
        b_high, a_high = signal.butter(4, high_cutoff, btype='high')
        
        result = np.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            # Split into bands
            low_band = signal.filtfilt(b_low, a_low, audio[ch])
            mid_band = signal.filtfilt(b_mid, a_mid, audio[ch])
            high_band = signal.filtfilt(b_high, a_high, audio[ch])
            
            # Compress each band
            low_compressed = self._compress_band(low_band, low_threshold_db, ratio)
            mid_compressed = self._compress_band(mid_band, mid_threshold_db, ratio)
            high_compressed = self._compress_band(high_band, high_threshold_db, ratio)
            
            # Sum bands
            result[ch] = low_compressed + mid_compressed + high_compressed
        
        return result
    
    def _compress_band(self, audio: np.ndarray, 
                       threshold_db: float, 
                       ratio: float) -> np.ndarray:
        """Compress a single band using soft-knee compression."""
        # Calculate envelope
        envelope = np.abs(audio)
        envelope = gaussian_filter1d(envelope, sigma=int(0.01 * self.sample_rate))
        
        # Convert to dB
        envelope_db = 20 * np.log10(envelope + 1e-10)
        
        # Calculate gain reduction
        threshold = threshold_db
        knee_width = 6  # dB
        
        gain_db = np.zeros_like(envelope_db)
        
        # Below threshold
        below = envelope_db < (threshold - knee_width / 2)
        gain_db[below] = 0
        
        # In knee region
        in_knee = (envelope_db >= (threshold - knee_width / 2)) & (envelope_db <= (threshold + knee_width / 2))
        knee_input = envelope_db[in_knee] - threshold + knee_width / 2
        gain_db[in_knee] = ((1 / ratio - 1) * (knee_input ** 2)) / (2 * knee_width)
        
        # Above threshold
        above = envelope_db > (threshold + knee_width / 2)
        gain_db[above] = (threshold + (envelope_db[above] - threshold) / ratio) - envelope_db[above]
        
        # Convert gain to linear and apply
        gain_linear = 10 ** (gain_db / 20)
        
        # Smooth the gain
        gain_linear = gaussian_filter1d(gain_linear, sigma=int(0.005 * self.sample_rate))
        
        return audio * gain_linear
    
    def limit(self, audio: np.ndarray, 
              threshold_db: float = -1.0,
              release_ms: float = 50) -> np.ndarray:
        """
        Apply brick-wall limiting to prevent clipping.
        
        Args:
            audio: Input audio
            threshold_db: Limiter threshold (usually -1 to -0.1)
            release_ms: Release time
        
        Returns:
            Limited audio
        """
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        result = np.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            channel = audio[ch].astype(np.float32)
            
            board = Pedalboard([
                Limiter(
                    threshold_db=threshold_db,
                    release_ms=release_ms
                )
            ])
            
            result[ch] = board(channel, self.sample_rate)
        
        return result
    
    def normalize_lufs(self, audio: np.ndarray, 
                       target_lufs: float = -16.0) -> np.ndarray:
        """
        Normalize audio to target LUFS (broadcast standard).
        
        Args:
            audio: Input audio (channels, samples)
            target_lufs: Target loudness in LUFS (e.g., -16 for podcasts, -14 for streaming)
        
        Returns:
            Normalized audio
        """
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        # Measure current loudness
        # pyloudnorm expects (samples, channels)
        audio_for_meter = audio.T
        
        try:
            current_lufs = self.meter.integrated_loudness(audio_for_meter)
        except Exception:
            # If measurement fails, use RMS-based estimation
            rms = np.sqrt(np.mean(audio ** 2))
            current_lufs = 20 * np.log10(rms + 1e-10) - 10
        
        # Calculate required gain
        if np.isfinite(current_lufs):
            gain_db = target_lufs - current_lufs
            gain_linear = 10 ** (gain_db / 20)
            
            # Apply gain with soft clipping to prevent distortion
            result = audio * gain_linear
            
            # Soft clip if needed
            result = np.tanh(result * 0.9) / 0.9
            
            return result
        
        return audio
    
    def normalize_peak(self, audio: np.ndarray, 
                       target_db: float = -1.0) -> np.ndarray:
        """
        Normalize audio to target peak level.
        
        Args:
            audio: Input audio
            target_db: Target peak in dB
        
        Returns:
            Peak-normalized audio
        """
        current_peak = np.max(np.abs(audio))
        if current_peak > 0:
            target_linear = 10 ** (target_db / 20)
            gain = target_linear / current_peak
            return audio * gain
        return audio
    
    def auto_gain(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Automatically set optimal gain level.
        
        Args:
            audio: Input audio
        
        Returns:
            Tuple of (gained audio, gain applied in dB)
        """
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        # Measure current levels
        rms = np.sqrt(np.mean(audio ** 2))
        peak = np.max(np.abs(audio))
        
        # Target: RMS around -20dB with peak headroom
        target_rms_db = -20
        target_rms = 10 ** (target_rms_db / 20)
        
        # Calculate required gain
        if rms > 0:
            gain = target_rms / rms
            
            # Ensure we don't clip
            if peak * gain > 0.95:
                gain = 0.95 / peak
            
            gain_db = 20 * np.log10(gain)
            
            return audio * gain, gain_db
        
        return audio, 0.0

