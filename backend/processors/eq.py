"""
Smart EQ - Intelligent equalization based on audio analysis.
Includes parametric EQ, graphic EQ, and auto-EQ.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import librosa
from typing import Dict, List, Optional, Tuple


class SmartEQ:
    """
    Intelligent equalizer with:
    1. Auto-EQ - Automatically optimizes frequency balance
    2. Parametric EQ - Precise frequency adjustments
    3. Voice-optimized presets
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.n_fft = 4096
        
        # Standard voice EQ frequency bands
        self.bands = {
            'sub': (20, 80),       # Rumble
            'bass': (80, 250),     # Body/warmth
            'low_mid': (250, 500), # Muddiness zone
            'mid': (500, 2000),    # Presence
            'high_mid': (2000, 4000),  # Clarity
            'presence': (4000, 8000),  # Sibilance/detail
            'air': (8000, 16000)   # Air/brightness
        }
    
    def apply_parametric_eq(self, audio: np.ndarray, 
                            bands: List[Dict]) -> np.ndarray:
        """
        Apply parametric EQ with multiple bands.
        
        Args:
            audio: Input audio (channels, samples)
            bands: List of band settings, each containing:
                   {'freq': float, 'gain_db': float, 'q': float, 'type': str}
                   type can be: 'peak', 'lowshelf', 'highshelf', 'lowpass', 'highpass'
        
        Returns:
            EQ'd audio
        """
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        result = audio.copy()
        
        for band in bands:
            freq = band.get('freq', 1000)
            gain_db = band.get('gain_db', 0)
            q = band.get('q', 1.0)
            band_type = band.get('type', 'peak')
            
            if gain_db == 0 and band_type == 'peak':
                continue
            
            for ch in range(result.shape[0]):
                if band_type == 'peak':
                    result[ch] = self._apply_peak(result[ch], freq, gain_db, q)
                elif band_type == 'lowshelf':
                    result[ch] = self._apply_low_shelf(result[ch], freq, gain_db)
                elif band_type == 'highshelf':
                    result[ch] = self._apply_high_shelf(result[ch], freq, gain_db)
                elif band_type == 'lowpass':
                    result[ch] = self._apply_lowpass(result[ch], freq, q)
                elif band_type == 'highpass':
                    result[ch] = self._apply_highpass(result[ch], freq, q)
        
        return result
    
    def auto_eq(self, audio: np.ndarray, 
                target_curve: str = 'voice',
                strength: float = 0.5) -> np.ndarray:
        """
        Automatically equalize based on target curve.
        
        Args:
            audio: Input audio
            target_curve: 'voice', 'podcast', 'broadcast', or 'flat'
            strength: How much to apply (0-1)
        
        Returns:
            Auto-EQ'd audio
        """
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        # Analyze current frequency balance
        mono = np.mean(audio, axis=0) if audio.shape[0] > 1 else audio[0]
        current_balance = self._analyze_spectrum(mono)
        
        # Get target balance
        target_balance = self._get_target_curve(target_curve)
        
        # Calculate corrections
        corrections = []
        for band_name, (low, high) in self.bands.items():
            center_freq = np.sqrt(low * high)  # Geometric mean
            current = current_balance.get(band_name, 0)
            target = target_balance.get(band_name, 0)
            
            diff = (target - current) * strength
            
            if abs(diff) > 0.5:  # Only apply significant corrections
                corrections.append({
                    'freq': center_freq,
                    'gain_db': np.clip(diff, -6, 6),  # Limit to ±6dB
                    'q': 1.0,
                    'type': 'peak'
                })
        
        # Apply corrections
        return self.apply_parametric_eq(audio, corrections)
    
    def apply_voice_preset(self, audio: np.ndarray, 
                           preset: str = 'podcast') -> np.ndarray:
        """
        Apply voice-optimized EQ preset.
        
        Args:
            audio: Input audio
            preset: 'podcast', 'broadcast', 'telephone', 'warmth', 'clarity'
        
        Returns:
            EQ'd audio
        """
        presets = {
            'podcast': [
                {'freq': 80, 'gain_db': 0, 'q': 1, 'type': 'highpass'},  # Remove rumble
                {'freq': 120, 'gain_db': 2, 'q': 1.5, 'type': 'peak'},   # Subtle warmth
                {'freq': 350, 'gain_db': -2, 'q': 1.5, 'type': 'peak'},  # Reduce mud
                {'freq': 3000, 'gain_db': 2, 'q': 1.5, 'type': 'peak'},  # Presence
                {'freq': 8000, 'gain_db': 1.5, 'q': 1, 'type': 'highshelf'},  # Air
            ],
            'broadcast': [
                {'freq': 100, 'gain_db': 0, 'q': 1, 'type': 'highpass'},
                {'freq': 150, 'gain_db': 3, 'q': 1.2, 'type': 'peak'},   # Radio bass
                {'freq': 400, 'gain_db': -3, 'q': 1.5, 'type': 'peak'},  # Clear mud
                {'freq': 2500, 'gain_db': 3, 'q': 1.5, 'type': 'peak'},  # Strong presence
                {'freq': 5000, 'gain_db': 2, 'q': 2, 'type': 'peak'},    # Clarity
                {'freq': 10000, 'gain_db': 2, 'q': 1, 'type': 'highshelf'},
            ],
            'telephone': [
                {'freq': 300, 'gain_db': 0, 'q': 1, 'type': 'highpass'},
                {'freq': 3400, 'gain_db': 0, 'q': 1, 'type': 'lowpass'},
                {'freq': 1000, 'gain_db': 2, 'q': 1, 'type': 'peak'},
            ],
            'warmth': [
                {'freq': 80, 'gain_db': 0, 'q': 1, 'type': 'highpass'},
                {'freq': 200, 'gain_db': 3, 'q': 1.2, 'type': 'peak'},
                {'freq': 400, 'gain_db': 1, 'q': 1.5, 'type': 'peak'},
                {'freq': 6000, 'gain_db': -1, 'q': 1, 'type': 'highshelf'},
            ],
            'clarity': [
                {'freq': 100, 'gain_db': 0, 'q': 1, 'type': 'highpass'},
                {'freq': 300, 'gain_db': -2, 'q': 1.5, 'type': 'peak'},
                {'freq': 2000, 'gain_db': 2, 'q': 1.5, 'type': 'peak'},
                {'freq': 4000, 'gain_db': 3, 'q': 1.5, 'type': 'peak'},
                {'freq': 10000, 'gain_db': 2, 'q': 1, 'type': 'highshelf'},
            ]
        }
        
        if preset not in presets:
            preset = 'podcast'
        
        return self.apply_parametric_eq(audio, presets[preset])
    
    def _analyze_spectrum(self, audio: np.ndarray) -> Dict[str, float]:
        """Analyze frequency balance of audio."""
        # Compute magnitude spectrum
        stft = np.abs(librosa.stft(audio, n_fft=self.n_fft))
        magnitude = np.mean(stft, axis=1)  # Average over time
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        
        # Calculate energy in each band
        balance = {}
        for band_name, (low, high) in self.bands.items():
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                band_energy = np.mean(magnitude[mask])
                balance[band_name] = 20 * np.log10(band_energy + 1e-10)
        
        # Normalize to reference (mid band)
        reference = balance.get('mid', -20)
        for band in balance:
            balance[band] -= reference
        
        return balance
    
    def _get_target_curve(self, curve_type: str) -> Dict[str, float]:
        """Get target frequency balance curve."""
        curves = {
            'voice': {
                'sub': -12,
                'bass': -3,
                'low_mid': -1,
                'mid': 0,
                'high_mid': 1,
                'presence': 0,
                'air': -2
            },
            'podcast': {
                'sub': -15,
                'bass': -2,
                'low_mid': -2,
                'mid': 0,
                'high_mid': 2,
                'presence': 1,
                'air': -1
            },
            'broadcast': {
                'sub': -18,
                'bass': 0,
                'low_mid': -3,
                'mid': 0,
                'high_mid': 3,
                'presence': 2,
                'air': 1
            },
            'flat': {band: 0 for band in self.bands}
        }
        
        return curves.get(curve_type, curves['voice'])
    
    def _apply_peak(self, audio: np.ndarray, 
                    freq: float, gain_db: float, q: float) -> np.ndarray:
        """Apply peak/bell filter."""
        nyquist = self.sample_rate / 2
        w0 = freq / nyquist
        
        if w0 >= 1 or w0 <= 0:
            return audio
        
        A = 10 ** (gain_db / 40)
        alpha = np.sin(np.pi * w0) / (2 * q)
        
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(np.pi * w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(np.pi * w0)
        a2 = 1 - alpha / A
        
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1, a1/a0, a2/a0])
        
        return signal.filtfilt(b, a, audio)
    
    def _apply_low_shelf(self, audio: np.ndarray, 
                         freq: float, gain_db: float) -> np.ndarray:
        """Apply low shelf filter."""
        nyquist = self.sample_rate / 2
        w0 = freq / nyquist
        
        if w0 >= 1 or w0 <= 0:
            return audio
        
        A = 10 ** (gain_db / 40)
        alpha = np.sin(np.pi * w0) / 2 * np.sqrt(2)
        
        cos_w0 = np.cos(np.pi * w0)
        sqrt_A = np.sqrt(A)
        
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
        
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1, a1/a0, a2/a0])
        
        return signal.filtfilt(b, a, audio)
    
    def _apply_high_shelf(self, audio: np.ndarray, 
                          freq: float, gain_db: float) -> np.ndarray:
        """Apply high shelf filter."""
        nyquist = self.sample_rate / 2
        w0 = freq / nyquist
        
        if w0 >= 1 or w0 <= 0:
            return audio
        
        A = 10 ** (gain_db / 40)
        alpha = np.sin(np.pi * w0) / 2 * np.sqrt(2)
        
        cos_w0 = np.cos(np.pi * w0)
        sqrt_A = np.sqrt(A)
        
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
        
        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1, a1/a0, a2/a0])
        
        return signal.filtfilt(b, a, audio)
    
    def _apply_highpass(self, audio: np.ndarray, 
                        freq: float, q: float = 0.707) -> np.ndarray:
        """Apply high-pass filter."""
        nyquist = self.sample_rate / 2
        normalized_freq = freq / nyquist
        
        if normalized_freq >= 1 or normalized_freq <= 0:
            return audio
        
        b, a = signal.butter(2, normalized_freq, btype='high')
        return signal.filtfilt(b, a, audio)
    
    def _apply_lowpass(self, audio: np.ndarray, 
                       freq: float, q: float = 0.707) -> np.ndarray:
        """Apply low-pass filter."""
        nyquist = self.sample_rate / 2
        normalized_freq = freq / nyquist
        
        if normalized_freq >= 1 or normalized_freq <= 0:
            return audio
        
        b, a = signal.butter(2, normalized_freq, btype='low')
        return signal.filtfilt(b, a, audio)

