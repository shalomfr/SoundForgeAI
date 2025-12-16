"""
Voice Enhancer - Improves voice clarity and presence.
Includes harmonic enhancement, presence boost, and de-reverb.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import librosa
from typing import Tuple, Optional


class VoiceEnhancer:
    """
    Professional voice enhancement processor:
    1. Clarity Enhancement - Boosts intelligibility frequencies
    2. Harmonic Enhancement - Adds warmth and presence
    3. De-Reverb - Reduces room sound
    4. Stereo Enhancement - Widens stereo image
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.n_fft = 2048
        self.hop_length = 512
    
    def process(self, audio: np.ndarray,
                clarity_strength: float = 0.5,
                warmth_strength: float = 0.3,
                presence_strength: float = 0.4) -> np.ndarray:
        """
        Apply comprehensive voice enhancement.
        
        Args:
            audio: Input audio (channels, samples)
            clarity_strength: Clarity boost amount (0-1)
            warmth_strength: Low-end warmth amount (0-1)
            presence_strength: Presence/air boost amount (0-1)
        
        Returns:
            Enhanced audio
        """
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        result = audio.copy()
        
        # Apply enhancements to each channel
        for ch in range(audio.shape[0]):
            channel = audio[ch]
            
            # Clarity enhancement (2-5kHz)
            if clarity_strength > 0:
                channel = self._enhance_clarity(channel, clarity_strength)
            
            # Warmth enhancement (100-300Hz)
            if warmth_strength > 0:
                channel = self._add_warmth(channel, warmth_strength)
            
            # Presence/air (8-12kHz)
            if presence_strength > 0:
                channel = self._add_presence(channel, presence_strength)
            
            result[ch] = channel
        
        return result
    
    def _enhance_clarity(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Enhance voice clarity in the 2-5kHz range."""
        # Design peak filter centered at 3.5kHz
        center_freq = 3500
        q = 1.5
        gain_db = strength * 4  # Up to 4dB boost
        
        enhanced = self._apply_peak_filter(audio, center_freq, q, gain_db)
        
        # Also add slight boost at 1.5kHz for fundamental clarity
        enhanced = self._apply_peak_filter(enhanced, 1500, 2.0, gain_db * 0.5)
        
        return enhanced
    
    def _add_warmth(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Add low-end warmth to voice."""
        # Subtle boost around 200Hz
        center_freq = 200
        q = 1.0
        gain_db = strength * 3  # Up to 3dB boost
        
        return self._apply_peak_filter(audio, center_freq, q, gain_db)
    
    def _add_presence(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """Add air and presence to voice."""
        # High shelf starting at 8kHz
        enhanced = self._apply_high_shelf(audio, 8000, strength * 3)
        
        # Subtle presence at 5kHz
        enhanced = self._apply_peak_filter(enhanced, 5000, 2.0, strength * 2)
        
        return enhanced
    
    def _apply_peak_filter(self, audio: np.ndarray, 
                           center_freq: float, 
                           q: float, 
                           gain_db: float) -> np.ndarray:
        """Apply a peak/bell EQ filter."""
        nyquist = self.sample_rate / 2
        w0 = center_freq / nyquist
        
        if w0 >= 1:
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
    
    def _apply_high_shelf(self, audio: np.ndarray, 
                          cutoff: float, 
                          gain_db: float) -> np.ndarray:
        """Apply a high shelf filter."""
        nyquist = self.sample_rate / 2
        w0 = cutoff / nyquist
        
        if w0 >= 1:
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
    
    def de_reverb(self, audio: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Reduce reverb/room sound using spectral processing.
        
        Args:
            audio: Input audio
            strength: De-reverb strength (0-1)
        
        Returns:
            Audio with reduced reverb
        """
        if strength <= 0:
            return audio
        
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        result = np.zeros_like(audio)
        
        for ch in range(audio.shape[0]):
            # STFT
            stft = librosa.stft(audio[ch], n_fft=self.n_fft, hop_length=self.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate reverb using spectral decay
            # Reverb tends to have slower decay than direct sound
            reverb_estimate = gaussian_filter1d(magnitude, sigma=3, axis=1)
            
            # Spectral subtraction of reverb estimate
            alpha = strength * 0.7  # Reverb suppression factor
            magnitude_cleaned = np.maximum(
                magnitude - alpha * reverb_estimate,
                magnitude * 0.1  # Keep some minimum to avoid artifacts
            )
            
            # Reconstruct
            stft_cleaned = magnitude_cleaned * np.exp(1j * phase)
            result[ch] = librosa.istft(stft_cleaned, hop_length=self.hop_length, length=audio.shape[1])
        
        return result
    
    def stereo_enhance(self, audio: np.ndarray, width: float = 0.3) -> np.ndarray:
        """
        Enhance stereo width using mid-side processing.
        
        Args:
            audio: Input audio (2, samples) - stereo only
            width: Stereo width enhancement (0-1)
        
        Returns:
            Stereo-enhanced audio
        """
        if audio.ndim == 1 or audio.shape[0] == 1:
            return audio  # Can't enhance mono
        
        if audio.shape[0] != 2:
            return audio
        
        left = audio[0]
        right = audio[1]
        
        # Convert to mid-side
        mid = (left + right) / 2
        side = (left - right) / 2
        
        # Enhance side signal
        side_gain = 1 + width
        side_enhanced = side * side_gain
        
        # Convert back to left-right
        left_out = mid + side_enhanced
        right_out = mid - side_enhanced
        
        # Normalize to prevent clipping
        max_val = max(np.max(np.abs(left_out)), np.max(np.abs(right_out)))
        if max_val > 1:
            left_out /= max_val
            right_out /= max_val
        
        return np.vstack([left_out, right_out])
    
    def de_ess(self, audio: np.ndarray, 
               threshold_db: float = -20,
               frequency: float = 6000,
               strength: float = 0.5) -> np.ndarray:
        """
        De-esser to reduce harsh sibilants.
        
        Args:
            audio: Input audio
            threshold_db: Detection threshold
            frequency: Center frequency for sibilance detection
            strength: Reduction strength (0-1)
        
        Returns:
            De-essed audio
        """
        if strength <= 0:
            return audio
        
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        result = audio.copy()
        
        for ch in range(audio.shape[0]):
            # Detect sibilance using bandpass filter
            nyquist = self.sample_rate / 2
            low = 4000 / nyquist
            high = min(10000 / nyquist, 0.99)
            
            b, a = signal.butter(4, [low, high], btype='band')
            sibilance_band = signal.filtfilt(b, a, audio[ch])
            
            # Calculate envelope of sibilance
            envelope = np.abs(signal.hilbert(sibilance_band))
            envelope = gaussian_filter1d(envelope, sigma=int(0.005 * self.sample_rate))
            
            # Calculate threshold
            threshold = 10 ** (threshold_db / 20)
            
            # Calculate gain reduction
            reduction = np.ones_like(envelope)
            above_threshold = envelope > threshold
            if np.any(above_threshold):
                # Soft knee compression of sibilance
                ratio = 4 * strength + 1  # 1:1 to 5:1 ratio
                excess_db = 20 * np.log10(envelope[above_threshold] / threshold + 1e-10)
                gain_reduction_db = excess_db * (1 - 1/ratio)
                reduction[above_threshold] = 10 ** (-gain_reduction_db / 20)
            
            # Smooth the gain reduction
            reduction = gaussian_filter1d(reduction, sigma=int(0.002 * self.sample_rate))
            
            # Apply to sibilance band only (multiband approach)
            sibilance_reduced = sibilance_band * reduction
            
            # Reconstruct: original minus sibilance plus reduced sibilance
            result[ch] = audio[ch] - sibilance_band + sibilance_reduced
        
        return result

