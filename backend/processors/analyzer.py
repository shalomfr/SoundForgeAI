"""
AI Audio Analyzer - Detects problems in audio and recommends processing settings.
Uses librosa for professional audio analysis.
"""

import numpy as np
import librosa
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class AudioProblems:
    """Detected audio problems and their severity (0-1)."""
    noise_level: float = 0.0
    clipping: float = 0.0
    low_volume: float = 0.0
    high_volume: float = 0.0
    sibilance: float = 0.0
    reverb: float = 0.0
    muddiness: float = 0.0
    harshness: float = 0.0
    breath_sounds: float = 0.0
    dynamic_range_issue: float = 0.0


@dataclass
class ProcessingRecommendations:
    """Recommended processing settings based on analysis."""
    noise_reduction_strength: float = 0.0
    de_esser_strength: float = 0.0
    de_reverb_strength: float = 0.0
    compression_ratio: float = 1.0
    eq_adjustments: Dict[str, float] = None
    normalize_target_lufs: float = -16.0
    gate_threshold_db: float = -60.0
    limiter_threshold_db: float = -1.0
    breath_removal_enabled: bool = False
    voice_enhance_strength: float = 0.0
    
    def __post_init__(self):
        if self.eq_adjustments is None:
            self.eq_adjustments = {}


class AudioAnalyzer:
    """
    AI-powered audio analyzer that detects problems and recommends processing.
    Combines multiple analysis techniques for comprehensive audio assessment.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.hop_length = 512
        self.n_fft = 2048
    
    def analyze(self, audio: np.ndarray) -> Tuple[AudioProblems, ProcessingRecommendations]:
        """
        Perform comprehensive audio analysis.
        
        Args:
            audio: Audio data (channels, samples) or (samples,)
        
        Returns:
            Tuple of (problems, recommendations)
        """
        # Convert to mono for analysis
        if audio.ndim == 2:
            mono = np.mean(audio, axis=0)
        else:
            mono = audio
        
        # Run all analysis modules
        problems = AudioProblems()
        
        # Detect various problems
        problems.noise_level = self._analyze_noise(mono)
        problems.clipping = self._detect_clipping(mono)
        problems.low_volume, problems.high_volume = self._analyze_volume(mono)
        problems.sibilance = self._detect_sibilance(mono)
        problems.reverb = self._estimate_reverb(mono)
        problems.muddiness = self._detect_muddiness(mono)
        problems.harshness = self._detect_harshness(mono)
        problems.breath_sounds = self._detect_breaths(mono)
        problems.dynamic_range_issue = self._analyze_dynamics(mono)
        
        # Generate recommendations based on problems
        recommendations = self._generate_recommendations(problems, mono)
        
        return problems, recommendations
    
    def _analyze_noise(self, audio: np.ndarray) -> float:
        """Estimate background noise level."""
        # Compute RMS in small windows
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop = frame_length // 2
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]
        
        # Find quietest 10% of frames (likely noise floor)
        sorted_rms = np.sort(rms)
        noise_floor_rms = np.mean(sorted_rms[:max(1, len(sorted_rms) // 10)])
        
        # Find loudest 10% (likely speech)
        signal_rms = np.mean(sorted_rms[-max(1, len(sorted_rms) // 10):])
        
        # Calculate SNR and convert to problem severity
        if signal_rms > 0 and noise_floor_rms > 0:
            snr = 20 * np.log10(signal_rms / noise_floor_rms)
            # SNR < 20dB is problematic, < 10dB is severe
            noise_severity = np.clip(1 - (snr - 10) / 30, 0, 1)
            return float(noise_severity)
        return 0.0
    
    def _detect_clipping(self, audio: np.ndarray) -> float:
        """Detect audio clipping."""
        threshold = 0.99
        clipped_samples = np.sum(np.abs(audio) > threshold)
        clip_ratio = clipped_samples / len(audio)
        # Even 0.1% clipping is noticeable
        return float(np.clip(clip_ratio * 1000, 0, 1))
    
    def _analyze_volume(self, audio: np.ndarray) -> Tuple[float, float]:
        """Analyze overall volume levels."""
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # Target RMS around -20dB for speech
        low_volume = float(np.clip((-30 - rms_db) / 20, 0, 1))
        high_volume = float(np.clip((rms_db - (-10)) / 10, 0, 1))
        
        return low_volume, high_volume
    
    def _detect_sibilance(self, audio: np.ndarray) -> float:
        """Detect harsh sibilant sounds (S, SH, etc.)."""
        # Sibilance is typically 4-10kHz
        stft = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        
        # Find sibilance band
        sibilance_mask = (freqs >= 4000) & (freqs <= 10000)
        low_mask = (freqs >= 100) & (freqs <= 2000)
        
        sibilance_energy = np.mean(stft[sibilance_mask, :])
        low_energy = np.mean(stft[low_mask, :])
        
        if low_energy > 0:
            ratio = sibilance_energy / low_energy
            # High ratio indicates sibilance issues
            return float(np.clip((ratio - 0.3) / 0.5, 0, 1))
        return 0.0
    
    def _estimate_reverb(self, audio: np.ndarray) -> float:
        """Estimate reverb/room sound level."""
        # Use spectral flatness and decay characteristics
        spectral_flatness = librosa.feature.spectral_flatness(y=audio, hop_length=self.hop_length)[0]
        avg_flatness = np.mean(spectral_flatness)
        
        # Also check for energy decay patterns
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Calculate autocorrelation for reverb tail detection
        if len(rms) > 10:
            autocorr = np.correlate(rms, rms, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]
            
            # Long autocorrelation decay suggests reverb
            decay_samples = np.sum(autocorr > 0.3)
            reverb_score = np.clip(decay_samples / 50, 0, 1)
        else:
            reverb_score = 0.0
        
        return float(reverb_score * 0.7 + (1 - avg_flatness) * 0.3)
    
    def _detect_muddiness(self, audio: np.ndarray) -> float:
        """Detect muddy/boomy low-mid frequencies."""
        stft = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        
        # Muddiness typically 200-500Hz
        muddy_mask = (freqs >= 200) & (freqs <= 500)
        clarity_mask = (freqs >= 2000) & (freqs <= 5000)
        
        muddy_energy = np.mean(stft[muddy_mask, :])
        clarity_energy = np.mean(stft[clarity_mask, :])
        
        if clarity_energy > 0:
            ratio = muddy_energy / clarity_energy
            return float(np.clip((ratio - 2) / 4, 0, 1))
        return 0.0
    
    def _detect_harshness(self, audio: np.ndarray) -> float:
        """Detect harsh/fatiguing frequencies."""
        stft = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        
        # Harshness typically 2-4kHz
        harsh_mask = (freqs >= 2000) & (freqs <= 4000)
        reference_mask = (freqs >= 500) & (freqs <= 1500)
        
        harsh_energy = np.mean(stft[harsh_mask, :])
        reference_energy = np.mean(stft[reference_mask, :])
        
        if reference_energy > 0:
            ratio = harsh_energy / reference_energy
            return float(np.clip((ratio - 1.5) / 2, 0, 1))
        return 0.0
    
    def _detect_breaths(self, audio: np.ndarray) -> float:
        """Detect prominent breath sounds."""
        # Breaths are typically broadband noise with specific characteristics
        frame_length = int(0.05 * self.sample_rate)  # 50ms frames
        hop = frame_length // 2
        
        # Get spectral centroid and flatness
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, 
                                                      hop_length=hop)[0]
        flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop)[0]
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]
        
        # Breaths have: moderate RMS, high flatness, lower centroid than speech
        avg_rms = np.mean(rms)
        breath_candidates = (flatness > 0.1) & (rms > avg_rms * 0.1) & (rms < avg_rms * 0.7)
        
        breath_ratio = np.sum(breath_candidates) / len(breath_candidates)
        return float(np.clip(breath_ratio * 3, 0, 1))
    
    def _analyze_dynamics(self, audio: np.ndarray) -> float:
        """Analyze dynamic range issues."""
        frame_length = int(0.1 * self.sample_rate)  # 100ms frames
        hop = frame_length // 2
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop)[0]
        rms_db = 20 * np.log10(rms + 1e-10)
        
        # Calculate dynamic range
        dynamic_range = np.percentile(rms_db, 95) - np.percentile(rms_db, 5)
        
        # Dynamic range > 30dB might need compression
        # Dynamic range < 6dB might be over-compressed
        if dynamic_range > 30:
            return float(np.clip((dynamic_range - 30) / 20, 0, 1))
        elif dynamic_range < 6:
            return float(np.clip((6 - dynamic_range) / 6, 0, 1))
        return 0.0
    
    def _generate_recommendations(self, problems: AudioProblems, audio: np.ndarray) -> ProcessingRecommendations:
        """Generate processing recommendations based on detected problems."""
        rec = ProcessingRecommendations()
        
        # Noise reduction
        if problems.noise_level > 0.1:
            rec.noise_reduction_strength = min(problems.noise_level * 1.2, 1.0)
            rec.gate_threshold_db = -50 + (problems.noise_level * 20)  # -50 to -30
        
        # De-esser
        if problems.sibilance > 0.2:
            rec.de_esser_strength = problems.sibilance
        
        # De-reverb
        if problems.reverb > 0.3:
            rec.de_reverb_strength = problems.reverb * 0.8
        
        # Compression
        if problems.dynamic_range_issue > 0.2:
            rec.compression_ratio = 2.0 + (problems.dynamic_range_issue * 4)  # 2:1 to 6:1
        
        # EQ adjustments
        rec.eq_adjustments = {}
        if problems.muddiness > 0.2:
            rec.eq_adjustments['low_mid_cut'] = -3 * problems.muddiness  # Cut 200-500Hz
        if problems.harshness > 0.2:
            rec.eq_adjustments['harsh_cut'] = -3 * problems.harshness  # Cut 2-4kHz
        
        # Voice enhancement
        if problems.muddiness > 0.1 or problems.harshness < 0.3:
            rec.voice_enhance_strength = 0.5
        
        # Breath removal
        if problems.breath_sounds > 0.3:
            rec.breath_removal_enabled = True
        
        # Limiter (always recommended to prevent clipping)
        rec.limiter_threshold_db = -1.0 if problems.clipping > 0 else -0.3
        
        # LUFS target based on content type (assuming podcast/voice)
        rec.normalize_target_lufs = -16.0
        
        return rec
    
    def get_analysis_report(self, audio: np.ndarray) -> Dict:
        """Get a detailed analysis report."""
        problems, recommendations = self.analyze(audio)
        
        return {
            'problems': {
                'noise_level': {'value': problems.noise_level, 'severity': self._severity_label(problems.noise_level)},
                'clipping': {'value': problems.clipping, 'severity': self._severity_label(problems.clipping)},
                'sibilance': {'value': problems.sibilance, 'severity': self._severity_label(problems.sibilance)},
                'reverb': {'value': problems.reverb, 'severity': self._severity_label(problems.reverb)},
                'muddiness': {'value': problems.muddiness, 'severity': self._severity_label(problems.muddiness)},
                'harshness': {'value': problems.harshness, 'severity': self._severity_label(problems.harshness)},
                'breath_sounds': {'value': problems.breath_sounds, 'severity': self._severity_label(problems.breath_sounds)},
                'dynamic_range': {'value': problems.dynamic_range_issue, 'severity': self._severity_label(problems.dynamic_range_issue)},
            },
            'recommendations': {
                'noise_reduction': recommendations.noise_reduction_strength,
                'de_esser': recommendations.de_esser_strength,
                'de_reverb': recommendations.de_reverb_strength,
                'compression_ratio': recommendations.compression_ratio,
                'eq_adjustments': recommendations.eq_adjustments,
                'target_lufs': recommendations.normalize_target_lufs,
                'gate_threshold': recommendations.gate_threshold_db,
                'limiter_threshold': recommendations.limiter_threshold_db,
                'breath_removal': recommendations.breath_removal_enabled,
                'voice_enhance': recommendations.voice_enhance_strength,
            }
        }
    
    @staticmethod
    def _severity_label(value: float) -> str:
        """Convert severity value to label."""
        if value < 0.1:
            return 'none'
        elif value < 0.3:
            return 'low'
        elif value < 0.6:
            return 'medium'
        else:
            return 'high'

