"""
Audio Processing Pipeline - Orchestrates all processors into one seamless flow.
The magic happens here - AI-driven automatic processing with optional manual control.
"""

import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import time

from .analyzer import AudioAnalyzer, AudioProblems, ProcessingRecommendations
from .denoiser import Denoiser
from .enhancer import VoiceEnhancer
from .dynamics import DynamicsProcessor
from .eq import SmartEQ


@dataclass
class ProcessingSettings:
    """User-configurable processing settings."""
    # Auto mode - let AI decide everything
    auto_mode: bool = True
    
    # Individual processor toggles
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
    
    # Manual overrides (used when auto_mode=False)
    noise_reduction_strength: float = 0.5
    de_reverb_strength: float = 0.3
    de_esser_strength: float = 0.4
    compression_ratio: float = 3.0
    compression_threshold_db: float = -20.0
    eq_preset: str = 'podcast'  # podcast, broadcast, warmth, clarity
    target_lufs: float = -16.0
    stereo_width: float = 0.2
    
    # Preset selection
    preset: str = 'auto'  # auto, podcast, interview, audiobook, voiceover


@dataclass
class ProcessingResult:
    """Result of audio processing."""
    audio: np.ndarray
    sample_rate: int
    problems_detected: Dict[str, Any] = field(default_factory=dict)
    processing_applied: Dict[str, bool] = field(default_factory=dict)
    settings_used: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0


class AudioPipeline:
    """
    Master audio processing pipeline.
    Combines all 12 processing engines into one intelligent flow.
    
    Processing Order:
    1. Analysis (AI detection)
    2. Noise Gate
    3. Spectral Denoising  
    4. De-Reverb
    5. Breath Removal
    6. De-Esser
    7. Voice Enhancement
    8. Smart EQ
    9. Compression
    10. Stereo Enhancement
    11. Limiter
    12. LUFS Normalization
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
        # Initialize all processors
        self.analyzer = AudioAnalyzer(sample_rate)
        self.denoiser = Denoiser(sample_rate)
        self.enhancer = VoiceEnhancer(sample_rate)
        self.dynamics = DynamicsProcessor(sample_rate)
        self.eq = SmartEQ(sample_rate)
    
    def process(self, audio: np.ndarray, 
                settings: Optional[ProcessingSettings] = None) -> ProcessingResult:
        """
        Process audio through the entire pipeline.
        
        Args:
            audio: Input audio (channels, samples)
            settings: Processing settings (uses defaults if None)
        
        Returns:
            ProcessingResult with processed audio and metadata
        """
        start_time = time.time()
        
        if settings is None:
            settings = ProcessingSettings()
        
        # Ensure correct shape
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)
        
        result = ProcessingResult(
            audio=audio.copy(),
            sample_rate=self.sample_rate
        )
        
        # Step 1: Analyze audio
        problems, recommendations = self.analyzer.analyze(audio)
        result.problems_detected = self._problems_to_dict(problems)
        
        # Get effective settings (auto or manual)
        effective_settings = self._get_effective_settings(settings, recommendations)
        result.settings_used = effective_settings
        
        processed = audio.copy()
        
        # Step 2: Noise Gate
        if settings.enable_noise_gate and effective_settings.get('noise_gate', False):
            processed = self.denoiser._apply_noise_gate(
                processed, 
                effective_settings.get('gate_threshold_db', -50)
            )
            result.processing_applied['noise_gate'] = True
        
        # Step 3: Spectral Denoising
        if settings.enable_denoiser and effective_settings.get('noise_reduction', 0) > 0:
            processed = self.denoiser.process(
                processed,
                strength=effective_settings['noise_reduction'],
                use_noise_gate=False,  # Already applied
                use_spectral_denoise=True
            )
            result.processing_applied['denoiser'] = True
        
        # Step 4: De-Reverb
        if settings.enable_de_reverb and effective_settings.get('de_reverb', 0) > 0:
            processed = self.enhancer.de_reverb(
                processed,
                strength=effective_settings['de_reverb']
            )
            result.processing_applied['de_reverb'] = True
        
        # Step 5: Breath Removal
        if settings.enable_breath_removal and effective_settings.get('breath_removal', False):
            processed = self.denoiser.remove_breath_sounds(
                processed,
                sensitivity=0.5
            )
            result.processing_applied['breath_removal'] = True
        
        # Step 6: De-Esser
        if settings.enable_de_esser and effective_settings.get('de_esser', 0) > 0:
            processed = self.enhancer.de_ess(
                processed,
                strength=effective_settings['de_esser']
            )
            result.processing_applied['de_esser'] = True
        
        # Step 7: Voice Enhancement
        if settings.enable_voice_enhance and effective_settings.get('voice_enhance', 0) > 0:
            strength = effective_settings['voice_enhance']
            processed = self.enhancer.process(
                processed,
                clarity_strength=strength * 0.6,
                warmth_strength=strength * 0.3,
                presence_strength=strength * 0.4
            )
            result.processing_applied['voice_enhance'] = True
        
        # Step 8: Smart EQ
        if settings.enable_eq:
            eq_preset = effective_settings.get('eq_preset', 'podcast')
            if settings.auto_mode:
                processed = self.eq.auto_eq(processed, target_curve='voice', strength=0.5)
            else:
                processed = self.eq.apply_voice_preset(processed, preset=eq_preset)
            result.processing_applied['eq'] = True
        
        # Step 9: Compression
        if settings.enable_compression and effective_settings.get('compression_ratio', 1) > 1:
            ratio = effective_settings['compression_ratio']
            threshold = effective_settings.get('compression_threshold_db', -20)
            
            if ratio > 3:
                # Use multiband for heavy compression
                processed = self.dynamics.multiband_compress(
                    processed,
                    low_threshold_db=threshold - 4,
                    mid_threshold_db=threshold,
                    high_threshold_db=threshold + 2,
                    ratio=ratio
                )
            else:
                processed = self.dynamics.compress(
                    processed,
                    threshold_db=threshold,
                    ratio=ratio,
                    attack_ms=10,
                    release_ms=100
                )
            result.processing_applied['compression'] = True
        
        # Step 10: Stereo Enhancement
        if settings.enable_stereo_enhance and processed.shape[0] == 2:
            width = effective_settings.get('stereo_width', 0.2)
            processed = self.enhancer.stereo_enhance(processed, width=width)
            result.processing_applied['stereo_enhance'] = True
        
        # Step 11: Limiter
        if settings.enable_limiter:
            threshold = effective_settings.get('limiter_threshold_db', -1.0)
            processed = self.dynamics.limit(processed, threshold_db=threshold)
            result.processing_applied['limiter'] = True
        
        # Step 12: LUFS Normalization
        if settings.enable_normalize:
            target_lufs = effective_settings.get('target_lufs', -16.0)
            processed = self.dynamics.normalize_lufs(processed, target_lufs=target_lufs)
            result.processing_applied['normalize'] = True
        
        result.audio = processed
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _get_effective_settings(self, 
                                settings: ProcessingSettings,
                                recommendations: ProcessingRecommendations) -> Dict[str, Any]:
        """Get effective settings based on auto mode or manual overrides."""
        
        if settings.auto_mode:
            # Use AI recommendations
            return {
                'noise_gate': recommendations.noise_reduction_strength > 0.1,
                'gate_threshold_db': recommendations.gate_threshold_db,
                'noise_reduction': recommendations.noise_reduction_strength,
                'de_reverb': recommendations.de_reverb_strength,
                'de_esser': recommendations.de_esser_strength,
                'breath_removal': recommendations.breath_removal_enabled,
                'voice_enhance': recommendations.voice_enhance_strength,
                'eq_preset': 'auto',
                'compression_ratio': recommendations.compression_ratio,
                'compression_threshold_db': -20,
                'stereo_width': 0.2,
                'limiter_threshold_db': recommendations.limiter_threshold_db,
                'target_lufs': recommendations.normalize_target_lufs
            }
        else:
            # Use manual settings
            return {
                'noise_gate': settings.enable_noise_gate,
                'gate_threshold_db': -50,
                'noise_reduction': settings.noise_reduction_strength,
                'de_reverb': settings.de_reverb_strength,
                'de_esser': settings.de_esser_strength,
                'breath_removal': settings.enable_breath_removal,
                'voice_enhance': 0.5 if settings.enable_voice_enhance else 0,
                'eq_preset': settings.eq_preset,
                'compression_ratio': settings.compression_ratio,
                'compression_threshold_db': settings.compression_threshold_db,
                'stereo_width': settings.stereo_width,
                'limiter_threshold_db': -1.0,
                'target_lufs': settings.target_lufs
            }
    
    def _problems_to_dict(self, problems: AudioProblems) -> Dict[str, Any]:
        """Convert AudioProblems to dictionary."""
        return {
            'noise_level': round(problems.noise_level, 3),
            'clipping': round(problems.clipping, 3),
            'low_volume': round(problems.low_volume, 3),
            'high_volume': round(problems.high_volume, 3),
            'sibilance': round(problems.sibilance, 3),
            'reverb': round(problems.reverb, 3),
            'muddiness': round(problems.muddiness, 3),
            'harshness': round(problems.harshness, 3),
            'breath_sounds': round(problems.breath_sounds, 3),
            'dynamic_range_issue': round(problems.dynamic_range_issue, 3)
        }
    
    def get_presets(self) -> Dict[str, ProcessingSettings]:
        """Get available processing presets."""
        return {
            'auto': ProcessingSettings(auto_mode=True, preset='auto'),
            'podcast': ProcessingSettings(
                auto_mode=False,
                preset='podcast',
                noise_reduction_strength=0.6,
                de_reverb_strength=0.3,
                de_esser_strength=0.4,
                compression_ratio=3.0,
                eq_preset='podcast',
                target_lufs=-16.0,
                enable_breath_removal=True
            ),
            'interview': ProcessingSettings(
                auto_mode=False,
                preset='interview',
                noise_reduction_strength=0.5,
                de_reverb_strength=0.4,
                de_esser_strength=0.3,
                compression_ratio=2.5,
                eq_preset='clarity',
                target_lufs=-16.0,
                enable_breath_removal=True
            ),
            'audiobook': ProcessingSettings(
                auto_mode=False,
                preset='audiobook',
                noise_reduction_strength=0.7,
                de_reverb_strength=0.5,
                de_esser_strength=0.5,
                compression_ratio=4.0,
                eq_preset='warmth',
                target_lufs=-18.0,
                enable_breath_removal=True
            ),
            'voiceover': ProcessingSettings(
                auto_mode=False,
                preset='voiceover',
                noise_reduction_strength=0.8,
                de_reverb_strength=0.6,
                de_esser_strength=0.4,
                compression_ratio=4.0,
                eq_preset='broadcast',
                target_lufs=-14.0,
                enable_breath_removal=True
            )
        }
    
    def analyze_only(self, audio: np.ndarray) -> Dict[str, Any]:
        """Run analysis without processing."""
        return self.analyzer.get_analysis_report(audio)

