"""
Language detection utilities for improved accuracy.
"""

from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from collections import Counter
import asyncio
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LanguageDetectionResult:
    """Result from language detection."""
    language: str
    confidence: float
    alternatives: Optional[List[Tuple[str, float]]] = None
    method: str = "unknown"


class LanguageDetector:
    """
    Enhanced language detection using multiple methods.
    Combines results from STT and text-based detection for higher accuracy.
    """
    
    # Common language codes mapping
    LANGUAGE_CODES = {
        "english": "en",
        "spanish": "es", 
        "castilian": "es",
        "french": "fr",
        "german": "de",
        "italian": "it",
        "portuguese": "pt",
        "dutch": "nl",
        "russian": "ru",
        "chinese": "zh",
        "mandarin": "zh",
        "japanese": "ja",
        "korean": "ko",
        "arabic": "ar",
        "hindi": "hi",
        "galician": "gl",
        "catalan": "ca",
        "basque": "eu",
        "asturian": "ast",
    }
    
    # Language families for fallback
    LANGUAGE_FAMILIES = {
        "romance": ["es", "pt", "it", "fr", "gl", "ca", "ro"],
        "germanic": ["en", "de", "nl", "sv", "no", "da"],
        "slavic": ["ru", "pl", "cs", "sk", "uk", "bg"],
        "asian": ["zh", "ja", "ko", "vi", "th"],
        "indic": ["hi", "ur", "bn", "ta", "te"],
        "semitic": ["ar", "he"],
    }
    
    def __init__(self):
        self.detection_history: Dict[str, List[LanguageDetectionResult]] = {}
        self.confidence_threshold = 0.7
    
    def normalize_language_code(self, lang_code: str) -> str:
        """
        Normalize language code to ISO 639-1 format.
        
        Args:
            lang_code: Input language code or name
            
        Returns:
            Normalized ISO 639-1 code
        """
        if not lang_code:
            return "unknown"
        
        # Convert to lowercase
        lang_code = lang_code.lower().strip()
        
        # Check if already in correct format
        if len(lang_code) == 2 and lang_code.isalpha():
            return lang_code
        
        # Extract base code if regional (e.g., "en-US" -> "en")
        if '-' in lang_code or '_' in lang_code:
            base_code = lang_code.split('-')[0].split('_')[0]
            if len(base_code) == 2:
                return base_code
        
        # Look up in mapping
        return self.LANGUAGE_CODES.get(lang_code, lang_code[:2] if len(lang_code) >= 2 else "unknown")
    
    async def detect_from_audio(
        self,
        audio_data: bytes,
        stt_provider,
        sample_rate: int = 8000
    ) -> LanguageDetectionResult:
        """
        Detect language from audio using STT provider.
        
        Args:
            audio_data: Audio bytes
            stt_provider: STT provider instance
            sample_rate: Audio sample rate
            
        Returns:
            LanguageDetectionResult
        """
        try:
            # Use STT provider's language detection
            language, confidence = await stt_provider.detect_language(audio_data, sample_rate)
            
            # Normalize language code
            normalized_lang = self.normalize_language_code(language)
            
            return LanguageDetectionResult(
                language=normalized_lang,
                confidence=confidence,
                method="audio_stt"
            )
            
        except Exception as e:
            logger.error(f"Audio language detection failed: {e}")
            return LanguageDetectionResult(
                language="unknown",
                confidence=0.0,
                method="audio_stt_failed"
            )
    
    async def detect_from_text(
        self,
        text: str,
        translation_provider
    ) -> LanguageDetectionResult:
        """
        Detect language from text using translation provider.
        
        Args:
            text: Text to analyze
            translation_provider: Translation provider instance
            
        Returns:
            LanguageDetectionResult
        """
        try:
            # Use translation provider's language detection
            language, confidence = await translation_provider.detect_language(text)
            
            # Normalize language code
            normalized_lang = self.normalize_language_code(language)
            
            return LanguageDetectionResult(
                language=normalized_lang,
                confidence=confidence,
                method="text_analysis"
            )
            
        except Exception as e:
            logger.error(f"Text language detection failed: {e}")
            return LanguageDetectionResult(
                language="unknown",
                confidence=0.0,
                method="text_analysis_failed"
            )
    
    async def detect_combined(
        self,
        audio_data: Optional[bytes],
        text: Optional[str],
        stt_provider=None,
        translation_provider=None,
        sample_rate: int = 8000
    ) -> LanguageDetectionResult:
        """
        Detect language using multiple methods and combine results.
        
        Args:
            audio_data: Optional audio bytes
            text: Optional text
            stt_provider: Optional STT provider
            translation_provider: Optional translation provider
            sample_rate: Audio sample rate
            
        Returns:
            Combined LanguageDetectionResult with highest confidence
        """
        results = []
        
        # Detect from audio if available
        if audio_data and stt_provider:
            audio_result = await self.detect_from_audio(
                audio_data, stt_provider, sample_rate
            )
            if audio_result.confidence > 0:
                results.append(audio_result)
        
        # Detect from text if available
        if text and translation_provider:
            text_result = await self.detect_from_text(
                text, translation_provider
            )
            if text_result.confidence > 0:
                results.append(text_result)
        
        if not results:
            return LanguageDetectionResult(
                language="unknown",
                confidence=0.0,
                method="no_detection"
            )
        
        # Combine results - weighted average
        language_scores: Dict[str, float] = {}
        for result in results:
            if result.language not in language_scores:
                language_scores[result.language] = 0
            language_scores[result.language] += result.confidence
        
        # Get best language
        best_language = max(language_scores, key=language_scores.get)
        avg_confidence = language_scores[best_language] / len(results)
        
        # Get alternatives
        alternatives = [
            (lang, score / len(results)) 
            for lang, score in language_scores.items() 
            if lang != best_language
        ]
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        return LanguageDetectionResult(
            language=best_language,
            confidence=avg_confidence,
            alternatives=alternatives[:3],  # Top 3 alternatives
            method="combined"
        )
    
    def track_detection(self, call_id: str, result: LanguageDetectionResult):
        """
        Track language detection results for a call.
        
        Args:
            call_id: Call identifier
            result: Detection result to track
        """
        if call_id not in self.detection_history:
            self.detection_history[call_id] = []
        
        self.detection_history[call_id].append(result)
        
        # Keep only last 10 detections per call
        if len(self.detection_history[call_id]) > 10:
            self.detection_history[call_id] = self.detection_history[call_id][-10:]
    
    def get_consensus_language(self, call_id: str) -> Tuple[str, float]:
        """
        Get consensus language from detection history.
        
        Args:
            call_id: Call identifier
            
        Returns:
            Tuple of (language, confidence)
        """
        if call_id not in self.detection_history:
            return "unknown", 0.0
        
        detections = self.detection_history[call_id]
        if not detections:
            return "unknown", 0.0
        
        # Count language occurrences weighted by confidence
        language_scores: Dict[str, float] = {}
        for detection in detections:
            if detection.language not in language_scores:
                language_scores[detection.language] = 0
            language_scores[detection.language] += detection.confidence
        
        # Get most likely language
        best_language = max(language_scores, key=language_scores.get)
        avg_confidence = language_scores[best_language] / len(detections)
        
        return best_language, avg_confidence
    
    def is_related_language(self, lang1: str, lang2: str) -> bool:
        """
        Check if two languages are related (same family).
        
        Args:
            lang1: First language code
            lang2: Second language code
            
        Returns:
            True if languages are in the same family
        """
        for family in self.LANGUAGE_FAMILIES.values():
            if lang1 in family and lang2 in family:
                return True
        return False
    
    def suggest_fallback_language(self, detected_lang: str, supported_langs: List[str]) -> Optional[str]:
        """
        Suggest a fallback language if detected language is not supported.
        
        Args:
            detected_lang: Detected language code
            supported_langs: List of supported language codes
            
        Returns:
            Suggested fallback language or None
        """
        # Check if detected language is supported
        if detected_lang in supported_langs:
            return detected_lang
        
        # Try to find a related language
        for family in self.LANGUAGE_FAMILIES.values():
            if detected_lang in family:
                for lang in family:
                    if lang in supported_langs:
                        logger.info(f"Using fallback language {lang} for unsupported {detected_lang}")
                        return lang
        
        # Default fallback to English if supported
        if "en" in supported_langs:
            return "en"
        
        # Or Spanish as secondary fallback
        if "es" in supported_langs:
            return "es"
        
        # Return first supported language
        return supported_langs[0] if supported_langs else None


# Global language detector instance
language_detector = LanguageDetector()
