"""
Base interface for translation providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from app.utils.logging import LoggerMixin


@dataclass
class TranslationResult:
    """Result from translation processing."""
    translated_text: str
    source_language: str
    target_language: str
    confidence: Optional[float] = None
    alternatives: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate translation result."""
        if self.confidence is not None:
            self.confidence = max(0.0, min(1.0, self.confidence))


class TranslationProvider(ABC, LoggerMixin):
    """Abstract base class for translation providers."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize translation provider.
        
        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs
        self._initialized = False
        self._supported_languages: Optional[List[str]] = None
    
    async def initialize(self):
        """Initialize the translation provider."""
        if not self._initialized:
            await self._initialize()
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    async def _initialize(self):
        """Provider-specific initialization logic."""
        pass
    
    @abstractmethod
    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        **kwargs
    ) -> TranslationResult:
        """
        Translate text from source to target language.
        
        Args:
            text: Text to translate
            target_language: Target language code (ISO 639-1)
            source_language: Source language code (auto-detect if None)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            TranslationResult with translation and metadata
        """
        pass
    
    @abstractmethod
    async def detect_language(
        self,
        text: str
    ) -> Tuple[str, float]:
        """
        Detect the language of text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Tuple of (language_code, confidence_score)
        """
        pass
    
    @abstractmethod
    async def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of ISO 639-1 language codes
        """
        pass
    
    async def batch_translate(
        self,
        texts: List[str],
        target_language: str,
        source_language: Optional[str] = None,
        **kwargs
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            target_language: Target language code
            source_language: Source language code (auto-detect if None)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            List of TranslationResult objects
        """
        # Default implementation: translate one by one
        results = []
        for text in texts:
            result = await self.translate(text, target_language, source_language, **kwargs)
            results.append(result)
        return results
    
    def should_bypass_translation(
        self,
        source_language: str,
        target_language: str,
        bypass_same_language: bool = False
    ) -> bool:
        """
        Check if translation should be bypassed.
        
        Args:
            source_language: Source language code
            target_language: Target language code
            bypass_same_language: Whether to bypass when languages are the same
        
        Returns:
            True if translation should be bypassed
        """
        if not bypass_same_language:
            return False
        
        # Normalize language codes (e.g., 'es-ES' -> 'es')
        source_base = source_language.split('-')[0].lower()
        target_base = target_language.split('-')[0].lower()
        
        return source_base == target_base
    
    def validate_text(self, text: str) -> bool:
        """
        Validate text before translation.
        
        Args:
            text: Text to validate
        
        Returns:
            True if text is valid for translation
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided for translation")
            return False
        
        if len(text) > 10000:  # Reasonable limit
            self.logger.warning(f"Text too long for translation: {len(text)} characters")
            return False
        
        return True
    
    async def close(self):
        """Clean up resources."""
        if self._initialized:
            await self._close()
            self._initialized = False
            self.logger.info(f"{self.__class__.__name__} closed")
    
    async def _close(self):
        """Provider-specific cleanup logic."""
        pass
