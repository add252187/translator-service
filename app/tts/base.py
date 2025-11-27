"""
Base interface for Text-to-Speech providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass
import asyncio
from app.utils.logging import LoggerMixin


@dataclass
class TTSResult:
    """Result from TTS processing."""
    audio_data: bytes
    sample_rate: int
    duration_ms: Optional[int] = None
    voice_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class TTSProvider(ABC, LoggerMixin):
    """Abstract base class for TTS providers."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize TTS provider.
        
        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs
        self._initialized = False
        self._voice_cache: Dict[str, str] = {}  # Language -> Voice ID mapping
    
    async def initialize(self):
        """Initialize the TTS provider."""
        if not self._initialized:
            await self._initialize()
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    async def _initialize(self):
        """Provider-specific initialization logic."""
        pass
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        language: str,
        voice_id: Optional[str] = None,
        **kwargs
    ) -> TTSResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            language: Language code (ISO 639-1)
            voice_id: Optional specific voice ID
            **kwargs: Additional provider-specific parameters
        
        Returns:
            TTSResult with audio data and metadata
        """
        pass
    
    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        language: str,
        voice_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech as a stream of audio chunks.
        
        Args:
            text: Text to synthesize
            language: Language code
            voice_id: Optional specific voice ID
            **kwargs: Additional provider-specific parameters
        
        Yields:
            Audio data chunks
        """
        pass
    
    @abstractmethod
    async def get_available_voices(
        self,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of available voices.
        
        Args:
            language: Optional language filter
        
        Returns:
            List of voice information dictionaries
        """
        pass
    
    async def get_voice_for_language(
        self,
        language: str,
        gender: Optional[str] = None,
        style: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the best voice ID for a given language.
        
        Args:
            language: Language code
            gender: Optional gender preference ('male', 'female', 'neutral')
            style: Optional style preference ('conversational', 'formal', etc.)
        
        Returns:
            Voice ID or None if not found
        """
        # Check cache first
        cache_key = f"{language}_{gender}_{style}"
        if cache_key in self._voice_cache:
            return self._voice_cache[cache_key]
        
        # Get available voices for the language
        voices = await self.get_available_voices(language)
        
        if not voices:
            return None
        
        # Filter by preferences
        if gender:
            gender_filtered = [v for v in voices if v.get("gender", "").lower() == gender.lower()]
            if gender_filtered:
                voices = gender_filtered
        
        if style:
            style_filtered = [v for v in voices if style.lower() in v.get("styles", []).lower()]
            if style_filtered:
                voices = style_filtered
        
        # Select the first matching voice
        if voices:
            voice_id = voices[0].get("voice_id")
            self._voice_cache[cache_key] = voice_id
            return voice_id
        
        return None
    
    def validate_text(self, text: str) -> bool:
        """
        Validate text before synthesis.
        
        Args:
            text: Text to validate
        
        Returns:
            True if text is valid for synthesis
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided for TTS")
            return False
        
        if len(text) > 5000:  # Reasonable limit for real-time TTS
            self.logger.warning(f"Text too long for TTS: {len(text)} characters")
            return False
        
        return True
    
    def split_text_for_streaming(
        self,
        text: str,
        max_chunk_size: int = 200
    ) -> List[str]:
        """
        Split text into chunks for streaming synthesis.
        
        Args:
            text: Text to split
            max_chunk_size: Maximum characters per chunk
        
        Returns:
            List of text chunks
        """
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def close(self):
        """Clean up resources."""
        if self._initialized:
            await self._close()
            self._initialized = False
            self.logger.info(f"{self.__class__.__name__} closed")
    
    async def _close(self):
        """Provider-specific cleanup logic."""
        pass
