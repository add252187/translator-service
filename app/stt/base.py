"""
Base interface for Speech-to-Text providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import asyncio
from app.utils.logging import LoggerMixin


@dataclass
class STTResult:
    """Result from STT processing."""
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    is_final: bool = True
    alternatives: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate result data."""
        if self.confidence is not None:
            self.confidence = max(0.0, min(1.0, self.confidence))


class STTProvider(ABC, LoggerMixin):
    """Abstract base class for STT providers."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize STT provider.
        
        Args:
            api_key: API key for the provider
            **kwargs: Additional provider-specific configuration
        """
        self.api_key = api_key
        self.config = kwargs
        self._initialized = False
    
    async def initialize(self):
        """Initialize the STT provider (e.g., load models, establish connections)."""
        if not self._initialized:
            await self._initialize()
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    async def _initialize(self):
        """Provider-specific initialization logic."""
        pass
    
    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 8000,
        language_hint: Optional[str] = None,
        **kwargs
    ) -> STTResult:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate in Hz
            language_hint: Optional language hint (ISO 639-1 code)
            **kwargs: Additional provider-specific parameters
        
        Returns:
            STTResult with transcription and metadata
        """
        pass
    
    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: asyncio.Queue,
        sample_rate: int = 8000,
        language_hint: Optional[str] = None,
        **kwargs
    ) -> asyncio.Queue:
        """
        Transcribe streaming audio data.
        
        Args:
            audio_stream: Queue of audio chunks
            sample_rate: Audio sample rate in Hz
            language_hint: Optional language hint
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Queue of STTResult objects
        """
        pass
    
    @abstractmethod
    async def detect_language(
        self,
        audio_data: bytes,
        sample_rate: int = 8000
    ) -> Tuple[str, float]:
        """
        Detect the language of audio data.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate in Hz
        
        Returns:
            Tuple of (language_code, confidence_score)
        """
        pass
    
    async def close(self):
        """Clean up resources."""
        if self._initialized:
            await self._close()
            self._initialized = False
            self.logger.info(f"{self.__class__.__name__} closed")
    
    async def _close(self):
        """Provider-specific cleanup logic."""
        pass
    
    def validate_audio(self, audio_data: bytes, sample_rate: int) -> bool:
        """
        Validate audio data before processing.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
        
        Returns:
            True if audio is valid
        """
        if not audio_data:
            self.logger.warning("Empty audio data received")
            return False
        
        if len(audio_data) < 100:  # Minimum reasonable audio size
            self.logger.warning(f"Audio data too small: {len(audio_data)} bytes")
            return False
        
        if sample_rate < 8000:
            self.logger.warning(f"Sample rate too low: {sample_rate} Hz")
            return False
        
        return True
