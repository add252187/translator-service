"""
Factory for creating STT provider instances.
"""

from typing import Optional
from app.config import settings, STTProvider as STTProviderEnum
from app.stt.base import STTProvider
from app.stt.deepgram_stt import DeepgramSTTProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)


def get_stt_provider(
    provider_type: Optional[STTProviderEnum] = None,
    **kwargs
) -> STTProvider:
    """
    Factory function to get an STT provider instance.
    
    Args:
        provider_type: Type of STT provider to create
        **kwargs: Additional configuration for the provider
    
    Returns:
        STTProvider instance
    
    Raises:
        ValueError: If provider type is not supported
    """
    provider_type = provider_type or settings.stt_provider
    
    logger.info(f"Creating STT provider: {provider_type}")
    
    if provider_type == STTProviderEnum.DEEPGRAM:
        return DeepgramSTTProvider(**kwargs)
    # Additional providers can be added here:
    # elif provider_type == STTProviderEnum.WHISPER:
    #     return WhisperSTTProvider(**kwargs)
    # elif provider_type == STTProviderEnum.ELEVENLABS:
    #     return ElevenLabsSTTProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported STT provider: {provider_type}")
