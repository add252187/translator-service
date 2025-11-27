"""
Factory for creating TTS provider instances.
"""

from typing import Optional
from app.config import settings, TTSProvider as TTSProviderEnum
from app.tts.base import TTSProvider
from app.tts.elevenlabs_tts import ElevenLabsTTSProvider
from app.utils.logging import get_logger

logger = get_logger(__name__)


def get_tts_provider(
    provider_type: Optional[TTSProviderEnum] = None,
    **kwargs
) -> TTSProvider:
    """
    Factory function to get a TTS provider instance.
    
    Args:
        provider_type: Type of TTS provider to create
        **kwargs: Additional configuration for the provider
    
    Returns:
        TTSProvider instance
    
    Raises:
        ValueError: If provider type is not supported
    """
    provider_type = provider_type or settings.tts_provider
    
    logger.info(f"Creating TTS provider: {provider_type}")
    
    if provider_type == TTSProviderEnum.ELEVENLABS:
        return ElevenLabsTTSProvider(**kwargs)
    # Additional providers can be added here:
    # elif provider_type == TTSProviderEnum.AZURE:
    #     return AzureTTSProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported TTS provider: {provider_type}")
