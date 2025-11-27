"""
Factory for creating translation provider instances.
"""

from typing import Optional
from app.config import settings, TranslationProvider as TranslationProviderEnum
from app.translation.base import TranslationProvider
from app.translation.deepl_translator import DeepLTranslator
from app.translation.openai_translator import OpenAITranslator
from app.utils.logging import get_logger

logger = get_logger(__name__)


def get_translation_provider(
    provider_type: Optional[TranslationProviderEnum] = None,
    **kwargs
) -> TranslationProvider:
    """
    Factory function to get a translation provider instance.
    
    Args:
        provider_type: Type of translation provider to create
        **kwargs: Additional configuration for the provider
    
    Returns:
        TranslationProvider instance
    
    Raises:
        ValueError: If provider type is not supported
    """
    provider_type = provider_type or settings.translation_provider
    
    logger.info(f"Creating translation provider: {provider_type}")
    
    if provider_type == TranslationProviderEnum.DEEPL:
        return DeepLTranslator(**kwargs)
    elif provider_type == TranslationProviderEnum.OPENAI:
        return OpenAITranslator(**kwargs)
    else:
        raise ValueError(f"Unsupported translation provider: {provider_type}")
