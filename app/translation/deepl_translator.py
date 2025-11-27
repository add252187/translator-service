"""
DeepL translation provider implementation.
"""

from typing import Optional, List, Tuple
import deepl
import asyncio
from app.translation.base import TranslationProvider, TranslationResult
from app.config import settings
import time


class DeepLTranslator(TranslationProvider):
    """DeepL translation provider for high-quality translations."""
    
    # Language code mapping (DeepL uses specific codes)
    LANGUAGE_MAP = {
        "en": "EN-US",  # English (American)
        "es": "ES",     # Spanish
        "fr": "FR",     # French
        "de": "DE",     # German
        "it": "IT",     # Italian
        "pt": "PT-PT",  # Portuguese
        "nl": "NL",     # Dutch
        "pl": "PL",     # Polish
        "ru": "RU",     # Russian
        "ja": "JA",     # Japanese
        "zh": "ZH",     # Chinese
        "ar": "AR",     # Arabic
    }
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key or settings.deepl_api_key, **kwargs)
        self.client: Optional[deepl.Translator] = None
        self._loop = None
    
    async def _initialize(self):
        """Initialize DeepL client."""
        self._loop = asyncio.get_event_loop()
        self.client = deepl.Translator(self.api_key)
        
        # Test connection and cache supported languages
        try:
            await self._run_sync(self.client.get_source_languages)
            self._supported_languages = await self.get_supported_languages()
        except Exception as e:
            self.logger.error(f"Failed to initialize DeepL: {e}")
            raise
    
    async def _run_sync(self, func, *args, **kwargs):
        """Run synchronous DeepL functions in executor."""
        return await self._loop.run_in_executor(None, func, *args, **kwargs)
    
    def _normalize_language_code(self, lang_code: str, is_target: bool = False) -> str:
        """
        Normalize language code for DeepL API.
        
        Args:
            lang_code: Input language code
            is_target: Whether this is a target language
        
        Returns:
            DeepL-compatible language code
        """
        if not lang_code:
            return None
        
        # Extract base language code
        base_code = lang_code.split('-')[0].lower()
        
        # Map to DeepL format
        if base_code in self.LANGUAGE_MAP:
            deepl_code = self.LANGUAGE_MAP[base_code]
        else:
            deepl_code = base_code.upper()
        
        # For source language, use base code (e.g., "EN" instead of "EN-US")
        if not is_target and '-' in deepl_code:
            deepl_code = deepl_code.split('-')[0]
        
        return deepl_code
    
    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        **kwargs
    ) -> TranslationResult:
        """
        Translate text using DeepL API.
        """
        if not self.validate_text(text):
            return TranslationResult(
                translated_text=text,
                source_language=source_language or "unknown",
                target_language=target_language,
                confidence=0.0
            )
        
        start_time = time.time()
        
        try:
            # Normalize language codes
            target_lang = self._normalize_language_code(target_language, is_target=True)
            source_lang = self._normalize_language_code(source_language) if source_language else None
            
            # Check if translation should be bypassed
            if source_lang and self.should_bypass_translation(
                source_lang, target_lang, 
                settings.bypass_translation_for_same_language
            ):
                self.logger.info(f"Bypassing translation for same language: {source_lang}")
                return TranslationResult(
                    translated_text=text,
                    source_language=source_lang,
                    target_language=target_lang,
                    confidence=1.0,
                    metadata={"bypassed": True}
                )
            
            # Perform translation using lambda to handle kwargs
            def do_translate():
                return self.client.translate_text(
                    text,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    formality=kwargs.get("formality", "default"),
                    preserve_formatting=kwargs.get("preserve_formatting", True),
                )
            
            result = await self._loop.run_in_executor(None, do_translate)
            
            # Build response
            translation_result = TranslationResult(
                translated_text=result.text,
                source_language=result.detected_source_lang.lower() if result.detected_source_lang else source_language,
                target_language=target_language,
                confidence=0.95,  # DeepL doesn't provide confidence scores
                metadata={
                    "processing_time": time.time() - start_time,
                    "billed_characters": len(text)
                }
            )
            
            self.log_latency("deepl_translate", start_time,
                           text_length=len(text),
                           source_lang=translation_result.source_language,
                           target_lang=target_language)
            
            return translation_result
            
        except Exception as e:
            self.log_error("deepl_translate", e)
            return TranslationResult(
                translated_text=text,  # Return original text on error
                source_language=source_language or "unknown",
                target_language=target_language,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language using DeepL (via translation with auto-detect).
        """
        if not self.validate_text(text):
            return "unknown", 0.0
        
        try:
            # Use a dummy translation to detect language
            result = await self._run_sync(
                self.client.translate_text,
                text[:100],  # Use only first 100 chars for detection
                target_lang="EN-US"  # Translate to English for detection
            )
            
            if result.detected_source_lang:
                return result.detected_source_lang.lower(), 0.95
            
            return "unknown", 0.0
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "unknown", 0.0
    
    async def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages from DeepL.
        """
        if self._supported_languages:
            return self._supported_languages
        
        try:
            source_langs = await self._run_sync(self.client.get_source_languages)
            target_langs = await self._run_sync(self.client.get_target_languages)
            
            # Combine and normalize language codes
            all_langs = set()
            for lang in source_langs + target_langs:
                all_langs.add(lang.code.lower())
            
            self._supported_languages = sorted(list(all_langs))
            return self._supported_languages
            
        except Exception as e:
            self.logger.error(f"Failed to get supported languages: {e}")
            return []
    
    async def batch_translate(
        self,
        texts: List[str],
        target_language: str,
        source_language: Optional[str] = None,
        **kwargs
    ) -> List[TranslationResult]:
        """
        Translate multiple texts efficiently.
        """
        if not texts:
            return []
        
        start_time = time.time()
        
        try:
            # Normalize language codes
            target_lang = self._normalize_language_code(target_language, is_target=True)
            source_lang = self._normalize_language_code(source_language) if source_language else None
            
            # DeepL supports batch translation
            results = await self._run_sync(
                self.client.translate_text,
                texts,
                target_lang=target_lang,
                source_lang=source_lang,
                **kwargs
            )
            
            # Convert to TranslationResult objects
            translation_results = []
            for i, result in enumerate(results):
                translation_results.append(TranslationResult(
                    translated_text=result.text,
                    source_language=result.detected_source_lang.lower() if result.detected_source_lang else source_language,
                    target_language=target_language,
                    confidence=0.95,
                    metadata={
                        "batch_index": i,
                        "processing_time": (time.time() - start_time) / len(texts)
                    }
                ))
            
            self.log_latency("deepl_batch_translate", start_time,
                           batch_size=len(texts))
            
            return translation_results
            
        except Exception as e:
            self.log_error("deepl_batch_translate", e)
            # Return original texts on error
            return [
                TranslationResult(
                    translated_text=text,
                    source_language=source_language or "unknown",
                    target_language=target_language,
                    confidence=0.0,
                    metadata={"error": str(e)}
                )
                for text in texts
            ]
    
    async def _close(self):
        """Close DeepL client."""
        # DeepL client doesn't need explicit closing
        self.client = None
