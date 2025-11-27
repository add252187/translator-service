"""
OpenAI GPT-based translation provider implementation.
"""

from typing import Optional, List, Tuple
import openai
from openai import AsyncOpenAI
import json
from app.translation.base import TranslationProvider, TranslationResult
from app.config import settings
import time


class OpenAITranslator(TranslationProvider):
    """OpenAI GPT-based translation provider for context-aware translations."""
    
    # Supported languages (can be extended)
    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "zh", "ar",
        "gl", "ca", "eu", "ast",  # Regional Spanish languages
        "hi", "ur", "bn", "ta", "te",  # Indian languages
        "ko", "vi", "th", "id", "ms",  # Asian languages
        "tr", "he", "fa", "uk", "cs", "sk",  # More European/Middle Eastern
        "sv", "no", "da", "fi", "is",  # Nordic languages
    ]
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key or settings.openai_api_key, **kwargs)
        self.client: Optional[AsyncOpenAI] = None
        self.model = kwargs.get("model", settings.openai_model)
        self.temperature = kwargs.get("temperature", 0.3)  # Lower for more consistent translations
    
    async def _initialize(self):
        """Initialize OpenAI client."""
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Test connection
        try:
            await self.client.models.list()
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI: {e}")
            raise
    
    def _create_translation_prompt(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        context: Optional[str] = None
    ) -> str:
        """
        Create an optimized prompt for translation.
        """
        source_lang_str = f"from {source_language}" if source_language else "from the detected language"
        
        prompt = f"""You are a professional translator specializing in real-time phone call translation.
Translate the following text {source_lang_str} to {target_language}.

Requirements:
- Maintain the conversational tone and register
- Preserve any cultural nuances
- Keep the translation natural and fluent
- If the text is already in {target_language}, return it unchanged
- Return ONLY the translation, no explanations

Text to translate: "{text}"

Translation:"""
        
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        return prompt
    
    async def translate(
        self,
        text: str,
        target_language: str,
        source_language: Optional[str] = None,
        **kwargs
    ) -> TranslationResult:
        """
        Translate text using OpenAI GPT.
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
            # Check if translation should be bypassed
            if source_language and self.should_bypass_translation(
                source_language, target_language,
                settings.bypass_translation_for_same_language
            ):
                self.logger.info(f"Bypassing translation for same language: {source_language}")
                return TranslationResult(
                    translated_text=text,
                    source_language=source_language,
                    target_language=target_language,
                    confidence=1.0,
                    metadata={"bypassed": True}
                )
            
            # Create the translation prompt
            prompt = self._create_translation_prompt(
                text, target_language, source_language,
                context=kwargs.get("context")
            )
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional translator for real-time phone calls."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=min(len(text) * 3, 2000),  # Estimate max tokens needed
                timeout=settings.translation_timeout
            )
            
            translated_text = response.choices[0].message.content.strip()
            
            # If source language wasn't provided, try to detect it
            detected_language = source_language
            if not source_language:
                detected_language, _ = await self.detect_language(text)
            
            translation_result = TranslationResult(
                translated_text=translated_text,
                source_language=detected_language or "unknown",
                target_language=target_language,
                confidence=0.9,  # GPT doesn't provide confidence scores
                metadata={
                    "processing_time": time.time() - start_time,
                    "model": self.model,
                    "tokens_used": response.usage.total_tokens if response.usage else None
                }
            )
            
            self.log_latency("openai_translate", start_time,
                           text_length=len(text),
                           source_lang=translation_result.source_language,
                           target_lang=target_language)
            
            return translation_result
            
        except Exception as e:
            self.log_error("openai_translate", e)
            return TranslationResult(
                translated_text=text,  # Return original text on error
                source_language=source_language or "unknown",
                target_language=target_language,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect language using GPT.
        """
        if not self.validate_text(text):
            return "unknown", 0.0
        
        try:
            prompt = f"""Detect the language of the following text and return ONLY the ISO 639-1 language code (e.g., 'en' for English, 'es' for Spanish, 'gl' for Galician).

Text: "{text[:200]}"

Language code:"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a language detection system. Return only ISO 639-1 codes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Very low temperature for consistent detection
                max_tokens=10,
                timeout=5
            )
            
            detected_lang = response.choices[0].message.content.strip().lower()
            
            # Validate the language code
            if len(detected_lang) == 2 and detected_lang.isalpha():
                return detected_lang, 0.85
            
            return "unknown", 0.0
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "unknown", 0.0
    
    async def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        """
        return self.SUPPORTED_LANGUAGES.copy()
    
    async def batch_translate(
        self,
        texts: List[str],
        target_language: str,
        source_language: Optional[str] = None,
        **kwargs
    ) -> List[TranslationResult]:
        """
        Translate multiple texts efficiently using GPT.
        """
        if not texts:
            return []
        
        start_time = time.time()
        
        try:
            # Create batch prompt
            texts_json = json.dumps(texts, ensure_ascii=False)
            source_lang_str = f"from {source_language}" if source_language else "from their detected languages"
            
            prompt = f"""Translate the following list of texts {source_lang_str} to {target_language}.
Return a JSON array with the translations in the same order.

Texts to translate:
{texts_json}

Return format: ["translation1", "translation2", ...]

Translations:"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional translator. Return only valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=sum(len(t) * 3 for t in texts),
                timeout=settings.translation_timeout * 2  # Allow more time for batch
            )
            
            # Parse the response
            try:
                translations = json.loads(response.choices[0].message.content.strip())
            except json.JSONDecodeError:
                # Fallback to individual translation
                return await super().batch_translate(texts, target_language, source_language, **kwargs)
            
            # Build results
            translation_results = []
            for i, (original, translated) in enumerate(zip(texts, translations)):
                translation_results.append(TranslationResult(
                    translated_text=translated,
                    source_language=source_language or "unknown",
                    target_language=target_language,
                    confidence=0.9,
                    metadata={
                        "batch_index": i,
                        "processing_time": (time.time() - start_time) / len(texts)
                    }
                ))
            
            self.log_latency("openai_batch_translate", start_time,
                           batch_size=len(texts))
            
            return translation_results
            
        except Exception as e:
            self.log_error("openai_batch_translate", e)
            # Fallback to individual translation
            return await super().batch_translate(texts, target_language, source_language, **kwargs)
    
    async def _close(self):
        """Close OpenAI client."""
        if self.client:
            await self.client.close()
            self.client = None
