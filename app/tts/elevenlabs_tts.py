"""
ElevenLabs TTS provider implementation.
"""

from typing import Optional, Dict, Any, List, AsyncGenerator
import aiohttp
import asyncio
import time
from app.tts.base import TTSProvider, TTSResult
from app.config import settings


class ElevenLabsTTSProvider(TTSProvider):
    """ElevenLabs Text-to-Speech provider for hyper-realistic voices."""
    
    BASE_URL = "https://api.elevenlabs.io/v1"
    
    # Language to voice mapping (can be extended)
    DEFAULT_VOICES = {
        "en": "21m00Tcm4TlvDq8ikWAM",  # Rachel - English
        "es": "IKne3meq5aSn9XLyUdCD",  # Spanish voice
        "fr": "GBv7mTt0atIp3Br8iCZE",  # French voice
        "de": "pNInz6obpgDQGcFmaJgB",  # German voice
        "it": "pqHfZKP75CvOlQylNhV4",  # Italian voice
        "pt": "Yko7PKc5Ze5SmIFhBH70",  # Portuguese voice
    }
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key or settings.elevenlabs_tts_api_key, **kwargs)
        self.session: Optional[aiohttp.ClientSession] = None
        self.model_id = kwargs.get("model_id", settings.elevenlabs_model_id)
        self.spanish_voice_id = kwargs.get("spanish_voice_id", settings.elevenlabs_voice_id_spanish)
    
    async def _initialize(self):
        """Initialize ElevenLabs connection."""
        self.session = aiohttp.ClientSession(
            headers={
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
        )
        
        # Load available voices
        await self._load_voices()
    
    async def _load_voices(self):
        """Load and cache available voices."""
        try:
            voices = await self.get_available_voices()
            for voice in voices:
                # Cache voices by their supported languages
                for lang in voice.get("languages", []):
                    if lang not in self._voice_cache:
                        self._voice_cache[lang] = voice["voice_id"]
        except Exception as e:
            self.logger.warning(f"Failed to load voices: {e}")
    
    async def synthesize(
        self,
        text: str,
        language: str,
        voice_id: Optional[str] = None,
        **kwargs
    ) -> TTSResult:
        """
        Synthesize speech using ElevenLabs API.
        """
        if not self.validate_text(text):
            return TTSResult(audio_data=b"", sample_rate=8000)
        
        start_time = time.time()
        
        try:
            # Get voice ID for the language
            if not voice_id:
                if language == "es":
                    voice_id = self.spanish_voice_id or self.DEFAULT_VOICES.get("es")
                else:
                    voice_id = await self.get_voice_for_language(language)
                    if not voice_id:
                        voice_id = self.DEFAULT_VOICES.get(language, self.DEFAULT_VOICES["en"])
            
            # Prepare request
            url = f"{self.BASE_URL}/text-to-speech/{voice_id}"
            
            payload = {
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {
                    "stability": kwargs.get("stability", 0.5),
                    "similarity_boost": kwargs.get("similarity_boost", 0.75),
                    "style": kwargs.get("style", 0.0),
                    "use_speaker_boost": kwargs.get("use_speaker_boost", True)
                }
            }
            
            # Add language code if using multilingual model
            if "multilingual" in self.model_id.lower():
                payload["language_code"] = language
            
            # Make API request
            async with self.session.post(
                url,
                json=payload,
                headers={"Accept": "audio/mpeg"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                audio_data = await response.read()
            
            # Calculate duration (approximate based on typical speech rate)
            words = len(text.split())
            duration_ms = int(words * 400)  # ~150 words per minute
            
            result = TTSResult(
                audio_data=audio_data,
                sample_rate=44100,  # ElevenLabs returns high-quality audio
                duration_ms=duration_ms,
                voice_id=voice_id,
                metadata={
                    "processing_time": time.time() - start_time,
                    "text_length": len(text),
                    "model": self.model_id
                }
            )
            
            self.log_latency("elevenlabs_synthesize", start_time,
                           text_length=len(text), language=language)
            
            return result
            
        except Exception as e:
            self.log_error("elevenlabs_synthesize", e)
            return TTSResult(
                audio_data=b"",
                sample_rate=8000,
                metadata={"error": str(e)}
            )
    
    async def synthesize_stream(
        self,
        text: str,
        language: str,
        voice_id: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech as a stream using ElevenLabs streaming API.
        """
        if not self.validate_text(text):
            return
        
        try:
            # Get voice ID for the language
            if not voice_id:
                if language == "es":
                    voice_id = self.spanish_voice_id or self.DEFAULT_VOICES.get("es")
                else:
                    voice_id = await self.get_voice_for_language(language)
                    if not voice_id:
                        voice_id = self.DEFAULT_VOICES.get(language, self.DEFAULT_VOICES["en"])
            
            # Prepare request for streaming
            url = f"{self.BASE_URL}/text-to-speech/{voice_id}/stream"
            
            payload = {
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {
                    "stability": kwargs.get("stability", 0.5),
                    "similarity_boost": kwargs.get("similarity_boost", 0.75)
                },
                "optimize_streaming_latency": kwargs.get("optimize_latency", 3)
            }
            
            # Add language code if using multilingual model
            if "multilingual" in self.model_id.lower():
                payload["language_code"] = language
            
            # Stream the response
            async with self.session.post(
                url,
                json=payload,
                headers={"Accept": "audio/mpeg"},
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                response.raise_for_status()
                
                # Stream audio chunks
                async for chunk in response.content.iter_chunked(1024):
                    if chunk:
                        yield chunk
                        
        except Exception as e:
            self.log_error("elevenlabs_stream", e)
            # Yield empty chunk on error
            yield b""
    
    async def get_available_voices(
        self,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of available voices from ElevenLabs.
        """
        try:
            url = f"{self.BASE_URL}/voices"
            
            async with self.session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                data = await response.json()
            
            voices = []
            for voice in data.get("voices", []):
                voice_info = {
                    "voice_id": voice["voice_id"],
                    "name": voice["name"],
                    "category": voice.get("category", ""),
                    "labels": voice.get("labels", {}),
                    "languages": [],
                    "gender": None
                }
                
                # Extract language support from labels
                if "language" in voice.get("labels", {}):
                    voice_info["languages"] = [voice["labels"]["language"]]
                
                # Extract gender from labels
                if "gender" in voice.get("labels", {}):
                    voice_info["gender"] = voice["labels"]["gender"]
                
                # Filter by language if specified
                if language:
                    if language in voice_info["languages"]:
                        voices.append(voice_info)
                else:
                    voices.append(voice_info)
            
            return voices
            
        except Exception as e:
            self.logger.error(f"Failed to get voices: {e}")
            return []
    
    async def get_voice_for_language(
        self,
        language: str,
        gender: Optional[str] = None,
        style: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the best voice for a given language.
        """
        # Check if we have a cached voice for this language
        if language in self._voice_cache:
            return self._voice_cache[language]
        
        # Check default voices
        if language in self.DEFAULT_VOICES:
            return self.DEFAULT_VOICES[language]
        
        # Try to find a voice through the API
        voices = await self.get_available_voices(language)
        
        if voices:
            # Apply gender filter if specified
            if gender:
                gender_filtered = [v for v in voices if v.get("gender", "").lower() == gender.lower()]
                if gender_filtered:
                    voices = gender_filtered
            
            # Return the first matching voice
            voice_id = voices[0]["voice_id"]
            self._voice_cache[language] = voice_id
            return voice_id
        
        # Fallback to English voice
        self.logger.warning(f"No voice found for language {language}, using English")
        return self.DEFAULT_VOICES["en"]
    
    async def _close(self):
        """Close ElevenLabs session."""
        if self.session:
            await self.session.close()
