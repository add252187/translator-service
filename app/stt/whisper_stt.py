"""
Whisper STT provider implementation for local or API-based speech recognition.
"""

import asyncio
import time
from typing import Optional, Tuple
import numpy as np
from app.stt.base import STTProvider, STTResult
from app.config import settings
import tempfile
import os

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class WhisperSTTProvider(STTProvider):
    """OpenAI Whisper Speech-to-Text provider for local processing."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        self.model = None
        self.model_name = kwargs.get("model", settings.whisper_model)
        self.device = kwargs.get("device", "cpu")  # or "cuda" for GPU
        
    async def _initialize(self):
        """Initialize Whisper model."""
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper is not installed. Install with: pip install openai-whisper")
        
        # Load model in executor to avoid blocking
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None, 
            whisper.load_model, 
            self.model_name,
            self.device
        )
        self.logger.info(f"Whisper model '{self.model_name}' loaded on {self.device}")
    
    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 8000,
        language_hint: Optional[str] = None,
        **kwargs
    ) -> STTResult:
        """
        Transcribe audio using local Whisper model.
        """
        if not self.validate_audio(audio_data, sample_rate):
            return STTResult(text="", confidence=0.0)
        
        if not self.model:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample if needed (Whisper expects 16kHz)
            if sample_rate != 16000:
                # Simple resampling (for production, use librosa or scipy)
                from scipy import signal
                audio_array = signal.resample(
                    audio_array, 
                    int(len(audio_array) * 16000 / sample_rate)
                )
            
            # Run transcription in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_array,
                language_hint
            )
            
            # Parse result
            text = result.get("text", "").strip()
            detected_language = result.get("language", language_hint)
            
            # Calculate confidence based on no_speech_prob
            no_speech_prob = result.get("no_speech_prob", 0)
            confidence = 1.0 - no_speech_prob
            
            stt_result = STTResult(
                text=text,
                language=detected_language,
                confidence=confidence,
                is_final=True,
                metadata={
                    "processing_time": time.time() - start_time,
                    "model": self.model_name
                }
            )
            
            self.log_latency("whisper_transcribe", start_time,
                           text_length=len(text))
            
            return stt_result
            
        except Exception as e:
            self.log_error("whisper_transcribe", e)
            return STTResult(text="", confidence=0.0, metadata={"error": str(e)})
    
    def _transcribe_sync(self, audio_array: np.ndarray, language: Optional[str] = None):
        """Synchronous transcription for running in executor."""
        # Transcribe with Whisper
        options = {
            "language": language,
            "task": "transcribe",
            "fp16": False,  # Set to True if using GPU with FP16 support
        }
        
        if language:
            # If language is specified, use it
            result = self.model.transcribe(audio_array, **options)
        else:
            # Auto-detect language
            result = self.model.transcribe(audio_array, task="transcribe")
        
        return result
    
    async def transcribe_stream(
        self,
        audio_stream: asyncio.Queue,
        sample_rate: int = 8000,
        language_hint: Optional[str] = None,
        **kwargs
    ) -> asyncio.Queue:
        """
        Transcribe streaming audio using Whisper.
        Note: Whisper doesn't support true streaming, so we process chunks.
        """
        result_queue = asyncio.Queue()
        
        async def process_stream():
            buffer = bytearray()
            chunk_size = sample_rate * 2  # 2 seconds of audio
            
            try:
                while True:
                    audio_chunk = await audio_stream.get()
                    if audio_chunk is None:
                        # Process remaining buffer
                        if buffer:
                            result = await self.transcribe(
                                bytes(buffer), 
                                sample_rate, 
                                language_hint
                            )
                            await result_queue.put(result)
                        break
                    
                    buffer.extend(audio_chunk)
                    
                    # Process when buffer is large enough
                    if len(buffer) >= chunk_size:
                        result = await self.transcribe(
                            bytes(buffer[:chunk_size]),
                            sample_rate,
                            language_hint
                        )
                        await result_queue.put(result)
                        buffer = buffer[chunk_size:]
                        
            except Exception as e:
                self.log_error("whisper_stream", e)
                await result_queue.put(STTResult(text="", metadata={"error": str(e)}))
        
        asyncio.create_task(process_stream())
        return result_queue
    
    async def detect_language(
        self,
        audio_data: bytes,
        sample_rate: int = 8000
    ) -> Tuple[str, float]:
        """
        Detect language using Whisper's built-in language detection.
        """
        if not self.validate_audio(audio_data, sample_rate):
            return "unknown", 0.0
        
        if not self.model:
            await self.initialize()
        
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample if needed
            if sample_rate != 16000:
                from scipy import signal
                audio_array = signal.resample(
                    audio_array,
                    int(len(audio_array) * 16000 / sample_rate)
                )
            
            # Detect language
            loop = asyncio.get_event_loop()
            
            # Use first 30 seconds for detection
            audio_segment = audio_array[:16000 * 30]
            
            result = await loop.run_in_executor(
                None,
                self.model.detect_language,
                audio_segment
            )
            
            # Get language and probability
            if isinstance(result, tuple):
                detected_lang, probs = result
                confidence = probs.get(detected_lang, 0.5) if isinstance(probs, dict) else 0.5
            else:
                detected_lang = result
                confidence = 0.8
            
            return detected_lang, confidence
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "unknown", 0.0
    
    async def _close(self):
        """Clean up Whisper model."""
        self.model = None
