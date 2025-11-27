"""
Call session management for individual calls.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from app.utils.logging import LoggerMixin
from app.utils.audio import AudioBuffer, AudioProcessor, VoiceActivityDetector
from app.stt import STTProvider, STTResult
from app.translation import TranslationProvider, TranslationResult
from app.tts import TTSProvider, TTSResult
from app.config import settings


class AudioDirection(str, Enum):
    """Direction of audio flow in a call."""
    CLIENT_TO_AGENT = "client_to_agent"
    AGENT_TO_CLIENT = "agent_to_client"


@dataclass
class CallMetadata:
    """Metadata for a call session."""
    call_sid: str
    stream_sid: Optional[str] = None
    agent_phone: Optional[str] = None
    client_phone: Optional[str] = None
    client_language: Optional[str] = None
    client_language_confidence: float = 0.0
    agent_language: str = "es"
    started_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: Optional[datetime] = None
    
    # Performance tracking
    stt_count: int = 0
    translation_count: int = 0
    tts_count: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0


class CallSession(LoggerMixin):
    """
    Manages a single call session with bidirectional translation.
    Coordinates the flow: Audio → STT → Translation → TTS → Audio
    """
    
    def __init__(
        self,
        call_sid: str,
        stt_provider: STTProvider,
        translation_provider: TranslationProvider,
        tts_provider: TTSProvider,
        **kwargs
    ):
        """
        Initialize a call session.
        
        Args:
            call_sid: Unique call identifier
            stt_provider: Speech-to-text provider instance
            translation_provider: Translation provider instance
            tts_provider: Text-to-speech provider instance
            **kwargs: Additional configuration
        """
        self.call_sid = call_sid
        self.stt_provider = stt_provider
        self.translation_provider = translation_provider
        self.tts_provider = tts_provider
        
        # Call metadata
        self.metadata = CallMetadata(
            call_sid=call_sid,
            agent_language=kwargs.get("agent_language", settings.default_agent_language)
        )
        
        # Audio buffers for each direction
        self.client_audio_buffer = AudioBuffer()
        self.agent_audio_buffer = AudioBuffer()
        
        # Output queues for processed audio
        self.client_output_queue: asyncio.Queue = asyncio.Queue()
        self.agent_output_queue: asyncio.Queue = asyncio.Queue()
        
        # Processing queues
        self.client_text_queue: asyncio.Queue = asyncio.Queue()
        self.agent_text_queue: asyncio.Queue = asyncio.Queue()
        
        # Voice activity detection
        self.client_vad = VoiceActivityDetector(settings.vad_aggressiveness)
        self.agent_vad = VoiceActivityDetector(settings.vad_aggressiveness)
        
        # Audio processor
        self.audio_processor = AudioProcessor()
        
        # Control flags
        self.is_active = False
        self.language_detected = False
        self._tasks: list[asyncio.Task] = []
        
        # Configuration
        self.bypass_same_language = kwargs.get(
            "bypass_same_language",
            settings.bypass_translation_for_same_language
        )
        self.language_detection_threshold = kwargs.get(
            "language_detection_threshold",
            settings.language_detection_confidence_threshold
        )
    
    async def start(self):
        """Start the call session and processing pipelines."""
        if self.is_active:
            return
        
        self.is_active = True
        self.metadata.connected_at = datetime.utcnow()
        
        # Initialize providers
        await self.stt_provider.initialize()
        await self.translation_provider.initialize()
        await self.tts_provider.initialize()
        
        # Start processing tasks
        self._tasks = [
            asyncio.create_task(self._process_client_audio()),
            asyncio.create_task(self._process_agent_audio()),
            asyncio.create_task(self._process_client_text()),
            asyncio.create_task(self._process_agent_text()),
        ]
        
        self.logger.info(f"Call session started", call_sid=self.call_sid)
    
    async def stop(self):
        """Stop the call session and clean up resources."""
        if not self.is_active:
            return
        
        self.is_active = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Close providers
        await self.stt_provider.close()
        await self.translation_provider.close()
        await self.tts_provider.close()
        
        # Clear buffers
        await self.client_audio_buffer.clear()
        await self.agent_audio_buffer.clear()
        
        self.logger.info(
            f"Call session stopped",
            call_sid=self.call_sid,
            duration_seconds=(datetime.utcnow() - self.metadata.started_at).total_seconds(),
            client_language=self.metadata.client_language,
            error_count=self.metadata.error_count
        )
    
    async def add_client_audio(self, audio_data: bytes):
        """Add audio from the client side."""
        await self.client_audio_buffer.write(audio_data)
    
    async def add_agent_audio(self, audio_data: bytes):
        """Add audio from the agent side."""
        await self.agent_audio_buffer.write(audio_data)
    
    async def get_client_output(self) -> Optional[bytes]:
        """Get processed audio for the client."""
        try:
            return await asyncio.wait_for(self.client_output_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    async def get_agent_output(self) -> Optional[bytes]:
        """Get processed audio for the agent."""
        try:
            return await asyncio.wait_for(self.agent_output_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    async def _process_client_audio(self):
        """Process audio from client: STT → Detect Language → Queue for translation."""
        audio_accumulator = bytearray()
        # Accumulate 2 seconds of audio before processing
        min_audio_bytes = settings.audio_sample_rate * 2 * 2  # 2 seconds, 16-bit
        
        while self.is_active:
            try:
                # Read audio chunk
                audio_chunk = await self.client_audio_buffer.read(
                    settings.audio_sample_rate * settings.audio_chunk_duration_ms // 1000 * 2
                )
                
                if not audio_chunk:
                    await asyncio.sleep(0.01)
                    continue
                
                # Accumulate audio
                audio_accumulator.extend(audio_chunk)
                
                # Only process when we have enough audio
                if len(audio_accumulator) < min_audio_bytes:
                    continue
                
                self.logger.info(f"Processing {len(audio_accumulator)} bytes of audio")
                
                start_time = time.time()
                
                # Perform STT on accumulated audio
                stt_result = await self.stt_provider.transcribe(
                    bytes(audio_accumulator),
                    sample_rate=settings.audio_sample_rate,
                    language_hint=self.metadata.client_language
                )
                
                # Clear accumulator
                audio_accumulator.clear()
                
                if not stt_result.text:
                    self.logger.info("STT returned empty text")
                    continue
                
                self.logger.info(f"STT result: {stt_result.text}")
                self.metadata.stt_count += 1
                
                # Detect language if not yet detected
                if not self.language_detected and stt_result.language:
                    if stt_result.confidence >= self.language_detection_threshold:
                        self.metadata.client_language = stt_result.language
                        self.metadata.client_language_confidence = stt_result.confidence
                        self.language_detected = True
                        self.logger.info(
                            f"Client language detected",
                            call_sid=self.call_sid,
                            language=stt_result.language,
                            confidence=stt_result.confidence
                        )
                
                # Queue text for translation
                await self.client_text_queue.put({
                    "text": stt_result.text,
                    "language": stt_result.language or self.metadata.client_language,
                    "timestamp": time.time(),
                    "processing_time": time.time() - start_time
                })
                
            except Exception as e:
                self.metadata.error_count += 1
                self.logger.error(f"Error processing client audio", error=str(e))
                await asyncio.sleep(0.1)
    
    async def _process_agent_audio(self):
        """Process audio from agent: STT → Queue for translation."""
        while self.is_active:
            try:
                # Read audio chunk
                audio_chunk = await self.agent_audio_buffer.read(
                    settings.audio_sample_rate * settings.audio_chunk_duration_ms // 1000 * 2
                )
                
                if not audio_chunk:
                    await asyncio.sleep(0.01)
                    continue
                
                # Check for speech using VAD
                if not self.agent_vad.is_speech(audio_chunk):
                    continue
                
                start_time = time.time()
                
                # Perform STT (agent speaks Spanish)
                stt_result = await self.stt_provider.transcribe(
                    audio_chunk,
                    sample_rate=settings.audio_sample_rate,
                    language_hint=self.metadata.agent_language
                )
                
                if not stt_result.text:
                    continue
                
                self.metadata.stt_count += 1
                
                # Queue text for translation
                await self.agent_text_queue.put({
                    "text": stt_result.text,
                    "language": self.metadata.agent_language,
                    "timestamp": time.time(),
                    "processing_time": time.time() - start_time
                })
                
            except Exception as e:
                self.metadata.error_count += 1
                self.logger.error(f"Error processing agent audio", error=str(e))
                await asyncio.sleep(0.1)
    
    async def _process_client_text(self):
        """Process client text: Translate to Spanish → TTS → Send to agent."""
        while self.is_active:
            try:
                # Get text from queue
                text_data = await self.client_text_queue.get()
                
                start_time = time.time()
                
                # Translate to Spanish for the agent
                if self.bypass_same_language and text_data["language"] == self.metadata.agent_language:
                    # Skip translation if same language
                    translated_text = text_data["text"]
                else:
                    translation_result = await self.translation_provider.translate(
                        text_data["text"],
                        target_language=self.metadata.agent_language,
                        source_language=text_data["language"]
                    )
                    translated_text = translation_result.translated_text
                    self.metadata.translation_count += 1
                
                # Generate Spanish audio for agent
                tts_result = await self.tts_provider.synthesize(
                    translated_text,
                    language=self.metadata.agent_language
                )
                
                if tts_result.audio_data:
                    # Convert audio format if needed (to μ-law for Twilio)
                    converted_audio = self.audio_processor.convert_audio_format(
                        tts_result.audio_data,
                        input_format="mp3",  # ElevenLabs returns MP3
                        output_format="ulaw",
                        sample_rate=8000
                    )
                    
                    # Send to agent output queue
                    await self.agent_output_queue.put(converted_audio)
                    self.metadata.tts_count += 1
                
                # Track latency
                total_latency = (time.time() - text_data["timestamp"]) * 1000
                self.metadata.total_latency_ms = (
                    (self.metadata.total_latency_ms + total_latency) / 2
                )
                
                self.log_latency(
                    "client_to_agent_pipeline",
                    start_time,
                    source_lang=text_data["language"],
                    target_lang=self.metadata.agent_language
                )
                
            except Exception as e:
                self.metadata.error_count += 1
                self.logger.error(f"Error processing client text", error=str(e))
    
    async def _process_agent_text(self):
        """Process agent text: Translate to client language → TTS → Send to client."""
        while self.is_active:
            try:
                # Get text from queue
                text_data = await self.agent_text_queue.get()
                
                # Skip if client language not detected yet
                if not self.metadata.client_language:
                    self.logger.warning("Client language not yet detected, skipping translation")
                    continue
                
                start_time = time.time()
                
                # Translate to client's language
                if self.bypass_same_language and self.metadata.client_language == self.metadata.agent_language:
                    # Skip translation if same language
                    translated_text = text_data["text"]
                else:
                    translation_result = await self.translation_provider.translate(
                        text_data["text"],
                        target_language=self.metadata.client_language,
                        source_language=self.metadata.agent_language
                    )
                    translated_text = translation_result.translated_text
                    self.metadata.translation_count += 1
                
                # Generate audio in client's language
                tts_result = await self.tts_provider.synthesize(
                    translated_text,
                    language=self.metadata.client_language
                )
                
                if tts_result.audio_data:
                    # Convert audio format if needed (to μ-law for Twilio)
                    converted_audio = self.audio_processor.convert_audio_format(
                        tts_result.audio_data,
                        input_format="mp3",  # ElevenLabs returns MP3
                        output_format="ulaw",
                        sample_rate=8000
                    )
                    
                    # Send to client output queue
                    await self.client_output_queue.put(converted_audio)
                    self.metadata.tts_count += 1
                
                # Track latency
                total_latency = (time.time() - text_data["timestamp"]) * 1000
                self.metadata.total_latency_ms = (
                    (self.metadata.total_latency_ms + total_latency) / 2
                )
                
                self.log_latency(
                    "agent_to_client_pipeline",
                    start_time,
                    source_lang=self.metadata.agent_language,
                    target_lang=self.metadata.client_language
                )
                
            except Exception as e:
                self.metadata.error_count += 1
                self.logger.error(f"Error processing agent text", error=str(e))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current session metrics."""
        duration = (datetime.utcnow() - self.metadata.started_at).total_seconds()
        
        return {
            "call_sid": self.call_sid,
            "duration_seconds": duration,
            "client_language": self.metadata.client_language,
            "client_language_confidence": self.metadata.client_language_confidence,
            "stt_count": self.metadata.stt_count,
            "translation_count": self.metadata.translation_count,
            "tts_count": self.metadata.tts_count,
            "avg_latency_ms": self.metadata.total_latency_ms,
            "error_count": self.metadata.error_count,
            "is_active": self.is_active
        }
