"""
Deepgram STT provider implementation.
"""

import asyncio
import json
from typing import Optional, Dict, Any, Tuple
import aiohttp
from app.stt.base import STTProvider, STTResult
from app.config import settings
import base64
import time


class DeepgramSTTProvider(STTProvider):
    """Deepgram Speech-to-Text provider for real-time transcription."""
    
    BASE_URL = "https://api.deepgram.com/v1"
    WS_URL = "wss://api.deepgram.com/v1/listen"
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key or settings.deepgram_api_key, **kwargs)
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection = None
        
    async def _initialize(self):
        """Initialize Deepgram connection."""
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Token {self.api_key}"}
        )
    
    async def transcribe(
        self,
        audio_data: bytes,
        sample_rate: int = 8000,
        language_hint: Optional[str] = None,
        **kwargs
    ) -> STTResult:
        """
        Transcribe audio using Deepgram's REST API.
        """
        if not self.validate_audio(audio_data, sample_rate):
            return STTResult(text="", confidence=0.0)
        
        start_time = time.time()
        
        try:
            # Prepare request parameters
            params = {
                "model": kwargs.get("model", "nova-2"),
                "punctuate": "true",
                "diarize": "false",
                "smart_format": "true",
                "language": language_hint or "multi",
                "detect_language": "true" if not language_hint else "false",
            }
            
            # Make API request
            url = f"{self.BASE_URL}/listen"
            
            # Add encoding parameters for raw PCM audio
            params["encoding"] = "linear16"
            params["sample_rate"] = str(sample_rate)
            params["channels"] = "1"
            
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/raw"
            }
            
            self.logger.info(f"Sending {len(audio_data)} bytes to Deepgram, sample_rate={sample_rate}")
            
            async with self.session.post(
                url,
                headers=headers,
                params=params,
                data=audio_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                result = await response.json()
            
            # Parse response
            if result and "results" in result and result["results"]["channels"]:
                channel = result["results"]["channels"][0]
                if channel["alternatives"]:
                    alt = channel["alternatives"][0]
                    
                    # Get detected language if available
                    detected_language = None
                    if "detected_language" in channel:
                        detected_language = channel["detected_language"]
                    
                    stt_result = STTResult(
                        text=alt["transcript"],
                        confidence=alt.get("confidence", 0.0),
                        language=detected_language,
                        is_final=True,
                        alternatives=[{"text": a["transcript"], "confidence": a.get("confidence", 0.0)} 
                                    for a in channel["alternatives"][1:3]],
                        metadata={"processing_time": time.time() - start_time}
                    )
                    
                    self.log_latency("deepgram_transcribe", start_time, 
                                   text_length=len(stt_result.text))
                    return stt_result
            
            return STTResult(text="", confidence=0.0)
            
        except Exception as e:
            self.log_error("deepgram_transcribe", e)
            return STTResult(text="", confidence=0.0, metadata={"error": str(e)})
    
    async def transcribe_stream(
        self,
        audio_stream: asyncio.Queue,
        sample_rate: int = 8000,
        language_hint: Optional[str] = None,
        **kwargs
    ) -> asyncio.Queue:
        """
        Transcribe streaming audio using Deepgram's WebSocket API.
        """
        result_queue = asyncio.Queue()
        
        # Build WebSocket URL with parameters
        params = {
            "encoding": "linear16",
            "sample_rate": str(sample_rate),
            "channels": "1",
            "model": kwargs.get("model", "nova-2"),
            "punctuate": "true",
            "interim_results": "true",
            "language": language_hint or "multi",
            "detect_language": "true" if not language_hint else "false",
            "vad_events": "true",
            "endpointing": kwargs.get("endpointing", 300),
        }
        
        param_str = "&".join([f"{k}={v}" for k, v in params.items()])
        ws_url = f"{self.WS_URL}?{param_str}"
        
        async def process_stream():
            try:
                async with self.session.ws_connect(
                    ws_url,
                    headers={"Authorization": f"Token {self.api_key}"}
                ) as ws:
                    self.ws_connection = ws
                    
                    # Start tasks for sending and receiving
                    send_task = asyncio.create_task(self._send_audio(ws, audio_stream))
                    receive_task = asyncio.create_task(self._receive_transcripts(ws, result_queue))
                    
                    await asyncio.gather(send_task, receive_task)
                    
            except Exception as e:
                self.log_error("deepgram_stream", e)
                await result_queue.put(STTResult(text="", metadata={"error": str(e)}))
            finally:
                self.ws_connection = None
        
        # Start processing in background
        asyncio.create_task(process_stream())
        
        return result_queue
    
    async def _send_audio(self, ws, audio_stream: asyncio.Queue):
        """Send audio chunks to Deepgram WebSocket."""
        try:
            while True:
                audio_chunk = await audio_stream.get()
                if audio_chunk is None:  # End of stream
                    await ws.send_json({"type": "CloseStream"})
                    break
                
                await ws.send_bytes(audio_chunk)
                
        except Exception as e:
            self.log_error("deepgram_send_audio", e)
    
    async def _receive_transcripts(self, ws, result_queue: asyncio.Queue):
        """Receive transcripts from Deepgram WebSocket."""
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    
                    if data.get("type") == "Results":
                        channel = data.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        
                        if alternatives:
                            alt = alternatives[0]
                            
                            # Get detected language
                            detected_language = data.get("metadata", {}).get("detected_language")
                            
                            result = STTResult(
                                text=alt.get("transcript", ""),
                                confidence=alt.get("confidence", 0.0),
                                language=detected_language,
                                is_final=data.get("is_final", False),
                                alternatives=[{"text": a["transcript"], "confidence": a.get("confidence", 0.0)} 
                                            for a in alternatives[1:3]],
                                metadata={
                                    "speech_final": data.get("speech_final", False),
                                    "duration": data.get("duration", 0)
                                }
                            )
                            
                            await result_queue.put(result)
                    
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {msg.data}")
                    break
                    
        except Exception as e:
            self.log_error("deepgram_receive", e)
    
    async def detect_language(
        self,
        audio_data: bytes,
        sample_rate: int = 8000
    ) -> Tuple[str, float]:
        """
        Detect language using Deepgram's language detection.
        """
        result = await self.transcribe(
            audio_data,
            sample_rate,
            language_hint=None  # Let Deepgram detect
        )
        
        if result.language:
            return result.language, result.confidence or 0.5
        
        return "unknown", 0.0
    
    async def _close(self):
        """Close Deepgram connections."""
        if self.ws_connection:
            await self.ws_connection.close()
        
        if self.session:
            await self.session.close()
