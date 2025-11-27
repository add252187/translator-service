"""
Twilio Media Streams WebSocket handler for real-time audio processing.
"""

import json
import base64
import asyncio
from typing import Optional, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from app.call.manager import call_manager
from app.call.session import CallSession
from app.utils.audio import AudioProcessor
from app.utils.logging import get_logger, call_id_var
from app.config import settings

logger = get_logger(__name__)


class TwilioStreamHandler:
    """
    Handles Twilio Media Streams WebSocket connections.
    Processes bidirectional audio streams for real-time translation.
    """
    
    def __init__(self, websocket: WebSocket):
        """
        Initialize Twilio stream handler.
        
        Args:
            websocket: FastAPI WebSocket connection
        """
        self.websocket = websocket
        self.call_session: Optional[CallSession] = None
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None
        self.audio_processor = AudioProcessor()
        self.sequence_number = 0
        self._output_task: Optional[asyncio.Task] = None
    
    async def handle(self):
        """Main handler for WebSocket connection."""
        try:
            await self.websocket.accept()
            logger.info("Twilio WebSocket connection accepted")
            
            while True:
                # Receive message from Twilio
                message = await self.websocket.receive_text()
                await self._process_message(message)
                
        except WebSocketDisconnect:
            logger.info("Twilio WebSocket disconnected")
            await self._cleanup()
        except Exception as e:
            logger.error(f"Error in Twilio stream handler: {e}")
            await self._cleanup()
    
    async def _process_message(self, message: str):
        """
        Process incoming Twilio message.
        
        Args:
            message: JSON message from Twilio
        """
        try:
            data = json.loads(message)
            event_type = data.get("event")
            
            if event_type == "start":
                await self._handle_start(data)
            elif event_type == "media":
                await self._handle_media(data)
            elif event_type == "stop":
                await self._handle_stop(data)
            elif event_type == "mark":
                await self._handle_mark(data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from Twilio: {e}")
        except Exception as e:
            logger.error(f"Error processing Twilio message: {e}")
    
    async def _handle_start(self, data: Dict[str, Any]):
        """
        Handle stream start event.
        
        Args:
            data: Start event data from Twilio
        """
        start_data = data.get("start", {})
        self.stream_sid = start_data.get("streamSid")
        self.call_sid = start_data.get("callSid")
        
        # Set call ID context for logging
        call_id_var.set(self.call_sid)
        
        # Get call metadata
        custom_parameters = start_data.get("customParameters", {})
        agent_phone = custom_parameters.get("agentPhone")
        client_phone = custom_parameters.get("clientPhone")
        
        logger.info(
            f"Stream started",
            stream_sid=self.stream_sid,
            call_sid=self.call_sid,
            agent_phone=agent_phone,
            client_phone=client_phone
        )
        
        # Create call session
        try:
            self.call_session = await call_manager.create_session(
                call_sid=self.call_sid,
                stream_sid=self.stream_sid,
                agent_phone=agent_phone,
                client_phone=client_phone
            )
            
            # Start output task
            self._output_task = asyncio.create_task(self._send_output())
            
        except Exception as e:
            logger.error(f"Failed to create call session: {e}")
            await self.websocket.close()
    
    async def _handle_media(self, data: Dict[str, Any]):
        """
        Handle media event with audio data.
        
        Args:
            data: Media event data from Twilio
        """
        if not self.call_session:
            return
        
        media = data.get("media", {})
        track = media.get("track")  # "inbound" or "outbound"
        payload = media.get("payload")  # Base64 encoded μ-law audio
        
        if not payload:
            return
        
        try:
            # Decode base64 audio
            audio_data = base64.b64decode(payload)
            
            # Convert μ-law to PCM for processing
            pcm_audio = self.audio_processor.ulaw_to_pcm(audio_data)
            
            # Route audio to appropriate buffer
            if track == "inbound":
                # Audio from client
                await self.call_session.add_client_audio(pcm_audio)
            elif track == "outbound":
                # Audio from agent
                await self.call_session.add_agent_audio(pcm_audio)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
    
    async def _handle_stop(self, data: Dict[str, Any]):
        """
        Handle stream stop event.
        
        Args:
            data: Stop event data from Twilio
        """
        logger.info(f"Stream stopped", stream_sid=self.stream_sid)
        await self._cleanup()
    
    async def _handle_mark(self, data: Dict[str, Any]):
        """
        Handle mark event (used for synchronization).
        
        Args:
            data: Mark event data from Twilio
        """
        mark = data.get("mark", {})
        name = mark.get("name")
        logger.debug(f"Mark received: {name}")
    
    async def _send_output(self):
        """Send processed audio back to Twilio."""
        while self.call_session and self.call_session.is_active:
            try:
                # Check for audio to send to client
                client_audio = await self.call_session.get_client_output()
                if client_audio:
                    await self._send_audio(client_audio, "inbound")
                
                # Check for audio to send to agent
                agent_audio = await self.call_session.get_agent_output()
                if agent_audio:
                    await self._send_audio(agent_audio, "outbound")
                
                # Small delay to prevent busy waiting
                if not client_audio and not agent_audio:
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error sending output: {e}")
                await asyncio.sleep(0.1)
    
    async def _send_audio(self, audio_data: bytes, track: str):
        """
        Send audio data to Twilio.
        
        Args:
            audio_data: Audio data in μ-law format
            track: Target track ("inbound" or "outbound")
        """
        try:
            # Encode audio to base64
            encoded_audio = base64.b64encode(audio_data).decode('utf-8')
            
            # Create media message
            message = {
                "event": "media",
                "sequenceNumber": str(self.sequence_number),
                "media": {
                    "track": track,
                    "chunk": str(self.sequence_number),
                    "timestamp": str(self.sequence_number * 20),  # 20ms chunks
                    "payload": encoded_audio
                }
            }
            
            # Send to Twilio
            await self.websocket.send_text(json.dumps(message))
            
            self.sequence_number += 1
            
        except Exception as e:
            logger.error(f"Error sending audio to Twilio: {e}")
    
    async def _send_clear(self):
        """Send clear message to Twilio to clear audio buffer."""
        try:
            message = {
                "event": "clear",
                "streamSid": self.stream_sid
            }
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending clear message: {e}")
    
    async def _cleanup(self):
        """Clean up resources when connection ends."""
        try:
            # Cancel output task
            if self._output_task:
                self._output_task.cancel()
                await asyncio.gather(self._output_task, return_exceptions=True)
            
            # End call session
            if self.call_sid:
                await call_manager.end_session(self.call_sid)
            
            # Close WebSocket
            await self.websocket.close()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def handle_twilio_stream(websocket: WebSocket):
    """
    Entry point for handling Twilio Media Streams.
    
    Args:
        websocket: FastAPI WebSocket connection
    """
    handler = TwilioStreamHandler(websocket)
    await handler.handle()
