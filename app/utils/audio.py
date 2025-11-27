"""
Audio processing utilities for format conversion and manipulation.
"""

import io
import base64
import struct
import numpy as np
from typing import Optional, Tuple, List
from pydub import AudioSegment
from pydub.utils import make_chunks
import webrtcvad
import asyncio
from app.utils.logging import get_logger

logger = get_logger(__name__)


class AudioProcessor:
    """Handles audio format conversions and processing."""
    
    @staticmethod
    def ulaw_to_pcm(ulaw_data: bytes) -> bytes:
        """
        Convert μ-law encoded audio to PCM.
        
        Args:
            ulaw_data: μ-law encoded audio bytes
        
        Returns:
            PCM audio bytes (16-bit signed)
        """
        # μ-law to linear PCM conversion table
        ULAW_TABLE = [
            -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
            -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
            -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
            -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
            -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
            -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
            -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
            -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
            -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
            -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
            -876, -844, -812, -780, -748, -716, -684, -652,
            -620, -588, -556, -524, -492, -460, -428, -396,
            -372, -356, -340, -324, -308, -292, -276, -260,
            -244, -228, -212, -196, -180, -164, -148, -132,
            -120, -112, -104, -96, -88, -80, -72, -64,
            -56, -48, -40, -32, -24, -16, -8, 0,
            32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
            23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
            15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
            11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
            7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
            5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
            3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
            2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
            1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
            1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
            876, 844, 812, 780, 748, 716, 684, 652,
            620, 588, 556, 524, 492, 460, 428, 396,
            372, 356, 340, 324, 308, 292, 276, 260,
            244, 228, 212, 196, 180, 164, 148, 132,
            120, 112, 104, 96, 88, 80, 72, 64,
            56, 48, 40, 32, 24, 16, 8, 0
        ]
        
        pcm_data = bytearray()
        for byte in ulaw_data:
            # Convert μ-law byte to 16-bit PCM
            pcm_value = ULAW_TABLE[byte]
            # Pack as 16-bit signed integer (little-endian)
            pcm_data.extend(struct.pack('<h', pcm_value))
        
        return bytes(pcm_data)
    
    @staticmethod
    def pcm_to_ulaw(pcm_data: bytes) -> bytes:
        """
        Convert PCM audio to μ-law encoding.
        
        Args:
            pcm_data: PCM audio bytes (16-bit signed)
        
        Returns:
            μ-law encoded audio bytes
        """
        # PCM to μ-law conversion
        def linear_to_ulaw(sample):
            # Clip sample to 16-bit range
            sample = max(-32768, min(32767, sample))
            
            # Get sign bit
            sign = 0
            if sample < 0:
                sign = 0x80
                sample = -sample
            
            # Add bias
            sample = sample + 0x84
            
            # Find position
            if sample > 0x7FFF:
                sample = 0x7FFF
            
            # Find exponent
            exp = 7
            for i in range(7, 0, -1):
                if sample & (0x4000 >> (7 - i)):
                    exp = i
                    break
            
            # Extract mantissa
            mantissa = (sample >> (exp + 3)) & 0x0F
            
            # Combine and invert
            ulaw = ~(sign | (exp << 4) | mantissa) & 0xFF
            
            return ulaw
        
        ulaw_data = bytearray()
        
        # Process PCM data as 16-bit samples
        for i in range(0, len(pcm_data), 2):
            if i + 1 < len(pcm_data):
                # Unpack 16-bit signed integer (little-endian)
                sample = struct.unpack('<h', pcm_data[i:i+2])[0]
                ulaw_byte = linear_to_ulaw(sample)
                ulaw_data.append(ulaw_byte)
        
        return bytes(ulaw_data)
    
    @staticmethod
    def resample_audio(
        audio_data: bytes,
        input_rate: int,
        output_rate: int,
        input_format: str = "pcm"
    ) -> bytes:
        """
        Resample audio to a different sample rate.
        
        Args:
            audio_data: Input audio bytes
            input_rate: Input sample rate in Hz
            output_rate: Output sample rate in Hz
            input_format: Format of input audio ("pcm", "ulaw", "mp3", etc.)
        
        Returns:
            Resampled audio bytes in PCM format
        """
        try:
            # Convert to AudioSegment
            if input_format == "pcm":
                audio = AudioSegment(
                    audio_data,
                    frame_rate=input_rate,
                    sample_width=2,  # 16-bit
                    channels=1
                )
            elif input_format == "ulaw":
                # First convert to PCM
                pcm_data = AudioProcessor.ulaw_to_pcm(audio_data)
                audio = AudioSegment(
                    pcm_data,
                    frame_rate=input_rate,
                    sample_width=2,
                    channels=1
                )
            else:
                # Try to load directly
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format=input_format)
            
            # Resample
            audio = audio.set_frame_rate(output_rate)
            
            # Convert back to raw PCM
            return audio.raw_data
            
        except Exception as e:
            logger.error(f"Audio resampling failed: {e}")
            return audio_data
    
    @staticmethod
    def convert_audio_format(
        audio_data: bytes,
        input_format: str,
        output_format: str,
        sample_rate: int = 8000
    ) -> bytes:
        """
        Convert audio between different formats.
        
        Args:
            audio_data: Input audio bytes
            input_format: Input format (e.g., "ulaw", "pcm", "mp3")
            output_format: Output format
            sample_rate: Sample rate for the audio
        
        Returns:
            Converted audio bytes
        """
        try:
            # Handle direct conversions
            if input_format == "ulaw" and output_format == "pcm":
                return AudioProcessor.ulaw_to_pcm(audio_data)
            elif input_format == "pcm" and output_format == "ulaw":
                return AudioProcessor.pcm_to_ulaw(audio_data)
            
            # Use pydub for other conversions
            if input_format == "pcm":
                audio = AudioSegment(
                    audio_data,
                    frame_rate=sample_rate,
                    sample_width=2,
                    channels=1
                )
            else:
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format=input_format)
            
            # Export to desired format
            output_buffer = io.BytesIO()
            audio.export(output_buffer, format=output_format)
            return output_buffer.getvalue()
            
        except Exception as e:
            logger.error(f"Audio format conversion failed: {e}")
            return audio_data
    
    @staticmethod
    def encode_base64(audio_data: bytes) -> str:
        """Encode audio data to base64 string."""
        return base64.b64encode(audio_data).decode('utf-8')
    
    @staticmethod
    def decode_base64(base64_string: str) -> bytes:
        """Decode base64 string to audio data."""
        return base64.b64decode(base64_string)


class VoiceActivityDetector:
    """Voice Activity Detection for audio streams."""
    
    def __init__(self, aggressiveness: int = 2, frame_duration_ms: int = 30):
        """
        Initialize VAD.
        
        Args:
            aggressiveness: VAD aggressiveness (0-3)
            frame_duration_ms: Frame duration in milliseconds (10, 20, or 30)
        """
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = frame_duration_ms
        self.sample_rate = 8000  # VAD works with 8kHz, 16kHz, 32kHz, or 48kHz
        
    def is_speech(self, audio_frame: bytes) -> bool:
        """
        Check if audio frame contains speech.
        
        Args:
            audio_frame: Audio frame bytes (PCM 16-bit)
        
        Returns:
            True if speech is detected
        """
        try:
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False
    
    def process_audio_stream(
        self,
        audio_data: bytes,
        padding_duration_ms: int = 300
    ) -> List[bytes]:
        """
        Process audio stream and extract speech segments.
        
        Args:
            audio_data: Continuous audio data
            padding_duration_ms: Padding to add around speech segments
        
        Returns:
            List of audio segments containing speech
        """
        # Convert to AudioSegment for easier manipulation
        audio = AudioSegment(
            audio_data,
            frame_rate=self.sample_rate,
            sample_width=2,
            channels=1
        )
        
        # Split into frames
        frames = make_chunks(audio, self.frame_duration_ms)
        
        # Detect speech segments
        segments = []
        current_segment = []
        num_padding_frames = padding_duration_ms // self.frame_duration_ms
        ring_buffer = []
        triggered = False
        
        for frame in frames:
            is_speech = self.is_speech(frame.raw_data)
            
            if not triggered:
                ring_buffer.append(frame)
                if len(ring_buffer) > num_padding_frames:
                    ring_buffer.pop(0)
                
                if is_speech:
                    triggered = True
                    current_segment.extend(ring_buffer)
                    ring_buffer = []
            else:
                current_segment.append(frame)
                ring_buffer.append(frame)
                if len(ring_buffer) > num_padding_frames:
                    ring_buffer.pop(0)
                
                if not is_speech:
                    if len([f for f in ring_buffer if self.is_speech(f.raw_data)]) == 0:
                        # End of speech segment
                        triggered = False
                        if current_segment:
                            combined = sum(current_segment[:-num_padding_frames])
                            if len(combined) > 0:
                                segments.append(combined.raw_data)
                        current_segment = []
                        ring_buffer = []
        
        # Add any remaining segment
        if current_segment:
            combined = sum(current_segment)
            if len(combined) > 0:
                segments.append(combined.raw_data)
        
        return segments


class AudioBuffer:
    """Thread-safe audio buffer for streaming."""
    
    def __init__(self, max_size: int = 1024 * 1024):  # 1MB default
        """
        Initialize audio buffer.
        
        Args:
            max_size: Maximum buffer size in bytes
        """
        self.buffer = bytearray()
        self.max_size = max_size
        self.lock = asyncio.Lock()
    
    async def write(self, data: bytes) -> bool:
        """
        Write data to buffer.
        
        Args:
            data: Audio data to write
        
        Returns:
            True if successful, False if buffer is full
        """
        async with self.lock:
            if len(self.buffer) + len(data) > self.max_size:
                logger.warning("Audio buffer overflow")
                return False
            
            self.buffer.extend(data)
            return True
    
    async def read(self, size: Optional[int] = None) -> bytes:
        """
        Read data from buffer.
        
        Args:
            size: Number of bytes to read (all if None)
        
        Returns:
            Audio data bytes
        """
        async with self.lock:
            if size is None or size >= len(self.buffer):
                data = bytes(self.buffer)
                self.buffer.clear()
            else:
                data = bytes(self.buffer[:size])
                self.buffer = self.buffer[size:]
            
            return data
    
    async def clear(self):
        """Clear the buffer."""
        async with self.lock:
            self.buffer.clear()
    
    async def size(self) -> int:
        """Get current buffer size."""
        async with self.lock:
            return len(self.buffer)
