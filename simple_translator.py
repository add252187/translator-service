"""
Simple Real-time Translator for Twilio Calls
Simplified version for testing and demo
"""

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import wave
import io
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client as TwilioClient
import json
import base64
import asyncio
import aiohttp
from typing import Optional
import os
from dotenv import load_dotenv
import deepl
import struct

# Load environment variables
load_dotenv()

app = FastAPI()

# Mount static files if directory exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_TTS_API_KEY")
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL")

# Twilio client
twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# DeepL translator
deepl_translator = deepl.Translator(DEEPL_API_KEY)

# Active sessions and global logs
active_sessions = {}
global_logs = []
recordings = []
settings = {"translation_enabled": True}

# WebSocket connections for browser clients
browser_clients: List[WebSocket] = []

# Directory for recordings
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

class TranslationSession:
    def __init__(self, call_sid: str, websocket: WebSocket):
        self.call_sid = call_sid
        self.websocket = websocket
        self.stream_sid = None
        self.audio_buffer = bytearray()
        self.is_active = True
        self.deepgram_session = None
        self.sequence_number = 0
        self.processing_task = None
        self.logs = []
        self.detected_language = None
        self.silence_counter = 0
        self.translation_enabled = settings["translation_enabled"]
        # Recording buffers
        self.agent_audio_raw = bytearray()  # Original audio from agent (PC)
        self.client_audio_raw = bytearray()  # Original audio from client (phone)
        self.start_time = None
        
    async def start_processing(self):
        """Start continuous audio processing"""
        self.processing_task = asyncio.create_task(self.process_audio_continuous())
    
    async def process_audio_continuous(self):
        """Process audio with voice activity detection for natural conversation"""
        sentence_buffer = bytearray()
        silence_duration = 0
        last_had_speech = False
        
        while self.is_active:
            try:
                # Wait for audio to accumulate
                await asyncio.sleep(0.1)
                
                if len(self.audio_buffer) == 0:
                    silence_duration += 0.1
                    # If we have accumulated speech and silence > 1 second, process
                    if len(sentence_buffer) > 16000 and silence_duration > 1.0:  # 1 sec of audio min
                        audio_chunk = bytes(sentence_buffer)
                        sentence_buffer.clear()
                        silence_duration = 0
                        await self.process_audio_chunk(audio_chunk)
                    continue
                
                # Take all available audio
                audio_chunk = bytes(self.audio_buffer)
                self.audio_buffer.clear()
                
                # Check for speech activity (simple energy-based VAD)
                has_speech = self.detect_speech(audio_chunk)
                
                if has_speech:
                    sentence_buffer.extend(audio_chunk)
                    silence_duration = 0
                    last_had_speech = True
                else:
                    silence_duration += len(audio_chunk) / 16000  # 8kHz * 2 bytes
                    
                    # If we had speech and now silence > 0.8 sec, process the sentence
                    if last_had_speech and len(sentence_buffer) > 8000 and silence_duration > 0.8:
                        audio_to_process = bytes(sentence_buffer)
                        sentence_buffer.clear()
                        silence_duration = 0
                        last_had_speech = False
                        await self.process_audio_chunk(audio_to_process)
                    elif len(sentence_buffer) > 0:
                        # Add some silence to buffer
                        sentence_buffer.extend(audio_chunk)
                
                # Force process if buffer gets too large (5 seconds)
                if len(sentence_buffer) > 80000:
                    audio_to_process = bytes(sentence_buffer)
                    sentence_buffer.clear()
                    await self.process_audio_chunk(audio_to_process)
                    
            except Exception as e:
                print(f"Processing error: {e}")
                await asyncio.sleep(0.1)
    
    def detect_speech(self, audio_data: bytes) -> bool:
        """Simple energy-based voice activity detection"""
        if len(audio_data) < 100:
            return False
        
        # Calculate RMS energy
        samples = []
        for i in range(0, len(audio_data) - 1, 2):
            sample = int.from_bytes(audio_data[i:i+2], 'little', signed=True)
            samples.append(sample * sample)
        
        if not samples:
            return False
            
        rms = (sum(samples) / len(samples)) ** 0.5
        
        # Threshold for speech detection (adjust as needed)
        return rms > 500
    
    async def add_audio(self, audio_data: bytes):
        """Add audio to buffer"""
        self.audio_buffer.extend(audio_data)
        
    async def process_audio_chunk(self, audio_data: bytes):
        """Process a chunk of audio with auto language detection and TTS"""
        try:
            # Only process if translation is enabled
            if not settings["translation_enabled"]:
                return
                
            # Only process if we have enough audio
            if len(audio_data) < 800:  # Skip very small chunks
                return
                
            print(f"Processing {len(audio_data)} bytes for translation")
            
            # 1. Send to Deepgram for STT with language detection
            text, detected_lang = await self.transcribe_with_deepgram_detect(audio_data)
            
            if text and len(text.strip()) > 0:
                # Update detected language if found
                if detected_lang and not self.detected_language:
                    self.detected_language = detected_lang
                    self.add_log(f"🌐 Idioma detectado: {detected_lang}")
                
                self.add_log(f"📝 Cliente ({detected_lang or '?'}): {text}")
                print(f"Transcribed ({detected_lang}): {text}")
                
                # 2. Determine translation direction
                # If Spanish detected, translate to English; otherwise translate to Spanish
                source_lang = detected_lang or self.detected_language or "en"
                target_lang = "EN" if source_lang.lower().startswith("es") else "ES"
                
                # 3. Translate with DeepL
                translated = await self.translate_text(text, target_lang=target_lang)
                self.add_log(f"🔄 → {target_lang}: {translated}")
                print(f"Translated to {target_lang}: {translated}")
                
                # 4. Generate TTS with ElevenLabs and send to browser (agent)
                tts_lang = "es" if target_lang == "ES" else "en"
                tts_audio = await self.generate_tts_for_browser(translated, language=tts_lang)
                
                if tts_audio:
                    # Send translated audio to browser for agent to hear
                    await send_audio_to_browser(tts_audio)
                    self.add_log(f"🔊 Audio enviado al agente")
                        
        except Exception as e:
            print(f"Error processing audio: {e}")
            self.add_log(f"❌ Error: {str(e)}")
    
    def add_log(self, message: str):
        """Add log message to both session and global logs"""
        log_entry = {
            "time": asyncio.get_event_loop().time(),
            "message": message,
            "call_sid": self.call_sid
        }
        self.logs.append(log_entry)
        global_logs.append(log_entry)
        
        # Keep only last 50 logs
        if len(self.logs) > 50:
            self.logs = self.logs[-50:]
        if len(global_logs) > 100:
            global_logs[:] = global_logs[-100:]
    
    async def send_text_to_client(self, text: str):
        """Send text message to web client (not implemented yet)"""
        # This would send to a web interface
        pass
    
    async def transcribe_with_deepgram_detect(self, audio_data: bytes) -> tuple[Optional[str], Optional[str]]:
        """Transcribe audio using Deepgram with language detection"""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.deepgram.com/v1/listen"
                headers = {
                    "Authorization": f"Token {DEEPGRAM_API_KEY}",
                    "Content-Type": "audio/raw"
                }
                params = {
                    "model": "nova-2",
                    "punctuate": "true",
                    "detect_language": "true",  # Enable language detection
                    "language": "multi",  # Support multiple languages
                    "encoding": "linear16",
                    "sample_rate": "8000",
                    "channels": "1"
                }
                
                async with session.post(url, headers=headers, params=params, data=audio_data) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("results", {}).get("channels", []):
                            channel = result["results"]["channels"][0]
                            detected_language = channel.get("detected_language", None)
                            if channel.get("alternatives", []):
                                transcript = channel["alternatives"][0].get("transcript", "")
                                return transcript, detected_language
        except Exception as e:
            print(f"Deepgram error: {e}")
        return None, None
    
    async def transcribe_with_deepgram(self, audio_data: bytes) -> Optional[str]:
        """Legacy method for compatibility"""
        text, _ = await self.transcribe_with_deepgram_detect(audio_data)
        return text
    
    async def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text using DeepL"""
        try:
            result = deepl_translator.translate_text(text, target_lang=target_lang)
            return result.text
        except Exception as e:
            print(f"DeepL error: {e}")
            return text
    
    async def generate_speech(self, text: str, language: str) -> Optional[bytes]:
        """Generate speech using ElevenLabs - returns MP3"""
        try:
            async with aiohttp.ClientSession() as session:
                # Use different voices for different languages
                voice_id = "EXAVITQu4vr4xnSDxMaL" if language == "es" else "21m00Tcm4TlvDq8ikWAM"
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                headers = {
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json"
                }
                data = {
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                }
                
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        print(f"ElevenLabs error: {response.status}")
        except Exception as e:
            print(f"ElevenLabs error: {e}")
        return None
    
    async def generate_tts_for_browser(self, text: str, language: str) -> Optional[bytes]:
        """Generate TTS and convert to PCM for browser playback"""
        try:
            # Get MP3 from ElevenLabs
            mp3_audio = await self.generate_speech(text, language)
            
            if mp3_audio:
                # Convert MP3 to PCM using pydub
                from pydub import AudioSegment
                
                audio = AudioSegment.from_mp3(io.BytesIO(mp3_audio))
                # Convert to 8kHz mono 16-bit PCM
                audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
                
                # Get raw PCM data
                pcm_data = audio.raw_data
                print(f"Generated {len(pcm_data)} bytes of PCM audio for browser")
                return pcm_data
                
        except Exception as e:
            print(f"TTS for browser error: {e}")
        return None
    
    async def generate_tts_for_twilio(self, text: str, language: str) -> Optional[bytes]:
        """Generate TTS and convert to μ-law for Twilio"""
        try:
            # Get MP3 from ElevenLabs
            mp3_audio = await self.generate_speech(text, language)
            
            if mp3_audio:
                # Convert MP3 to μ-law using pydub
                from pydub import AudioSegment
                
                audio = AudioSegment.from_mp3(io.BytesIO(mp3_audio))
                # Convert to 8kHz mono
                audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
                
                # Get raw PCM and convert to μ-law
                pcm_data = audio.raw_data
                ulaw_data = pcm_to_ulaw(pcm_data)
                print(f"Generated {len(ulaw_data)} bytes of μ-law audio for Twilio")
                return ulaw_data
                
        except Exception as e:
            print(f"TTS for Twilio error: {e}")
        return None
    
    async def send_audio_to_twilio(self, audio_data: bytes):
        """Send audio back to Twilio through WebSocket"""
        try:
            # Split audio into chunks
            chunk_size = 640  # 20ms at 8kHz μ-law
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i+chunk_size]
                
                # Create media message
                media_message = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "sequenceNumber": str(self.sequence_number),
                    "media": {
                        "payload": base64.b64encode(chunk).decode('utf-8')
                    }
                }
                
                await self.websocket.send_text(json.dumps(media_message))
                self.sequence_number += 1
                
                # Add mark for synchronization
                if self.sequence_number % 50 == 0:
                    mark_message = {
                        "event": "mark",
                        "streamSid": self.stream_sid,
                        "mark": {
                            "name": f"mark_{self.sequence_number}"
                        }
                    }
                    await self.websocket.send_text(json.dumps(mark_message))
                    
        except Exception as e:
            print(f"Error sending audio to Twilio: {e}")

@app.get("/")
async def root():
    """Serve the static HTML file"""
    html_file = os.path.join(static_dir, "index.html")
    if os.path.exists(html_file):
        return FileResponse(html_file)
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Translator Control Panel</title>
        <style>
            body { font-family: Arial; padding: 20px; background: #1a1a2e; color: white; }
            .container { max-width: 600px; margin: 0 auto; }
            h1 { color: #00d9ff; }
            button { 
                background: linear-gradient(90deg, #00d9ff, #00ff88);
                border: none;
                padding: 15px 30px;
                border-radius: 8px;
                color: #1a1a2e;
                font-weight: bold;
                cursor: pointer;
                font-size: 18px;
                margin: 10px;
            }
            button:hover { transform: translateY(-2px); }
            .status { 
                background: rgba(255,255,255,0.1);
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }
            #logs {
                background: #000;
                padding: 15px;
                border-radius: 8px;
                height: 300px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌐 Real-time Translator</h1>
            <div class="status">
                <h2>🚀 Quick Start:</h2>
                <ol>
                    <li>Click "Call My Phone" to start a call</li>
                    <li>Answer the call on your phone</li>
                    <li>Press any key when you hear the Twilio message</li>
                    <li>Speak in ANY language - automatic detection & translation!</li>
                </ol>
                <p style="color: #00ff88; margin-top: 10px;">
                    ✨ Supports: English, Spanish, French, German, Italian, Portuguese, Dutch, Polish, Russian, Japanese, Chinese, Arabic and more!
                </p>
            </div>
            <button onclick="makeCall()">📞 Call My Phone</button>
            <h3 style="margin-top: 20px;">📝 Live Transcription & Translation:</h3>
            <div id="logs"></div>
        </div>
        <script>
            let pollInterval = null;
            
            function addLog(msg) {
                const logs = document.getElementById('logs');
                const entry = document.createElement('div');
                entry.innerHTML = `[${new Date().toLocaleTimeString()}] ${msg}`;
                logs.insertBefore(entry, logs.firstChild);
                
                // Keep only last 30 entries
                while (logs.children.length > 30) {
                    logs.removeChild(logs.lastChild);
                }
            }
            
            async function fetchLogs() {
                try {
                    const response = await fetch('/logs');
                    const data = await response.json();
                    
                    // Clear and update logs
                    const logs = document.getElementById('logs');
                    logs.innerHTML = '';
                    
                    data.logs.forEach(log => {
                        const entry = document.createElement('div');
                        entry.innerHTML = log.message;
                        entry.style.marginBottom = '5px';
                        if (log.message.includes('📝')) entry.style.color = '#00ff88';
                        if (log.message.includes('🔄')) entry.style.color = '#00d9ff';
                        if (log.message.includes('❌')) entry.style.color = '#ff4444';
                        logs.appendChild(entry);
                    });
                } catch (error) {
                    console.error('Error fetching logs:', error);
                }
            }
            
            async function makeCall() {
                addLog('Initiating call...');
                try {
                    const response = await fetch('/call/outbound?to_number=%2B34651351636', {
                        method: 'POST'
                    });
                    const data = await response.json();
                    addLog(`Call initiated: ${data.call_sid}`);
                    
                    // Start polling for logs
                    if (!pollInterval) {
                        pollInterval = setInterval(fetchLogs, 500);
                    }
                } catch (error) {
                    addLog(`Error: ${error.message}`);
                }
            }
            
            // Stop polling when page is hidden
            document.addEventListener('visibilitychange', () => {
                if (document.hidden && pollInterval) {
                    clearInterval(pollInterval);
                    pollInterval = null;
                }
            });
            
            addLog('System ready');
        </script>
    </body>
    </html>
    """)

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "active_sessions": len(active_sessions)}

@app.post("/call/outbound")
async def make_outbound_call(to_number: str = "+34651351636"):
    """Initiate an outbound call"""
    try:
        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{WEBHOOK_BASE_URL}/voice/webhook"
        )
        return {"success": True, "call_sid": call.sid}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/voice/webhook")
async def voice_webhook(request: Request):
    """Twilio voice webhook"""
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        
        # Create TwiML response
        response = VoiceResponse()
        response.say("Connecting to real-time translator. Please wait.", voice="alice", language="en-US")
        
        # Connect to WebSocket
        connect = Connect()
        stream = Stream(url=f"wss://{WEBHOOK_BASE_URL.replace('http://', '').replace('https://', '')}/ws/media-stream")
        connect.append(stream)
        response.append(connect)
        
        return PlainTextResponse(str(response), media_type="application/xml")
        
    except Exception as e:
        print(f"Webhook error: {e}")
        response = VoiceResponse()
        response.say("Error connecting call.", voice="alice")
        response.hangup()
        return PlainTextResponse(str(response), media_type="application/xml")

@app.websocket("/ws/media-stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for Twilio Media Streams"""
    await websocket.accept()
    session = None
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data["event"] == "start":
                # Initialize session
                call_sid = data["start"]["callSid"]
                stream_sid = data["start"]["streamSid"]
                session = TranslationSession(call_sid, websocket)
                session.stream_sid = stream_sid
                active_sessions[call_sid] = session
                print(f"Session started: {call_sid}")
                session.add_log(f"📞 Call started: {call_sid}")
                # Start continuous processing
                await session.start_processing()
                
            elif data["event"] == "media" and session:
                # Add incoming audio to buffer
                payload = data["media"]["payload"]
                audio_data = base64.b64decode(payload)
                
                # Convert μ-law to PCM and add to buffer
                pcm_data = ulaw_to_pcm(audio_data)
                
                # Store raw audio for recording (client side)
                session.client_audio_raw.extend(pcm_data)
                
                # If translation is OFF, send directly to browser
                if not settings["translation_enabled"]:
                    await send_audio_to_browser(pcm_data)
                
                # Add to processing buffer (for translation when enabled)
                await session.add_audio(pcm_data)
                
            elif data["event"] == "stop":
                print(f"Session stopped: {session.call_sid if session else 'unknown'}")
                if session:
                    session.is_active = False
                    session.add_log(f"📴 Call ended")
                    # Save recording before removing session
                    await save_recording(session)
                    if session.call_sid in active_sessions:
                        del active_sessions[session.call_sid]
                break
                
    except Exception as e:
        print(f"WebSocket error: {e}")
        if session:
            session.is_active = False
    finally:
        await websocket.close()

@app.get("/logs")
async def get_logs():
    """Get global logs"""
    return {"logs": [{"message": log["message"]} for log in global_logs[-50:]]}  # Return last 50 logs

@app.post("/process-audio")
async def process_audio_from_browser(request: Request):
    """Process audio from browser microphone"""
    try:
        data = await request.json()
        audio_base64 = data.get("audio")
        
        if not audio_base64:
            return {"error": "No audio data provided"}
        
        # Decode base64 audio
        audio_data = base64.b64decode(audio_base64)
        
        # Create a temporary session for processing
        temp_session = TranslationSession("browser", None)
        
        # Process the audio
        text, lang = await temp_session.transcribe_with_deepgram_detect(audio_data)
        
        if text:
            # Log transcription
            global_logs.append({
                "message": f"📝 Browser ({lang or 'unknown'}): {text}",
                "call_sid": "browser"
            })
            
            # Translate only if enabled
            if settings["translation_enabled"]:
                source_lang = lang or "en"
                target_lang = "EN" if source_lang.lower().startswith("es") else "ES"
                translated = await temp_session.translate_text(text, target_lang=target_lang)
                
                # Log translation
                global_logs.append({
                    "message": f"🔄 Translated to {target_lang}: {translated}",
                    "call_sid": "browser"
                })
            else:
                translated = text
                target_lang = lang
            
            # Send to active call if exists
            if active_sessions:
                for session in active_sessions.values():
                    # Generate TTS and send to call
                    audio_response = await session.generate_speech(translated, 
                                                                  language=target_lang.lower() if target_lang else "es")
                    if audio_response:
                        await session.send_audio_to_twilio(audio_response)
            
            return {
                "success": True,
                "transcription": text,
                "language": lang,
                "translation": translated,
                "target_language": target_lang
            }
        
        return {"success": False, "error": "No speech detected"}
        
    except Exception as e:
        print(f"Error processing browser audio: {e}")
        return {"success": False, "error": str(e)}

@app.post("/settings")
async def update_settings(request: Request):
    """Update global settings"""
    try:
        data = await request.json()
        if "translation_enabled" in data:
            settings["translation_enabled"] = data["translation_enabled"]
            # Update all active sessions
            for session in active_sessions.values():
                session.translation_enabled = settings["translation_enabled"]
        return {"success": True, "settings": settings}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/audio/from-pc")
async def receive_audio_from_pc(request: Request):
    """Receive audio from PC microphone and send to phone"""
    try:
        data = await request.json()
        audio_base64 = data.get("audio")
        call_sid = data.get("call_sid")
        translate = data.get("translate", True)
        
        if not audio_base64:
            return {"error": "No audio"}
        
        audio_data = base64.b64decode(audio_base64)
        
        # Find the session
        session = None
        for sid, s in active_sessions.items():
            if s.call_sid == call_sid or call_sid is None:
                session = s
                break
        
        if not session:
            return {"error": "No active call"}
        
        # Store raw audio for recording
        session.agent_audio_raw.extend(audio_data)
        
        if translate and settings["translation_enabled"]:
            # Transcribe
            text, lang = await session.transcribe_with_deepgram_detect(audio_data)
            
            if text and len(text.strip()) > 0:
                global_logs.append({
                    "message": f"📝 PC ({lang or 'unknown'}): {text}",
                    "call_sid": call_sid
                })
                
                # Translate
                source_lang = lang or "en"
                target_lang = "ES" if not source_lang.lower().startswith("es") else "EN"
                translated = await session.translate_text(text, target_lang=target_lang)
                
                global_logs.append({
                    "message": f"🔄 → {target_lang}: {translated}",
                    "call_sid": call_sid
                })
                
                # Generate TTS and send
                tts_audio = await session.generate_speech(translated, language=target_lang.lower())
                if tts_audio and len(tts_audio) > 0:
                    await session.send_audio_to_twilio(tts_audio)
        else:
            # Direct passthrough - convert and send without translation
            # For now, just log
            global_logs.append({
                "message": f"🎤 PC Audio (passthrough): {len(audio_data)} bytes",
                "call_sid": call_sid
            })
        
        return {"success": True}
        
    except Exception as e:
        print(f"Error receiving PC audio: {e}")
        return {"success": False, "error": str(e)}

@app.post("/call/hangup")
async def hangup_call(call_sid: str = None):
    """Hangup a call"""
    try:
        if call_sid:
            # End the Twilio call
            twilio_client.calls(call_sid).update(status="completed")
            
            # Save recordings if we have audio
            session = active_sessions.get(call_sid)
            if session:
                await save_recording(session)
                session.is_active = False
                del active_sessions[call_sid]
            
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/recordings")
async def get_recordings():
    """Get list of recordings"""
    return {"recordings": recordings[-20:]}  # Last 20 recordings

@app.get("/recordings/{filename}")
async def serve_recording(filename: str):
    """Serve a recording file"""
    file_path = os.path.join(RECORDINGS_DIR, filename)
    if os.path.exists(file_path):
        media_type = "audio/wav" if filename.endswith(".wav") else "audio/raw"
        return FileResponse(file_path, media_type=media_type, filename=filename)
    return {"error": "Recording not found"}

@app.websocket("/ws/browser")
async def browser_websocket(websocket: WebSocket):
    """WebSocket for browser audio streaming"""
    await websocket.accept()
    browser_clients.append(websocket)
    print(f"Browser client connected. Total: {len(browser_clients)}")
    
    # Buffer for accumulating audio for translation with VAD
    agent_sentence_buffer = bytearray()
    silence_duration = 0
    last_had_speech = False
    
    try:
        while True:
            # Receive audio from browser
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "audio":
                # Audio from PC microphone
                audio_base64 = data.get("data")
                if audio_base64:
                    audio_data = base64.b64decode(audio_base64)
                    
                    # Find active session
                    for session in active_sessions.values():
                        if session.is_active and session.websocket:
                            # Store for recording
                            session.agent_audio_raw.extend(audio_data)
                            
                            if settings["translation_enabled"]:
                                # Check for speech
                                has_speech = detect_speech_simple(audio_data)
                                
                                if has_speech:
                                    agent_sentence_buffer.extend(audio_data)
                                    silence_duration = 0
                                    last_had_speech = True
                                else:
                                    silence_duration += len(audio_data) / 16000
                                    
                                    # Process when silence detected after speech
                                    if last_had_speech and len(agent_sentence_buffer) > 8000 and silence_duration > 0.7:
                                        audio_to_process = bytes(agent_sentence_buffer)
                                        agent_sentence_buffer.clear()
                                        silence_duration = 0
                                        last_had_speech = False
                                        
                                        # Process in background
                                        asyncio.create_task(
                                            process_agent_audio_for_translation(session, audio_to_process)
                                        )
                                    elif len(agent_sentence_buffer) > 0:
                                        agent_sentence_buffer.extend(audio_data)
                                
                                # Force process if buffer too large
                                if len(agent_sentence_buffer) > 80000:
                                    audio_to_process = bytes(agent_sentence_buffer)
                                    agent_sentence_buffer.clear()
                                    asyncio.create_task(
                                        process_agent_audio_for_translation(session, audio_to_process)
                                    )
                            else:
                                # Direct passthrough - convert to μ-law and send
                                ulaw_audio = pcm_to_ulaw(audio_data)
                                await send_raw_audio_to_twilio(session, ulaw_audio)
                            
    except Exception as e:
        print(f"Browser WebSocket error: {e}")
    finally:
        if websocket in browser_clients:
            browser_clients.remove(websocket)
        print(f"Browser client disconnected. Total: {len(browser_clients)}")

def detect_speech_simple(audio_data: bytes) -> bool:
    """Simple energy-based voice activity detection"""
    if len(audio_data) < 100:
        return False
    
    samples = []
    for i in range(0, len(audio_data) - 1, 2):
        sample = int.from_bytes(audio_data[i:i+2], 'little', signed=True)
        samples.append(sample * sample)
    
    if not samples:
        return False
        
    rms = (sum(samples) / len(samples)) ** 0.5
    return rms > 500

async def process_agent_audio_for_translation(session, audio_data: bytes):
    """Process agent audio: STT -> Translate -> TTS -> Send to client"""
    try:
        # 1. Transcribe
        text, detected_lang = await session.transcribe_with_deepgram_detect(audio_data)
        
        if text and len(text.strip()) > 0:
            # Log
            global_logs.append({
                "message": f"📝 Agente ({detected_lang or '?'}): {text}",
                "call_sid": session.call_sid
            })
            print(f"Agent said ({detected_lang}): {text}")
            
            # 2. Translate (agent usually speaks the target language for the client)
            source_lang = detected_lang or "en"
            target_lang = "ES" if not source_lang.lower().startswith("es") else "EN"
            
            translated = await session.translate_text(text, target_lang=target_lang)
            
            global_logs.append({
                "message": f"🔄 → Cliente ({target_lang}): {translated}",
                "call_sid": session.call_sid
            })
            print(f"Translated for client: {translated}")
            
            # 3. Generate TTS for Twilio (client)
            tts_lang = "es" if target_lang == "ES" else "en"
            tts_audio = await session.generate_tts_for_twilio(translated, language=tts_lang)
            
            if tts_audio:
                # Send to client via Twilio
                await send_raw_audio_to_twilio(session, tts_audio)
                global_logs.append({
                    "message": f"🔊 Audio enviado al cliente",
                    "call_sid": session.call_sid
                })
                
    except Exception as e:
        print(f"Error processing agent audio: {e}")

async def send_raw_audio_to_twilio(session, ulaw_audio: bytes):
    """Send raw μ-law audio to Twilio"""
    if not session.websocket or not session.stream_sid:
        return
    
    try:
        # Split into chunks (160 bytes = 20ms at 8kHz μ-law)
        chunk_size = 160
        for i in range(0, len(ulaw_audio), chunk_size):
            chunk = ulaw_audio[i:i+chunk_size]
            
            media_message = {
                "event": "media",
                "streamSid": session.stream_sid,
                "media": {
                    "payload": base64.b64encode(chunk).decode('utf-8')
                }
            }
            
            await session.websocket.send_text(json.dumps(media_message))
            
    except Exception as e:
        print(f"Error sending audio to Twilio: {e}")

async def save_recording(session):
    """Save recording to WAV files"""
    import time
    try:
        timestamp = int(time.time())
        call_id = session.call_sid[-8:] if session.call_sid else "unknown"
        
        # Save agent audio as WAV
        if len(session.agent_audio_raw) > 1000:
            agent_file = os.path.join(RECORDINGS_DIR, f"{call_id}_agent_{timestamp}.wav")
            save_pcm_as_wav(bytes(session.agent_audio_raw), agent_file, sample_rate=8000)
            print(f"Saved agent recording: {agent_file} ({len(session.agent_audio_raw)} bytes)")
        
        # Save client audio as WAV
        if len(session.client_audio_raw) > 1000:
            client_file = os.path.join(RECORDINGS_DIR, f"{call_id}_client_{timestamp}.wav")
            save_pcm_as_wav(bytes(session.client_audio_raw), client_file, sample_rate=8000)
            print(f"Saved client recording: {client_file} ({len(session.client_audio_raw)} bytes)")
        
        # Calculate duration
        duration = max(
            len(session.client_audio_raw) / (8000 * 2),
            len(session.agent_audio_raw) / (8000 * 2)
        )
        
        if duration > 1:  # Only add if more than 1 second
            recordings.append({
                "call_sid": call_id,
                "timestamp": timestamp,
                "duration": int(duration),
                "agent_url": f"/recordings/{call_id}_agent_{timestamp}.wav" if len(session.agent_audio_raw) > 1000 else None,
                "client_url": f"/recordings/{call_id}_client_{timestamp}.wav" if len(session.client_audio_raw) > 1000 else None
            })
            
            global_logs.append({
                "message": f"💾 Grabación guardada: {int(duration)}s",
                "call_sid": session.call_sid
            })
    except Exception as e:
        print(f"Error saving recording: {e}")
        import traceback
        traceback.print_exc()

def save_pcm_as_wav(pcm_data: bytes, filename: str, sample_rate: int = 8000):
    """Save PCM data as WAV file"""
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

def pcm_to_ulaw(pcm_data: bytes) -> bytes:
    """Convert PCM 16-bit to μ-law"""
    BIAS = 0x84
    CLIP = 32635
    
    ulaw_data = bytearray()
    
    for i in range(0, len(pcm_data), 2):
        if i + 1 < len(pcm_data):
            sample = struct.unpack('<h', pcm_data[i:i+2])[0]
        else:
            break
            
        # Get sign and magnitude
        sign = (sample >> 8) & 0x80
        if sign:
            sample = -sample
        
        # Clip
        if sample > CLIP:
            sample = CLIP
        
        # Add bias
        sample += BIAS
        
        # Find segment
        exponent = 7
        exp_mask = 0x4000
        for _ in range(8):
            if sample & exp_mask:
                break
            exponent -= 1
            exp_mask >>= 1
        
        # Get mantissa
        mantissa = (sample >> (exponent + 3)) & 0x0F
        
        # Combine
        ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
        ulaw_data.append(ulaw_byte)
    
    return bytes(ulaw_data)

async def send_audio_to_browser(audio_data: bytes):
    """Send audio to all connected browser clients"""
    if not browser_clients:
        return
    
    # Convert to base64 for sending
    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
    message = json.dumps({
        "type": "audio",
        "data": audio_base64
    })
    
    # Send to all connected clients
    disconnected = []
    for client in browser_clients:
        try:
            await client.send_text(message)
        except:
            disconnected.append(client)
    
    # Remove disconnected clients
    for client in disconnected:
        if client in browser_clients:
            browser_clients.remove(client)

def ulaw_to_pcm(ulaw_data: bytes) -> bytes:
    """Convert μ-law to PCM"""
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
        pcm_value = ULAW_TABLE[byte]
        pcm_data.extend(struct.pack('<h', pcm_value))
    return bytes(pcm_data)

if __name__ == "__main__":
    import uvicorn
    print("Starting Simple Translator Server...")
    print(f"Webhook URL: {WEBHOOK_BASE_URL}")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)
