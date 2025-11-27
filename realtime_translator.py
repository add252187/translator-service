"""
Real-time Bidirectional Translator with ElevenLabs
Optimized for fluent conversation between agent and client
"""

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import wave
import io
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client as TwilioClient
import json
import base64
import asyncio
import aiohttp
import os
from dotenv import load_dotenv
import deepl
import struct
from collections import deque

load_dotenv()

app = FastAPI()

# Mount static files
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

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
deepl_translator = deepl.Translator(DEEPL_API_KEY)

# Global state
active_calls = {}
browser_clients: List[WebSocket] = []
global_logs = []
recordings = []
settings = {"translation_enabled": True, "agent_lang": "en", "client_lang": "es"}

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)


class RealtimeCall:
    """Manages a real-time translated call"""
    
    def __init__(self, call_sid: str, twilio_ws: WebSocket):
        self.call_sid = call_sid
        self.twilio_ws = twilio_ws
        self.stream_sid = None
        self.is_active = True
        
        # Audio buffers for recording
        self.agent_audio = bytearray()
        self.client_audio = bytearray()
        
        # Transcription buffers - accumulate until sentence complete
        self.client_text_buffer = ""
        self.agent_text_buffer = ""
        
        # Audio accumulation for STT
        self.client_audio_buffer = bytearray()
        self.agent_audio_buffer = bytearray()
        
        # Processing locks to prevent overlapping TTS
        self.client_processing = False
        self.agent_processing = False
        
        # Last activity timestamps
        self.last_client_audio = 0
        self.last_agent_audio = 0
        
    async def process_client_audio(self, pcm_audio: bytes):
        """Process audio from client (phone) - translate and send to agent (browser)"""
        if not settings["translation_enabled"]:
            # Direct passthrough
            await send_to_browser(pcm_audio)
            return
            
        # Store for recording
        self.client_audio.extend(pcm_audio)
        
        # Accumulate audio
        self.client_audio_buffer.extend(pcm_audio)
        self.last_client_audio = asyncio.get_event_loop().time()
        
        # Process when we have ~2 seconds of audio
        if len(self.client_audio_buffer) >= 32000:  # 2 sec at 8kHz 16-bit
            await self._process_client_chunk()
    
    async def check_client_silence(self):
        """Check if client stopped speaking and process remaining audio"""
        if len(self.client_audio_buffer) > 8000:  # At least 0.5 sec
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_client_audio > 1.0:  # 1 sec silence
                await self._process_client_chunk()
    
    async def _process_client_chunk(self):
        """Process accumulated client audio"""
        if self.client_processing or len(self.client_audio_buffer) < 4000:
            return
            
        self.client_processing = True
        audio_data = bytes(self.client_audio_buffer)
        self.client_audio_buffer.clear()
        
        try:
            # 1. Transcribe
            text, lang = await transcribe_audio(audio_data)
            
            if text and len(text.strip()) > 2:
                add_log(f"📝 Cliente: {text}")
                
                # 2. Translate to agent's language
                target = settings["agent_lang"].upper()
                if target == "EN" and lang and lang.lower().startswith("en"):
                    translated = text  # No need to translate
                else:
                    translated = await translate_text(text, target)
                
                add_log(f"🔄 → Agente: {translated}")
                
                # 3. Generate TTS and send to browser
                tts_audio = await generate_tts_pcm(translated, settings["agent_lang"])
                if tts_audio:
                    await send_to_browser(tts_audio)
                    add_log(f"🔊 Audio enviado al agente")
                    
        except Exception as e:
            print(f"Client processing error: {e}")
        finally:
            self.client_processing = False
    
    async def process_agent_audio(self, pcm_audio: bytes):
        """Process audio from agent (browser) - translate and send to client (phone)"""
        if not settings["translation_enabled"]:
            # Direct passthrough
            ulaw = pcm_to_ulaw(pcm_audio)
            await self.send_to_twilio(ulaw)
            return
            
        # Store for recording
        self.agent_audio.extend(pcm_audio)
        
        # Accumulate audio
        self.agent_audio_buffer.extend(pcm_audio)
        self.last_agent_audio = asyncio.get_event_loop().time()
        
        # Process when we have ~2 seconds of audio
        if len(self.agent_audio_buffer) >= 32000:
            await self._process_agent_chunk()
    
    async def check_agent_silence(self):
        """Check if agent stopped speaking and process remaining audio"""
        if len(self.agent_audio_buffer) > 8000:
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_agent_audio > 1.0:
                await self._process_agent_chunk()
    
    async def _process_agent_chunk(self):
        """Process accumulated agent audio"""
        if self.agent_processing or len(self.agent_audio_buffer) < 4000:
            return
            
        self.agent_processing = True
        audio_data = bytes(self.agent_audio_buffer)
        self.agent_audio_buffer.clear()
        
        try:
            # 1. Transcribe
            text, lang = await transcribe_audio(audio_data)
            
            if text and len(text.strip()) > 2:
                add_log(f"📝 Agente: {text}")
                
                # 2. Translate to client's language
                target = settings["client_lang"].upper()
                if target == "ES" and lang and lang.lower().startswith("es"):
                    translated = text
                else:
                    translated = await translate_text(text, target)
                
                add_log(f"🔄 → Cliente: {translated}")
                
                # 3. Generate TTS and send to phone
                tts_audio = await generate_tts_ulaw(translated, settings["client_lang"])
                if tts_audio:
                    await self.send_to_twilio(tts_audio)
                    add_log(f"🔊 Audio enviado al cliente")
                    
        except Exception as e:
            print(f"Agent processing error: {e}")
        finally:
            self.agent_processing = False
    
    async def send_to_twilio(self, ulaw_audio: bytes):
        """Send μ-law audio to Twilio"""
        if not self.twilio_ws or not self.stream_sid:
            return
            
        try:
            # Send in 20ms chunks (160 bytes)
            for i in range(0, len(ulaw_audio), 160):
                chunk = ulaw_audio[i:i+160]
                msg = {
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(chunk).decode()}
                }
                await self.twilio_ws.send_text(json.dumps(msg))
                await asyncio.sleep(0.015)  # Pace the audio
        except Exception as e:
            print(f"Twilio send error: {e}")


# === Audio Processing Functions ===

async def transcribe_audio(audio_data: bytes) -> tuple[Optional[str], Optional[str]]:
    """Transcribe audio using Deepgram"""
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
                "detect_language": "true",
                "encoding": "linear16",
                "sample_rate": "8000",
                "channels": "1"
            }
            
            async with session.post(url, headers=headers, params=params, data=audio_data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    channels = result.get("results", {}).get("channels", [])
                    if channels:
                        channel = channels[0]
                        lang = channel.get("detected_language")
                        alts = channel.get("alternatives", [])
                        if alts:
                            return alts[0].get("transcript", ""), lang
                else:
                    print(f"Deepgram error: {resp.status}")
    except Exception as e:
        print(f"Transcribe error: {e}")
    return None, None


async def translate_text(text: str, target_lang: str) -> str:
    """Translate text using DeepL"""
    try:
        result = deepl_translator.translate_text(text, target_lang=target_lang)
        return result.text
    except Exception as e:
        print(f"Translate error: {e}")
        return text


async def generate_tts_pcm(text: str, language: str) -> Optional[bytes]:
    """Generate TTS as PCM for browser"""
    try:
        mp3_audio = await _call_elevenlabs(text, language)
        if mp3_audio:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_audio))
            audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
            return audio.raw_data
    except Exception as e:
        print(f"TTS PCM error: {e}")
    return None


async def generate_tts_ulaw(text: str, language: str) -> Optional[bytes]:
    """Generate TTS as μ-law for Twilio"""
    try:
        mp3_audio = await _call_elevenlabs(text, language)
        if mp3_audio:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_audio))
            audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
            return pcm_to_ulaw(audio.raw_data)
    except Exception as e:
        print(f"TTS ulaw error: {e}")
    return None


async def _call_elevenlabs(text: str, language: str) -> Optional[bytes]:
    """Call ElevenLabs API"""
    try:
        async with aiohttp.ClientSession() as session:
            # Sarah for Spanish, Rachel for English
            voice_id = "EXAVITQu4vr4xnSDxMaL" if language == "es" else "21m00Tcm4TlvDq8ikWAM"
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json"
            }
            data = {
                "text": text,
                "model_id": "eleven_turbo_v2",  # Faster model
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }
            
            async with session.post(url, headers=headers, json=data) as resp:
                if resp.status == 200:
                    return await resp.read()
                print(f"ElevenLabs error: {resp.status}")
    except Exception as e:
        print(f"ElevenLabs error: {e}")
    return None


# === Audio Conversion ===

def pcm_to_ulaw(pcm_data: bytes) -> bytes:
    """Convert PCM 16-bit to μ-law"""
    BIAS = 0x84
    CLIP = 32635
    
    ulaw = bytearray()
    for i in range(0, len(pcm_data) - 1, 2):
        sample = struct.unpack('<h', pcm_data[i:i+2])[0]
        
        sign = (sample >> 8) & 0x80
        if sign:
            sample = -sample
        if sample > CLIP:
            sample = CLIP
        
        sample += BIAS
        
        exponent = 7
        for exp_mask in [0x4000, 0x2000, 0x1000, 0x800, 0x400, 0x200, 0x100, 0x80]:
            if sample & exp_mask:
                break
            exponent -= 1
        
        mantissa = (sample >> (exponent + 3)) & 0x0F
        ulaw.append(~(sign | (exponent << 4) | mantissa) & 0xFF)
    
    return bytes(ulaw)


def ulaw_to_pcm(ulaw_data: bytes) -> bytes:
    """Convert μ-law to PCM 16-bit"""
    ULAW_TABLE = [
        -32124,-31100,-30076,-29052,-28028,-27004,-25980,-24956,
        -23932,-22908,-21884,-20860,-19836,-18812,-17788,-16764,
        -15996,-15484,-14972,-14460,-13948,-13436,-12924,-12412,
        -11900,-11388,-10876,-10364,-9852,-9340,-8828,-8316,
        -7932,-7676,-7420,-7164,-6908,-6652,-6396,-6140,
        -5884,-5628,-5372,-5116,-4860,-4604,-4348,-4092,
        -3900,-3772,-3644,-3516,-3388,-3260,-3132,-3004,
        -2876,-2748,-2620,-2492,-2364,-2236,-2108,-1980,
        -1884,-1820,-1756,-1692,-1628,-1564,-1500,-1436,
        -1372,-1308,-1244,-1180,-1116,-1052,-988,-924,
        -876,-844,-812,-780,-748,-716,-684,-652,
        -620,-588,-556,-524,-492,-460,-428,-396,
        -372,-356,-340,-324,-308,-292,-276,-260,
        -244,-228,-212,-196,-180,-164,-148,-132,
        -120,-112,-104,-96,-88,-80,-72,-64,
        -56,-48,-40,-32,-24,-16,-8,0,
        32124,31100,30076,29052,28028,27004,25980,24956,
        23932,22908,21884,20860,19836,18812,17788,16764,
        15996,15484,14972,14460,13948,13436,12924,12412,
        11900,11388,10876,10364,9852,9340,8828,8316,
        7932,7676,7420,7164,6908,6652,6396,6140,
        5884,5628,5372,5116,4860,4604,4348,4092,
        3900,3772,3644,3516,3388,3260,3132,3004,
        2876,2748,2620,2492,2364,2236,2108,1980,
        1884,1820,1756,1692,1628,1564,1500,1436,
        1372,1308,1244,1180,1116,1052,988,924,
        876,844,812,780,748,716,684,652,
        620,588,556,524,492,460,428,396,
        372,356,340,324,308,292,276,260,
        244,228,212,196,180,164,148,132,
        120,112,104,96,88,80,72,64,
        56,48,40,32,24,16,8,0
    ]
    
    pcm = bytearray()
    for byte in ulaw_data:
        sample = ULAW_TABLE[byte]
        pcm.extend(struct.pack('<h', sample))
    return bytes(pcm)


# === Browser Communication ===

async def send_to_browser(pcm_audio: bytes):
    """Send audio to all browser clients"""
    if not browser_clients:
        return
    
    msg = json.dumps({
        "type": "audio",
        "data": base64.b64encode(pcm_audio).decode()
    })
    
    dead = []
    for client in browser_clients:
        try:
            await client.send_text(msg)
        except:
            dead.append(client)
    
    for client in dead:
        if client in browser_clients:
            browser_clients.remove(client)


def add_log(message: str):
    """Add log entry"""
    global_logs.append({"message": message})
    if len(global_logs) > 100:
        global_logs[:] = global_logs[-100:]
    print(message)


# === API Endpoints ===

@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post("/call/outbound")
async def make_call(to_number: str = "+34651351636"):
    """Initiate outbound call"""
    try:
        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{WEBHOOK_BASE_URL}/voice/webhook",
            status_callback=f"{WEBHOOK_BASE_URL}/call/status",
            status_callback_event=["completed"]
        )
        add_log(f"📞 Llamada iniciada: {call.sid}")
        return {"success": True, "call_sid": call.sid}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/voice/webhook")
async def voice_webhook(request: Request):
    """Twilio webhook for call setup"""
    response = VoiceResponse()
    
    connect = Connect()
    stream = Stream(url=f"wss://{WEBHOOK_BASE_URL.replace('https://', '')}/ws/twilio")
    connect.append(stream)
    response.append(connect)
    
    return PlainTextResponse(str(response), media_type="application/xml")


@app.websocket("/ws/twilio")
async def twilio_websocket(websocket: WebSocket):
    """WebSocket for Twilio media stream"""
    await websocket.accept()
    call: Optional[RealtimeCall] = None
    
    # Background task to check for silence
    async def silence_checker():
        while call and call.is_active:
            await asyncio.sleep(0.5)
            await call.check_client_silence()
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data["event"] == "start":
                stream_sid = data["start"]["streamSid"]
                call_sid = data["start"]["callSid"]
                
                call = RealtimeCall(call_sid, websocket)
                call.stream_sid = stream_sid
                active_calls[call_sid] = call
                
                add_log(f"📞 Llamada conectada: {call_sid[-8:]}")
                asyncio.create_task(silence_checker())
                
            elif data["event"] == "media" and call:
                payload = data["media"]["payload"]
                ulaw_audio = base64.b64decode(payload)
                pcm_audio = ulaw_to_pcm(ulaw_audio)
                
                # Process client audio
                await call.process_client_audio(pcm_audio)
                
            elif data["event"] == "stop":
                if call:
                    call.is_active = False
                    await save_recording(call)
                    if call.call_sid in active_calls:
                        del active_calls[call.call_sid]
                    add_log(f"📴 Llamada terminada")
                break
                
    except Exception as e:
        print(f"Twilio WS error: {e}")
        if call:
            call.is_active = False
    finally:
        await websocket.close()


@app.websocket("/ws/browser")
async def browser_websocket(websocket: WebSocket):
    """WebSocket for browser audio"""
    await websocket.accept()
    browser_clients.append(websocket)
    add_log("🖥️ Navegador conectado")
    
    # Buffer for agent audio
    agent_buffer = bytearray()
    last_audio_time = 0
    
    async def silence_checker():
        nonlocal agent_buffer, last_audio_time
        while websocket in browser_clients:
            await asyncio.sleep(0.5)
            if len(agent_buffer) > 8000:
                current = asyncio.get_event_loop().time()
                if current - last_audio_time > 1.0:
                    # Process accumulated audio
                    for call in active_calls.values():
                        if call.is_active:
                            audio = bytes(agent_buffer)
                            agent_buffer.clear()
                            await call.process_agent_audio(audio)
    
    asyncio.create_task(silence_checker())
    
    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            if data.get("type") == "audio":
                audio_b64 = data.get("data")
                if audio_b64:
                    pcm_audio = base64.b64decode(audio_b64)
                    last_audio_time = asyncio.get_event_loop().time()
                    
                    # Find active call
                    for call in active_calls.values():
                        if call.is_active:
                            if settings["translation_enabled"]:
                                # Accumulate for translation
                                agent_buffer.extend(pcm_audio)
                                
                                # Process if buffer large enough
                                if len(agent_buffer) >= 32000:
                                    audio = bytes(agent_buffer)
                                    agent_buffer.clear()
                                    await call.process_agent_audio(audio)
                            else:
                                # Direct passthrough
                                ulaw = pcm_to_ulaw(pcm_audio)
                                await call.send_to_twilio(ulaw)
                            
    except Exception as e:
        print(f"Browser WS error: {e}")
    finally:
        if websocket in browser_clients:
            browser_clients.remove(websocket)
        add_log("🖥️ Navegador desconectado")


@app.get("/logs")
async def get_logs():
    return {"logs": global_logs[-50:]}


@app.post("/settings")
async def update_settings(request: Request):
    data = await request.json()
    settings.update(data)
    add_log(f"⚙️ Configuración actualizada: {data}")
    return {"success": True, "settings": settings}


@app.get("/recordings")
async def get_recordings():
    return {"recordings": recordings[-20:]}


@app.get("/recordings/{filename}")
async def serve_recording(filename: str):
    path = os.path.join(RECORDINGS_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="audio/wav", filename=filename)
    return {"error": "Not found"}


async def save_recording(call: RealtimeCall):
    """Save call recordings as WAV"""
    import time
    try:
        ts = int(time.time())
        call_id = call.call_sid[-8:]
        
        if len(call.agent_audio) > 1000:
            path = os.path.join(RECORDINGS_DIR, f"{call_id}_agent_{ts}.wav")
            with wave.open(path, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(8000)
                f.writeframes(bytes(call.agent_audio))
        
        if len(call.client_audio) > 1000:
            path = os.path.join(RECORDINGS_DIR, f"{call_id}_client_{ts}.wav")
            with wave.open(path, 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(8000)
                f.writeframes(bytes(call.client_audio))
        
        duration = max(len(call.agent_audio), len(call.client_audio)) / 16000
        if duration > 1:
            recordings.append({
                "call_sid": call_id,
                "timestamp": ts,
                "duration": int(duration),
                "agent_url": f"/recordings/{call_id}_agent_{ts}.wav",
                "client_url": f"/recordings/{call_id}_client_{ts}.wav"
            })
            add_log(f"💾 Grabación guardada: {int(duration)}s")
            
    except Exception as e:
        print(f"Save recording error: {e}")


if __name__ == "__main__":
    import uvicorn
    print("Starting Realtime Translator...")
    print(f"Webhook URL: {WEBHOOK_BASE_URL}")
    print("Open http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
