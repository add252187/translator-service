"""
Streaming Translator con Google Cloud TTS (GDPR compliant)
Versión completa - Igual que ElevenLabs pero con Google TTS
"""

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client as TwilioClient
from google.cloud import texttospeech
import json
import base64
import asyncio
import websockets
import os
import struct
from dotenv import load_dotenv
import deepl
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

# Thread pool for blocking TTS calls
tts_executor = ThreadPoolExecutor(max_workers=4)

load_dotenv()

# Set Google credentials
GOOGLE_CREDS_FILE = os.path.join(os.path.dirname(__file__), "traductorbidireccional-a4b169f37aef.json")
if os.path.exists(GOOGLE_CREDS_FILE):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS_FILE

app = FastAPI()

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Config
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "https://fcf414cb0e41.ngrok-free.app")

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
deepl_translator = deepl.Translator(DEEPL_API_KEY)

# Google TTS client
google_tts_client = texttospeech.TextToSpeechClient()

# State
active_calls = {}
browser_clients: List[WebSocket] = []
logs = []
settings = {"translation_enabled": True, "agent_lang": "en", "client_lang": "es"}

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)


def log(msg: str):
    """Add log entry"""
    logs.append({"message": msg})
    if len(logs) > 100:
        logs[:] = logs[-100:]
    print(msg)


def synthesize_google_tts_sync(text: str, lang: str = "es") -> bytes:
    """Generate TTS audio using Google Cloud (sync)"""
    lang_code = "es-ES" if lang == "es" else "en-US"
    
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=8000
    )
    
    response = google_tts_client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    return response.audio_content  # PCM 16-bit 8kHz


async def synthesize_google_tts(text: str, lang: str = "es") -> bytes:
    """Generate TTS audio using Google Cloud (async)"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(tts_executor, synthesize_google_tts_sync, text, lang)


class StreamingCall:
    """Real-time streaming call with Deepgram WebSocket"""
    
    def __init__(self, call_sid: str, twilio_ws: WebSocket):
        self.call_sid = call_sid
        self.twilio_ws = twilio_ws
        self.stream_sid = None
        self.active = True
        
        self.client_deepgram_ws = None
        self.agent_deepgram_ws = None
        
        self.client_audio = bytearray()
        self.agent_audio = bytearray()
    
    async def start_deepgram_client(self):
        """Start Deepgram streaming for client audio"""
        try:
            url = "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=8000&channels=1&punctuate=true&interim_results=false&endpointing=500&model=nova-2&language=multi"
            
            log("🔄 Conectando Deepgram cliente...")
            self.client_deepgram_ws = await websockets.connect(
                url,
                extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
            )
            
            asyncio.create_task(self._receive_client_transcripts())
            log("🎙️ Deepgram cliente conectado")
            
        except Exception as e:
            log(f"❌ Error Deepgram cliente: {e}")
    
    async def start_deepgram_agent(self):
        """Start Deepgram streaming for agent audio with retries"""
        url = "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=8000&channels=1&punctuate=true&interim_results=false&endpointing=500&model=nova-2&language=multi"
        
        for attempt in range(3):
            try:
                log("🔄 Conectando Deepgram agente...")
                self.agent_deepgram_ws = await asyncio.wait_for(
                    websockets.connect(
                        url,
                        extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
                    ),
                    timeout=10
                )
                
                asyncio.create_task(self._receive_agent_transcripts())
                log("🎙️ Deepgram agente conectado")
                return
                
            except Exception as e:
                log(f"⚠️ Intento {attempt+1}/3 Deepgram agente: {e}")
                if attempt < 2:
                    await asyncio.sleep(1)
        
        log("❌ No se pudo conectar Deepgram agente")
    
    async def _receive_client_transcripts(self):
        """Receive and process client transcripts"""
        try:
            async for message in self.client_deepgram_ws:
                if not self.active:
                    break
                
                data = json.loads(message)
                
                if data.get("type") == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    
                    if alternatives:
                        transcript = alternatives[0].get("transcript", "").strip()
                        is_final = data.get("is_final", False)
                        
                        if transcript and is_final:
                            asyncio.create_task(
                                self._translate_and_speak_to_agent(transcript)
                            )
                                    
        except Exception as e:
            if self.active:
                log(f"❌ Error transcripción cliente: {e}")
    
    async def _receive_agent_transcripts(self):
        """Receive and process agent transcripts"""
        try:
            async for message in self.agent_deepgram_ws:
                if not self.active:
                    break
                
                data = json.loads(message)
                
                if data.get("type") == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    
                    if alternatives:
                        transcript = alternatives[0].get("transcript", "").strip()
                        is_final = data.get("is_final", False)
                        
                        if transcript and is_final:
                            asyncio.create_task(
                                self._translate_and_speak_to_client(transcript)
                            )
                                    
        except Exception as e:
            if self.active:
                log(f"❌ Error transcripción agente: {e}")
    
    async def _translate_and_speak_to_agent(self, text: str):
        """Translate client text and send TTS to agent"""
        try:
            target = settings["agent_lang"].upper()
            if target == "EN":
                target = "EN-US"
            translated = await translate_fast(text, target)
            
            log(f"📞 Cliente: {text}")
            if text != translated:
                log(f"🔊 → Agente escucha: {translated}")
            
            await generate_and_send_to_browser(translated, settings["agent_lang"])
            
        except Exception as e:
            log(f"❌ Error TTS agente: {e}")
    
    async def _translate_and_speak_to_client(self, text: str):
        """Translate agent text and send TTS to client"""
        try:
            target = settings["client_lang"].upper()
            translated = await translate_fast(text, target)
            
            log(f"🎤 Agente: {text}")
            if text != translated:
                log(f"🔊 → Cliente escucha: {translated}")
            
            await self.generate_and_send_to_twilio(translated, settings["client_lang"])
            
        except Exception as e:
            log(f"❌ Error TTS cliente: {e}")
    
    async def send_client_audio(self, pcm_audio: bytes):
        """Send client audio to Deepgram"""
        self.client_audio.extend(pcm_audio)
        
        if not settings["translation_enabled"]:
            await send_to_browser(pcm_audio)
            return
        
        if self.client_deepgram_ws:
            try:
                await self.client_deepgram_ws.send(pcm_audio)
            except Exception as e:
                log(f"❌ Error Deepgram cliente: {e}")
    
    async def send_agent_audio(self, pcm_audio: bytes):
        """Send agent audio to Deepgram"""
        self.agent_audio.extend(pcm_audio)
        
        if not settings["translation_enabled"]:
            ulaw = pcm_to_ulaw(pcm_audio)
            await self.send_to_twilio(ulaw)
            return
        
        if self.agent_deepgram_ws:
            try:
                await self.agent_deepgram_ws.send(pcm_audio)
            except Exception as e:
                log(f"❌ Error Deepgram agente: {e}")
    
    async def generate_and_send_to_twilio(self, text: str, lang: str):
        """Generate Google TTS and send to Twilio"""
        try:
            pcm_data = await synthesize_google_tts(text, lang)
            ulaw_data = pcm_to_ulaw(pcm_data)
            await self.send_to_twilio(ulaw_data)
            log("🔊 Audio enviado al cliente (Google TTS)")
        except Exception as e:
            log(f"❌ Google TTS Twilio error: {e}")
    
    async def send_to_twilio(self, ulaw_audio: bytes):
        """Send audio to Twilio"""
        if not self.twilio_ws or not self.stream_sid:
            return
        
        try:
            for i in range(0, len(ulaw_audio), 160):
                chunk = ulaw_audio[i:i+160]
                if len(chunk) > 0:
                    msg = {
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": base64.b64encode(chunk).decode()}
                    }
                    await self.twilio_ws.send_text(json.dumps(msg))
        except Exception as e:
            pass
    
    async def close(self):
        """Close all connections"""
        self.active = False
        
        if self.client_deepgram_ws:
            try:
                await self.client_deepgram_ws.close()
            except:
                pass
        
        if self.agent_deepgram_ws:
            try:
                await self.agent_deepgram_ws.close()
            except:
                pass


def translate_sync(text: str, target: str) -> str:
    """Sync translation using DeepL"""
    result = deepl_translator.translate_text(text, target_lang=target)
    return result.text


async def translate_fast(text: str, target: str) -> str:
    """Fast async translation using DeepL"""
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(tts_executor, translate_sync, text, target)
    except:
        return text


async def generate_and_send_to_browser(text: str, lang: str):
    """Generate Google TTS and send to browser"""
    try:
        pcm_data = await synthesize_google_tts(text, lang)
        
        # 8kHz mono 16-bit = 16000 bytes/sec
        # Send 1600 bytes = 100ms of audio, wait 100ms
        chunk_size = 1600  # 100ms of audio at 8kHz
        for i in range(0, len(pcm_data), chunk_size):
            await send_to_browser(pcm_data[i:i+chunk_size])
            await asyncio.sleep(0.1)  # Wait 100ms to match audio duration
        
        log("🔊 Audio enviado al agente (Google TTS)")
    except Exception as e:
        log(f"❌ Google TTS browser error: {e}")


async def send_to_browser(pcm_audio: bytes):
    """Send audio to browser clients"""
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
    
    for c in dead:
        if c in browser_clients:
            browser_clients.remove(c)


def pcm_to_ulaw(pcm: bytes) -> bytes:
    """Convert PCM to μ-law"""
    BIAS = 0x84
    CLIP = 32635
    
    ulaw = bytearray()
    for i in range(0, len(pcm) - 1, 2):
        sample = struct.unpack('<h', pcm[i:i+2])[0]
        
        sign = (sample >> 8) & 0x80
        if sign:
            sample = -sample
        if sample > CLIP:
            sample = CLIP
        
        sample += BIAS
        
        exp = 7
        for mask in [0x4000, 0x2000, 0x1000, 0x800, 0x400, 0x200, 0x100, 0x80]:
            if sample & mask:
                break
            exp -= 1
        
        mantissa = (sample >> (exp + 3)) & 0x0F
        ulaw.append(~(sign | (exp << 4) | mantissa) & 0xFF)
    
    return bytes(ulaw)


def ulaw_to_pcm(ulaw: bytes) -> bytes:
    """Convert μ-law to PCM"""
    TABLE = [-32124,-31100,-30076,-29052,-28028,-27004,-25980,-24956,-23932,-22908,-21884,-20860,-19836,-18812,-17788,-16764,-15996,-15484,-14972,-14460,-13948,-13436,-12924,-12412,-11900,-11388,-10876,-10364,-9852,-9340,-8828,-8316,-7932,-7676,-7420,-7164,-6908,-6652,-6396,-6140,-5884,-5628,-5372,-5116,-4860,-4604,-4348,-4092,-3900,-3772,-3644,-3516,-3388,-3260,-3132,-3004,-2876,-2748,-2620,-2492,-2364,-2236,-2108,-1980,-1884,-1820,-1756,-1692,-1628,-1564,-1500,-1436,-1372,-1308,-1244,-1180,-1116,-1052,-988,-924,-876,-844,-812,-780,-748,-716,-684,-652,-620,-588,-556,-524,-492,-460,-428,-396,-372,-356,-340,-324,-308,-292,-276,-260,-244,-228,-212,-196,-180,-164,-148,-132,-120,-112,-104,-96,-88,-80,-72,-64,-56,-48,-40,-32,-24,-16,-8,0,32124,31100,30076,29052,28028,27004,25980,24956,23932,22908,21884,20860,19836,18812,17788,16764,15996,15484,14972,14460,13948,13436,12924,12412,11900,11388,10876,10364,9852,9340,8828,8316,7932,7676,7420,7164,6908,6652,6396,6140,5884,5628,5372,5116,4860,4604,4348,4092,3900,3772,3644,3516,3388,3260,3132,3004,2876,2748,2620,2492,2364,2236,2108,1980,1884,1820,1756,1692,1628,1564,1500,1436,1372,1308,1244,1180,1116,1052,988,924,876,844,812,780,748,716,684,652,620,588,556,524,492,460,428,396,372,356,340,324,308,292,276,260,244,228,212,196,180,164,148,132,120,112,104,96,88,80,72,64,56,48,40,32,24,16,8,0]
    
    pcm = bytearray()
    for b in ulaw:
        pcm.extend(struct.pack('<h', TABLE[b]))
    return bytes(pcm)


# === API Endpoints ===

@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post("/call/outbound")
async def make_call(to_number: str = "+34651351636"):
    try:
        log(f"📞 Iniciando llamada a {to_number}...")
        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{WEBHOOK_BASE_URL}/voice/webhook",
            status_callback=f"{WEBHOOK_BASE_URL}/call/status",
            status_callback_event=["completed"]
        )
        log(f"📞 Llamada iniciada: {call.sid[-8:]}")
        return {"success": True, "call_sid": call.sid}
    except Exception as e:
        log(f"❌ Error Twilio: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


from fastapi import Request

@app.api_route("/voice/webhook", methods=["GET", "POST"])
async def voice_webhook(request: Request):
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url="wss://fcf414cb0e41.ngrok-free.app/ws/twilio")
    connect.append(stream)
    response.append(connect)
    return PlainTextResponse(str(response), media_type="application/xml")


@app.post("/call/status")
async def call_status():
    return {"status": "ok"}


@app.websocket("/ws/twilio")
async def twilio_ws(websocket: WebSocket):
    await websocket.accept()
    call: Optional[StreamingCall] = None
    
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            
            if data["event"] == "start":
                call = StreamingCall(data["start"]["callSid"], websocket)
                call.stream_sid = data["start"]["streamSid"]
                active_calls[call.call_sid] = call
                
                await call.start_deepgram_client()
                log(f"📞 Llamada conectada")
                
            elif data["event"] == "media" and call:
                ulaw = base64.b64decode(data["media"]["payload"])
                pcm = ulaw_to_pcm(ulaw)
                await call.send_client_audio(pcm)
                
            elif data["event"] == "stop":
                if call:
                    await call.close()
                    if call.call_sid in active_calls:
                        del active_calls[call.call_sid]
                    log("📴 Llamada terminada")
                break
                
    except Exception as e:
        log(f"❌ Twilio WS error: {e}")
        if call:
            await call.close()


@app.websocket("/ws/browser")
async def browser_ws(websocket: WebSocket):
    await websocket.accept()
    browser_clients.append(websocket)
    log("🖥️ Navegador conectado")
    
    agent_deepgram_started = False
    
    try:
        while True:
            try:
                msg = await websocket.receive_text()
                data = json.loads(msg)
                
                if data.get("type") == "audio":
                    pcm = base64.b64decode(data["data"])
                    
                    for call in active_calls.values():
                        if call.active:
                            if not agent_deepgram_started and settings["translation_enabled"]:
                                await call.start_deepgram_agent()
                                agent_deepgram_started = True
                                log("🎤 Micrófono agente activo")
                            
                            await call.send_agent_audio(pcm)
                            
            except Exception as inner_e:
                if "1000" in str(inner_e) or "1001" in str(inner_e):
                    break
                raise
                        
    except Exception as e:
        pass
    finally:
        if websocket in browser_clients:
            browser_clients.remove(websocket)
        log("🖥️ Navegador desconectado")


@app.get("/logs")
async def get_logs():
    return {"logs": logs[-50:]}


@app.post("/settings")
async def update_settings(request: Request):
    data = await request.json()
    settings.update(data)
    log(f"⚙️ Config: {data}")
    return {"success": True}


@app.get("/recordings")
async def get_recordings():
    return {"recordings": []}


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 STREAMING TRANSLATOR - Google TTS (GDPR)")
    print("="*50)
    print(f"Webhook: {WEBHOOK_BASE_URL}")
    print("Open: http://localhost:8000")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
