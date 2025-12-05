"""
Ultra-Fast Streaming Translator
Uses WebSocket streaming for real-time STT and TTS
"""

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import PlainTextResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client as TwilioClient
import json
import base64
import asyncio
import aiohttp
import websockets
import os
import wave
import struct
from dotenv import load_dotenv
import deepl
from typing import Optional, List
from collections import deque
from google.cloud import texttospeech
from concurrent.futures import ThreadPoolExecutor
from openai import AsyncOpenAI
import time

load_dotenv()

tts_executor = ThreadPoolExecutor(max_workers=4)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client: Optional[AsyncOpenAI] = None

def synthesize_google_tts_sync(text: str, lang: str = "es", sample_rate: int = 8000) -> bytes:
    # Mapeo de códigos de idioma a códigos de Google TTS
    LANG_CODES = {
        "es": "es-ES",
        "en": "en-US",
        "gl": "gl-ES",  # Gallego
        "ca": "ca-ES",  # Catalán
        "eu": "eu-ES",  # Vasco/Euskera
    }
    lang_code = LANG_CODES.get(lang.lower(), "es-ES")
    
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang_code,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate
    )
    google_tts_client = texttospeech.TextToSpeechClient()
    response = google_tts_client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )
    return response.audio_content

async def synthesize_google_tts(text: str, lang: str = "es", sample_rate: int = 8000) -> bytes:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        tts_executor,
        synthesize_google_tts_sync,
        text,
        lang,
        sample_rate,
    )

TTS_ENGINE = os.getenv("TTS_ENGINE", "elevenlabs").lower()

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
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_TTS_API_KEY")
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "https://diana-return-orange-hoped.trycloudflare.com")

twilio_client = TwilioClient(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
deepl_translator = deepl.Translator(DEEPL_API_KEY)

# State
active_calls = {}
browser_clients: List[WebSocket] = []
logs = []
# Idiomas soportados para clientes: gl (gallego), ca (catalán), eu (vasco), es (español)
# El agente siempre habla español
settings = {
    "translation_enabled": True, 
    "agent_lang": "es",  # Agente siempre español
    "client_lang": "gl",  # Cliente: gl, ca, eu, es
    "browser_sample_rate": 16000  # Mayor calidad para navegador
}

# Mapeo de idiomas para DeepL y OpenAI
LANGUAGE_NAMES = {
    "gl": "Galician",
    "ca": "Catalan", 
    "eu": "Basque",
    "es": "Spanish",
    "en": "English"
}

# Idiomas que DeepL NO soporta (usaremos OpenAI)
DEEPL_UNSUPPORTED = ["gl", "ca", "eu"]

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)


def log(msg: str):
    """Add log entry"""
    logs.append({"message": msg})
    if len(logs) > 100:
        logs[:] = logs[-100:]
    print(msg)


class StreamingCall:
    """Real-time streaming call with Deepgram WebSocket"""
    
    def __init__(self, call_sid: str, twilio_ws: WebSocket):
        self.call_sid = call_sid
        self.twilio_ws = twilio_ws
        self.stream_sid = None
        self.active = True
        
        # Deepgram streaming connections
        self.client_deepgram_ws = None
        self.agent_deepgram_ws = None
        self.client_keepalive_task: Optional[asyncio.Task] = None
        self.agent_keepalive_task: Optional[asyncio.Task] = None
        self.client_last_audio = time.time()
        self.agent_last_audio = time.time()
        
        # Audio queues for smooth playback
        self.agent_audio_queue = asyncio.Queue()
        self.client_audio_queue = asyncio.Queue()
        
        # Recording
        self.client_audio = bytearray()
        self.agent_audio = bytearray()
        
        # Pending text for translation
        self.client_pending_text = ""
        self.agent_pending_text = ""
        
        # Last transcript time for debouncing
        self.last_client_transcript = 0
        self.last_agent_transcript = 0
    
    async def start_deepgram_client(self):
        """Start Deepgram streaming for client audio"""
        try:
            # Cliente habla gallego/catalán/vasco - Deepgram detecta automáticamente
            # endpointing=400 - balance entre latencia y frases completas
            # vad_events=true - detecta inicio/fin de voz
            # smart_format=true - mejor formato de números/fechas
            url = "wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=8000&channels=1&punctuate=true&interim_results=false&endpointing=400&vad_events=true&smart_format=true&model=nova-2&language=multi"
            
            log("🔄 Conectando Deepgram cliente...")
            self.client_deepgram_ws = await websockets.connect(
                url,
                extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
            )
            self.client_last_audio = time.time()
            if self.client_keepalive_task:
                self.client_keepalive_task.cancel()
            self.client_keepalive_task = asyncio.create_task(self._deepgram_keepalive("client"))
            await self._send_initial_silence(self.client_deepgram_ws)
            
            # Start receiving transcripts
            asyncio.create_task(self._receive_client_transcripts())
            log("🎙️ Deepgram cliente conectado")
            
        except Exception as e:
            log(f"❌ Error Deepgram cliente: {e}")
            import traceback
            traceback.print_exc()
    
    agent_sample_rate = 48000  # Default, will be updated from browser
    
    async def start_deepgram_agent(self):
        """Start Deepgram streaming for agent audio"""
        try:
            # Agente habla español - usar sample rate del navegador (48kHz típicamente)
            sample_rate = getattr(self, 'agent_sample_rate', 48000)
            url = f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate={sample_rate}&channels=1&punctuate=true&interim_results=false&endpointing=400&vad_events=true&smart_format=true&model=nova-2&language=es"
            
            log(f"🔄 Conectando Deepgram agente (sample_rate={sample_rate})...")
            self.agent_deepgram_ws = await websockets.connect(
                url,
                extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
            )
            self.agent_last_audio = time.time()
            if self.agent_keepalive_task:
                self.agent_keepalive_task.cancel()
            self.agent_keepalive_task = asyncio.create_task(self._deepgram_keepalive("agent", sample_rate))
            await self._send_initial_silence(self.agent_deepgram_ws, sample_rate)
            
            asyncio.create_task(self._receive_agent_transcripts())
            log("🎙️ Deepgram agente conectado - listener iniciado")
            
        except Exception as e:
            log(f"❌ Error Deepgram agente: {e}")
            import traceback
            traceback.print_exc()
    
    async def _receive_client_transcripts(self):
        """Receive and process client transcripts in real-time"""
        try:
            async for message in self.client_deepgram_ws:
                if not self.active:
                    break
                
                data = json.loads(message)
                
                if data.get("type") == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    is_final = data.get("is_final", False)
                    
                    if alternatives and is_final:
                        transcript = alternatives[0].get("transcript", "").strip()
                        
                        # Solo procesar transcripciones finales con contenido
                        if transcript:
                            asyncio.create_task(
                                self._translate_and_speak_to_agent(transcript)
                            )
                                    
        except Exception as e:
            if self.active:
                log(f"❌ Error transcripción cliente: {e}")
                import traceback
                traceback.print_exc()
    
    async def _receive_agent_transcripts(self):
        """Receive and process agent transcripts in real-time"""
        try:
            log("🔍 Esperando transcripciones del agente...")
            async for message in self.agent_deepgram_ws:
                if not self.active:
                    break
                
                data = json.loads(message)
                msg_type = data.get("type", "unknown")
                
                # Debug: log all message types from Deepgram
                if msg_type == "Results":
                    channel = data.get("channel", {})
                    alternatives = channel.get("alternatives", [])
                    is_final = data.get("is_final", False)
                    transcript = alternatives[0].get("transcript", "") if alternatives else ""
                    
                    # Log para debug
                    if transcript.strip():
                        log(f"🔍 DG Agente: '{transcript}' (final={is_final})")
                    
                    if alternatives and is_final and transcript.strip():
                        asyncio.create_task(
                            self._translate_and_speak_to_client(transcript.strip())
                        )
                # Solo loguear eventos importantes, no cada SpeechStarted
                elif msg_type == "UtteranceEnd":
                    log("🔍 DG Agente: Fin de frase")
                                    
        except Exception as e:
            if self.active:
                log(f"❌ Error transcripción agente: {e}")
                import traceback
                traceback.print_exc()
    
    async def _translate_and_speak_to_agent(self, text: str):
        """Translate client text and send TTS to agent"""
        try:
            # Cliente habla gl/ca/eu/es, traducir a español para el agente
            client_lang = settings["client_lang"].lower()
            agent_lang = settings["agent_lang"].lower()  # siempre "es"
            
            # Si cliente habla español, no traducir
            if client_lang == agent_lang:
                translated = text
            else:
                translated = await translate_fast(text, agent_lang, source=client_lang)
            
            # Log con idioma detectado
            lang_name = LANGUAGE_NAMES.get(client_lang, client_lang.upper())
            log(f"📞 Cliente ({lang_name}): {text}")
            log(f"🔊 → Agente escucha (ES): {translated}")
            
            # Generate TTS en español y enviar al navegador
            await generate_and_send_to_browser(translated, "es")  # Siempre español para el agente
            
        except Exception as e:
            log(f"❌ Error TTS agente: {e}")
            import traceback
            traceback.print_exc()
    
    async def _translate_and_speak_to_client(self, text: str):
        """Translate agent text and send TTS to client"""
        try:
            # Agente habla español, traducir al idioma del cliente (gl/ca/eu/es)
            client_lang = settings["client_lang"].lower()
            agent_lang = settings["agent_lang"].lower()  # siempre "es"
            
            log(f"🎤 Agente (ES): {text}")
            
            # Si cliente habla español, no traducir
            if client_lang == agent_lang:
                translated = text
                log(f"🔊 → Cliente (ES): {translated} [sin traducción]")
            else:
                log(f"🔄 Traduciendo ES → {client_lang.upper()}...")
                translated = await translate_fast(text, client_lang, source=agent_lang)
                lang_name = LANGUAGE_NAMES.get(client_lang, client_lang.upper())
                log(f"🔊 → Cliente ({lang_name}): {translated}")
            
            # Generate TTS en idioma del cliente y enviar a Twilio
            await self.generate_and_send_to_twilio(translated, client_lang)
            
        except Exception as e:
            log(f"❌ Error traducción agente: {e}")
            import traceback
            traceback.print_exc()
    
    async def send_client_audio(self, pcm_audio: bytes):
        """Send client audio to Deepgram for streaming STT"""
        self.client_audio.extend(pcm_audio)
        
        if not settings["translation_enabled"]:
            # Direct passthrough to browser (only when translation OFF)
            await send_to_browser(pcm_audio)
            return
        
        # When translation ON, only send to Deepgram (no direct audio)
        if self.client_deepgram_ws:
            try:
                await self.client_deepgram_ws.send(pcm_audio)
                self.client_last_audio = time.time()
            except Exception as e:
                log(f"❌ Error enviando a Deepgram cliente: {e}")
                await self.start_deepgram_client()

    audio_count = 0  # Debug counter
    
    async def send_agent_audio(self, pcm_audio: bytes):
        """Send agent audio to Deepgram for streaming STT"""
        self.agent_audio.extend(pcm_audio)
        
        if not settings["translation_enabled"]:
            # Direct passthrough to Twilio
            ulaw = pcm_to_ulaw(pcm_audio)
            await self.send_to_twilio(ulaw)
            return
        
        # Send to Deepgram for transcription
        if self.agent_deepgram_ws:
            try:
                await self.agent_deepgram_ws.send(pcm_audio)
                self.agent_last_audio = time.time()
                
                # Debug: log cada 100 chunks (~4 segundos) con nivel de audio
                self.audio_count = getattr(self, 'audio_count', 0) + 1
                if self.audio_count == 1:
                    # Calcular nivel de audio (RMS)
                    samples = struct.unpack(f'<{len(pcm_audio)//2}h', pcm_audio)
                    rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
                    log(f"🎤 Primer audio agente: {len(pcm_audio)} bytes, RMS={rms:.0f}")
                elif self.audio_count % 100 == 0:
                    samples = struct.unpack(f'<{len(pcm_audio)//2}h', pcm_audio)
                    rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
                    log(f"🔊 Audio agente: chunk #{self.audio_count}, RMS={rms:.0f}")
            except Exception as e:
                log(f"❌ Error enviando a Deepgram agente: {e}")
                # Try to reconnect
                await self.start_deepgram_agent()
        else:
            # Log si no hay conexion Deepgram
            if not hasattr(self, '_no_dg_warned'):
                log("⚠️ Audio agente recibido pero Deepgram no conectado")
                self._no_dg_warned = True
    
    async def generate_and_send_to_twilio(self, text: str, lang: str):
        """Generate TTS and send to Twilio - ElevenLabs para todo (soporta cualquier idioma)"""
        try:
            # ElevenLabs con modelo turbo para menor latencia
            voice_id = "EXAVITQu4vr4xnSDxMaL"  # Sarah - voz femenina natural
            async with aiohttp.ClientSession() as session:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                headers = {
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg"
                }
                data = {
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",  # Turbo para menor latencia
                    "voice_settings": {
                        "stability": 0.4,  # Menos estabilidad = más rápido
                        "similarity_boost": 0.7
                    }
                }
                async with session.post(url, headers=headers, json=data) as resp:
                    if resp.status == 200:
                        mp3_data = await resp.read()
                        from pydub import AudioSegment
                        import io
                        audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
                        audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
                        pcm_data = audio.raw_data
                        ulaw_data = pcm_to_ulaw(pcm_data)
                        await self.send_to_twilio(ulaw_data)
                        log(f"🔊 Audio enviado al cliente (ElevenLabs, {lang})")
                    else:
                        error_text = await resp.text()
                        log(f"❌ ElevenLabs error: {resp.status} - {error_text}")
        except Exception as e:
            log(f"❌ TTS Twilio error: {e}")
            import traceback
            traceback.print_exc()
    
    async def send_to_twilio(self, ulaw_audio: bytes):
        """Send audio to Twilio"""
        if not self.twilio_ws or not self.stream_sid:
            return
        
        try:
            # Send in small chunks for smooth playback
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
        if self.client_keepalive_task:
            self.client_keepalive_task.cancel()
            self.client_keepalive_task = None
        if self.agent_keepalive_task:
            self.agent_keepalive_task.cancel()
            self.agent_keepalive_task = None
    
    async def _send_initial_silence(self, ws, sample_rate: int = 8000):
        """Send a short silence burst to keep Deepgram stream alive until real audio arrives"""
        try:
            # 100ms of silence at the given sample rate (16-bit mono = 2 bytes per sample)
            num_samples = int(sample_rate * 0.1)  # 100ms
            silence = bytes(num_samples * 2)  # 2 bytes per sample
            await ws.send(silence)
            log(f"Silencio inicial enviado: {len(silence)} bytes @ {sample_rate}Hz")
        except Exception as e:
            log(f"Error enviando silencio inicial: {e}")
    
    async def _deepgram_keepalive(self, kind: str, sample_rate: int = 8000):
        """Periodically send silence to prevent Deepgram timeouts"""
        try:
            while self.active:
                await asyncio.sleep(2)
                ws = self.client_deepgram_ws if kind == "client" else self.agent_deepgram_ws
                if not ws:
                    break
                last_audio = self.client_last_audio if kind == "client" else self.agent_last_audio
                if time.time() - last_audio > 5:
                    try:
                        # Silencio de 100ms al sample rate correcto
                        num_samples = int(sample_rate * 0.1)
                        silence = bytes(num_samples * 2)
                        await ws.send(silence)
                        if kind == "client":
                            self.client_last_audio = time.time()
                        else:
                            self.agent_last_audio = time.time()
                        log(f"🛟 Deepgram keepalive ({kind})")
                    except Exception as e:
                        log(f"⚠️ Keepalive {kind} falló: {e}")
                        if kind == "client":
                            await self.start_deepgram_client()
                        else:
                            await self.start_deepgram_agent()
                        break
        except asyncio.CancelledError:
            pass


async def translate_fast(text: str, target: str, source: str = None) -> str:
    """Fast translation using DeepL or OpenAI (for unsupported languages)"""
    global openai_client
    
    # Normalizar códigos de idioma
    target_base = target.lower().split("-")[0]
    source_base = source.lower().split("-")[0] if source else None
    
    # Si origen o destino es gallego/catalán/vasco, SOLO usar OpenAI (DeepL no soporta)
    needs_openai = target_base in DEEPL_UNSUPPORTED or (source_base and source_base in DEEPL_UNSUPPORTED)
    
    if needs_openai:
        if not OPENAI_API_KEY:
            log(f"⚠️ OpenAI API key no configurada - no se puede traducir {source_base} → {target_base}")
            return text
        
        try:
            if not openai_client:
                openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            
            target_name = LANGUAGE_NAMES.get(target_base, target)
            source_name = LANGUAGE_NAMES.get(source_base, "the detected language") if source_base else "the detected language"
            
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are a professional translator. Translate the following text from {source_name} to {target_name}. Return ONLY the translation, nothing else."},
                    {"role": "user", "content": text}
                ],
                temperature=0.3,
                max_tokens=500
            )
            translated = response.choices[0].message.content.strip()
            log(f"🤖 OpenAI: {source_base or '?'} → {target_base}")
            return translated
        except Exception as e:
            log(f"❌ OpenAI error: {e}")
            return text
    
    # Usar DeepL solo para idiomas soportados (es, en, etc.)
    try:
        deepl_target = target.upper()
        if deepl_target == "EN":
            deepl_target = "EN-US"
        elif deepl_target == "ES":
            deepl_target = "ES"
        
        result = deepl_translator.translate_text(text, target_lang=deepl_target)
        return result.text
    except Exception as e:
        log(f"❌ DeepL error: {e}")
        return text


async def generate_and_send_to_browser(text: str, lang: str):
    """Generate TTS and send to browser (Google o ElevenLabs)"""
    try:
        # Usar sample rate más alto para mejor calidad en navegador
        browser_rate = settings.get("browser_sample_rate", 16000)
        
        if TTS_ENGINE == "google":
            # Generar audio a mayor calidad para el navegador
            pcm_data = await synthesize_google_tts(text, lang, sample_rate=browser_rate)
            
            # Calcular chunk size basado en sample rate (100ms de audio)
            # 16kHz * 2 bytes * 0.1s = 3200 bytes por chunk
            chunk_size = browser_rate * 2 // 10  # 100ms chunks
            
            # Enviar todo el audio de una vez para mínima latencia
            await send_to_browser(pcm_data, sample_rate=browser_rate)
            log(f"🔊 Audio enviado al agente (Google TTS, {lang})")
        else:
            # ElevenLabs turbo para menor latencia
            voice_id = "EXAVITQu4vr4xnSDxMaL"  # Sarah
            async with aiohttp.ClientSession() as session:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                headers = {
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg"
                }
                data = {
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",  # Turbo para menor latencia
                    "voice_settings": {
                        "stability": 0.4,
                        "similarity_boost": 0.7
                    }
                }
                async with session.post(url, headers=headers, json=data) as resp:
                    if resp.status == 200:
                        mp3_data = await resp.read()
                        from pydub import AudioSegment
                        import io
                        audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
                        browser_rate = settings.get("browser_sample_rate", 16000)
                        audio = audio.set_frame_rate(browser_rate).set_channels(1).set_sample_width(2)
                        pcm_data = audio.raw_data
                        await send_to_browser(pcm_data, sample_rate=browser_rate)
                        log(f"🔊 Audio enviado al agente (ElevenLabs, {lang})")
                    else:
                        log(f"❌ ElevenLabs error: {resp.status} - {await resp.text()}")
    except Exception as e:
        log(f"❌ TTS browser error: {e}")
        import traceback
        traceback.print_exc()


async def send_to_browser(pcm_audio: bytes, sample_rate: int = 16000):
    """Send audio to browser clients with sample rate info"""
    if not browser_clients:
        return
    
    msg = json.dumps({
        "type": "audio",
        "data": base64.b64encode(pcm_audio).decode(),
        "sampleRate": sample_rate  # Enviar sample rate al navegador
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


def resample_16k_to_8k(pcm_16k: bytes) -> bytes:
    """Resample from 16kHz to 8kHz by taking every other sample"""
    result = bytearray()
    for i in range(0, len(pcm_16k) - 3, 4):
        result.extend(pcm_16k[i:i+2])
    return bytes(result)


def resample_audio(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM audio from one rate to another"""
    if from_rate == to_rate:
        return pcm
    
    # Convert bytes to samples
    samples = struct.unpack(f'<{len(pcm)//2}h', pcm)
    
    # Calculate ratio
    ratio = to_rate / from_rate
    new_length = int(len(samples) * ratio)
    
    # Simple linear interpolation resampling
    result = []
    for i in range(new_length):
        src_idx = i / ratio
        idx = int(src_idx)
        frac = src_idx - idx
        
        if idx + 1 < len(samples):
            sample = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
        else:
            sample = samples[idx] if idx < len(samples) else 0
        
        result.append(max(-32768, min(32767, sample)))
    
    return struct.pack(f'<{len(result)}h', *result)


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
        return {"success": False, "error": str(e)}


@app.post("/voice/webhook")
async def voice_webhook():
    response = VoiceResponse()
    connect = Connect()
    stream = Stream(url=f"wss://{WEBHOOK_BASE_URL.replace('https://', '')}/ws/twilio")
    connect.append(stream)
    response.append(connect)
    return PlainTextResponse(str(response), media_type="application/xml")


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
                
                # Start Deepgram streaming
                await call.start_deepgram_client()
                log(f"📞 Llamada conectada")
                
            elif data["event"] == "media" and call:
                ulaw = base64.b64decode(data["media"]["payload"])
                pcm = ulaw_to_pcm(ulaw)
                await call.send_client_audio(pcm)
                
            elif data["event"] == "stop":
                if call:
                    await call.close()
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
    
    try:
        while True:
            try:
                msg = await websocket.receive_text()
                data = json.loads(msg)
                
                if data.get("type") == "audio":
                    pcm = base64.b64decode(data["data"])
                    browser_rate = data.get("sampleRate", 48000)
                    
                    for call in active_calls.values():
                        if call.active:
                            # Start agent Deepgram on first audio if not already connected
                            if not call.agent_deepgram_ws and settings["translation_enabled"]:
                                call.agent_sample_rate = browser_rate
                                await call.start_deepgram_agent()
                                log(f"🎤 Micrófono agente activo ({browser_rate}Hz)")
                            
                            await call.send_agent_audio(pcm)
                            
            except Exception as inner_e:
                # Check if it's a normal close
                if "1000" in str(inner_e) or "1001" in str(inner_e):
                    break
                raise
                        
    except Exception as e:
        pass  # Silent close
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
    print("🚀 STREAMING TRANSLATOR - Ultra Fast")
    print("="*50)
    print(f"Webhook: {WEBHOOK_BASE_URL}")
    print("Open: http://localhost:8000")
    print("="*50 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
