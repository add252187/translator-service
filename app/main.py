"""
Main FastAPI application for the real-time bidirectional translation service.
"""

from fastapi import FastAPI, WebSocket, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import uuid

from app.config import settings
from app.utils.logging import get_logger, request_id_var
from app.models.database import init_database, close_database
from app.call.manager import call_manager
from app.call.twilio_stream import handle_twilio_stream
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream
from twilio.rest import Client as TwilioClient

logger = get_logger(__name__)

# Twilio client for outbound calls
twilio_client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    """
    # Startup
    logger.info("Starting translation service...")
    
    # Initialize database
    await init_database()
    
    # Start call manager
    await call_manager.start()
    
    logger.info("Translation service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down translation service...")
    
    # Stop call manager
    await call_manager.stop()
    
    # Close database
    await close_database()
    
    logger.info("Translation service shut down")


# Create FastAPI app
app = FastAPI(
    title="Real-time Bidirectional Translation Service",
    description="A scalable system for real-time phone call translation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
if settings.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to context for logging."""
    request_id = str(uuid.uuid4())
    request_id_var.set(request_id)
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the control panel."""
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        with open(static_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Translator Service Running</h1>")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    active_sessions = len(call_manager.active_sessions)
    
    return {
        "status": "healthy",
        "active_sessions": active_sessions,
        "max_sessions": settings.max_concurrent_calls,
        "environment": settings.environment.value
    }


@app.post("/voice/webhook")
async def voice_webhook(request: Request):
    """
    Twilio voice webhook endpoint.
    Returns TwiML to connect the call to Media Streams.
    """
    try:
        # Get call parameters from Twilio
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        from_number = form_data.get("From")
        to_number = form_data.get("To")
        
        logger.info(
            f"Voice webhook called",
            call_sid=call_sid,
            from_number=from_number,
            to_number=to_number
        )
        
        # Create TwiML response
        response = VoiceResponse()
        
        # Add initial greeting (optional)
        response.say(
            "Connecting to translation service...",
            voice="alice",
            language="en-US"
        )
        
        # Connect to Media Streams
        connect = Connect()
        stream = Stream(
            url=f"wss://{settings.webhook_base_url.replace('http://', '').replace('https://', '')}/ws/media-stream"
        )
        
        # Add custom parameters
        stream.parameter(name="agentPhone", value=to_number)
        stream.parameter(name="clientPhone", value=from_number)
        
        connect.append(stream)
        response.append(connect)
        
        return PlainTextResponse(
            content=str(response),
            media_type="application/xml"
        )
        
    except Exception as e:
        logger.error(f"Error in voice webhook: {e}")
        
        # Return error TwiML
        response = VoiceResponse()
        response.say("Sorry, there was an error connecting your call.", voice="alice")
        response.hangup()
        
        return PlainTextResponse(
            content=str(response),
            media_type="application/xml"
        )


@app.websocket("/ws/media-stream")
async def media_stream_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    Handles real-time bidirectional audio.
    """
    await handle_twilio_stream(websocket)


@app.get("/calls")
async def get_active_calls():
    """Get information about active calls."""
    try:
        sessions = await call_manager.get_active_sessions()
        return {
            "total": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"Error getting active calls: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/calls/{call_sid}")
async def get_call_info(call_sid: str):
    """Get information about a specific call."""
    session = await call_manager.get_session(call_sid)
    
    if not session:
        raise HTTPException(status_code=404, detail="Call not found")
    
    return session.get_metrics()


@app.post("/calls/{call_sid}/end")
async def end_call(call_sid: str):
    """End a specific call."""
    success = await call_manager.end_session(call_sid)
    
    if not success:
        raise HTTPException(status_code=404, detail="Call not found")
    
    return {"message": "Call ended successfully"}


@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    sessions = await call_manager.get_active_sessions()
    
    total_stt = sum(s["metrics"]["stt_count"] for s in sessions)
    total_translations = sum(s["metrics"]["translation_count"] for s in sessions)
    total_tts = sum(s["metrics"]["tts_count"] for s in sessions)
    avg_latency = sum(s["metrics"]["avg_latency_ms"] for s in sessions) / len(sessions) if sessions else 0
    
    return {
        "active_sessions": len(sessions),
        "total_stt_requests": total_stt,
        "total_translation_requests": total_translations,
        "total_tts_requests": total_tts,
        "average_latency_ms": avg_latency,
        "max_concurrent_calls": settings.max_concurrent_calls,
        "providers": {
            "stt": settings.stt_provider.value,
            "translation": settings.translation_provider.value,
            "tts": settings.tts_provider.value
        }
    }


@app.post("/call/outbound")
async def make_outbound_call(
    to_number: str = "+34651351636",
    agent_language: str = "es"
):
    """
    Initiate an outbound call from Twilio to a phone number.
    The call will connect through the translation service.
    
    Args:
        to_number: Phone number to call (E.164 format)
        agent_language: Language of the agent (default: Spanish)
    """
    try:
        logger.info(f"Initiating outbound call to {to_number}")
        
        # Create the call using Twilio REST API
        call = twilio_client.calls.create(
            to=to_number,
            from_=settings.twilio_phone_number,
            url=f"{settings.webhook_base_url}/voice/outbound-webhook",
            status_callback=f"{settings.webhook_base_url}/voice/status",
            status_callback_event=["initiated", "ringing", "answered", "completed"]
        )
        
        logger.info(f"Outbound call initiated", call_sid=call.sid)
        
        return {
            "success": True,
            "call_sid": call.sid,
            "to_number": to_number,
            "from_number": settings.twilio_phone_number,
            "status": call.status
        }
        
    except Exception as e:
        logger.error(f"Failed to initiate outbound call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/outbound-webhook")
async def outbound_voice_webhook(request: Request):
    """
    Twilio webhook for outbound calls.
    Returns TwiML to connect the call to Media Streams for translation.
    """
    # Get form data
    form_data = await request.form()
    call_sid = form_data.get("CallSid", "unknown")
    to_number = form_data.get("To", "unknown")
    from_number = form_data.get("From", "unknown")
    
    logger.info(f"Outbound webhook called: call_sid={call_sid}, to={to_number}, from={from_number}")
    
    # Build WebSocket URL
    base_url = settings.webhook_base_url.replace('http://', '').replace('https://', '')
    ws_url = f"wss://{base_url}/ws/media-stream"
    
    logger.info(f"WebSocket URL: {ws_url}")
    
    # Create TwiML response
    response = VoiceResponse()
    
    # Greeting in Spanish
    response.say(
        "Conectando con el servicio de traducción en tiempo real. Por favor espere.",
        voice="alice",
        language="es-ES"
    )
    
    # Connect to Media Streams
    connect = Connect()
    stream = Stream(url=ws_url)
    stream.parameter(name="direction", value="outbound")
    stream.parameter(name="agentPhone", value=str(to_number))
    stream.parameter(name="agentLanguage", value="es")
    
    connect.append(stream)
    response.append(connect)
    
    twiml = str(response)
    logger.info(f"TwiML response: {twiml[:200]}...")
    
    return PlainTextResponse(content=twiml, media_type="application/xml")


@app.post("/voice/status")
async def call_status_webhook(request: Request):
    """
    Twilio call status webhook.
    Receives status updates for calls.
    """
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        call_status = form_data.get("CallStatus")
        
        logger.info(f"Call status update", call_sid=call_sid, status=call_status)
        
        return {"received": True}
        
    except Exception as e:
        logger.error(f"Error in status webhook: {e}")
        return {"received": False}


@app.post("/test/pipeline")
async def test_full_pipeline(text: str = "Hello, how are you today?"):
    """
    Test the full translation pipeline: Text → Translation → TTS
    """
    try:
        from app.translation import get_translation_provider
        from app.tts import get_tts_provider
        
        results = {"input": text, "steps": []}
        
        # Step 1: Translation
        translator = get_translation_provider()
        await translator.initialize()
        
        translation_result = await translator.translate(
            text=text,
            source_language="en",
            target_language="es"
        )
        
        results["steps"].append({
            "step": "translation",
            "input": text,
            "output": translation_result.translated_text,
            "success": True
        })
        
        await translator.close()
        
        # Step 2: TTS (optional - just test if it works)
        try:
            tts = get_tts_provider()
            await tts.initialize()
            
            tts_result = await tts.synthesize(
                text=translation_result.translated_text,
                language="es"
            )
            
            results["steps"].append({
                "step": "tts",
                "input": translation_result.translated_text,
                "audio_bytes": len(tts_result.audio_data) if tts_result.audio_data else 0,
                "success": tts_result.audio_data is not None
            })
            
            await tts.close()
        except Exception as tts_error:
            results["steps"].append({
                "step": "tts",
                "error": str(tts_error),
                "success": False
            })
        
        results["final_translation"] = translation_result.translated_text
        return results
        
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/test/translate")
async def test_translation(
    text: str,
    source_language: str = "en",
    target_language: str = "es"
):
    """
    Test endpoint for translation functionality.
    
    Args:
        text: Text to translate
        source_language: Source language code
        target_language: Target language code
    """
    try:
        from app.translation import get_translation_provider
        
        provider = get_translation_provider()
        await provider.initialize()
        
        result = await provider.translate(
            text=text,
            source_language=source_language,
            target_language=target_language
        )
        
        await provider.close()
        
        return {
            "original_text": text,
            "translated_text": result.translated_text,
            "source_language": result.source_language,
            "target_language": result.target_language,
            "confidence": result.confidence
        }
        
    except Exception as e:
        logger.error(f"Translation test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
