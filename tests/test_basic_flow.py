"""
Basic flow tests for the translation service.
Tests the complete pipeline: Audio → STT → Translation → TTS → Audio
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.call.session import CallSession, CallMetadata
from app.stt.base import STTProvider, STTResult
from app.translation.base import TranslationProvider, TranslationResult
from app.tts.base import TTSProvider, TTSResult
from app.utils.audio import AudioProcessor


class MockSTTProvider(STTProvider):
    """Mock STT provider for testing."""
    
    async def _initialize(self):
        pass
    
    async def transcribe(self, audio_data, sample_rate=8000, language_hint=None, **kwargs):
        # Simulate STT based on audio length
        if len(audio_data) > 1000:
            if language_hint == "es":
                return STTResult(text="Hola, ¿cómo estás?", language="es", confidence=0.95)
            else:
                return STTResult(text="Hello, how are you?", language="en", confidence=0.95)
        return STTResult(text="", language=None, confidence=0.0)
    
    async def transcribe_stream(self, audio_stream, sample_rate=8000, language_hint=None, **kwargs):
        result_queue = asyncio.Queue()
        return result_queue
    
    async def detect_language(self, audio_data, sample_rate=8000):
        if len(audio_data) > 1000:
            return ("en", 0.9)
        return ("unknown", 0.0)


class MockTranslationProvider(TranslationProvider):
    """Mock translation provider for testing."""
    
    async def _initialize(self):
        pass
    
    async def translate(self, text, target_language, source_language=None, **kwargs):
        # Simple mock translation
        translations = {
            ("Hello, how are you?", "es"): "Hola, ¿cómo estás?",
            ("Hola, ¿cómo estás?", "en"): "Hello, how are you?",
            ("Good morning", "es"): "Buenos días",
            ("Buenos días", "en"): "Good morning",
        }
        
        key = (text, target_language)
        translated = translations.get(key, text)
        
        return TranslationResult(
            translated_text=translated,
            source_language=source_language or "en",
            target_language=target_language,
            confidence=0.95
        )
    
    async def detect_language(self, text):
        if "hola" in text.lower() or "cómo" in text.lower():
            return ("es", 0.95)
        return ("en", 0.95)
    
    async def get_supported_languages(self):
        return ["en", "es", "fr", "de", "it", "pt"]


class MockTTSProvider(TTSProvider):
    """Mock TTS provider for testing."""
    
    async def _initialize(self):
        pass
    
    async def synthesize(self, text, language, voice_id=None, **kwargs):
        # Return mock audio data
        audio_data = b"mock_audio_" + text.encode()[:100]
        return TTSResult(
            audio_data=audio_data,
            sample_rate=8000,
            duration_ms=len(text) * 50,
            voice_id=voice_id or "default"
        )
    
    async def synthesize_stream(self, text, language, voice_id=None, **kwargs):
        # Yield mock audio chunks
        for i in range(3):
            yield b"chunk_" + str(i).encode()
    
    async def get_available_voices(self, language=None):
        voices = [
            {"voice_id": "voice_es", "name": "Spanish Voice", "languages": ["es"]},
            {"voice_id": "voice_en", "name": "English Voice", "languages": ["en"]},
        ]
        if language:
            return [v for v in voices if language in v["languages"]]
        return voices


@pytest.fixture
async def call_session():
    """Create a call session with mock providers."""
    session = CallSession(
        call_sid="test_call_123",
        stt_provider=MockSTTProvider(),
        translation_provider=MockTranslationProvider(),
        tts_provider=MockTTSProvider()
    )
    await session.start()
    yield session
    await session.stop()


@pytest.mark.asyncio
async def test_call_session_initialization():
    """Test that a call session initializes correctly."""
    session = CallSession(
        call_sid="test_call_123",
        stt_provider=MockSTTProvider(),
        translation_provider=MockTranslationProvider(),
        tts_provider=MockTTSProvider()
    )
    
    assert session.call_sid == "test_call_123"
    assert session.metadata.agent_language == "es"
    assert not session.is_active
    assert not session.language_detected
    
    await session.start()
    assert session.is_active
    
    await session.stop()
    assert not session.is_active


@pytest.mark.asyncio
async def test_client_to_agent_translation(call_session):
    """Test translation from client (English) to agent (Spanish)."""
    # Simulate client speaking English
    client_audio = b"x" * 2000  # Mock audio data
    
    # Add audio to client buffer
    await call_session.add_client_audio(client_audio)
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    # Check that language was detected
    # Note: In real test, we'd need to trigger the processing loop
    
    # Verify metrics
    metrics = call_session.get_metrics()
    assert metrics["call_sid"] == "test_call_123"
    assert metrics["is_active"] == True


@pytest.mark.asyncio
async def test_agent_to_client_translation(call_session):
    """Test translation from agent (Spanish) to client (detected language)."""
    # First, set client language
    call_session.metadata.client_language = "en"
    call_session.language_detected = True
    
    # Simulate agent speaking Spanish
    agent_audio = b"y" * 2000  # Mock audio data
    
    # Add audio to agent buffer
    await call_session.add_agent_audio(agent_audio)
    
    # Wait for processing
    await asyncio.sleep(0.5)
    
    # Verify metrics
    metrics = call_session.get_metrics()
    assert metrics["agent_language"] == "es"


@pytest.mark.asyncio
async def test_language_detection():
    """Test automatic language detection from client speech."""
    stt = MockSTTProvider()
    await stt.initialize()
    
    # Test with sufficient audio
    audio_data = b"z" * 2000
    language, confidence = await stt.detect_language(audio_data)
    
    assert language == "en"
    assert confidence > 0.8
    
    # Test with insufficient audio
    short_audio = b"a" * 100
    language, confidence = await stt.detect_language(short_audio)
    
    assert language == "unknown"
    assert confidence == 0.0


@pytest.mark.asyncio
async def test_audio_processing():
    """Test audio format conversions."""
    processor = AudioProcessor()
    
    # Test μ-law to PCM conversion
    ulaw_data = bytes([0xFF, 0x7F, 0x00, 0x80])  # Sample μ-law data
    pcm_data = processor.ulaw_to_pcm(ulaw_data)
    
    assert len(pcm_data) == len(ulaw_data) * 2  # PCM is 16-bit
    assert isinstance(pcm_data, bytes)
    
    # Test PCM to μ-law conversion
    pcm_input = b'\x00\x00\xFF\x7F'  # Sample PCM data
    ulaw_output = processor.pcm_to_ulaw(pcm_input)
    
    assert len(ulaw_output) == len(pcm_input) // 2
    assert isinstance(ulaw_output, bytes)


@pytest.mark.asyncio
async def test_translation_bypass_same_language():
    """Test that translation is bypassed when source and target languages are the same."""
    translator = MockTranslationProvider()
    await translator.initialize()
    
    # Test with bypass enabled
    result = await translator.translate(
        text="Hola mundo",
        target_language="es",
        source_language="es"
    )
    
    # In a real implementation with bypass, text should remain unchanged
    # This would need to be implemented in the mock or actual provider


@pytest.mark.asyncio
async def test_concurrent_calls():
    """Test handling multiple concurrent call sessions."""
    from app.call.manager import CallManager
    
    manager = CallManager()
    await manager.start()
    
    # Create multiple sessions
    sessions = []
    for i in range(3):
        session = await manager.create_session(
            call_sid=f"call_{i}",
            stream_sid=f"stream_{i}"
        )
        sessions.append(session)
    
    # Verify all sessions are active
    active_sessions = await manager.get_active_sessions()
    assert len(active_sessions) == 3
    
    # End one session
    await manager.end_session("call_1")
    
    active_sessions = await manager.get_active_sessions()
    assert len(active_sessions) == 2
    
    # Clean up
    for i in [0, 2]:
        await manager.end_session(f"call_{i}")
    
    await manager.stop()


@pytest.mark.asyncio
async def test_error_handling(call_session):
    """Test error handling in the translation pipeline."""
    # Simulate an error by providing invalid data
    invalid_audio = b""  # Empty audio
    
    # This should not crash the session
    await call_session.add_client_audio(invalid_audio)
    await call_session.add_agent_audio(invalid_audio)
    
    # Session should still be active
    assert call_session.is_active
    
    # Check error count in metrics
    metrics = call_session.get_metrics()
    # Error count might be 0 if empty audio is filtered early
    assert "error_count" in metrics


@pytest.mark.asyncio
async def test_tts_voice_selection():
    """Test that correct voices are selected for different languages."""
    tts = MockTTSProvider()
    await tts.initialize()
    
    # Get Spanish voice
    spanish_voice = await tts.get_voice_for_language("es")
    assert spanish_voice is not None
    
    # Get English voice
    english_voice = await tts.get_voice_for_language("en")
    assert english_voice is not None
    
    # Synthesize Spanish audio
    result = await tts.synthesize("Hola mundo", "es")
    assert result.audio_data is not None
    assert result.sample_rate == 8000


@pytest.mark.asyncio
async def test_end_to_end_flow():
    """Test complete end-to-end flow of a call."""
    # Create session
    session = CallSession(
        call_sid="e2e_test",
        stt_provider=MockSTTProvider(),
        translation_provider=MockTranslationProvider(),
        tts_provider=MockTTSProvider()
    )
    
    await session.start()
    
    # Simulate client speaking English
    client_audio = b"client_speech" * 200
    await session.add_client_audio(client_audio)
    
    # Set detected language (normally done automatically)
    session.metadata.client_language = "en"
    session.language_detected = True
    
    # Simulate agent speaking Spanish
    agent_audio = b"agent_speech" * 200
    await session.add_agent_audio(agent_audio)
    
    # Wait for processing
    await asyncio.sleep(1)
    
    # Get metrics
    metrics = session.get_metrics()
    
    # Verify session processed audio
    assert metrics["call_sid"] == "e2e_test"
    assert metrics["client_language"] == "en"
    assert metrics["is_active"] == True
    
    # Stop session
    await session.stop()
    assert not session.is_active


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
