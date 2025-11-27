"""
Configuration module for the translation service.
Centralizes all environment variables and configuration settings.
"""

from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class STTProvider(str, Enum):
    DEEPGRAM = "deepgram"
    WHISPER = "whisper"
    ELEVENLABS = "elevenlabs"


class TranslationProvider(str, Enum):
    DEEPL = "deepl"
    OPENAI = "openai"


class TTSProvider(str, Enum):
    ELEVENLABS = "elevenlabs"
    AZURE = "azure"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    environment: Environment = Field(default=Environment.DEVELOPMENT, env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./translator.db", env="DATABASE_URL")
    
    # Twilio Configuration
    twilio_account_sid: Optional[str] = Field(default=None, env="TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[str] = Field(default=None, env="TWILIO_AUTH_TOKEN")
    twilio_phone_number: Optional[str] = Field(default=None, env="TWILIO_PHONE_NUMBER")
    webhook_base_url: str = Field(default="http://localhost:8000", env="WEBHOOK_BASE_URL")
    
    # STT Provider Configuration
    stt_provider: STTProvider = Field(default=STTProvider.DEEPGRAM, env="STT_PROVIDER")
    deepgram_api_key: Optional[str] = Field(default=None, env="DEEPGRAM_API_KEY")
    whisper_model: str = Field(default="base", env="WHISPER_MODEL")
    elevenlabs_api_key: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    
    # Translation Provider Configuration
    translation_provider: TranslationProvider = Field(
        default=TranslationProvider.DEEPL, env="TRANSLATION_PROVIDER"
    )
    deepl_api_key: Optional[str] = Field(default=None, env="DEEPL_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    translation_timeout: int = Field(default=5, env="TRANSLATION_TIMEOUT")
    
    # TTS Provider Configuration
    tts_provider: TTSProvider = Field(default=TTSProvider.ELEVENLABS, env="TTS_PROVIDER")
    elevenlabs_tts_api_key: Optional[str] = Field(default=None, env="ELEVENLABS_TTS_API_KEY")
    elevenlabs_voice_id_spanish: Optional[str] = Field(
        default=None, env="ELEVENLABS_VOICE_ID_SPANISH"
    )
    elevenlabs_model_id: str = Field(
        default="eleven_multilingual_v2", env="ELEVENLABS_MODEL_ID"
    )
    azure_tts_key: Optional[str] = Field(default=None, env="AZURE_TTS_KEY")
    azure_tts_region: str = Field(default="westeurope", env="AZURE_TTS_REGION")
    
    # Audio Configuration
    audio_sample_rate: int = Field(default=8000, env="AUDIO_SAMPLE_RATE")
    audio_chunk_duration_ms: int = Field(default=500, env="AUDIO_CHUNK_DURATION_MS")
    vad_aggressiveness: int = Field(default=2, env="VAD_AGGRESSIVENESS")
    audio_buffer_size: int = Field(default=4096, env="AUDIO_BUFFER_SIZE")
    
    # Performance Configuration
    max_concurrent_calls: int = Field(default=50, env="MAX_CONCURRENT_CALLS")
    worker_threads: int = Field(default=4, env="WORKER_THREADS")
    connection_pool_size: int = Field(default=20, env="CONNECTION_POOL_SIZE")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    # Language Configuration
    default_agent_language: str = Field(default="es", env="DEFAULT_AGENT_LANGUAGE")
    bypass_translation_for_same_language: bool = Field(
        default=False, env="BYPASS_TRANSLATION_FOR_SAME_LANGUAGE"
    )
    language_detection_confidence_threshold: float = Field(
        default=0.7, env="LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD"
    )
    
    # Security
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    internal_api_key: Optional[str] = Field(default=None, env="INTERNAL_API_KEY")
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    log_conversation_content: bool = Field(default=False, env="LOG_CONVERSATION_CONTENT")
    
    @validator("vad_aggressiveness")
    def validate_vad_aggressiveness(cls, v):
        if not 0 <= v <= 3:
            raise ValueError("VAD aggressiveness must be between 0 and 3")
        return v
    
    @validator("database_url")
    def validate_database_url(cls, v, values):
        """Ensure database URL is properly formatted."""
        if values.get("environment") == Environment.PRODUCTION and v.startswith("sqlite"):
            raise ValueError("SQLite should not be used in production")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def get_stt_api_key(self) -> Optional[str]:
        """Get the API key for the configured STT provider."""
        if self.stt_provider == STTProvider.DEEPGRAM:
            return self.deepgram_api_key
        elif self.stt_provider == STTProvider.ELEVENLABS:
            return self.elevenlabs_api_key
        return None
    
    def get_translation_api_key(self) -> Optional[str]:
        """Get the API key for the configured translation provider."""
        if self.translation_provider == TranslationProvider.DEEPL:
            return self.deepl_api_key
        elif self.translation_provider == TranslationProvider.OPENAI:
            return self.openai_api_key
        return None
    
    def get_tts_api_key(self) -> Optional[str]:
        """Get the API key for the configured TTS provider."""
        if self.tts_provider == TTSProvider.ELEVENLABS:
            return self.elevenlabs_tts_api_key
        elif self.tts_provider == TTSProvider.AZURE:
            return self.azure_tts_key
        return None


# Global settings instance
settings = Settings()
