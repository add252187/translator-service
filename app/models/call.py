"""
Database models for call tracking and metrics.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, Enum as SQLEnum, Text
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, Dict, Any
import enum
from app.models.database import Base


class CallStatus(str, enum.Enum):
    """Status of a call in the system."""
    INITIATED = "initiated"
    CONNECTING = "connecting"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Call(Base):
    """Model for tracking call sessions and their metadata."""
    
    __tablename__ = "calls"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Call identifiers
    call_sid = Column(String(100), unique=True, index=True, nullable=False)
    stream_sid = Column(String(100), unique=True, index=True, nullable=True)
    
    # Call participants
    agent_phone = Column(String(50), nullable=True)
    client_phone = Column(String(50), nullable=True)
    
    # Language information
    client_language = Column(String(10), nullable=True)  # Detected language code (e.g., 'en', 'fr', 'gl')
    client_language_confidence = Column(Float, nullable=True)  # Confidence score for language detection
    agent_language = Column(String(10), default="es")  # Always Spanish for agents
    
    # Call status and timing
    status = Column(SQLEnum(CallStatus), default=CallStatus.INITIATED, nullable=False)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    connected_at = Column(DateTime(timezone=True), nullable=True)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    # Performance metrics (stored as JSON for flexibility)
    metrics = Column(JSON, nullable=True)
    # Example metrics structure:
    # {
    #     "stt_latency_avg_ms": 250,
    #     "translation_latency_avg_ms": 150,
    #     "tts_latency_avg_ms": 200,
    #     "total_latency_avg_ms": 600,
    #     "stt_requests": 150,
    #     "translation_requests": 150,
    #     "tts_requests": 150,
    #     "errors": {"stt": 2, "translation": 1, "tts": 0}
    # }
    
    # Error tracking
    error_count = Column(Integer, default=0)
    last_error = Column(Text, nullable=True)
    
    # Configuration used for this call
    configuration = Column(JSON, nullable=True)
    # Example configuration:
    # {
    #     "stt_provider": "deepgram",
    #     "translation_provider": "deepl",
    #     "tts_provider": "elevenlabs",
    #     "bypass_same_language": false
    # }
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<Call(id={self.id}, call_sid={self.call_sid}, status={self.status})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "agent_phone": self.agent_phone,
            "client_phone": self.client_phone,
            "client_language": self.client_language,
            "client_language_confidence": self.client_language_confidence,
            "agent_language": self.agent_language,
            "status": self.status.value if self.status else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "configuration": self.configuration,
        }
    
    def calculate_duration(self):
        """Calculate and set the call duration."""
        if self.started_at and self.ended_at:
            delta = self.ended_at - self.started_at
            self.duration_seconds = int(delta.total_seconds())
    
    def update_metrics(self, new_metrics: Dict[str, Any]):
        """Update call metrics."""
        if not self.metrics:
            self.metrics = {}
        
        # Merge new metrics with existing ones
        for key, value in new_metrics.items():
            if key in self.metrics and isinstance(self.metrics[key], (int, float)) and isinstance(value, (int, float)):
                # For numeric values, calculate running average
                self.metrics[key] = (self.metrics[key] + value) / 2
            else:
                self.metrics[key] = value
    
    def increment_error(self, error_message: str):
        """Increment error count and update last error."""
        self.error_count = (self.error_count or 0) + 1
        self.last_error = error_message[:500]  # Limit error message length


class CallMetrics(Base):
    """Model for storing detailed call metrics for analytics."""
    
    __tablename__ = "call_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(Integer, nullable=False, index=True)
    
    # Timestamp for this metric entry
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # Component that generated this metric
    component = Column(String(50), nullable=False)  # e.g., "stt", "translation", "tts"
    
    # Operation type
    operation = Column(String(100), nullable=False)  # e.g., "process_audio", "translate_text"
    
    # Performance metrics
    latency_ms = Column(Float, nullable=True)
    success = Column(Boolean, default=True)
    
    # Additional context
    meta_info = Column(JSON, nullable=True)
    
    def __repr__(self):
        return f"<CallMetrics(call_id={self.call_id}, component={self.component}, latency_ms={self.latency_ms})>"
