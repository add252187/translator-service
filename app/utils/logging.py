"""
Structured logging configuration for the translation service.
Provides JSON-formatted logs with contextual information.
"""

import logging
import sys
from typing import Any, Dict, Optional
import structlog
from structlog.stdlib import LoggerFactory
from pythonjsonlogger import jsonlogger
from app.config import settings
import time
from contextvars import ContextVar

# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
call_id_var: ContextVar[Optional[str]] = ContextVar("call_id", default=None)


def add_context_processor(logger, method_name, event_dict):
    """Add context variables to log entries."""
    request_id = request_id_var.get()
    call_id = call_id_var.get()
    
    if request_id:
        event_dict["request_id"] = request_id
    if call_id:
        event_dict["call_id"] = call_id
    
    event_dict["environment"] = settings.environment.value
    return event_dict


def add_timestamp(logger, method_name, event_dict):
    """Add timestamp to log entries."""
    event_dict["timestamp"] = time.time()
    return event_dict


def censor_sensitive_data(logger, method_name, event_dict):
    """Remove or mask sensitive data from logs."""
    if not settings.log_conversation_content:
        # List of keys that might contain conversation content
        sensitive_keys = [
            "transcript", "text", "translation", "audio_data",
            "message", "content", "user_input", "agent_input"
        ]
        
        for key in sensitive_keys:
            if key in event_dict:
                event_dict[key] = "[REDACTED]"
    
    # Always mask API keys and tokens
    for key in event_dict:
        if any(sensitive in key.lower() for sensitive in ["key", "token", "password", "secret"]):
            if isinstance(event_dict[key], str) and len(event_dict[key]) > 4:
                event_dict[key] = event_dict[key][:4] + "..." + event_dict[key][-4:]
    
    return event_dict


def setup_logging():
    """Configure structured logging for the application."""
    
    # Configure Python's standard logging
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Create a JSON formatter
    json_formatter = jsonlogger.JsonFormatter(
        fmt="%(timestamp)s %(level)s %(name)s %(message)s",
        rename_fields={"levelname": "level", "name": "logger"}
    )
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler with JSON formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    root_logger.addHandler(console_handler)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            add_timestamp,
            add_context_processor,
            censor_sensitive_data,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get a logger instance for this class."""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
    
    def log_latency(self, operation: str, start_time: float, **kwargs):
        """Log the latency of an operation."""
        latency_ms = (time.time() - start_time) * 1000
        self.logger.info(
            f"{operation}_completed",
            latency_ms=latency_ms,
            **kwargs
        )
    
    def log_error(self, operation: str, error: Exception, **kwargs):
        """Log an error with context."""
        self.logger.error(
            f"{operation}_failed",
            error_type=type(error).__name__,
            error_message=str(error),
            **kwargs
        )


# Initialize logging when module is imported
setup_logging()
