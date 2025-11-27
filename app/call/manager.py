"""
Call manager for handling multiple concurrent call sessions.
"""

import asyncio
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from app.call.session import CallSession
from app.stt import get_stt_provider
from app.translation import get_translation_provider
from app.tts import get_tts_provider
from app.models import Call, CallStatus
from app.models.database import get_async_session
from app.config import settings
from app.utils.logging import LoggerMixin
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


class CallManager(LoggerMixin):
    """
    Manages multiple concurrent call sessions.
    Handles creation, tracking, and cleanup of calls.
    """
    
    def __init__(self):
        """Initialize the call manager."""
        self.active_sessions: Dict[str, CallSession] = {}
        self.session_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the call manager and background tasks."""
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_inactive_sessions())
        
        # Start metrics collection task
        self._metrics_task = asyncio.create_task(self._collect_metrics())
        
        self.logger.info("Call manager started")
    
    async def stop(self):
        """Stop the call manager and all active sessions."""
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._metrics_task:
            self._metrics_task.cancel()
        
        # Stop all active sessions
        async with self.session_lock:
            for session in list(self.active_sessions.values()):
                await session.stop()
            self.active_sessions.clear()
        
        self.logger.info("Call manager stopped")
    
    async def create_session(
        self,
        call_sid: str,
        stream_sid: Optional[str] = None,
        agent_phone: Optional[str] = None,
        client_phone: Optional[str] = None,
        **kwargs
    ) -> CallSession:
        """
        Create a new call session.
        
        Args:
            call_sid: Unique call identifier
            stream_sid: Twilio stream identifier
            agent_phone: Agent's phone number
            client_phone: Client's phone number
            **kwargs: Additional configuration
        
        Returns:
            Created CallSession instance
        
        Raises:
            ValueError: If session already exists or max concurrent calls reached
        """
        async with self.session_lock:
            # Check if session already exists
            if call_sid in self.active_sessions:
                raise ValueError(f"Session already exists for call {call_sid}")
            
            # Check concurrent call limit
            if len(self.active_sessions) >= settings.max_concurrent_calls:
                raise ValueError(f"Maximum concurrent calls ({settings.max_concurrent_calls}) reached")
            
            # Create providers for this session
            stt_provider = get_stt_provider()
            translation_provider = get_translation_provider()
            tts_provider = get_tts_provider()
            
            # Create session
            session = CallSession(
                call_sid=call_sid,
                stt_provider=stt_provider,
                translation_provider=translation_provider,
                tts_provider=tts_provider,
                **kwargs
            )
            
            # Update metadata
            session.metadata.stream_sid = stream_sid
            session.metadata.agent_phone = agent_phone
            session.metadata.client_phone = client_phone
            
            # Store in active sessions
            self.active_sessions[call_sid] = session
            
            # Create database record
            await self._create_call_record(session)
            
            # Start the session
            await session.start()
            
            self.logger.info(
                f"Created call session",
                call_sid=call_sid,
                active_sessions=len(self.active_sessions)
            )
            
            return session
    
    async def get_session(self, call_sid: str) -> Optional[CallSession]:
        """
        Get an active call session.
        
        Args:
            call_sid: Call identifier
        
        Returns:
            CallSession if found, None otherwise
        """
        return self.active_sessions.get(call_sid)
    
    async def end_session(self, call_sid: str) -> bool:
        """
        End a call session.
        
        Args:
            call_sid: Call identifier
        
        Returns:
            True if session was ended, False if not found
        """
        async with self.session_lock:
            session = self.active_sessions.get(call_sid)
            if not session:
                return False
            
            # Stop the session
            await session.stop()
            
            # Update database record
            await self._update_call_record(session, CallStatus.COMPLETED)
            
            # Remove from active sessions
            del self.active_sessions[call_sid]
            
            self.logger.info(
                f"Ended call session",
                call_sid=call_sid,
                active_sessions=len(self.active_sessions)
            )
            
            return True
    
    async def get_active_sessions(self) -> List[Dict]:
        """Get information about all active sessions."""
        sessions_info = []
        for call_sid, session in self.active_sessions.items():
            sessions_info.append({
                "call_sid": call_sid,
                "stream_sid": session.metadata.stream_sid,
                "client_language": session.metadata.client_language,
                "duration": (datetime.utcnow() - session.metadata.started_at).total_seconds(),
                "metrics": session.get_metrics()
            })
        return sessions_info
    
    async def _create_call_record(self, session: CallSession):
        """Create a database record for a new call."""
        try:
            async for db_session in get_async_session():
                call = Call(
                    call_sid=session.call_sid,
                    stream_sid=session.metadata.stream_sid,
                    agent_phone=session.metadata.agent_phone,
                    client_phone=session.metadata.client_phone,
                    agent_language=session.metadata.agent_language,
                    status=CallStatus.ACTIVE,
                    configuration={
                        "stt_provider": settings.stt_provider.value,
                        "translation_provider": settings.translation_provider.value,
                        "tts_provider": settings.tts_provider.value,
                        "bypass_same_language": session.bypass_same_language
                    }
                )
                db_session.add(call)
                await db_session.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to create call record", error=str(e))
    
    async def _update_call_record(self, session: CallSession, status: CallStatus):
        """Update a call record in the database."""
        try:
            async for db_session in get_async_session():
                # Find the call record
                result = await db_session.execute(
                    select(Call).where(Call.call_sid == session.call_sid)
                )
                call = result.scalar_one_or_none()
                
                if call:
                    # Update call information
                    call.status = status
                    call.client_language = session.metadata.client_language
                    call.client_language_confidence = session.metadata.client_language_confidence
                    call.ended_at = datetime.utcnow()
                    call.calculate_duration()
                    
                    # Update metrics
                    metrics = session.get_metrics()
                    call.update_metrics({
                        "stt_requests": metrics["stt_count"],
                        "translation_requests": metrics["translation_count"],
                        "tts_requests": metrics["tts_count"],
                        "avg_latency_ms": metrics["avg_latency_ms"],
                        "errors": metrics["error_count"]
                    })
                    
                    call.error_count = session.metadata.error_count
                    
                    await db_session.commit()
                    
        except Exception as e:
            self.logger.error(f"Failed to update call record", error=str(e))
    
    async def _cleanup_inactive_sessions(self):
        """Background task to clean up inactive sessions."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                async with self.session_lock:
                    inactive_sessions = []
                    
                    for call_sid, session in self.active_sessions.items():
                        # Check if session has been inactive for too long
                        duration = datetime.utcnow() - session.metadata.started_at
                        if duration > timedelta(hours=1):  # Max call duration
                            inactive_sessions.append(call_sid)
                    
                    # Clean up inactive sessions
                    for call_sid in inactive_sessions:
                        self.logger.warning(f"Cleaning up inactive session", call_sid=call_sid)
                        await self.end_session(call_sid)
                        
            except Exception as e:
                self.logger.error(f"Error in cleanup task", error=str(e))
    
    async def _collect_metrics(self):
        """Background task to collect and log metrics."""
        while True:
            try:
                await asyncio.sleep(30)  # Collect every 30 seconds
                
                if self.active_sessions:
                    total_latency = 0
                    total_errors = 0
                    
                    for session in self.active_sessions.values():
                        metrics = session.get_metrics()
                        total_latency += metrics["avg_latency_ms"]
                        total_errors += metrics["error_count"]
                    
                    avg_latency = total_latency / len(self.active_sessions) if self.active_sessions else 0
                    
                    self.logger.info(
                        "Call manager metrics",
                        active_sessions=len(self.active_sessions),
                        avg_latency_ms=avg_latency,
                        total_errors=total_errors
                    )
                    
            except Exception as e:
                self.logger.error(f"Error in metrics collection", error=str(e))


# Global call manager instance
call_manager = CallManager()
