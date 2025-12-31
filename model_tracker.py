"""BlockSwap Model Tracker - Smart cleanup decision system.

This module provides a singleton tracker for managing BlockSwap state across
model instances and loop iterations. It enables smart cleanup decisions that
prevent premature block cleanup when subsequent loops still need the blocks.

Key Features:
- Singleton pattern using id(model.model) to track actual model instances
- Session-based reference counting for multi-loop workflows
- Thread-safe operations for concurrent access
- Smart cleanup decisions (FULL, PRESERVE, SKIP)
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Set, List, Any


class CleanupMode(Enum):
    """Configuration for cleanup behavior."""
    FULL = "full"        # Full cleanup (delete blocks, clear caches)
    PRESERVE = "preserve" # Clear state but preserve blocks
    DEFERRED = "deferred" # Defer cleanup to session end
    SMART = "smart"       # Let tracker decide based on state
    SKIP = "skip"         # Skip cleanup entirely


class CleanupDecision(Enum):
    """Result of cleanup decision logic."""
    FULL = "full"        # Safe to perform full cleanup
    PRESERVE = "preserve" # Clear tracking state but keep blocks
    SKIP = "skip"         # Skip cleanup (already done or not our model)
    ERROR = "error"       # Error state


@dataclass
class ModelPrepState:
    """Tracks preparation state for a single model instance."""
    model_id: int
    session_id: Optional[str] = None
    loop_index: int = 0
    is_prepared: bool = False
    cleanup_done: bool = False
    ref_count: int = 0
    blocks_to_swap: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class SessionState:
    """Tracks state for a BlockSwap looper session."""
    session_id: str
    expected_loops: int = 1
    current_loop: int = 0
    registered_models: List[int] = field(default_factory=list)
    is_complete: bool = False
    started_at: float = field(default_factory=time.time)


class BlockSwapModelTracker:
    """Singleton tracking BlockSwap state across model instances.

    Uses id(model.model) as key since ModelPatcher clones share the same
    underlying model reference. This enables detecting when cleanup on
    one clone would affect another clone's blocks.
    """

    _instance: Optional[BlockSwapModelTracker] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> BlockSwapModelTracker:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._model_states: Dict[int, ModelPrepState] = {}
        self._active_sessions: Dict[str, SessionState] = {}
        self._model_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._debug = False
        self._initialized = True

    @classmethod
    def get_instance(cls) -> BlockSwapModelTracker:
        """Get the singleton tracker instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self) -> None:
        """Reset all state (for testing only)."""
        with self._model_lock:
            self._model_states.clear()
        with self._session_lock:
            self._active_sessions.clear()

    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug logging."""
        self._debug = enabled

    def _log(self, msg: str) -> None:
        """Log a debug message."""
        if self._debug:
            print(f"[BlockSwap Tracker] {msg}")

    # ===== Session Management =====

    def start_session(self, expected_loops: int = 1) -> str:
        """Start a new tracking session."""
        session_id = str(uuid.uuid4())
        with self._session_lock:
            session = SessionState(
                session_id=session_id,
                expected_loops=expected_loops,
            )
            self._active_sessions[session_id] = session
        self._log(f"Started session {session_id} with {expected_loops} loops")
        return session_id

    def end_session(self, session_id: str) -> None:
        """End a session and clean up."""
        with self._session_lock:
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]

        # Clean up model states
        with self._model_lock:
            to_remove = [
                mid for mid, state in self._model_states.items()
                if state.session_id == session_id
            ]
            for mid in to_remove:
                del self._model_states[mid]

        self._log(f"Ended session {session_id}")

    def update_loop_progress(self, session_id: str, current_loop: int) -> None:
        """Update current loop progress in a session."""
        with self._session_lock:
            if session_id in self._active_sessions:
                self._active_sessions[session_id].current_loop = current_loop

    def mark_session_complete(self, session_id: str) -> None:
        """Mark a session as complete."""
        with self._session_lock:
            if session_id in self._active_sessions:
                self._active_sessions[session_id].is_complete = True

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a session."""
        with self._session_lock:
            session = self._active_sessions.get(session_id)
            if session is None:
                return None
            return {
                "session_id": session.session_id,
                "expected_loops": session.expected_loops,
                "current_loop": session.current_loop,
                "registered_models_count": len(session.registered_models),
                "is_complete": session.is_complete,
            }

    # ===== Model Registration =====

    def register_model(
        self,
        model: Any,
        loop_index: int,
        session_id: Optional[str] = None,
        blocks_to_swap: int = 0,
    ) -> None:
        """Register a model for tracking."""
        model_id = id(model.model)

        with self._model_lock:
            if model_id in self._model_states:
                state = self._model_states[model_id]
                state.ref_count += 1
                self._log(f"Model {model_id} ref count -> {state.ref_count}")
            else:
                state = ModelPrepState(
                    model_id=model_id,
                    session_id=session_id,
                    loop_index=loop_index,
                    blocks_to_swap=blocks_to_swap,
                    ref_count=1,
                )
                self._model_states[model_id] = state

        # Register in session
        if session_id:
            with self._session_lock:
                if session_id in self._active_sessions:
                    session = self._active_sessions[session_id]
                    if model_id not in session.registered_models:
                        session.registered_models.append(model_id)

        self._log(f"Registered model {model_id} in session {session_id}")

    def is_model_tracked(self, model: Any) -> bool:
        """Check if a model is tracked."""
        model_id = id(model.model)
        with self._model_lock:
            return model_id in self._model_states

    def find_session_for_model(self, model_id: int) -> Optional[str]:
        """Find the session ID that a model belongs to.

        Args:
            model_id: The id() of the model's underlying model object.

        Returns:
            The session_id if found, None otherwise.
        """
        with self._model_lock:
            if model_id in self._model_states:
                return self._model_states[model_id].session_id
        return None

    def increment_ref(self, model: Any) -> None:
        """Increment reference count for a model."""
        model_id = id(model.model)
        with self._model_lock:
            if model_id in self._model_states:
                self._model_states[model_id].ref_count += 1

    def decrement_ref(self, model: Any) -> None:
        """Decrement reference count for a model."""
        model_id = id(model.model)
        with self._model_lock:
            if model_id in self._model_states:
                state = self._model_states[model_id]
                state.ref_count = max(0, state.ref_count - 1)

    def mark_prepared(self, model: Any) -> None:
        """Mark a model as prepared."""
        model_id = id(model.model)
        with self._model_lock:
            if model_id in self._model_states:
                self._model_states[model_id].is_prepared = True

    # ===== Smart Cleanup Decisions =====

    def get_cleanup_decision(self, model: Any) -> CleanupDecision:
        """Determine cleanup action for a model.

        Decision logic:
        1. Model not tracked -> SKIP
        2. Cleanup already done -> SKIP
        3. Multiple refs -> PRESERVE
        4. In session with more loops -> PRESERVE
        5. Last loop or no session -> FULL
        """
        model_id = id(model.model)

        with self._model_lock:
            if model_id not in self._model_states:
                self._log(f"Model {model_id} not tracked -> SKIP")
                return CleanupDecision.SKIP

            state = self._model_states[model_id]

            if state.cleanup_done:
                self._log(f"Model {model_id} already cleaned -> SKIP")
                return CleanupDecision.SKIP

            if state.ref_count > 1:
                self._log(f"Model {model_id} has {state.ref_count} refs -> PRESERVE")
                return CleanupDecision.PRESERVE

            session_id = state.session_id

        # Check session state
        if session_id:
            with self._session_lock:
                session = self._active_sessions.get(session_id)
                if session:
                    if session.is_complete:
                        self._log(f"Session {session_id} complete -> FULL")
                        return CleanupDecision.FULL

                    if session.current_loop < session.expected_loops - 1:
                        self._log(
                            f"Session loop {session.current_loop}/"
                            f"{session.expected_loops} -> PRESERVE"
                        )
                        return CleanupDecision.PRESERVE

        self._log(f"Model {model_id} safe for cleanup -> FULL")
        return CleanupDecision.FULL

    def should_skip_preparation(
        self,
        model: Any,
        loop_index: int,
        session_id: Optional[str] = None,
    ) -> bool:
        """Check if model preparation should be skipped."""
        model_id = id(model.model)

        with self._model_lock:
            if model_id not in self._model_states:
                return False

            state = self._model_states[model_id]

            # Skip if already prepared in same session
            if state.is_prepared and state.session_id == session_id:
                self._log(f"Model {model_id} already prepared in session")
                return True

            return False

    def mark_cleanup_done(self, model: Any) -> None:
        """Mark cleanup as done for a model."""
        model_id = id(model.model)

        with self._model_lock:
            if model_id in self._model_states:
                state = self._model_states[model_id]
                state.cleanup_done = True
                state.ref_count = max(0, state.ref_count - 1)
                self._log(f"Model {model_id} marked cleanup done")

    def cleanup_expired_sessions(self, timeout_seconds: int = 3600) -> int:
        """Clean up sessions older than timeout."""
        now = time.time()
        to_remove = []

        with self._session_lock:
            for session_id, session in self._active_sessions.items():
                age = now - session.started_at
                if age > timeout_seconds:
                    to_remove.append(session_id)

        for session_id in to_remove:
            self.end_session(session_id)

        return len(to_remove)
