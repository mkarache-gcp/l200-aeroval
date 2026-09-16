"""
Unified Session, Context, and Memory Management for AeroEval.

Combines conversation state tracking, active hardware entity context
(active_drone_id, active_circuit_board, active_test_id), history compaction,
context truncation, in-memory caching, and asynchronous Firestore persistence.

Addresses context bloat via sliding-window compaction and prevents UI blocking
via asynchronous and background task persistence.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Configurable limits for context management
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))
RETAIN_RECENT_TURNS = int(os.getenv("RETAIN_RECENT_TURNS", "4"))

# Background executor for non-blocking thread pool persistence
_persistence_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="aeroeval-memory-worker")


class SessionStore:
    """Unified session, context, and memory persistence manager for AeroEval.

    Acts as both the context representation for an active session (storing hardware
    entities and conversation turns) and the storage engine (in-memory cache and
    asynchronous Cloud Firestore persistence).
    """

    # Global session cache and Firestore client shared across instances
    _memory_cache: Dict[str, "SessionStore"] = {}
    _db: Optional[Any] = None
    _db_initialized: bool = False

    def __init__(
        self,
        session_id: str = "default-session",
        active_drone_id: Optional[str] = None,
        active_board_type: Optional[str] = None,
        active_test_id: Optional[str] = None,
        last_file_path: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        pending_action: Optional[Dict[str, Any]] = None,
        filed_tickets: Optional[List[Dict[str, Any]]] = None,
        max_turns: int = MAX_HISTORY_TURNS,
        retain_recent: int = RETAIN_RECENT_TURNS,
    ):
        self.session_id = session_id
        self.active_drone_id = active_drone_id
        self.active_board_type = active_board_type
        self.active_test_id = active_test_id
        self.last_file_path = last_file_path
        self.history: List[Dict[str, str]] = list(history) if history is not None else []
        self.pending_action: Optional[Dict[str, Any]] = pending_action
        self.filed_tickets: List[Dict[str, Any]] = list(filed_tickets) if filed_tickets is not None else []
        self.max_turns = max_turns
        self.retain_recent = retain_recent

        # GCP & Firestore configuration
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID", "onboardingproject-507522")
        self.database_name = os.getenv("FIRESTORE_DATABASE", "aeroeval")
        self.collection_name = os.getenv("FIRESTORE_COLLECTION", "aeroeval_sessions")
        self.use_firestore = os.getenv("USE_FIRESTORE", "false").lower() in ("true", "1")

        if not SessionStore._db_initialized and self.use_firestore:
            SessionStore._init_firestore(self.project_id, self.database_name)

    @classmethod
    def _init_firestore(cls, project_id: str, database_name: str) -> None:
        """Initializes the shared Google Cloud Firestore client once."""
        try:
            from google.cloud import firestore
            cls._db = firestore.Client(
                project=project_id,
                database=database_name,
            )
            cls._db_initialized = True
            logger.info(
                f"Connected to Firestore (Project: {project_id}, Database: {database_name})"
            )
        except Exception as e:
            logger.warning(
                f"Firestore connection failed: {e}. Falling back to in-memory session store."
            )
            cls._db = None
            cls._db_initialized = True

    # -------------------------------------------------------------------------
    # Context State & Entity Tracking
    # -------------------------------------------------------------------------

    def update_context(
        self,
        drone_id: Optional[str] = None,
        board_type: Optional[str] = None,
        test_id: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> None:
        """Updates active hardware entity context (drone, circuit board, test ID)."""
        if drone_id:
            self.active_drone_id = drone_id
        if board_type:
            self.active_board_type = board_type
        if test_id:
            self.active_test_id = test_id
        if file_path:
            self.last_file_path = file_path

    def add_message(self, role: str, content: str, auto_compact: bool = True) -> None:
        """Appends a turn to conversation history and automatically compacts if exceeding limit.

        Normalizes roles: 'assistant' and 'model' both map to canonical 'model'.
        """
        canonical_role = "user" if role == "user" else "model"
        self.history.append({"role": canonical_role, "content": content})

        # History Compaction / Context Truncation to prevent LLM context bloat
        if auto_compact and len(self.history) > self.max_turns:
            self.compact_history(max_turns=self.max_turns, retain_recent=self.retain_recent)

    # -------------------------------------------------------------------------
    # History Compaction & Context Truncation (Anti-Bloat Management)
    # -------------------------------------------------------------------------

    def compact_history(
        self,
        max_turns: Optional[int] = None,
        retain_recent: Optional[int] = None,
    ) -> int:
        """Compacts older conversation turns into a structured summary turn.

        Preserves:
        1. All active hardware entity metadata (active_drone_id, active_board_type, active_test_id).
        2. A structured engineering summary of prior inquiries and findings.
        3. The most recent N turns in high-fidelity verbatim dialogue.

        Returns:
            The number of older turns compacted.
        """
        threshold = max_turns or self.max_turns
        keep_count = retain_recent or self.retain_recent

        if len(self.history) <= threshold:
            return 0

        # Ensure retain_recent is an even number of turns to preserve alternating pairs
        if keep_count % 2 != 0:
            keep_count += 1

        older_turns = self.history[:-keep_count]
        recent_turns = self.history[-keep_count:]

        # Ensure recent_turns starts with a 'user' turn
        if recent_turns and recent_turns[0]["role"] != "user" and len(older_turns) > 0:
            recent_turns = recent_turns[1:]

        compacted_count = len(older_turns)

        # Distill key topics and observations from older turns
        prior_queries = [
            turn["content"][:120].strip() + ("..." if len(turn["content"]) > 120 else "")
            for turn in older_turns
            if turn["role"] == "user"
        ]
        queries_summary = " | ".join(prior_queries[-3:]) if prior_queries else "Prior test inquiries"

        active_context_desc = []
        if self.active_drone_id:
            active_context_desc.append(f"Drone={self.active_drone_id}")
        if self.active_board_type:
            active_context_desc.append(f"Board={self.active_board_type}")
        if self.active_test_id:
            active_context_desc.append(f"Test={self.active_test_id}")
        context_str = ", ".join(active_context_desc) if active_context_desc else "General telemetry inspection"

        # Structured summary turn injected at head of compacted history
        summary_user_turn = {
            "role": "user",
            "content": (
                f"[Session Context Summary: {compacted_count} earlier turns compacted to manage context. "
                f"Active Hardware Context: {context_str}. "
                f"Prior Topics: {queries_summary}]"
            ),
        }
        summary_model_ack = {
            "role": "model",
            "content": (
                f"Understood. I have preserved context for {context_str} and all prior findings. "
                f"Continuing flight test telemetry evaluation."
            ),
        }

        self.history = [summary_user_turn, summary_model_ack] + recent_turns
        logger.info(
            f"Compacted {compacted_count} history turns for session '{self.session_id}'. "
            f"New history size: {len(self.history)} turns."
        )
        return compacted_count

    def truncate_context(self, max_turns: Optional[int] = None) -> int:
        """Sliding-window context truncation keeping the most recent turns.

        Maintains valid alternating sequence starting on a 'user' turn.
        Hardware entity tracking is retained even when older turns are truncated.

        Returns:
            The number of turns truncated.
        """
        threshold = max_turns or self.max_turns
        if len(self.history) <= threshold:
            return 0

        excess = len(self.history) - threshold
        truncated = self.history[excess:]

        # Ensure first turn is a user turn for GenAI/Claude API compliance
        while truncated and truncated[0]["role"] != "user":
            truncated = truncated[1:]

        removed_count = len(self.history) - len(truncated)
        self.history = truncated
        logger.info(
            f"Truncated {removed_count} turns from session '{self.session_id}'. "
            f"Remaining turns: {len(self.history)}."
        )
        return removed_count

    # -------------------------------------------------------------------------
    # Model Format Converters
    # -------------------------------------------------------------------------

    def to_genai_history(self) -> List[Any]:
        """Converts conversation turns to google.genai types.Content objects.

        Ensures valid alternating user/model sequence for client.chats.create(history=...).
        """
        try:
            from google.genai import types

            contents = []
            for msg in self.history:
                role = "user" if msg.get("role") == "user" else "model"
                text = msg.get("content", "")
                if text:
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=text)],
                        )
                    )
            return contents
        except ImportError:
            return []

    def to_anthropic_messages(self, current_message: Optional[str] = None) -> List[Dict[str, str]]:
        """Converts conversation turns to Anthropic Claude message format."""
        messages = []
        for msg in self.history:
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
        if current_message:
            messages.append({"role": "user", "content": current_message})
        return messages

    # -------------------------------------------------------------------------
    # Human-In-The-Loop (HITL) Action State Management
    # -------------------------------------------------------------------------

    def set_pending_action(self, action_type: str, draft: Dict[str, Any], draft_id: Optional[str] = None) -> str:
        """Stores a pending high-stakes action awaiting human approval."""
        assigned_id = draft_id or f"DRAFT-{action_type.upper()}-{len(self.filed_tickets) + 1}"
        self.pending_action = {
            "draft_id": assigned_id,
            "action_type": action_type,
            "draft": draft,
            "status": "REQUIRES_HUMAN_APPROVAL",
        }
        return assigned_id

    def get_pending_action(self) -> Optional[Dict[str, Any]]:
        """Returns current pending action requiring human signoff if present."""
        return self.pending_action

    def clear_pending_action(self) -> None:
        """Clears active pending action after human approval or rejection."""
        self.pending_action = None

    def record_filed_ticket(self, ticket: Dict[str, Any]) -> None:
        """Appends an approved and filed ticket into the session record."""
        self.filed_tickets.append(ticket)
        self.clear_pending_action()

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serializes session state to a dictionary."""
        return {
            "session_id": self.session_id,
            "active_drone_id": self.active_drone_id,
            "active_board_type": self.active_board_type,
            "active_test_id": self.active_test_id,
            "last_file_path": self.last_file_path,
            "history": self.history,
            "pending_action": self.pending_action,
            "filed_tickets": self.filed_tickets,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionStore":
        """Deserializes a dictionary into a SessionStore instance."""
        return cls(
            session_id=data.get("session_id", "default-session"),
            active_drone_id=data.get("active_drone_id"),
            active_board_type=data.get("active_board_type"),
            active_test_id=data.get("active_test_id"),
            last_file_path=data.get("last_file_path"),
            history=data.get("history", []),
            pending_action=data.get("pending_action"),
            filed_tickets=data.get("filed_tickets", []),
        )

    # -------------------------------------------------------------------------
    # Persistence & Caching Operations (Sync, Async, Background)
    # -------------------------------------------------------------------------

    @classmethod
    def get_or_create(cls, session_id: str) -> "SessionStore":
        """Retrieves an existing session from memory or Firestore, or creates a new one."""
        # 1. In-memory cache check (instantaneous read)
        if session_id in cls._memory_cache:
            return cls._memory_cache[session_id]

        # 2. Check Firestore if active
        if cls._db:
            try:
                collection = os.getenv("FIRESTORE_COLLECTION", "aeroeval_sessions")
                doc_ref = cls._db.collection(collection).document(session_id)
                doc = doc_ref.get()
                if doc.exists:
                    data = doc.to_dict() or {}
                    session = cls.from_dict(data)
                    cls._memory_cache[session_id] = session
                    return session
            except Exception as e:
                logger.warning(f"Failed to read session {session_id} from Firestore: {e}")

        # 3. Create fresh session
        new_session = cls(session_id=session_id)
        cls._memory_cache[session_id] = new_session
        return new_session

    def save(self, session: Optional["SessionStore"] = None) -> None:
        """Saves session state to memory cache and persists to Firestore.

        Supports both instance call `session.save()` and class/manager call `session_store.save(session)`.
        """
        target = session or self
        SessionStore._memory_cache[target.session_id] = target
        self._persist_to_firestore(target)

    async def save_async(self, session: Optional["SessionStore"] = None) -> None:
        """Asynchronously persists session state to Firestore without blocking the asyncio event loop.

        Executes the network I/O on a worker thread using asyncio.to_thread.
        """
        target = session or self
        SessionStore._memory_cache[target.session_id] = target
        await asyncio.to_thread(self._persist_to_firestore, target)

    def save_in_background(self, session: Optional["SessionStore"] = None) -> None:
        """Persists session state out-of-band on a background thread pool to prevent UI blocking."""
        target = session or self
        SessionStore._memory_cache[target.session_id] = target
        _persistence_executor.submit(self._persist_to_firestore, target)

    def _persist_to_firestore(self, target: "SessionStore") -> None:
        """Internal worker persisting state dictionary to Firestore."""
        if SessionStore._db:
            try:
                collection = os.getenv("FIRESTORE_COLLECTION", "aeroeval_sessions")
                doc_ref = SessionStore._db.collection(collection).document(target.session_id)
                doc_ref.set(target.to_dict())
                logger.debug(f"Persisted session '{target.session_id}' to Firestore collection '{collection}'.")
            except Exception as e:
                logger.warning(f"Failed to persist session {target.session_id} to Firestore: {e}")

    @classmethod
    def clear(cls, session_id: Optional[str] = None) -> None:
        """Clears a session from memory cache and Firestore."""
        sid = session_id or "default-session"
        if sid in cls._memory_cache:
            del cls._memory_cache[sid]

        if cls._db:
            try:
                collection = os.getenv("FIRESTORE_COLLECTION", "aeroeval_sessions")
                cls._db.collection(collection).document(sid).delete()
            except Exception as e:
                logger.warning(f"Failed to delete session {sid} from Firestore: {e}")


# Backwards-compatible alias for evaluator & existing imports
SessionState = SessionStore

# Global session manager instance
session_store = SessionStore()
