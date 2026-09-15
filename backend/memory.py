"""
Session and Context Management for AeroEval.

Tracks multi-turn conversation state, active hardware entities
(active_drone_id, active_circuit_board, active_test_id), and message history.
Supports native Google GenAI history format and optional persistence via Google Cloud Firestore.
"""

from dataclasses import asdict, dataclass, field
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Represents the context state for a single conversation session."""

    session_id: str
    active_drone_id: Optional[str] = None
    active_board_type: Optional[str] = None
    active_test_id: Optional[str] = None
    last_file_path: Optional[str] = None
    history: List[Dict[str, str]] = field(default_factory=list)

    def update_context(
        self,
        drone_id: Optional[str] = None,
        board_type: Optional[str] = None,
        test_id: Optional[str] = None,
        file_path: Optional[str] = None,
    ) -> None:
        """Updates the active hardware entity tracking."""
        if drone_id:
            self.active_drone_id = drone_id
        if board_type:
            self.active_board_type = board_type
        if test_id:
            self.active_test_id = test_id
        if file_path:
            self.last_file_path = file_path

    def add_message(self, role: str, content: str) -> None:
        """Appends a turn to conversation history."""
        # Normalize role: "assistant" and "model" both refer to AI response
        canonical_role = "user" if role == "user" else "model"
        self.history.append({"role": canonical_role, "content": content})

    def to_genai_history(self) -> List[Any]:
        """Converts stored conversation turns to google.genai types.Content objects.

        Used to initialize client.chats.create(history=...) for native multi-turn memory.
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

    def to_dict(self) -> Dict[str, Any]:
        """Serializes session state to a dictionary for Firestore or JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Deserializes a dictionary into a SessionState instance."""
        return cls(
            session_id=data.get("session_id", "default-session"),
            active_drone_id=data.get("active_drone_id"),
            active_board_type=data.get("active_board_type"),
            active_test_id=data.get("active_test_id"),
            last_file_path=data.get("last_file_path"),
            history=data.get("history", []),
        )


class SessionStore:
    """Session manager with in-memory caching and optional Google Cloud Firestore persistence."""

    def __init__(self):
        self._memory_cache: Dict[str, SessionState] = {}
        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT_ID", "onboardingproject-507522")
        self.database_name = os.getenv("FIRESTORE_DATABASE", "aeroeval")
        self.collection_name = os.getenv("FIRESTORE_COLLECTION", "aeroeval_sessions")
        self.use_firestore = os.getenv("USE_FIRESTORE", "false").lower() in ("true", "1")
        self.db = None

        if self.use_firestore:
            self._init_firestore()

    def _init_firestore(self) -> None:
        """Initializes the Firestore client if google-cloud-firestore is available."""
        try:
            from google.cloud import firestore
            self.db = firestore.Client(
                project=self.project_id,
                database=self.database_name,
            )
            logger.info(
                f"Connected to Firestore (Project: {self.project_id}, "
                f"Database: {self.database_name}, Collection: {self.collection_name})"
            )
        except Exception as e:
            logger.warning(
                f"Firestore connection failed: {e}. Falling back to in-memory session store."
            )
            self.db = None

    def get_or_create(self, session_id: str) -> SessionState:
        """Retrieves an existing session from memory or Firestore, or creates a new one."""
        # 1. Check local cache
        if session_id in self._memory_cache:
            return self._memory_cache[session_id]

        # 2. Check Firestore if active
        if self.db:
            try:
                doc_ref = self.db.collection(self.collection_name).document(session_id)
                doc = doc_ref.get()
                if doc.exists:
                    data = doc.to_dict() or {}
                    session = SessionState.from_dict(data)
                    self._memory_cache[session_id] = session
                    return session
            except Exception as e:
                logger.warning(f"Failed to read session {session_id} from Firestore: {e}")

        # 3. Create fresh session
        new_session = SessionState(session_id=session_id)
        self._memory_cache[session_id] = new_session
        return new_session

    def save(self, session: SessionState) -> None:
        """Saves session state to memory cache and asynchronously/synchronously to Firestore."""
        self._memory_cache[session.session_id] = session

        if self.db:
            try:
                doc_ref = self.db.collection(self.collection_name).document(session.session_id)
                doc_ref.set(session.to_dict())
            except Exception as e:
                logger.warning(f"Failed to persist session {session.session_id} to Firestore: {e}")

    def clear(self, session_id: str) -> None:
        """Clears a session from memory and Firestore."""
        if session_id in self._memory_cache:
            del self._memory_cache[session_id]

        if self.db:
            try:
                self.db.collection(self.collection_name).document(session_id).delete()
            except Exception as e:
                logger.warning(f"Failed to delete session {session_id} from Firestore: {e}")


# Global session manager instance
session_store = SessionStore()
