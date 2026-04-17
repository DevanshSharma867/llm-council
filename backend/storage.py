"""JSON-based storage for conversations."""

import json
import os
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from .config import DATA_DIR

logger = logging.getLogger(__name__)


def ensure_data_dir():
    """Ensure the data directory exists."""
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def validate_conversation_id(conversation_id: str) -> bool:
    """Validate that a conversation ID is a well-formed UUID (prevents path traversal)."""
    try:
        uuid.UUID(conversation_id)
        return True
    except (ValueError, AttributeError):
        return False


def get_conversation_path(conversation_id: str) -> str:
    """Get the file path for a conversation, with ID validation."""
    if not validate_conversation_id(conversation_id):
        raise ValueError(f"Invalid conversation ID: {conversation_id!r}")
    return os.path.join(DATA_DIR, f"{conversation_id}.json")


def _write_atomic(path: str, data: dict):
    """Write JSON data atomically via a temp file + rename."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)  # atomic on POSIX


def create_conversation(conversation_id: str) -> Dict[str, Any]:
    """Create a new conversation."""
    ensure_data_dir()

    conversation = {
        "id": conversation_id,
        "created_at": datetime.utcnow().isoformat(),
        "title": "New Conversation",
        "messages": [],
    }

    _write_atomic(get_conversation_path(conversation_id), conversation)
    return conversation


def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Load a conversation from storage."""
    try:
        path = get_conversation_path(conversation_id)
    except ValueError:
        logger.warning(f"Rejected invalid conversation ID: {conversation_id!r}")
        return None

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to read conversation {conversation_id}: {e}")
        return None


def save_conversation(conversation: Dict[str, Any]):
    """Save a conversation to storage (atomic write)."""
    ensure_data_dir()
    path = get_conversation_path(conversation["id"])
    _write_atomic(path, conversation)


def list_conversations() -> List[Dict[str, Any]]:
    """List all conversations (metadata only)."""
    ensure_data_dir()

    conversations = []
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".json") or filename.endswith(".tmp"):
            continue

        path = os.path.join(DATA_DIR, filename)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            conversations.append({
                "id": data["id"],
                "created_at": data["created_at"],
                "title": data.get("title", "New Conversation"),
                "message_count": len(data["messages"]),
            })
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.error(f"Corrupted conversation file {filename}: {e}")

    conversations.sort(key=lambda x: x["created_at"], reverse=True)
    return conversations


def add_user_message(conversation_id: str, content: str):
    """Add a user message to a conversation."""
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["messages"].append({
        "role": "user",
        "content": content,
    })

    save_conversation(conversation)


def add_assistant_message(
    conversation_id: str,
    stage1: List[Dict[str, Any]],
    stage2: List[Dict[str, Any]],
    stage3: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    stage2_5: Optional[Dict[str, Any]] = None,
):
    """
    Add an assistant message with all stages to a conversation.
    Metadata and debate results are persisted so the full view is available on reload.
    """
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    message: Dict[str, Any] = {
        "role": "assistant",
        "stage1": stage1,
        "stage2": stage2,
        "stage3": stage3,
    }
    if metadata:
        message["metadata"] = metadata
    if stage2_5:
        message["stage2_5"] = stage2_5

    conversation["messages"].append(message)
    save_conversation(conversation)


def update_conversation_title(conversation_id: str, title: str):
    """Update the title of a conversation."""
    conversation = get_conversation(conversation_id)
    if conversation is None:
        raise ValueError(f"Conversation {conversation_id} not found")

    conversation["title"] = title
    save_conversation(conversation)
