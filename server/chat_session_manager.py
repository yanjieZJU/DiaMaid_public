import os
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict

# Chat sessions directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_SESSIONS_DIR = os.path.join(BASE_DIR, 'data', 'chat_sessions')

# Ensure the directory exists
os.makedirs(CHAT_SESSIONS_DIR, exist_ok=True)

def _get_chat_session_file_path(session_id: str) -> str:
    """Gets the file path for a given chat session ID."""
    return os.path.join(CHAT_SESSIONS_DIR, f"{session_id}.json")

def _load_chat_session(session_id: str) -> Optional[dict]:
    """Loads a chat session from a JSON file. Returns None if it doesn't exist."""
    file_path = _get_chat_session_file_path(session_id)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return None

def _save_chat_session(session_data: dict):
    """Saves the chat session data to a JSON file."""
    session_id = session_data["id"]
    file_path = _get_chat_session_file_path(session_id)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

def create_chat_session(session_id: str, title: str = "") -> dict:
    """Creates a new chat session."""
    session_data = {
        "id": session_id,
        "startedAt": datetime.now().isoformat(),
        "title": title,
        "turns": []
    }
    _save_chat_session(session_data)
    return session_data

def save_chat_turn(session_id: str, turn: dict):
    """Appends a turn to the chat session."""
    session_data = _load_chat_session(session_id)
    if not session_data:
        session_data = create_chat_session(session_id)

    session_data["turns"].append(turn)

    # Auto-set title from first user message if not set
    if not session_data.get("title"):
        first_user = next((t for t in session_data["turns"] if t.get("role") == "user"), None)
        if first_user:
            content = first_user.get("content", "")
            session_data["title"] = content[:40] if content else "Untitled"

    _save_chat_session(session_data)

def get_chat_session(session_id: str) -> Optional[dict]:
    """Retrieves a chat session by ID."""
    return _load_chat_session(session_id)

def list_chat_sessions() -> List[dict]:
    """Lists all chat sessions, sorted by startedAt (newest first)."""
    sessions = []
    for filename in os.listdir(CHAT_SESSIONS_DIR):
        if filename.endswith('.json'):
            session_id = filename[:-5]
            session = _load_chat_session(session_id)
            if session:
                sessions.append(session)

    # Sort by startedAt descending
    sessions.sort(key=lambda s: s.get("startedAt", ""), reverse=True)
    return sessions

def delete_chat_session(session_id: str) -> bool:
    """Deletes a chat session."""
    file_path = _get_chat_session_file_path(session_id)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False

def rename_chat_session(session_id: str, new_title: str) -> bool:
    """Renames a chat session."""
    session_data = _load_chat_session(session_id)
    if not session_data:
        return False

    session_data["title"] = new_title
    _save_chat_session(session_data)
    return True
