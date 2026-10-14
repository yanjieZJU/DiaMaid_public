import os
import json
import uuid
from datetime import datetime

# Assume data dir is relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, 'data', 'sessions')

# Ensure the directory exists
os.makedirs(SESSIONS_DIR, exist_ok=True)

def _get_session_file_path(session_id: str) -> str:
    """Gets the file path for a given session ID."""
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def _load_session(session_id: str) -> dict:
    """Loads a session from a JSON file. Returns an empty structure if it doesn't exist."""
    file_path = _get_session_file_path(session_id)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass
    return {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "nodes": {}  # Dictionary of node_id -> node_data
    }

def _save_session(session_data: dict):
    """Saves the session data to a JSON file."""
    session_id = session_data["session_id"]
    file_path = _get_session_file_path(session_id)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

def create_initial_plan(session_id: str, original_query: str, perspective: str, description: str, arch_json: dict) -> str:
    """
    Creates an initial plan node for a session (a root node with no parent).
    If no session_id is provided, generates an anonymous one.
    """
    if not session_id:
        session_id = "anonymous_session"
        
    session_data = _load_session(session_id)
    
    node_id = str(uuid.uuid4())
    
    node_data = {
        "id": node_id,
        "parent_id": None,
        "original_query": original_query,
        "perspective": perspective,
        "description": description,
        "arch_json": arch_json,
        "created_at": datetime.now().isoformat(),
        "history": [
             {"role": "system", "content": f"初始方案已生成: {perspective}", "timestamp": datetime.now().timestamp() * 1000}
        ]
    }
    
    session_data["nodes"][node_id] = node_data
    _save_session(session_data)
    
    return node_id

def create_modified_plan(session_id: str, parent_node_id: str, user_instruction: str, original_query: str, modified_description: str, modified_arch_json: dict, perspective_prefix: str = "(已修改)", editor_memory: dict = None) -> str:
    """
    Creates a new plan node derived from a parent node.

    `editor_memory`: optional snapshot from the EditorAgent's memory subsystem,
    persisted on the new node so the next edit on this branch can rehydrate it.
    """
    if not session_id:
        session_id = "anonymous_session"

    session_data = _load_session(session_id)

    # Retrieve parent perspective to inherit/modify
    parent_perspective = "Standard"
    if parent_node_id and parent_node_id in session_data["nodes"]:
        parent_node = session_data["nodes"][parent_node_id]
        parent_perspective = parent_node.get("perspective", "Standard")

        # If it's already marked as modified, don't prefix again
        if perspective_prefix in parent_perspective:
            new_perspective = parent_perspective
        else:
            new_perspective = f"{perspective_prefix} {parent_perspective}"
    else:
        new_perspective = f"{perspective_prefix} 方案"

    new_node_id = str(uuid.uuid4())

    node_data = {
        "id": new_node_id,
        "parent_id": parent_node_id,
        "original_query": original_query,
        "user_instruction": user_instruction,
        "perspective": new_perspective,
        "description": modified_description,
        "arch_json": modified_arch_json,
        "created_at": datetime.now().isoformat(),
        "history": [
            {"role": "user", "content": user_instruction, "timestamp": datetime.now().timestamp() * 1000},
            {"role": "system", "content": f"修改完成: {modified_description[:50]}...", "timestamp": datetime.now().timestamp() * 1000}
        ]
    }
    if isinstance(editor_memory, dict) and editor_memory:
        node_data["editor_memory"] = editor_memory

    session_data["nodes"][new_node_id] = node_data
    _save_session(session_data)

    return new_node_id

def get_session_tree(session_id: str) -> dict:
    """Returns the entire session tree data."""
    if not session_id:
        return {}
    return _load_session(session_id)

def get_plan_lineage(session_id: str, plan_id: str) -> list:
    """
    Returns the ordered list of plan nodes from the root to the specified plan_id.
    This reconstructs the evolution history of a specific branch.
    """
    session_data = _load_session(session_id)
    if not session_data or "nodes" not in session_data:
        return []
    
    nodes = session_data["nodes"]
    lineage = []
    current_id = plan_id
    
    # Traverse backwards from current node to root
    while current_id and current_id in nodes:
        node = nodes[current_id]
        lineage.append(node)
        current_id = node.get("parent_id")
        
    # Reverse to get chronological order (Root -> Current)
    return lineage[::-1]
