import sqlite3
import os
import json
import uuid
from datetime import datetime

DB_NAME = "diamaid.db"

def get_db_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, DB_NAME)

def init_db():
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    # Plans Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS plans (
        id TEXT PRIMARY KEY,
        original_query TEXT,
        current_description TEXT,
        current_arch_json TEXT,
        perspective TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # History Table (Chat log for each plan)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plan_id TEXT,
        role TEXT, -- 'user' or 'system'
        content TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(plan_id) REFERENCES plans(id)
    )
    ''')

    conn.commit()
    conn.close()

def create_plan(original_query, perspective, description, arch_json):
    plan_id = str(uuid.uuid4())
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    # Store initial plan
    cursor.execute('''
        INSERT INTO plans (id, original_query, perspective, current_description, current_arch_json)
        VALUES (?, ?, ?, ?, ?)
    ''', (plan_id, original_query, perspective, description, json.dumps(arch_json)))
    
    # Add initial system message
    cursor.execute('''
        INSERT INTO history (plan_id, role, content)
        VALUES (?, ?, ?)
    ''', (plan_id, 'system', f"初始方案已生成: {perspective}"))
    
    conn.commit()
    conn.close()
    return plan_id

def update_plan(plan_id, description, arch_json):
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE plans 
        SET current_description = ?, current_arch_json = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    ''', (description, json.dumps(arch_json), plan_id))
    
    conn.commit()
    conn.close()

def add_history(plan_id, role, content):
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO history (plan_id, role, content)
        VALUES (?, ?, ?)
    ''', (plan_id, role, content))
    
    conn.commit()
    conn.close()

def get_plan_history(plan_id):
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT role, content, timestamp FROM history
        WHERE plan_id = ?
        ORDER BY timestamp ASC
    ''', (plan_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            "role": r[0],
            "content": r[1],
            "timestamp": r[2]
        })
    return history

# Initialize on import
init_db()
