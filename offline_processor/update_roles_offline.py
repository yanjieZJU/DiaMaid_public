import os
import json
import httpx
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- Configuration ---
DATA_DIR = "./case_llm_artifact/json_files"
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
# Taxonomy Definition
TAXONOMY = """
1. **user_entity**
    - primary_user: the user who is the final user of the system
    - secondary_user: stakeholder, observer or assistant, e.g., instructor, researcher
    - agents: non-human agents that can running independently (outside the system), e.g., robots, collaborative AI
2. **sensing_input**
    - bio_sensor: internal human physiological signals, e.g., EEG, EMG, ECG
    - behavior_sensor: external human behaviors/movements. e.g., IMU, eye tracker, motion capture
    - env_sensor: environment perception, e.g., RGB Camera, temperature sensor
    - direct_input: direct human-computer interaction input based on interfaces, e.g., Keyboard, Mouse, Touchscreen
    - system_context: equipment, system status and application log, etc.
    - data_import: offline documents, knowledge base, third-party API stream that serve as inputs
3. **processing_core**
    - signal_optimization: making the signal cleaner without changing the data type, e.g., filter, normalizer
    - feature_extraction: convert raw high-dimensional data (images, audio streams) into structured features or symbolic representations, e.g., gaze tracker, pose estimator, speech-to-text
    - state_assessment: qualitative or quantitative evaluation of the extracted features (judge how good/bad/intense something is), e.g., emotion recognition, fatigue detection
    - decision_logic: determine the next step based on the state assessment, e.g., recommender logic, dialog manager (even if it uses an LLM, if the output is a control flag or strategy, it belongs here)
    - content_generation: e.g., text generation (llm, chatbot), image generation (diffusion model)
4. **interface_output**
    - immersive_display: immersive visual output, e.g., VR headset, AR glasses, Unity Scene
    - gui_display: traditional visual display, e.g., Web UI, Mobile screen
    - haptic_feedback: e.g., Vibration motors, exoskeleton feedback
    - audio_output: e.g., TTS, Sound effects
5. **infrastructure**
    - data_storage: storage unit, e.g., vector DB, Database, Knowledge graph
    - communication: e.g., HTTP Server, WebSocket, Bluetooth
    - model_lifecycle: model repository/version, training and post-training pipeline, online learning, etc. 
    - personalization_store: user persona, prefenrence, threshold / calibration parameters, etc.
"""

http_client = httpx.Client(proxy=None, trust_env=False)
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=http_client)

def classify_role(module_name, function_desc, original_role):
    prompt = f"""
    You are a data labeling assistant.
    ## Taxonomy
    {TAXONOMY}
    
    ## Task
    Classify the following module into one of the Level 2 categories from the taxonomy above.
    
    Module Name: {module_name}
    Original Role: {original_role}
    Function Description: {function_desc}
    
    ## Output Format
    Return ONLY a JSON object with the following format:
    {{
        "role": "level1_category: level2_category" 
    }}
    Example: {{"role": "processing_core: ai_inference"}}
    If you are unsure, choose the closest match. Ensure the output is strictly lowercase.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("role", "unknown")
    except Exception as e:
        print(f"Error classifying {module_name}: {e}")
        return "unknown"

def process_files():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found.")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
    print(f"Found {len(files)} JSON files. Starting batch processing...")

    for i, filename in enumerate(files):
        filepath = os.path.join(DATA_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            modules = data.get("modules", [])
            modified = False
            
            print(f"[{i+1}/{len(files)}] Processing {filename} ({len(modules)} modules)...")
            
            for mod in modules:
                # Basic check to avoid re-processing if already in new format (this is a heuristic)
                # But user asked to re-process, so we simply run it.
                new_role = classify_role(
                    mod.get("mod_id", ""), 
                    mod.get("function_description", ""), 
                    mod.get("role", "")
                )
                
                print(f"   - {mod.get('mod_id')}: {mod.get('role')} -> {new_role}")
                mod["role"] = new_role
                modified = True
            
            if modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print("   Saved.")
                
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

if __name__ == "__main__":
    process_files()
