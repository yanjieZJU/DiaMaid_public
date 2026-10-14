"""
Bootstrap: load environment variables from .env at the project root and expose
Python-level constants for downstream modules.

History
-------
This file used to hardcode credentials and set them via os.environ[...]. They
now live in .env at the project root. We keep this module so that:
  - offline_processor/*.py can continue to do `from config import *`
  - tools/scripts can run `python config.py` to smoke-test the LLM client
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (this file's directory). Idempotent — safe
# to call from multiple modules.
_DOTENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_DOTENV_PATH)


# === Multi-Key Pool (round-robin load balancing) ===
# Either set OPENAI_API_KEYS in .env as a comma-separated list, or fall back
# to the single OPENAI_API_KEY. All keys must work with the same
# OPENAI_BASE_URL.
def _parse_keys() -> list:
    multi = os.environ.get("OPENAI_API_KEYS", "").strip()
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = (os.environ.get("OPENAI_API_KEY") or "").strip()
    return [single] if single else []


OPENAI_API_KEYS = _parse_keys()

# === Model Configuration ===
CHAT_MODEL = os.environ.get("CHAT_MODEL", "deepseek-v3.2")

# === Neo4j (kept for `from config import *` consumers) ===
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
NEO4J_URL = os.environ.get("NEO4J_URL")


# === API KEY TEST ===
if __name__ == "__main__":
    from openai import OpenAI
    import httpx

    if not OPENAI_API_KEYS:
        raise SystemExit(
            "No OpenAI API key configured. Set OPENAI_API_KEY (or "
            "comma-separated OPENAI_API_KEYS) in .env at the project root."
        )

    http_client = httpx.Client(proxy=None, trust_env=False)
    client = OpenAI(
        http_client=http_client,
        api_key=OPENAI_API_KEYS[0],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    print(client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "How are you?"}],
        temperature=0.2,
    ).choices[0].message.content)
