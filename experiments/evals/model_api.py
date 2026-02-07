"""
Cerebras model configuration for Inspect AI.

Cerebras has an OpenAI-compatible API at https://api.cerebras.ai/v1.
This module sets up the environment so Inspect's built-in openai/ provider
works with Cerebras directly — no custom ModelAPI needed.

On import, this module:
1. Loads CEREBRAS_API_KEY from experiments/.env
2. Sets OPENAI_API_KEY and OPENAI_BASE_URL for Inspect's openai provider
3. Exports SYSTEM_PROMPT, RESUME, MODEL constants

Usage:
    import model_api  # sets up env vars
    from model_api import SYSTEM_PROMPT, RESUME, MODEL

    # Then use Inspect's built-in openai/ provider:
    inspect eval ex1.py --model openai/gpt-oss-120b
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load env from experiments/.env
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
load_dotenv()

# Map Cerebras credentials to OpenAI-compatible env vars
# Cerebras API is OpenAI-compatible at https://api.cerebras.ai/v1
_cerebras_key = os.getenv("CEREBRAS_API_KEY", "")
if _cerebras_key and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = _cerebras_key
if not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "https://api.cerebras.ai/v1"

# Paths
EXPERIMENTS_DIR = Path(__file__).parent.parent  # experiments/
SYSTEM_PROMPT_PATH = EXPERIMENTS_DIR / "system_prompt.txt"
RESUME_PATH = EXPERIMENTS_DIR / "resume.txt"

# Load context files
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else ""
RESUME = RESUME_PATH.read_text() if RESUME_PATH.exists() else ""

# Model config — use openai/ prefix since Cerebras is OpenAI-compatible
MODEL_NAME = "gpt-oss-120b"
MODEL = f"openai/{MODEL_NAME}"


if __name__ == "__main__":
    print("Cerebras via OpenAI-compatible API for Inspect")
    print()
    print(f"Model: {MODEL}")
    print(f"Base URL: {os.getenv('OPENAI_BASE_URL')}")
    print(f"API key set: {bool(os.getenv('OPENAI_API_KEY'))}")
    print(f"System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"Resume: {len(RESUME)} chars")
    print()
    print("CLI usage:")
    print(f"  inspect eval ex1.py --model {MODEL}")
