"""
Model configuration for Inspect AI evals.

Loads .env, exports SYSTEM_PROMPT, RESUME, MODEL, CEREBRAS_MODELS, JUDGE_MODEL.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
load_dotenv()

EXPERIMENTS_DIR = Path(__file__).parent.parent
SYSTEM_PROMPT = (EXPERIMENTS_DIR / "system_prompt.txt").read_text() if (EXPERIMENTS_DIR / "system_prompt.txt").exists() else ""
RESUME = (EXPERIMENTS_DIR / "resume.txt").read_text() if (EXPERIMENTS_DIR / "resume.txt").exists() else ""

# MODEL = "openai-api/cerebras/gpt-oss-120b"
MODEL = "openai-api/cerebras/zai-glm-4.7"

CEREBRAS_MODELS = [
    "openai-api/cerebras/llama3.1-8b",
    "openai-api/cerebras/llama-3.3-70b",
    "openai-api/cerebras/gpt-oss-120b",
    "openai-api/cerebras/qwen-3-32b",
    "openai-api/cerebras/qwen-3-235b-a22b-instruct-2507",
    "openai-api/cerebras/zai-glm-4.7",
]

JUDGE_MODEL = "anthropic/claude-sonnet-4-5-20250929"
