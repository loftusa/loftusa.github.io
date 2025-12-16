import json
from pathlib import Path
import dotenv
import os
dotenv.load_dotenv()

from transformers import AutoModelForCausalLM, AutoTokenizerForCausalLM

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOG_PATH = Path(os.getenv("LOG_PATH"))

def iter_logs():
    with LOG_PATH.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

model = AutoModelForCausalLM(MODEL_NAME)
tokenizer = AutoTokenizerForCausalLM(MODEL_NAME)


if __name__ == "__main__":
    assert LOG_PATH.exists()
    print(LOG_PATH)
    first_log = next(iter_logs())
    print(type(first_log))
    print(first_log)