import dotenv
dotenv.load_dotenv()
import json
from pathlib import Path
import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from peft import PeftConfig, PeftModel
from typeguard import check_type, TypeCheckError
from typing import Any

OUTPUT_DIR = Path("./tinyllama-chat-lora")
assert OUTPUT_DIR.is_dir()
with (OUTPUT_DIR/"adapter_config.json").open('r') as f:
    MODEL_NAME = json.load(f)['base_model_name_or_path']
SYSTEM_PROMPT = (Path(__file__).parent.parent / "system_prompt.txt").read_text()
RESUME = (Path(__file__).parent.parent / "resume.txt").read_text()
EVAL_DATASET_PATH = Path(os.getenv("EVAL_DATASET_PATH"))

def isinstance(obj: Any, type: type) -> bool:
    try:
        check_type(obj, type)
        return True
    except TypeCheckError:
        return False

def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float16,
        device_map='auto'
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_lora_model(model: PreTrainedModel, lora_dir):
    config = PeftConfig.from_pretrained(lora_dir)
    model_ft = PeftModel.from_pretrained(model, lora_dir)
    return model_ft, config


def generate_response(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: list[dict[str, str]]):
    """
    `prompt` is pre-formatted into "role" and "content" standard format

    """
    model.eval()
    input_ids = tokenizer.apply_chat_template(prompt, return_tensors='pt').to(model.device)
    input_len = input_ids.shape[1]
    assert input_ids.ndim == 2
    with torch.no_grad():
        output_ids = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id)
    output_ids = output_ids[0, input_len:]
    token_count = len(output_ids)
    return tokenizer.decode(output_ids, skip_special_tokens=True), token_count


def compare_models(prompt: str | list[str]):
    if isinstance(prompt, list[str]): 
        raise NotImplementedError
    inputs: list[dict[str, str]] = [
        # {"role": "system", "content": SYSTEM_PROMPT},
        # {"role": "system", "content": RESUME},
        {"role": "user", "content": prompt},
    ]
    
    # base model
    model, tokenizer = load_base_model()
    response, token_count = generate_response(model, tokenizer, inputs)
    print(f'BASE MODEL: {token_count} tokens', response, '\n\n')

    # finetuned model
    model_ft, _ = load_lora_model(model, OUTPUT_DIR)
    response_ft, token_count_ft = generate_response(model_ft, tokenizer, inputs)
    print(f'LORA MODEL: {token_count_ft} tokens', response_ft)

    print(f"TOKEN DIFFERENCE BASE-FT: {token_count - token_count_ft}")


if __name__ == "__main__":
    compare_models("Tell me about Alex's background.")