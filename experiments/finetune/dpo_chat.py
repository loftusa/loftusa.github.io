from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

from dpo_train import OUTPUT_DIR, MODEL_NAME

# Load system prompt and resume context (same as chat_api.py)
SYSTEM_PROMPT = (Path(__file__).parent.parent / "system_prompt.txt").read_text()
RESUME = (Path(__file__).parent.parent / "resume.txt").read_text()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
CKPT_NUM = 1
CHECKPOINT_FMT = str(OUTPUT_DIR / "checkpoint-{}")
CHECKPOINT = CHECKPOINT_FMT.format(CKPT_NUM)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(base_model, CHECKPOINT)
model.eval()

# loop that checks for adapter

print(f"Loaded checkpoint: {CHECKPOINT}")
print("Type 'quit' to exit, 'base' for base model, 'ckpt#' for a particular checkpoint.")

use_adapter = True

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == 'quit':
        break
    if user_input.lower() == 'base':
        use_adapter = not use_adapter
        print(f"{'Using adapter' if use_adapter else 'Using base model'}")
        continue
    if user_input.lower().startswith('ckpt'):
        num = user_input.lower()[4:]
        CHECKPOINT = CHECKPOINT_FMT.format(num)
        model = PeftModel.from_pretrained(base_model, CHECKPOINT)
        model.eval()
        use_adapter = True
        print(f"Loaded checkpoint: {CHECKPOINT}")
        continue

    if use_adapter:
        model.enable_adapter_layers()
    else:
        model.disable_adapter_layers()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": RESUME},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print(f"Assistant: {response}\n")