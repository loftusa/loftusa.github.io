from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from pathlib import Path
from datasets import load_dataset
from dataset import MODEL_NAME
from peft import LoraConfig, get_peft_model
import torch

current_dir = Path(__file__).parent
print(current_dir)
dataset_path = current_dir / "local_datasets/dpo_combined.jsonl"
assert dataset_path.exists(), "Dataset file does not exist"

OUTPUT_DIR = "./qwen-instruct-lora-dpo"


lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_dora=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map='auto')
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
dataset = load_dataset("json", data_files=str(dataset_path), split="train")

dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR, 
    do_train=True,
    num_train_epochs=1,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    beta=0.1,
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4
    )
trainer = DPOTrainer(
    model=model, 
    args=dpo_config,
    train_dataset=dataset, 
    processing_class=tokenizer
    )

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)