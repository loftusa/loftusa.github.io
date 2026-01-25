from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer
from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch

current_dir = Path(__file__).parent
MODEL_NAME = "Qwen/Qwen2-32B-Instruct"
dataset_path = current_dir / "local_datasets/dpo_combined.jsonl"
OUTPUT_DIR = current_dir / "checkpoints/qwen32b-instruct-lora-dpo"


def main():
    assert dataset_path.exists(), "Dataset file does not exist"

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR, 
        do_train=True,
        num_train_epochs=1000,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=False,
        logging_strategy="epoch",
        # beta=0.1,
        # max_length=1024,
        # max_prompt_length=512,
        # per_device_train_batch_size=2,
        # gradient_accumulation_steps=4
    )
    trainer = DPOTrainer(
        model=model, 
        args=dpo_config,
        train_dataset=dataset, 
        processing_class=tokenizer
    )

    trainer.train(resume_from_checkpoint=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()