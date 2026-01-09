import os
from pathlib import Path
import dotenv
dotenv.load_dotenv()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

from dataset import ChatDataset, MODEL_NAME
from collator import DataCollatorWithLengths
from trainer import LengthWeightedTrainer

def main():
    LOG_PATH = Path(os.getenv("LOG_PATH"))
    OUTPUT_DIR = "./tinyllama-chat-lora"

    # model+tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map='auto',
    )

    # dora
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        use_dora=True
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # dataset+collator
    dataset = ChatDataset(LOG_PATH, tokenizer)
    collator = DataCollatorWithLengths(tokenizer)

    # train args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
    )

    # trainer
    trainer = LengthWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()