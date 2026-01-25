"""
Shared PEFT ModelAPI for all exercises.

This module provides the custom ModelAPI that wraps your DPO-trained model.
Import this in each exercise to use your local model.

Usage:
    from model_api import MODEL_NAME, CHECKPOINT_PATH

    # The peft/ model prefix is registered on import.
    # Then use in inspect eval:
    # Base model:  --model peft/Qwen/Qwen2-0.5B-Instruct
    # DPO model:   --model peft/Qwen/Qwen2-0.5B-Instruct --model-args checkpoint_path=/path/to/checkpoint
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
    GenerateConfig,
    ModelOutput,
    modelapi,
    ModelAPI,
)

# Paths
CURRENT_DIR = Path(__file__).parent.parent  # Up to finetune/
CHECKPOINT_DIR = CURRENT_DIR / "checkpoints/qwen-instruct-lora-dpo"
SYSTEM_PROMPT_PATH = CURRENT_DIR.parent / "system_prompt.txt"
RESUME_PATH = CURRENT_DIR.parent / "resume.txt"

# Model config
MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
CHECKPOINT_NUM = 126  # Your best checkpoint
CHECKPOINT_PATH = str(CHECKPOINT_DIR / f"checkpoint-{CHECKPOINT_NUM}")

# Load context files
SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text() if SYSTEM_PROMPT_PATH.exists() else ""
RESUME = RESUME_PATH.read_text() if RESUME_PATH.exists() else ""


class PeftModelAPI(ModelAPI):
    """Custom ModelAPI for PEFT/LoRA models.

    This wraps your DPO-trained model for use with Inspect.

    Args:
        model_name: Base model name (default: Qwen/Qwen2-0.5B-Instruct)
        checkpoint_path: Path to PEFT checkpoint. If None, uses base model.
        device: Device to load model on (default: cuda)
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        checkpoint_path: str | None = None,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__(model_name, base_url, api_key, api_key_vars, config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=device
        )

        if checkpoint_path:
            self.model = PeftModel.from_pretrained(self.base_model, checkpoint_path)
            self.model.eval()
            self._is_dpo = True
        else:
            self.model = self.base_model
            self._is_dpo = False

        self._model_name = model_name
        self._checkpoint_path = checkpoint_path

    async def generate(
        self,
        input: list,
        tools: list = [],
        tool_choice=None,
        config: GenerateConfig = GenerateConfig(),
        **kwargs
    ) -> ModelOutput:
        """Generate a response from the model."""
        # Convert Inspect messages to chat format
        messages = []
        for msg in input:
            if isinstance(msg, ChatMessageSystem):
                messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, ChatMessageUser):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, ChatMessageAssistant):
                messages.append({"role": "assistant", "content": msg.content})

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize and generate
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.max_tokens or 256,
                do_sample=True,
                temperature=config.temperature or 0.7,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response (skip input tokens)
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return ModelOutput.from_content(model=self._model_name, content=response)


# Register the model API with Inspect
# The decorator goes on a function that RETURNS the class
@modelapi(name="peft")
def peft_model_api():
    """Register PEFT model provider."""
    return PeftModelAPI


# For backwards compatibility
def register_peft_model():
    """No-op for backwards compatibility. Model is registered on import."""
    pass


if __name__ == "__main__":
    print("PEFT ModelAPI for Inspect")
    print()
    print(f"Base model: {MODEL_NAME}")
    print(f"DPO checkpoint: {CHECKPOINT_PATH}")
    print()
    print("Usage in exercises:")
    print("  from model_api import MODEL_NAME, CHECKPOINT_PATH")
    print()
    print("CLI usage:")
    print("  # Base model:")
    print(f"  inspect eval ex1.py --model peft/{MODEL_NAME}")
    print()
    print("  # DPO model:")
    print(f"  inspect eval ex1.py --model peft/{MODEL_NAME} --model-args checkpoint_path={CHECKPOINT_PATH}")
