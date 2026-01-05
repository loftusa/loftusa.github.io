from transformers import Trainer, PreTrainedModel, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss

class LengthWeightedTrainer(Trainer):
    def compute_loss(self, model: PreTrainedModel, inputs: dict, return_outputs=False, **kwargs):
        response_lengths = inputs.pop("response_lengths")
        labels = inputs.pop("labels")
        outputs = model(**inputs)


        batch_size = outputs.logits.shape[0]
        loss_fn = CrossEntropyLoss(reduction="none")
        logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        per_token_loss = loss_fn(logits.view(-1, logits.shape[-1]), shift_labels.view(-1))  # returns (batch*(seq-q),)
        per_token_loss = per_token_loss.view(batch_size, -1)

        # per-example loss (mean over valid tokens)
        valid_mask = (shift_labels != -100)
        valid_counts = valid_mask.sum(dim=1)  # (batch,)
        per_example_loss = (per_token_loss * valid_mask).sum(dim=1) / valid_counts

        # inverse-length weighting
        weights = 1.0 / response_lengths.float()
        weights = weights / weights.sum() * batch_size

        loss = (per_example_loss * weights).mean()

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import torch
    import dotenv
    from pathlib import Path
    import os

    dotenv.load_dotenv()
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to('cuda')
    model.eval()

    # Create a fake batch (mimics what collator outputs)
    batch_size = 2
    seq_len = 20
    vocab_size = tokenizer.vocab_size

    # Fake input_ids and attention_mask
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
    attention_mask = torch.ones_like(input_ids, device='cuda')

    # Fake labels: first 10 tokens masked (-100), last 10 are targets
    labels = input_ids.clone()
    labels[:, :10] = -100  # mask prompt portion

    # Fake response_lengths
    response_lengths = torch.tensor([10, 10], device='cuda')  # each example has 10 response tokens

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_lengths": response_lengths,
    }

    # Test compute_loss (don't need full Trainer, just call the method)
    trainer = LengthWeightedTrainer(model=model, args=None)

    with torch.no_grad():
        loss, outputs = trainer.compute_loss(model, inputs, return_outputs=True)

    # Assertions
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss > 0, f"Loss should be positive, got {loss.item()}"

    print(f"✓ compute_loss works!")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits shape: {outputs.logits.shape}")




