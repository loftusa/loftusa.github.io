# Inverse-Length Weighted Fine-tuning with HuggingFace Trainer

Fine-tune TinyLlama to produce shorter responses by weighting loss inversely by response length.

## Architecture

```
experiments/finetune/
├── dataset.py      # ChatDataset + SyntheticGenerator
├── collator.py     # DataCollatorWithLengths  
├── trainer.py      # LengthWeightedTrainer (subclass)
└── train.py        # Main script with TrainingArguments
```

---

## Module 1: `dataset.py`

**Two data sources:**

1. `chat_logs.jsonl` - real interactions: `{user_message, bot_response, ...}`
2. Synthetic QA pairs from `resume.txt`

**Each example returns:**

```python
{
    "input_ids": Tensor,      # [prompt + response] tokenized
    "labels": Tensor,         # [-100 for prompt, token_ids for response]
    "response_length": int    # token count in response (for weighting)
}
```

**Key detail:** Mask prompt tokens in labels with `-100` so loss only computes on response.

---

## Module 2: `collator.py`

Batches variable-length sequences:

- Left-pad `input_ids` (causal LM convention)
- Left-pad `labels` with `-100`
- Stack `response_length` into tensor

```python
@dataclass
class DataCollatorWithLengths:
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, features: list[dict]) -> dict:
        # Returns: {"input_ids", "attention_mask", "labels", "response_lengths"}
```

---

## Module 3: `trainer.py`

Override `compute_loss` for inverse-length weighting:

```python
class LengthWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        response_lengths = inputs.pop("response_lengths")  # (batch,)
        outputs = model(**inputs)
        
        # Per-example cross-entropy (not averaged)
        logits = outputs.logits[..., :-1, :]  # (batch, seq-1, vocab)
        labels = inputs["labels"][..., 1:]     # (batch, seq-1)
        
        loss_fct = CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(logits.reshape(-1, vocab_size), labels.reshape(-1))
        per_token_loss = per_token_loss.view(batch_size, -1)
        
        # Mean per example, then weight inversely by length
        valid_tokens = (labels != -100).sum(dim=1)
        per_example_loss = per_token_loss.sum(dim=1) / valid_tokens
        
        weights = 1.0 / response_lengths.float()
        weights = weights / weights.sum() * batch_size  # normalize to preserve LR scale
        
        loss = (per_example_loss * weights).mean()
        return (loss, outputs) if return_outputs else loss
```

---

## Module 4: `train.py`

1. Load TinyLlama + LoRA (PEFT) for memory efficiency
2. Create combined dataset (chat logs + synthetic)
3. Configure `TrainingArguments`
4. Run `LengthWeightedTrainer.train()`

---

## Data Sources

**Chat logs** (from `chat_logs.jsonl`):

```json
{"user_message": "Where did Alex go to school?", "bot_response": "Johns Hopkins for masters, WWU for undergrad."}
```

**Synthetic** (generated from `resume.txt` sections):

- "What was Alex's role at Creyon Bio?" → "Data Scientist"
- "What awards has Alex won?" → "Kaggle Vesuvius 1st place, NeurIPS LatinX Best Poster"

---

## Interview Skills Covered

| Skill | Module |
|-------|--------|
| Custom `Dataset` class | `dataset.py` |
| Tokenization + label masking | `dataset.py` |
| `DataCollator` padding logic | `collator.py` |
| Subclassing `Trainer` | `trainer.py` |
| Per-example loss computation | `trainer.py` |
| `CrossEntropyLoss` internals | `trainer.py` |
| LoRA/PEFT configuration | `train.py` |
| `TrainingArguments` tuning | `train.py` |
