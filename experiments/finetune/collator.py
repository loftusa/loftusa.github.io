from dataclasses import dataclass
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorWithLengths:
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: list[dict]) -> dict[str, Tensor]:
        """
        Args:
            features: list of dicts, each with:
                - input_ids: (seq_len_i,)
                - labels: (seq_len_i,)
                - response_length: int
        
        Returns:
            - input_ids: (batch, max_seq_len) - left-padded
            - attention_mask: (batch, max_seq_len)
            - labels: (batch, max_seq_len) - left-padded with -100
            - response_lengths: (batch,)
        """
        input_ids_list = [f['input_ids'] for f in features]
        labels_list = [f['labels'] for f in features]
        len_list = torch.tensor([f['response_length'] for f in features])

        max_len = max([ids.shape[0] for ids in input_ids_list])
        pad_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            pad_amount = max_len - input_ids.shape[0]
            
            # left padding
            padded_input_ids.append(torch.cat((torch.full((pad_amount,), fill_value=pad_id, dtype=input_ids.dtype), input_ids), dim=0))
            padded_labels.append(torch.cat((torch.full((pad_amount,), fill_value=-100, dtype=labels.dtype), labels), dim=0))
            attention_masks.append(torch.cat((torch.zeros((pad_amount,), dtype=torch.long), torch.ones_like(input_ids)), dim=0))

        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_masks),
            "response_lengths": len_list
        }



if __name__ == "__main__":
    from dataset import ChatDataset
    import dotenv
    from transformers import AutoTokenizer
    import os
    dotenv.load_dotenv()

    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LOG_PATH = os.getenv("LOG_PATH")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device='cuda')
    collator = DataCollatorWithLengths(tokenizer)

    ds = ChatDataset(LOG_PATH, tokenizer=tokenizer)
    features = [f for f in ds][:5]

    batch = collator(features)
    print(batch)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    print(f"✓ Batch shape: {batch['input_ids'].shape}")