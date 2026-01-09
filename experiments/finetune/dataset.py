from pathlib import Path
import dotenv
import os

from transformers import AutoTokenizer, PreTrainedTokenizer
import torch
from torch.utils.data import Dataset
from torch import Tensor

from datasets import load_dataset


dotenv.load_dotenv()
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOG_PATH = Path(os.getenv("LOG_PATH"))


def tokenize_chat_example(
    user_message: str,
    bot_response: str,
    tokenizer: PreTrainedTokenizer,
) -> dict[str, Tensor | int]:
    """
    Returns:
        input_ids: (seq_len,) - full conversation tokenized
        labels: (seq_len,) - prompt tokens masked with -100, response tokens exposed
        response_length: int - number of tokens in response
    """

    message_enc = tokenizer.apply_chat_template(user_message, return_tensors='pt').squeeze(0)
    response_enc = tokenizer.apply_chat_template(bot_response, return_tensors='pt', add_special_tokens=False).squeeze(0)
    eos = torch.tensor([tokenizer.eos_token_id], dtype=response_enc.dtype)
    response_enc = torch.cat((response_enc, eos))

    input_ids = torch.cat((message_enc, response_enc), dim=0)
    labels = torch.cat((torch.full_like(message_enc, fill_value=-100), response_enc), dim=0)

    response_length = response_enc.shape[0]
    
    return {'input_ids':input_ids, 'labels':labels, 'response_length':response_length}


class ChatDataset(Dataset):
    def __init__(self, path: Path|str, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.dataset = load_dataset('json', data_files=str(path), split='train')
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        dct = self.dataset[idx]
        return tokenize_chat_example(
            user_message=dct['user_message'],
            bot_response=dct['bot_response'], 
            tokenizer=self.tokenizer
        )


if __name__ == "__main__":
    assert LOG_PATH.exists()

    # model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, device='cuda')
    # dataset = load_dataset('json', data_files=str(LOG_PATH), split='train')

    dataset = ChatDataset(LOG_PATH, tokenizer=tokenizer)

    print(dataset[0])

