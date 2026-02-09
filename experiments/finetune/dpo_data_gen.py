from anthropic import Anthropic
import os
import json
from datasets import load_dataset
from pathlib import Path
import dotenv
dotenv.load_dotenv()
from litellm import completion
from tqdm import tqdm

if __name__ == "__main__":
    # PROMPT = "Here is a user message: `{}`. \n An LLM had this response: `{}` \n Rewrite this response to be maximally concise while preserving all key information. Say and do nothing else."
    PROMPT = "Rewrite this response to be maximally concise while preserving all key information. If the response "
    LOG_PATH = Path(os.getenv("LOG_PATH"))
    api_key = os.getenv("ANTHROPIC_API_KEY")

    dataset = load_dataset('json', data_files=str(LOG_PATH), split='train')

    messages = []
    for d in tqdm(dataset):
        message = d['user_message']
        response = d['bot_response']
        chat = [
            {"role": "user", "content": message}, 
            {"role": "assistant", "content": response},
            {"role": "user", "content": PROMPT}
        ]
        api_response = completion(model="anthropic/claude-opus-4-5-20251101", messages=chat)
        response_message = api_response.choices[0].message.content
        if len(response_message) < len(response):
            messages.append({
                'prompt': message,
                'chosen': response_message,
                'rejected': response
            })


    with (LOG_PATH.parent / 'dpo_pairs.jsonl').open('w', encoding='utf-8') as f:
        for m in messages:
            f.write(json.dumps(m) + '\n')
