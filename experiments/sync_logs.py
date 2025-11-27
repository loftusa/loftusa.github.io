from dotenv import load_dotenv
import os
import requests
from pathlib import Path
from datetime import datetime

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()  # Also try parent directory
LOG_ACCESS_TOKEN = os.getenv("LOG_ACCESS_TOKEN")
LOG_SYNC_URL = "https://llm-resume-restless-thunder-9259.fly.dev/logs/download"
assert LOG_ACCESS_TOKEN is not None, "set log access token"
LOG_PATH = Path(os.getenv("LOG_PATH", "experiments/logs/chat_logs.jsonl"))
if LOG_PATH.is_dir():
    LOG_PATH = LOG_PATH / "chat_logs.jsonl"
LOG_PATH.parent.mkdir(exist_ok=True, parents=True)


def sync():
    try:
        response = requests.get(
            LOG_SYNC_URL, headers={"Authorization": f"Bearer {LOG_ACCESS_TOKEN}"}
        )
    except requests.exceptions.ConnectionError as e:
        print(f"Error: Connection failed to {LOG_SYNC_URL}: {e}")
        return
    except Exception as e:
        print(f"Error: Unexpected error: {e}")
        return

    if response.status_code != 200:
        print(f"Error: server returned status {response.status_code}")
        return

    LOG_PATH_TMP = LOG_PATH.with_suffix(".jsonl.tmp")
    with LOG_PATH_TMP.open("w", encoding="utf-8") as f:
        f.write(response.text)

    bytes_difference = LOG_PATH_TMP.stat().st_size - LOG_PATH.stat().st_size
    LOG_PATH_TMP.replace(LOG_PATH)

    bytes_downloaded = len(response.content)
    print(f"""
    Synced logs from {LOG_SYNC_URL}. 
    Logs saved in {LOG_PATH}.
    Bytes downloaded: {bytes_downloaded}. 
    Bytes difference: {bytes_difference}.
    Time: {datetime.now().isoformat()}.
    """)


if __name__ == "__main__":
    sync()
