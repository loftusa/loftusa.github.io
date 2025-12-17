from dotenv import load_dotenv
import os
from pathlib import Path
from dataclasses import dataclass, asdict
import time
import requests
import json

from .analyze_logs import percentile
load_dotenv()

EVAL_DATASET_PATH = Path(os.getenv("EVAL_DATASET_PATH"))
EVAL_OUTPUT_PATH = Path(os.getenv("EVAL_OUTPUT_PATH"))
API_URL = "http://127.0.0.1:8000/chat?logging=false"

@dataclass
class EvalSummary:
    question: str
    expected_facts: list[str]
    category: str
    full_response: str
    elapsed_ms: float
    all_passed: bool
    missing: list[str]

def call_chat_api(question: str) -> tuple[str, float]:
    """return (response, latency(ms)"""
    payload = {"messages": [{"role": "user", "content": question}]}
    start = time.perf_counter()
    response = requests.post(API_URL, json=payload, stream=True)
    response.raise_for_status()
    full_response = ""
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
        if chunk:
                full_response += chunk
    elapsed_ms = (time.perf_counter() - start) * 1000
    return full_response, elapsed_ms


def check_facts(response: str, expected: list[str]) -> tuple[bool, list[str]]:
    """
    return (passed, missing_facts)
    passed: all facts mentioned in the response
    missing_facts: facts that aren't there
    """
    response = response.lower()
    missing = [f for f in expected if f.lower() not in response]
    all_passed = len(missing) == 0

    return all_passed, missing

def run_evaluation(dataset_path: Path, output_path: Path) -> EvalSummary:
    output_path.unlink(missing_ok=True)
    with dataset_path.open('r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    evaluations = []
    for eval_question in dataset:
        full_response, elapsed_ms = call_chat_api((question:=eval_question['question']))
        all_passed, missing = check_facts(full_response, (expected_facts:=eval_question['expected_facts']))
<<<<<<< HEAD
        summary = EvalSummary(
=======
        evaluations.append(
            EvalSummary(
>>>>>>> eval-harness
                question=question,
                expected_facts=expected_facts,
                category=eval_question['category'],
                full_response=full_response,
                elapsed_ms=elapsed_ms,
                all_passed=all_passed,
                missing=missing
            )
<<<<<<< HEAD
        evaluations.append(summary)
        with output_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(summary)) + "\n")

    total = len(evaluations)
    passed_count = sum([1 for r in evaluations if r.all_passed])
    latencies = sorted([r.elapsed_ms for r in evaluations])
    print(f"Accuracy: {passed_count}/{total} = {100*passed_count/total:.1f}%")
    print("Latency(ms)")
    print(f"Average: {sum(latencies)/len(latencies)}")
    print(f"25th percentile: {percentile(latencies, p=0.25)}")
    print(f"50th percentile: {percentile(latencies, p=0.50)}")
    print(f"75th percentile: {percentile(latencies, p=0.75)}")
=======
        )
        with output_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(evaluations)) + "\n")

    total = len(evaluations)
    passed_count = sum(1 for r in evaluations if r['all_passed'])
    print(f"Accuracy: {passed_count}/{total} = {100*passed_count/total:.1f}%")

>>>>>>> eval-harness




if __name__ == "__main__":
    # resp, latency = call_chat_api("Where did Alex get his master's?")
    # print(f"{resp=}")
    # print(f"{latency=}ms")
    run_evaluation(EVAL_DATASET_PATH, output_path=EVAL_OUTPUT_PATH)