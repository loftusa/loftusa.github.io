"""
DPO Evaluation Script

Evaluates DPO-trained models on held-out evaluation set using:
1. Preference Accuracy - core DPO metric (chosen > rejected log-prob)
2. Response Length Analysis - compare avg token counts
3. Refusal Rate - check for on-topic refusal patterns
4. Win Rate (LLM-as-Judge) - Claude judges base vs fine-tuned

Usage: uv run dpo_eval.py [--checkpoint N] [--num-samples N] [--skip-judge]
"""

import os
from pathlib import Path

# Set cache directories before importing other libraries
CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE_DIR / "matplotlib")
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")

import argparse
import json
import re
from typing import Optional

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset import MODEL_NAME
from dpo_train import OUTPUT_DIR

dotenv.load_dotenv(Path(__file__).parent.parent / ".env")

# Paths
CURRENT_DIR = Path(__file__).parent
EVAL_DATASET_PATH = CURRENT_DIR / "local_datasets/dpo_eval.jsonl"
SYSTEM_PROMPT = (CURRENT_DIR.parent / "system_prompt.txt").read_text()
RESUME = (CURRENT_DIR.parent / "resume.txt").read_text()
CHECKPOINT_FMT = str(OUTPUT_DIR / "checkpoint-{}")

# Refusal patterns to detect on-topic responses
REFUSAL_PATTERNS = [
    r"I can only (help|answer|assist) with",
    r"questions about Alex",
    r"Alex('s| Loftus)",
    r"background, experience, skills",
    r"I'm sorry,? but I can('t| not)",
    r"outside (of )?my (scope|knowledge)",
    r"not related to",
    r"politely decline",
]


def load_eval_dataset(path: Path, limit: Optional[int] = None) -> list[dict]:
    """Load evaluation dataset from JSONL file."""
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data.append(json.loads(line))
    return data


def format_prompt(user_message: str, tokenizer) -> str:
    """Format prompt with system context."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": RESUME},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def compute_log_prob(model, tokenizer, prompt: str, response: str, device: str) -> float:
    """Compute log probability of response given prompt."""
    # Tokenize prompt and full sequence
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_text = prompt + response + tokenizer.eos_token
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    # Get log probs for response tokens only
    prompt_len = prompt_ids.shape[1]
    response_logits = logits[0, prompt_len-1:-1]  # Shift by 1 for next-token prediction
    response_ids = full_ids[0, prompt_len:]

    log_probs = torch.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(1, response_ids.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum().item()


def compute_preference_accuracy(
    model, tokenizer, eval_data: list[dict], device: str, desc: str = "Preference Accuracy"
) -> dict:
    """Compute preference accuracy: how often chosen > rejected in log-prob."""
    correct = 0
    margins = []

    for sample in tqdm(eval_data, desc=desc):
        prompt = format_prompt(sample["prompt"], tokenizer)

        chosen_log_prob = compute_log_prob(model, tokenizer, prompt, sample["chosen"], device)
        rejected_log_prob = compute_log_prob(model, tokenizer, prompt, sample["rejected"], device)

        margin = chosen_log_prob - rejected_log_prob
        margins.append(margin)

        if chosen_log_prob > rejected_log_prob:
            correct += 1

    return {
        "accuracy": correct / len(eval_data),
        "margins": margins,
        "avg_margin": np.mean(margins),
        "std_margin": np.std(margins),
    }


def generate_response(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 256) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def compute_response_lengths(
    model, tokenizer, eval_data: list[dict], device: str, desc: str = "Response Lengths"
) -> dict:
    """Generate responses and compute token lengths."""
    lengths = []
    responses = []

    for sample in tqdm(eval_data, desc=desc):
        prompt = format_prompt(sample["prompt"], tokenizer)
        response = generate_response(model, tokenizer, prompt, device)
        responses.append(response)

        # Token count
        tokens = tokenizer(response, return_tensors="pt").input_ids.shape[1]
        lengths.append(tokens)

    return {
        "lengths": lengths,
        "responses": responses,
        "avg_length": np.mean(lengths),
        "std_length": np.std(lengths),
        "median_length": np.median(lengths),
    }


def compute_refusal_rate(responses: list[str]) -> dict:
    """Check what fraction of responses contain refusal patterns."""
    refusals = []

    for response in responses:
        is_refusal = any(re.search(pattern, response, re.IGNORECASE) for pattern in REFUSAL_PATTERNS)
        refusals.append(is_refusal)

    return {
        "refusal_rate": np.mean(refusals),
        "refusals": refusals,
        "num_refusals": sum(refusals),
    }


def compute_win_rate_llm_judge(
    base_responses: list[str],
    finetuned_responses: list[str],
    prompts: list[str],
    num_samples: int = 50,
) -> dict:
    """Use Claude as judge to compare base vs fine-tuned responses."""
    try:
        import anthropic
    except ImportError:
        print("anthropic package not installed. Skipping LLM judge evaluation.")
        return {"error": "anthropic not installed"}

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set. Skipping LLM judge evaluation.")
        return {"error": "ANTHROPIC_API_KEY not set"}

    client = anthropic.Anthropic(api_key=api_key)

    wins = {"base": 0, "finetuned": 0, "tie": 0}
    judgments = []

    # Sample subset for cost efficiency
    indices = np.random.choice(len(prompts), min(num_samples, len(prompts)), replace=False)

    for idx in tqdm(indices, desc="LLM Judge"):
        prompt = prompts[idx]
        base_resp = base_responses[idx]
        ft_resp = finetuned_responses[idx]

        # Randomize order to avoid position bias
        if np.random.random() > 0.5:
            resp_a, resp_b = base_resp, ft_resp
            order = "base_first"
        else:
            resp_a, resp_b = ft_resp, base_resp
            order = "finetuned_first"

        judge_prompt = f"""You are judging responses from a personal resume chatbot for Alexander Loftus.
The chatbot should:
1. Answer questions about Alex's background, experience, skills, research, and projects
2. Politely refuse off-topic or inappropriate requests
3. Keep responses concise and on-topic

User prompt: "{prompt}"

Response A:
{resp_a}

Response B:
{resp_b}

Which response is more appropriate for a personal resume chatbot?
Answer with ONLY one of: "A", "B", or "TIE"
"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            judgment = response.content[0].text.strip().upper()

            if "A" in judgment and "B" not in judgment:
                winner = "base" if order == "base_first" else "finetuned"
            elif "B" in judgment and "A" not in judgment:
                winner = "finetuned" if order == "base_first" else "base"
            else:
                winner = "tie"

            wins[winner] += 1
            judgments.append({"idx": int(idx), "order": order, "judgment": judgment, "winner": winner})
        except Exception as e:
            print(f"Error in judge call: {e}")
            judgments.append({"idx": int(idx), "error": str(e)})

    total = wins["base"] + wins["finetuned"] + wins["tie"]
    return {
        "win_rate_finetuned": wins["finetuned"] / total if total > 0 else 0,
        "win_rate_base": wins["base"] / total if total > 0 else 0,
        "tie_rate": wins["tie"] / total if total > 0 else 0,
        "wins": wins,
        "judgments": judgments,
        "num_samples": total,
    }


def plot_results(
    base_pref: dict,
    ft_pref: dict,
    base_lengths: dict,
    ft_lengths: dict,
    base_refusal: dict,
    ft_refusal: dict,
    win_rate: Optional[dict],
    checkpoint_num: int,
):
    """Create visualization of evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"DPO Evaluation Results (Checkpoint {checkpoint_num})", fontsize=14)

    # 1. Preference Accuracy Comparison
    ax = axes[0, 0]
    models = ["Base Model", f"Checkpoint {checkpoint_num}"]
    accuracies = [base_pref["accuracy"], ft_pref["accuracy"]]
    colors = ["steelblue", "coral"]
    bars = ax.bar(models, accuracies, color=colors)
    ax.set_ylabel("Preference Accuracy")
    ax.set_title("Preference Accuracy\n(chosen > rejected log-prob)")
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{acc:.2%}", ha="center", fontsize=11)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random")
    ax.legend()

    # 2. Response Length Distribution
    ax = axes[0, 1]
    ax.hist(base_lengths["lengths"], bins=30, alpha=0.6, label=f"Base (avg={base_lengths['avg_length']:.1f})", color="steelblue")
    ax.hist(ft_lengths["lengths"], bins=30, alpha=0.6, label=f"FT (avg={ft_lengths['avg_length']:.1f})", color="coral")
    ax.set_xlabel("Response Length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title("Response Length Distribution")
    ax.legend()

    # 3. Refusal Rate Comparison
    ax = axes[1, 0]
    refusal_rates = [base_refusal["refusal_rate"], ft_refusal["refusal_rate"]]
    bars = ax.bar(models, refusal_rates, color=colors)
    ax.set_ylabel("Refusal Rate")
    ax.set_title("On-Topic Refusal Rate\n(contains refusal patterns)")
    ax.set_ylim(0, 1)
    for bar, rate in zip(bars, refusal_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{rate:.2%}", ha="center", fontsize=11)

    # 4. Win Rate (if available)
    ax = axes[1, 1]
    if win_rate and "error" not in win_rate:
        categories = ["Fine-tuned Wins", "Base Wins", "Ties"]
        rates = [win_rate["win_rate_finetuned"], win_rate["win_rate_base"], win_rate["tie_rate"]]
        colors_wr = ["coral", "steelblue", "gray"]
        bars = ax.bar(categories, rates, color=colors_wr)
        ax.set_ylabel("Rate")
        ax.set_title(f"LLM Judge Win Rate\n(n={win_rate['num_samples']})")
        ax.set_ylim(0, 1)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{rate:.2%}", ha="center", fontsize=11)
    else:
        ax.text(0.5, 0.5, "LLM Judge\nNot Available", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    plt.tight_layout()
    return fig


def print_summary(
    base_pref: dict,
    ft_pref: dict,
    base_lengths: dict,
    ft_lengths: dict,
    base_refusal: dict,
    ft_refusal: dict,
    win_rate: Optional[dict],
    checkpoint_num: int,
):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print(f"DPO EVALUATION SUMMARY - Checkpoint {checkpoint_num}")
    print("=" * 60)

    print("\n1. PREFERENCE ACCURACY (chosen > rejected log-prob)")
    print(f"   Base Model:      {base_pref['accuracy']:.2%} (margin: {base_pref['avg_margin']:.2f} +/- {base_pref['std_margin']:.2f})")
    print(f"   Fine-tuned:      {ft_pref['accuracy']:.2%} (margin: {ft_pref['avg_margin']:.2f} +/- {ft_pref['std_margin']:.2f})")
    improvement = ft_pref['accuracy'] - base_pref['accuracy']
    print(f"   Improvement:     {improvement:+.2%}")

    print("\n2. RESPONSE LENGTH (tokens)")
    print(f"   Base Model:      {base_lengths['avg_length']:.1f} +/- {base_lengths['std_length']:.1f} (median: {base_lengths['median_length']:.0f})")
    print(f"   Fine-tuned:      {ft_lengths['avg_length']:.1f} +/- {ft_lengths['std_length']:.1f} (median: {ft_lengths['median_length']:.0f})")
    length_reduction = (base_lengths['avg_length'] - ft_lengths['avg_length']) / base_lengths['avg_length'] * 100
    print(f"   Reduction:       {length_reduction:.1f}%")

    print("\n3. REFUSAL RATE (on-topic refusal patterns)")
    print(f"   Base Model:      {base_refusal['refusal_rate']:.2%} ({base_refusal['num_refusals']}/{len(base_refusal['refusals'])})")
    print(f"   Fine-tuned:      {ft_refusal['refusal_rate']:.2%} ({ft_refusal['num_refusals']}/{len(ft_refusal['refusals'])})")

    if win_rate and "error" not in win_rate:
        print("\n4. LLM JUDGE WIN RATE")
        print(f"   Fine-tuned wins: {win_rate['win_rate_finetuned']:.2%}")
        print(f"   Base wins:       {win_rate['win_rate_base']:.2%}")
        print(f"   Ties:            {win_rate['tie_rate']:.2%}")
        print(f"   (n={win_rate['num_samples']} samples)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DPO-trained model")
    parser.add_argument("--checkpoint", type=int, default=126, help="Checkpoint number to evaluate")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples for generation-based metrics")
    parser.add_argument("--judge-samples", type=int, default=30, help="Number of samples for LLM judge")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judge evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading evaluation dataset from {EVAL_DATASET_PATH}")
    eval_data = load_eval_dataset(EVAL_DATASET_PATH)
    print(f"Loaded {len(eval_data)} evaluation samples")

    # Use subset for generation-based metrics (they're slow)
    eval_data_gen = eval_data[:args.num_samples]

    print(f"\nLoading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=args.device
    )

    checkpoint_path = CHECKPOINT_FMT.format(args.checkpoint)
    print(f"Loading fine-tuned model from: {checkpoint_path}")
    ft_model = PeftModel.from_pretrained(base_model, checkpoint_path)

    # Set models to eval mode
    base_model.eval()
    ft_model.eval()

    # ========== METRIC 1: Preference Accuracy ==========
    print("\n--- Computing Preference Accuracy ---")

    # For base model, disable adapter
    ft_model.disable_adapter_layers()
    base_pref = compute_preference_accuracy(ft_model, tokenizer, eval_data, args.device, "Base Pref Acc")

    # For fine-tuned model, enable adapter
    ft_model.enable_adapter_layers()
    ft_pref = compute_preference_accuracy(ft_model, tokenizer, eval_data, args.device, "FT Pref Acc")

    # ========== METRIC 2: Response Length ==========
    print("\n--- Computing Response Lengths ---")

    ft_model.disable_adapter_layers()
    base_lengths = compute_response_lengths(ft_model, tokenizer, eval_data_gen, args.device, "Base Lengths")

    ft_model.enable_adapter_layers()
    ft_lengths = compute_response_lengths(ft_model, tokenizer, eval_data_gen, args.device, "FT Lengths")

    # ========== METRIC 3: Refusal Rate ==========
    print("\n--- Computing Refusal Rates ---")
    base_refusal = compute_refusal_rate(base_lengths["responses"])
    ft_refusal = compute_refusal_rate(ft_lengths["responses"])

    # ========== METRIC 4: Win Rate (LLM Judge) ==========
    win_rate = None
    if not args.skip_judge:
        print("\n--- Computing LLM Judge Win Rate ---")
        prompts = [sample["prompt"] for sample in eval_data_gen]
        win_rate = compute_win_rate_llm_judge(
            base_lengths["responses"],
            ft_lengths["responses"],
            prompts,
            num_samples=args.judge_samples,
        )

    # ========== Results ==========
    print_summary(base_pref, ft_pref, base_lengths, ft_lengths, base_refusal, ft_refusal, win_rate, args.checkpoint)

    # Plot results
    fig = plot_results(base_pref, ft_pref, base_lengths, ft_lengths, base_refusal, ft_refusal, win_rate, args.checkpoint)

    # Save figure
    output_path = CURRENT_DIR / f"dpo_eval_checkpoint_{args.checkpoint}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")

    plt.show()

    # Save detailed results to JSON
    results = {
        "checkpoint": args.checkpoint,
        "num_eval_samples": len(eval_data),
        "num_gen_samples": len(eval_data_gen),
        "preference_accuracy": {
            "base": base_pref["accuracy"],
            "finetuned": ft_pref["accuracy"],
            "base_margin": base_pref["avg_margin"],
            "finetuned_margin": ft_pref["avg_margin"],
        },
        "response_length": {
            "base_avg": base_lengths["avg_length"],
            "finetuned_avg": ft_lengths["avg_length"],
            "reduction_pct": (base_lengths["avg_length"] - ft_lengths["avg_length"]) / base_lengths["avg_length"] * 100,
        },
        "refusal_rate": {
            "base": base_refusal["refusal_rate"],
            "finetuned": ft_refusal["refusal_rate"],
        },
    }
    if win_rate and "error" not in win_rate:
        results["win_rate"] = {
            "finetuned": win_rate["win_rate_finetuned"],
            "base": win_rate["win_rate_base"],
            "tie": win_rate["tie_rate"],
        }

    results_path = CURRENT_DIR / f"dpo_eval_results_{args.checkpoint}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
