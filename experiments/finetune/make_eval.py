"""Find DPO pairs not in combined file and save as eval dataset."""
import json
from pathlib import Path

base = Path("local_datasets")

# Load combined file
combined = set()
with open(base / "dpo_combined.jsonl") as f:
    for line in f:
        combined.add(line.strip())

print(f"Combined has {len(combined)} samples")

# Find all other samples
all_other = []
other_files = list(base.glob("dpo_pairs*.jsonl"))
other_files = [f for f in other_files if f.name != "dpo_combined.jsonl"]

for fpath in sorted(other_files):
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if line and line not in combined:
                all_other.append(line)

print(f"Found {len(all_other)} samples not in combined")

# Dedupe
unique = list(set(all_other))
print(f"After dedup: {len(unique)} unique samples")

# Save
with open(base / "dpo_eval.jsonl", "w") as f:
    for line in unique:
        f.write(line + "\n")

print(f"Saved to {base / 'dpo_eval.jsonl'}")
