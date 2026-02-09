#!/bin/bash
# Run DPO data generation 10 times to measure variance in generated pairs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

echo "=============================================="
echo "DPO Data Generation Variance Test"
echo "Running 10 iterations..."
echo "=============================================="

for i in {1..10}; do
    echo ""
    echo ">>> Run $i/10"
    uv run experiments/finetune/dpo_data_gen_agents.py "_$i"
done

echo ""
echo "=============================================="
echo "All runs complete!"
echo "=============================================="
echo ""
echo "Output files:"
ls -la experiments/logs/dpo_pairs_*.jsonl 2>/dev/null || echo "No files found in experiments/logs/"

echo ""
echo "Line counts (number of DPO pairs per run):"
wc -l experiments/logs/dpo_pairs_*.jsonl 2>/dev/null || echo "No files found"

echo ""
echo "To analyze variance, compare the generated pairs across files."
