#!/bin/bash
cd /disk/u/lofty/loftusa.github.io/experiments/finetune
export CUDA_VISIBLE_DEVICES=2
export TMPDIR=/disk/u/lofty/tmp
mkdir -p $TMPDIR
/disk/u/lofty/loftusa.github.io/.venv/bin/python dpo_eval.py "$@"
