---
title:  "GPTNext"
date:   2025-01-04
permalink: /posts/2024/10/gptnext/
mathjax: true
comments: true
---

# GPTNext

- using the the shakespeare dataset as example code

## Research Log
- change to use uv (done)

## TODO
- good logging, baseline performance. Measure:
  - token throughput
  - training time
  - accuracy
  - perplexity
  - total model memory
  - memory per batch

- RMSNorm
- KV cache
- MQA
- FlashAttention
- Quantization
- rotary encoding?
- better optimizers and schedulers
- MoE
- meta's latent BPE paper
- multi-token prediction objective
- context length extension tricks?
- deepseek's multi-headed latent attention
- torch compile