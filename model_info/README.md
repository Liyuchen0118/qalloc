# Model Info Directory

This directory contains pre-computed per-layer metadata for each supported model.
Each model subdirectory contains three JSON files:

| File | Description |
|------|-------------|
| `fisher.json` | Fisher Information — layer quantization sensitivity scores |
| `LayerParams.json` | Parameter count for each sub-layer weight matrix |
| `perplexities.json` | Per-layer perplexity measurements under quantization |

## Sensitivity Variants

For models with sensitivity variant suffixes:

| Suffix | Full Name | Description |
|--------|-----------|-------------|
| *(none)* | Default Fisher | Standard Fisher Information from quantization perturbation |
| `-os` | Outlier Score | Layer sensitivity computed using Outlier Score method |
| `-owq` | OWQ | Layer sensitivity computed using OWQ (Optimal Weight Quantization) method |

> **Note:** All sensitivity variants store their sensitivity data in `fisher.json`
> for program compatibility, regardless of the actual computation method used.

## Supported Models

- `llama-7b`, `llama-13b`, `llama-30b` — LLaMA model family
- `mistral-7b` — Mistral model
- `qwen2.5-7b`, `qwen2.5-14b` — Qwen2.5 model family

Each model has 7 sub-layers per Transformer block:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
