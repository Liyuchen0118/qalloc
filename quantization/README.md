# MixQuant — Mixed-Precision Quantization Framework

This directory contains the MixQuant framework for applying mixed-precision
quantization to large language models. It takes a bit-width allocation plan
(produced by the QuBit QUBO pipeline or classical baselines) and performs
the actual model quantization and evaluation.

## Supported Quantization Methods

| Method | Flag | Description |
|--------|------|-------------|
| GPTQ | `--quant_method gptq` | GPTQ weight quantization |
| AWQ | `--quant_method awq` | Activation-aware Weight Quantization |
| OmniQuant | `--quant_method omni` | OmniQuant quantization |
| Nearest | `--quant_method nearest` | Simple nearest-rounding quantization |

## Usage Examples

### Quantize and Evaluate Perplexity

```bash
python main.py \
    --model /path/to/qwen2.5-7b \
    --allocation "2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 2, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 2, 4, 2, 2, 2, 2, 4, 2, 2, 2, 2, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 4, 2, 4, 4, 4, 2, 2, 4, 2, 2, 4, 4, 4, 4, 4, 2, 4, 2, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 4, 2, 4, 4, 2, 2, 2, 2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2" \
    --quant_method awq \
    --wbits 3,4,5 \
    --nsamples 128 \
    --dataset wikitext2 \
    --dataset_dir /path/to/datasets/ \
    --evaluate_ppl \
    --save_path ./quantized_model
```

### Evaluate Reasoning Tasks

```bash
python main.py \
    --model /path/to/model \
    --allocation "..." \
    --quant_method awq \
    --wbits 3,4,5 \
    --nsamples 128 \
    --dataset wikitext2 \
    --reasoning
```

### Evaluate Perplexity Only (Pre-quantized Model)

```bash
python evaluation/perplexity.py --model /path/to/model --dataset wikitext2
```

## Key Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--model` | HuggingFace model path or local path | `/path/to/llama-7b` |
| `--allocation` | Comma-separated bit-width list for each layer | `"4,4,3,4,..."` |
| `--quant_method` | Quantization algorithm | `awq`, `gptq`, `omni` |
| `--wbits` | Available bit-width candidates | `3,4,5` |
| `--target_bit` | Target average bit-width | `4` |
| `--evaluate_ppl` | Run perplexity evaluation after quantization | flag |
| `--reasoning` | Run reasoning task evaluation | flag |
| `--nsamples` | Number of calibration samples | `128` |
| `--save_path` | Save quantized model to this path | `./output` |

## Supported Model Architectures

Configured in `model_config.json`:
- LLaMA, LLaMA-2, LLaMA-3
- Qwen, Qwen-2
- Mistral, Mixtral
- OPT, Bloom, Falcon
