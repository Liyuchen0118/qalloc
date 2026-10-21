# Quantum-Assisted Budget-Aware Mixed-Precision Allocation for Large Language Models

This repository contains the original QuBit allocation prototype and its
supporting experiments. The current implementation targets analytic bit-width
allocation with the candidate set `{2, 3, 4, 8}`. It does **not** yet implement
GGUF/llama.cpp quantization, actual device-memory budget enforcement, or a
client for the 1000-bit photonic Ising machine. Those are separate development
tasks for the revised study.

## Contents

| Path | Purpose |
| --- | --- |
| `qubit_core/` | Shared data loading, sensitivity loss, and evaluation code |
| `qubo_formulation/` | QUBO generation, decoding, and ablation variants |
| `classical_baselines/` | Classical optimization baselines from the original study |
| `sensitivity_extraction/` | Model parameter and Fisher sensitivity extraction |
| `quantization/` | Legacy MixQuant/PyTorch quantization and quality evaluation |
| `model_info/` | Small, precomputed model statistics used by allocation scripts |
| `results/` | Historical perplexity summary |

The six large OmniQuant activation-scale/shift `.pt` files, model weights,
datasets, GGUF artifacts, and generated QUBO CSV files are intentionally not
tracked. The full precision model and its tokenizer must be supplied separately.

## Reproduce the original allocation

Install the root Python dependencies in an environment compatible with PyTorch
and the Kaiwu SDK. From the repository root, run:

```bash
python qubo_formulation/base/generate_qubo.py \
  --model_name qwen2.5-7b \
  --model_info_root model_info
```

This writes a QUBO CSV. It does not submit it to a remote solver. Decode a
solver result with:

```bash
python qubo_formulation/base/decode_result.py \
  --model_name qwen2.5-7b \
  --model_info_root model_info \
  --solution_file /path/to/solution.json
```

The legacy PyTorch quantizer lives in `quantization/` and has its own
dependencies. Its AWQ/GPTQ-labeled paths update reconstructed floating-point
weights for quality evaluation; they do not produce packed low-bit GGUF models.
Do not infer actual storage or llama.cpp inference performance from those
results. Several legacy scripts still contain paths from the original compute
environment and need configuration before rerunning.

To regenerate model statistics, run:

```bash
bash sensitivity_extraction/run_evaluation.sh \
  /path/to/full-precision-model \
  /path/to/output-directory
```

This operation requires the model weights, tokenizer, datasets, and substantial
compute resources.

## Development direction

The revised experiments will map the seven projection tensors in each decoder
block to GGUF tensor names, measure actual size and quality for supported GGUF
types, solve device-budget-aware allocations, and compare them under a fixed
llama.cpp version and inference configuration. The existing Fisher scores are
historical inputs, not validated GGUF loss estimates.

## License and attribution

The project source is distributed under the MIT license in `LICENSE`.
Vendored AutoAWQ and OmniQuant components retain their upstream MIT licenses
and attribution in `THIRD_PARTY_NOTICES.md` and `licenses/`.
