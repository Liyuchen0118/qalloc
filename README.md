# Quantum-Assisted Budget-Aware Mixed-Precision Allocation for Large Language Models

This repository contains the original QuBit allocation prototype and its
supporting experiments. The original optimizer targets analytic bit-width
allocation with the candidate set `{2, 3, 4, 8}`. The `llama_cpp/` wrappers
now turn a seven-projection-per-block allocation into a mixed-type GGUF and run
inference, throughput benchmarks, and perplexity evaluation with an external
llama.cpp checkout. The revised optimizer uses measured GGUF storage bytes;
it does not yet guarantee runtime device-memory budgets or have a verified
physical run on the 1000-bit photonic Ising machine.

## Contents

| Path | Purpose |
| --- | --- |
| `qubit_core/` | Shared data loading, sensitivity loss, and evaluation code |
| `qubo_formulation/` | QUBO generation, decoding, and ablation variants |
| `classical_baselines/` | Classical optimization baselines from the original study |
| `sensitivity_extraction/` | Model parameter and Fisher sensitivity extraction |
| `quantization/` | Legacy MixQuant/PyTorch quantization and quality evaluation |
| `llama_cpp/` | GGUF conversion, allocation, quantization, and evaluation wrappers; see [usage guide](llama_cpp/README.md) |
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

## Revised experiments

The GGUF wrapper maps the seven projection tensors in each dense decoder block
to GGUF tensor names and verifies the quantized tensor types. It records actual
GGUF file size, while `llama_cpp/evaluate.py` records inference output,
benchmark results, and corpus perplexity. Model weights and evaluation data
are not included. The existing Fisher scores are historical inputs, not
validated GGUF loss estimates. See the [llama.cpp guide](llama_cpp/README.md)
for the full workflow and its scope. The [SCIS experiment plan](SCIS_EXPERIMENT_PLAN.md)
sets MILP and Lagrangian relaxation with local exchange as the new primary
classical baselines under the same measured GGUF byte budgets; the original
generic heuristics are historical controls only.

The [October 22 experiment](experiments/20261022/two_bit_qubo/README.md)
measures `Q6_K` alongside `Q3_K/Q4_K/Q5_K` and encodes four types with two
decision bits per tensor. Its five 392-409-variable QUBOs are exact,
unrounded mathematical references. Their coefficients do not fit the
photonic platform's signed 8-bit range. Separate
[adjacent-type hardware subproblems](experiments/20261022/int8_adjacent/README.md)
fit that range but impose fixed per-block quotas, so they do not solve the
unrestricted global-budget allocation problem.

## License and attribution

The project source is distributed under the MIT license in `LICENSE`.
Vendored AutoAWQ and OmniQuant components retain their upstream MIT licenses
and attribution in `THIRD_PARTY_NOTICES.md` and `licenses/`.
