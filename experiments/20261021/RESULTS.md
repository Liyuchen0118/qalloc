# Qwen2.5-7B mixed-precision baseline experiment, 2026-10-21

Author: Yuchen Li

## Scope and method

The local allocation problem has 196 Qwen2.5-7B projection tensors (28 blocks x
7 projections) and four actual GGUF tensor types: `Q3_K`, `Q4_K`, `Q5_K`, and
`Q8_0`. All non-target tensors use the same llama.cpp `Q4_K_M` base. The measured
per-tensor byte cost and additive objective were reconstructed from quantized
GGUFs. The objective is squared weight reconstruction error, **not** a validated
predictor of perplexity.

Five full-file byte budgets were fixed at the uniform Q3, Q4, and Q5 anchor
models and their two midpoints. MILP (Gurobi 13, zero relative gap), Lagrangian
relaxation with repair/local exchange, and gain-per-byte greedy were solved
locally for every budget. All 15 solutions were feasible. Identical allocations
were deduplicated to 8 GGUFs. The quantizer validated all 196 requested target
types and every actual file size against its budget before evaluation.

## Main results

File sizes below are decimal GB. PPL uses the complete WikiText-2 raw test
corpus. Prefill is llama-bench `pp512`; decode is `tg128`, each with five
repetitions. Every result used the same GPU.

| Budget GB | Method | Actual GB | Proxy error | PPL | Prefill tok/s | Decode tok/s |
|---:|---|---:|---:|---:|---:|---:|
| 3.565 | MILP | 3.565 | 42332.15 | 6.9646 | 9335 | 261.8 |
| 3.565 | Lagrangian + exchange | 3.565 | 42332.15 | 6.9646 | 9335 | 261.8 |
| 3.565 | Greedy | 3.565 | 42332.15 | 6.9646 | 9335 | 261.8 |
| 3.998 | MILP | 3.998 | 24336.79 | 6.6841 | 10133 | 245.2 |
| 3.998 | Lagrangian + exchange | 3.998 | 24336.79 | 6.6841 | 10133 | 245.2 |
| 3.998 | Greedy | 3.998 | 24343.36 | 6.6767 | 10111 | 245.1 |
| 4.431 | MILP | 4.431 | 9496.18 | 6.2932 | 11143 | 228.0 |
| 4.431 | Lagrangian + exchange | 4.431 | 9496.18 | 6.2932 | 11143 | 228.0 |
| 4.431 | Greedy | 4.431 | 9557.10 | 6.2926 | 11118 | 228.1 |
| 4.839 | MILP | 4.839 | 5644.59 | 6.2354 | 10935 | 211.6 |
| 4.839 | Lagrangian + exchange | 4.839 | 5644.59 | 6.2354 | 10935 | 211.6 |
| 4.839 | Greedy | 4.839 | 5645.40 | 6.2356 | 10977 | 211.7 |
| 5.247 | MILP | 5.247 | 2436.91 | 6.1810 | 10775 | 201.8 |
| 5.247 | Lagrangian + exchange | 5.247 | 2436.91 | 6.1810 | 10775 | 201.8 |
| 5.247 | Greedy | 5.247 | 2436.91 | 6.1810 | 10775 | 201.8 |

MILP and Lagrangian produced exactly the same allocation at all five budgets.
At the lowest and highest budgets, all three methods produced the same
allocation. The measured PPL order reverses the proxy order at 3.998 GB and
4.431 GB: greedy has a worse weight-error objective but a slightly better PPL.
The differences, especially 0.0006 PPL at 4.431 GB, are small; these five
budgets alone do not establish a reliable within-budget ranking. A calibrated
quality proxy and a larger fixed-budget ranking study are needed before using
this objective as the paper's main quality claim.

Local optimizer wall times were approximately 4-70 ms for Gurobi MILP,
8-45 ms for Lagrangian + exchange, and under 24 ms for greedy (Gurobi 13 on
the local machine). These times cover only allocation solving; they exclude
GGUF cost-table construction, quantization, network transit, and inference.

## Deployment references

These references occupy different actual byte positions and are **not**
matched-budget optimization competitors. `Q4_K`/`Q8_0` set all seven target
projections to one type while retaining the common `Q4_K_M` non-target base;
`Q4_K_M`/`Q5_K_S` are native whole-model llama.cpp presets.

| Reference | Actual GB | PPL | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| Uniform target `Q4_K` | 4.431 | 6.2948 | 11269 | 228.8 |
| Uniform target `Q8_0` | 7.694 | 6.1342 | 11128 | 145.9 |
| Native `Q4_K_M` | 4.683 | 6.2457 | 10736 | 219.0 |
| Native `Q5_K_S` | 5.315 | 6.1837 | 10761 | 201.7 |
| Source F16 | 15.238 | 6.1278 | 11075 | 81.4 |

Uniform target `Q3_K` and `Q5_K` are already represented by the 3.565 GB and
5.247 GB solver rows, respectively. The 5.247 GB mixed allocation measured
6.1810 PPL versus 6.1837 for the 5.315 GB native `Q5_K_S` preset. This is an
observed comparison on one corpus, not a general quality advantage.

## Reproducibility and limits

- GPU: NVIDIA RTX PRO 5000 72GB Blackwell, physical index 0, UUID
  `GPU-54e16f83-71b9-d6fd-26c8-a0d483c5e7c3`. The runner checked that the
  GPU was idle before every evaluation. GPU 3 and 4 were occupied by other jobs
  and were never selected.
- llama.cpp commit: `493253fe5a5af7e800b3c92a2452b5684a99e961`.
- Corpus: `wikitext-2-raw/wiki.test.raw` from ggml-org's llama.cpp CI archive;
  1,290,590 bytes, SHA-256
  `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`.
- PPL: context 2048, batch 2048, microbatch 512, flash attention on, all GPU
  layers requested, full corpus (no chunk limit). Benchmarks: 5 repetitions
  each, 8 CPU threads, the same other execution parameters.
- `summary.json` is the validated machine-readable comparison table. The
  `full/` and `references/` directories retain per-run commands, raw output,
  timestamps, model sizes, CUDA backend evidence, and sampled GPU memory.
- The reported GPU memory is the maximum of 0.5-second `nvidia-smi` samples,
  not an exact allocator peak. The constraint in this run is **GGUF file size**,
  not end-device RAM or VRAM capacity. The cloud GPU is not a target edge device.
- The 1000-bit photonic-machine allocator was **not** rerun on this new GGUF
  cost table. No quantum-vs-classical speed or quality conclusion follows from
  these data. A correct global-budget QUBO and full hardware timing are still
  required. Old AWQ/GPTQ fake-quantized results are not comparable to these
  executable GGUF measurements.
