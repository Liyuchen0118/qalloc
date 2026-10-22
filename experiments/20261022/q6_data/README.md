<!-- Author: Yuchen Li -->
# Q6_K two-bit candidate extension, 2026-10-22

This experiment replaces the previous Q8_0 choice with Q6_K for the 196
Qwen2.5-7B projection tensors. It preserves the five full-GGUF byte budgets
from `experiments/20261021/` and the untouched, fixed non-target portion
(760,916,128 bytes). The four choices are Q3_K, Q4_K, Q5_K, and Q6_K.

`qwen25_7b_q3_q4_q5_q6_costs.json` reuses the actual Q3/Q4/Q5 measurements
from the previous table and measures Q6_K squared weight error from the new
uniform Q6_K GGUF after dequantization against the original F16 GGUF. The
candidate and source hashes and measurement definition are in its metadata.
All 196 target tensors have been validated as Q6_K in the uniform GGUF. The
actual per-tensor GGUF byte counts satisfy
`bytes(Q3_K) + bytes(Q6_K) = bytes(Q4_K) + bytes(Q5_K)` for all 196 tensors.
Consequently the four types can use two type bits per tensor with a linear
storage cost; quality loss still has a two-bit interaction term.

The uniform Q6_K full GGUF is 6,113,691,808 bytes and is kept on the cloud
machine at
`/home/hefeng/quant/qalloc/experiments/20261022/q6_data/uniform/Qwen2.5-7B-proj-Q6_K.gguf`.
The B5 mixed GGUF is 5,246,937,248 bytes and is kept at
`/home/hefeng/quant/qalloc/experiments/20261022/q6_data/models/budget_05_milp.gguf`.
Their validation manifests are copied here. Both GGUFs contain
`general.author = Yuchen Li`; the large GGUF binaries are not in Git.

The five budgets were solved from this table by Gurobi 13 with integer cost
scaling and requested relative MIP gap 0. Results are in
`qwen25_7b_q3_q4_q5_q6_allocations.json`; runnable MILP assignments are in
`allocations/budget_01_milp.json` through `budget_05_milp.json`. Every MILP
result has status `optimal`, reported relative gap 0, and actual target
bytes at or below budget. B1 through B4 assignments match the previous
Q3/Q4/Q5/Q8 experiment. B5 uses Q6_K for eight small K/V projection tensors.

| Full-file budget, bytes | MILP proxy loss | B5 Q6_K count |
| ---: | ---: | ---: |
| 3,564,751,008 | 42,332.146513 | 0 |
| 3,998,070,944 | 24,336.790390 | 0 |
| 4,431,390,880 | 9,496.181179 | 0 |
| 4,839,221,408 | 5,644.587939 | 0 |
| 5,247,051,936 | 2,435.485034 | 8 |

For B5, the previous and new GGUFs were measured with the same llama.cpp
commit `493253fe5a5af7e800b3c92a2452b5684a99e961`, same WikiText-2 test
corpus, same CUDA device `GPU-9ef20e77-b08e-d0e4-ed70-833fde0a8837`, same
2048-token PPL context, and matched benchmark settings (512 prompt tokens or
128 generated tokens, 5 repetitions). The scripts checked that the GPU was
idle before each run; an occupied-check halted the first attempt before new
B5 decode, which was completed after the card became idle again. The raw
`eval/*.json` results and `b5_matched_eval_manifest.json` record provenance.

| B5 model | GGUF bytes | PPL | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Previous Q3/Q4/Q5/Q8 choice | 5,247,051,936 | 6.1810 | 10,923.88 | 202.38 |
| New Q3/Q4/Q5/Q6 choice | 5,246,937,248 | 6.1813 | 10,937.98 | 203.04 |

The lower B5 weight-error proxy did **not** improve measured PPL on this
corpus. Prefill differs by less than its run-to-run variation; decode differs
by about 0.33% in this single five-repeat setting. This experiment does not
establish a practically meaningful speed gain. It supports the two-bit
storage formulation and shows comparable deployed behavior. B1-B4 can reuse
the previous GGUF measurements because their
assignments are identical. The 8-bit hardware-scaled QUBO requires separate
validation; these MILP/GGUF results do not prove hardware solution quality.

Gurobi certified the five additive-proxy optima in 0.004-0.050 seconds, and
the Lagrangian baseline reached the same proxy objective at all five budgets.
This formulation currently offers no evidence for a quantum optimization or
wall-time advantage. The B5 proxy/PPL mismatch also makes within-budget
ranking validation necessary before claiming that proxy-selected allocations
improve deployed quality.

Reproduce the local allocations with:

```sh
python experiments/20261022/q6_data/solve_q6_allocations.py \
  --input experiments/20261022/q6_data/qwen25_7b_q3_q4_q5_q6_costs.json \
  --output experiments/20261022/q6_data/qwen25_7b_q3_q4_q5_q6_allocations.json \
  --mip-rel-gap 0
```
