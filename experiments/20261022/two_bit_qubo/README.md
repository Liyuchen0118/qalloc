---
author: Yuchen Li
---

# Q3_K/Q4_K/Q5_K/Q6_K two-bit global-budget QUBO

The five `*_two_bit_qubo.csv` files are unrounded, headerless, square,
upper-triangular QUBO matrices. For a binary vector `x`, the matrix energy is
`x.T @ Q @ x`; add `energy_offset` from the companion JSON to recover the
original objective, including its constant term. Each JSON also records the
source table hash, variable order, allocation checks, and coefficient range.

Two consecutive decision variables encode each tensor as `(b1,b0)`:
`00=Q3_K`, `01=Q4_K`, `10=Q5_K`, `11=Q6_K`. The CSV orders them as `b0,b1`.
The first 392 variables are decisions for 196 tensors. The remaining 0, 15,
16, 17, and 17 variables encode unused global budget for budget levels 1-5.
The generator verifies the measured per-tensor byte identity
`c(Q3_K)+c(Q6_K)=c(Q4_K)+c(Q5_K)`. This makes byte cost linear in the two
decision bits; arbitrary four-choice proxy losses are encoded by their local
`b0*b1` coefficient.

The global constraint is exact in integer bytes after extracting the greatest
common byte unit: `upgrade_units + unused_budget_units = capacity_units`.
Every integer unused-budget value in range is representable. Coefficients are
intentionally not scaled or rounded: the maximum positive coefficient reaches
about `1.44e14`, and the maximum absolute coefficient reaches about `4.01e14`.
**These files are mathematical reference matrices, not
certified SPQC-1000 8-bit hardware uploads.** Mapping the full problem to
signed int8 requires a separate method and validation that preserves quality
terms, decoded feasibility, and actual bytes.

Generate the files again from the measured Q6_K table with:

```sh
python -m qubo_formulation.gguf_two_bit_qubo \
  --references experiments/20261022/q6_data/qwen25_7b_q3_q4_q5_q6_allocations.json
```

The source table is `../q6_data/qwen25_7b_q3_q4_q5_q6_costs.json`. The
per-tensor Q6_K proxy loss comes from dequantizing the actual Q6_K GGUF, not
from interpolation between old candidates. Run the focused exactness checks:

```sh
python -m unittest tests.test_gguf_two_bit_qubo -v
```
