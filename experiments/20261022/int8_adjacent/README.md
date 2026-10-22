---
author: Yuchen Li
---

# Adjacent-type signed-int8 hardware subproblems

Run from the repository root:

```sh
python3 -m qubo_formulation.gguf_int8_adjacent
```

The two CSV files are headerless 196-by-196 upper-triangular signed-int8 QUBO
matrices. For a binary vector `x`, energy is `x.T @ Q @ x`. Each companion JSON
records the SHA-256, the index-to-tensor map, real GGUF bytes, proxy losses,
block constraints, and an exactly checked sample allocation. No auxiliary bits
are used. `0` selects the lower type, `1` the upper type.

| Case | Actual GGUF types | Full-file budget | Target-tensor budget | 8-bit Ising max `|h|/|J|` |
| --- | --- | ---: | ---: | ---: |
| B2 | `Q3_K` / `Q4_K` | 3,998,070,944 bytes | 3,237,154,816 bytes | 44 / 42 |
| B4 | `Q4_K` / `Q5_K` | 4,839,221,408 bytes | 4,078,305,280 bytes | 43 / 43 |

The 196 tensors are split into 42 independent blocks: for each consecutive pair
of transformer layers, the K/V tensors form a four-bit block, Q/O a four-bit
block, and MLP a six-bit block. Exactly half the variables in each block are
upgraded. Thus the three shape groups contain respectively 28, 28, and 42
upgrades, and **every valid block allocation has exactly the stated measured
byte cost**. The neutral reference upgrades all eligible tensors in the first
layer of each pair. This structure is chosen before optimizing proxy loss.

Each block has a centered cardinality penalty and measured weight-error gains.
The global 8-bit scale and round-half-even rule are in the JSON. Every local
binary state (`2^4` or `2^6`) was enumerated: the integer Ising expression
equals `4 * QUBO energy`, and **all quantized ground states satisfy every block
quota**. The 196-bit global ground state follows by concatenating local ground
states. Both the CSV entries and the Ising fields/couplings fit signed int8.

This is an **approximate, block-restricted hardware formulation**. It is not
the exact full global-budget QUBO: an unrestricted optimizer can trade bytes
across shape groups and blocks. The table uses the same measured proxy and
real byte costs for both methods. The "adjacent global optimum" is computed
exactly by enumerating the three uniform cost-class counts, equivalent to
solving the adjacent two-choice MILP; it is not a physical hardware result.

| Case | Block-restricted optimum proxy loss | Quantized ground-state proxy-loss range | Adjacent global optimum proxy loss |
| --- | ---: | ---: | ---: |
| B2 | 24,728.105 | 24,728.105 to 24,747.617 | 24,336.790 |
| B4 | 5,715.586 | 5,715.586 to 5,720.586 | 5,644.588 |

Some distinct allocations tie after 8-bit rounding. All tied ground states
remain byte-feasible, but their original proxy loss differs; do not report only
the best tie as hardware quality. The companion JSON includes a deterministic
representative vector and its loss, plus the best and worst possible loss among
the complete ground-state set.

These 4/6-bit blocks are also trivial for a classical top-`k` solver. Their
purpose is to test QBoson input, mapping, returned-bit decoding, and actual
budget feasibility; they do not establish a quantum optimization advantage.

To recompute true bytes and loss from a returned bit vector:

```python
import json
from qubo_formulation.gguf_int8_adjacent import decode_vector

metadata = json.load(open("experiments/20261022/int8_adjacent/qwen25_7b_b2_q3_k_q4_k_196bit_int8.json"))
result = decode_vector(metadata, solution_vector)
```

These matrices have **not** been accepted or solved by QBoson yet. The measured
proxy is squared weight reconstruction error, not PPL or end-to-end quality.
