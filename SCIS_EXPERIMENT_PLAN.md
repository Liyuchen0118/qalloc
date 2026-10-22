---
title: QuBit SCIS experiment plan
author: Yuchen Li
status: planned
---

# SCIS experiment plan: quantum-assisted budget-aware mixed-precision allocation

This plan replaces the main baseline comparison in the original QuBit study.
It is a plan, not a claim that the new measurements have already been run. The
primary question is whether the 1000-bit photonic Ising solver produces useful
**feasible allocations** when measured against strong classical optimization
on the same deployment problem.

## Research questions and pilot gates

The revised article should answer four questions. Complete G1 first; start a
full-instance MILP timing pilot as soon as its cost table exists, in parallel
with G2 proxy validation. G3 must precede the complete cloud comparison in G4.
Do not launch the full 7B/8B sweep until G1-G3 pass; G4 determines which
hardware claims the paper can responsibly make.

| Gate | Pilot evidence | Decision if it fails |
| --- | --- | --- |
| G1: Are four-type allocations real GGUF models under predictable byte budgets? | On a small model and then Qwen2.5-7B, validate all eligible shapes/types, the per-tensor byte table, full-file size accounting, and manifest types for several allocations. | Fix the type inventory or byte model; do not optimize nominal bitwidths. |
| G2: Does the proxy rank quality at a **fixed** budget? | At three interior budgets, evaluate 12-20 distinct feasible allocations per budget in a narrow actual-byte band, drawn from random swaps and different allocators. Report within-budget rank correlation, pairwise accuracy, and **validation** PPL of the proxy's selected plan. | Redesign/calibrate `c_it` on calibration/validation data; do not present cross-budget Pearson correlation as selection evidence or touch the final test set. |
| G3: Is the global-budget QUBO correct and usable on the 1000-bit machine? | Enumerate valid allocations on small instances and exhaustively check binary states only where tractable; then report full-instance spin/auxiliary count, nonzero couplings, coefficient range/rounding, hardware acceptance, and decoded feasibility before/after repair. | Change the encoding or allocation units, then rerun **all** baselines on that same problem; no full-model hardware claim until mapped. |
| G4: What remains after strong classical optimization? | Time the full MILP certificate and Lagrangian solution against both photonic core time and complete cloud round trip on the same proxy and budget. | If MILP certifies optimum faster than the cloud call, remove optimization/speed-advantage language and reassess the paper's hardware contribution before the large sweep. |

## 1. Problem and budget definition

Use the seven projection tensors in each supported dense decoder block as the
allocation units. For every eligible tensor `i`, select one actual GGUF tensor
type from `T_i subseteq {Q3_K, Q4_K, Q5_K, Q6_K}`. These names are quantization
types, not their physical bits per weight. Exclude a type only after recording
the shape/kernel incompatibility; do not silently allow a quantizer fallback.
Keep the source checkpoint, tokenizer, untargeted-tensor `--base-type`, and
importance matrix (`--imatrix`, if used) fixed across methods.

Encode the four types per tensor with two decision bits:
`00 -> Q3_K`, `01 -> Q4_K`, `10 -> Q5_K`, `11 -> Q6_K`. For compatible
256-weight K-quant blocks, their packed sizes are respectively 110, 144, 176,
and 210 bytes, so `m_i,Q3 + m_i,Q6 = m_i,Q4 + m_i,Q5` and the storage cost is
linear in those two bits. Verify this equality using the measured byte count
of **every** eligible tensor; reject a type fallback or nonconforming tensor
instead of assuming the identity. The four proxy losses need not be linear;
their two-bit expansion includes a quadratic cross term.

Let `m_it` be the measured or GGUF-block-calculated storage bytes of tensor `i`
at type `t`, including its alignment. Let `m_fixed` include the GGUF container,
metadata, and tensors outside the allocation. Validate this accounting against
several actual generated GGUF files before optimization; after quantization,
the **actual full file size** and the manifest's tensor types decide feasibility.
For a file-size budget `B`, define `B' = B - m_fixed` and solve:

```text
minimize    sum_i sum_(t in T_i) c_it x_it
subject to  sum_(t in T_i) x_it = 1               for every i
            sum_i sum_(t in T_i) m_it x_it <= B'
            x_it in {0, 1}
```

Here `c_it` is the same calibrated **additive proxy loss** supplied to every
allocator. Estimate it from the existing sensitivity method or measured local
quantization error, then test whether it ranks *complete allocations* at a
fixed budget on held-out data. Split proxy fitting/calibration and final PPL
evaluation data. The direct multiple-choice knapsack formulation and MILP
optimality certificate apply to this additive proxy, not to actual PPL. If a
new paper objective includes pairwise interactions or coupled type constraints,
encode the same terms/constraints for all solvers (for example by MILP
linearization or MIQP) and label the problem accordingly.

Construct a **new four-type** `c_it` table rather than reusing the historical
single-4-bit Fisher values. Compare two calibration-set candidates: (a)
activation-weighted output perturbation from actual quantized/dequantized GGUF
weights, `mean ||X_i (W_i - dequant(Q_t(W_i)))||^2`; and (b) diagonal-Fisher
weight error, `sum_j F_ij (W_ij - dequant(Q_t(W_i))_j)^2`, if a new Fisher
diagonal can be measured. Four targeted-uniform pilot GGUFs with the same
untargeted-tensor preset and `imatrix` can supply the per-type perturbations;
record their validated tensor types. Compare these with the original
Fisher/exponential proxy on the **validation** split, then freeze the proxy
before any final test-corpus evaluation. GGUF weight reading/dequantization and
activation collection required for this table are work items, not current
wrapper features.

Keep file-size budget distinct from device-memory budget. For each evaluated
GGUF, also measure peak runtime memory with fixed context, batch, KV-cache
type, offload, and hardware. A claim that user-supplied *device memory* is
guaranteed requires a separately validated memory predictor or an explicit
post-quantization runtime feasibility check.

## 2. Replacement baseline set

| Role | Method | Required implementation and reporting |
| --- | --- | --- |
| Proposed | 1000-bit photonic Ising allocation | Optimize the same proxy and global budget; decode, check, and repair feasibility; report original proxy objective and hardware/system time. |
| Primary classical | MILP, using Gurobi, SCIP, HiGHS, or another specified solver | Encode the model above directly. Record incumbent, best bound, relative gap, feasibility, time limit, threads, and version. Report "optimal" only when the solver certifies it. |
| Primary fast classical | Lagrangian relaxation + budget repair + local exchange | Use the algorithm below; report the best feasible solution, dual bound where valid, and elapsed time. |
| Simple control | Feasible marginal-gain-per-byte greedy | Start from the smallest-byte assignment, choose an affordable upgrade by proxy improvement per additional byte, and continue scanning after an unaffordable upgrade. Implement afresh; the historical `greeady.py` stops too early. |
| Deployment references | Targeted-projection uniform `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`; whole-model llama.cpp `Q4_K_M`/`Q5_K_S` presets; F16 source | Use an allocation `default` for uniform eligible projections while keeping the same untargeted-tensor base preset. Generate whole-model presets directly with the pinned `llama-quantize`. Presets are practical references, **not** per-tensor optimizers solving the exact same candidate-set problem. Show their actual budget positions; F16 is a quality ceiling, not a matched-budget control. Historical `Q8_0` results remain an external high-quality reference outside the new four-choice optimization. |
| Conditional supplement | DP + Pareto pruning | Use where byte discretization and state count are tractable; disclose rounding and any loss of an exact guarantee. Useful for a quality-storage frontier. |
| Conditional supplement | OR-Tools CP-SAT | Add when integer costs, type compatibility, tying `gate/up` types, or other deployment constraints make it informative; give the same objective, limit, and feasibility checks. |
| Historical/appendix only | Tabu, SA, GA, RL, Bayesian optimization | Retain at most one or two rerun controls if informative. Do not use their old unequal-budget numbers as the main evidence. Replace the old "Cplex, 1000 iterations" row with a newly certified or bounded MILP run. |

For the Lagrangian baseline, minimize `c_it + lambda * m_it` independently per
tensor while searching `lambda >= 0`; retain all distinct assignments and the
best valid lower bound
`g(lambda) = sum_i min_t(c_it + lambda * m_it) - lambda * B'`.
The penalized objective without the `-lambda * B'` term is not a bound. If an
assignment exceeds `B'` in allocatable-tensor bytes, repeatedly downgrade the
choice with the smallest proxy-loss increase per byte saved until feasible.
Spend remaining bytes on beneficial upgrades, then try budget-preserving
one-tensor changes and two-tensor exchanges until no strict improvement occurs
or the shared time limit expires. Run from at least the minimum-byte assignment
and the best assignments found along the multiplier search; specify tie breaks,
seed, termination tolerance, and feasibility tolerance. Never compare an
over-budget relaxed solution as if it were deployable. This per-tensor
minimization applies to the uncoupled additive model; if interactions or tied
tensor choices become part of the main objective, adapt and validate the
relaxation and bound for that exact formulation.

## 3. Fair comparison protocol

1. Preflight the target model: inventory all seven projection names per block,
   supported tensor shapes and four types; pin model revision, llama.cpp commit,
   GGUF source, calibration corpus, `--base-type`, and `--imatrix` policy. The
   existing wrapper rejects missing or mismatched tensor types in its manifest.
2. Build one shared table of `m_it` and `c_it`. Train or tune the proxy only on
   calibration data. Give every optimizer identical eligible types, objective,
   global byte budget, and constraints. Any extra classical-only computation
   must be identified rather than hidden in preprocessing.
3. Calculate the smallest and largest feasible file sizes from the shared cost
   table. Precommit the Qwen2.5-7B main budgets after measuring targeted-uniform
   `Q3_K`, `Q4_K`, and `Q5_K` file sizes: use those three anchors plus the two
   midpoints, for five budgets in the practical Q3-to-Q5 range. The `Q6_K`
   choice remains available at every budget; add a Q6 anchor as a separately
   labeled higher-budget sensitivity check if useful. Use the three original
   anchors on the independent model. These are
   upper bounds (`actual_bytes <= B`), not exact target sizes; report both
   requested and actual bytes. Give controllable solvers a common core-time
   limit, and additionally compare their anytime solutions at the **observed
   full cloud-call latency**. Report the exact or bounded proxy optimum
   separately from the **observed** PPL-byte Pareto frontier of evaluated
   GGUFs; do not call a sparsely sampled frontier complete.
4. Compare optimizer outputs on a common sheet: budget feasibility, additive
   proxy objective, MILP incumbent/bound/gap, objective difference from the
   best certified bound, success rate, and time-to-feasible/time-to-best.
   Where the QUBO maps, run the photonic solver independently at least five
   times per budget, and
   report dispersion rather than only the best sample. Predeclare selection of
   the minimum-proxy **feasible raw** output; if none exists, apply a fixed
   repair rule and label its result separately. Never select a hardware sample
   using held-out PPL. Compare its decoded assignment on the **original**
   objective and bytes, not its scaled QUBO/Ising energy alone.
5. Quantize selected solutions through `llama_cpp/quantize.py build`, confirm
   `status: validated`, actual tensor types and actual file bytes, then run all
   models on one pinned llama.cpp build. If measured size exceeds `B`, mark the
   candidate infeasible and reoptimize/repair before comparing quality. Report
   held-out corpus PPL, at least two relevant downstream task scores if a
   same-engine harness is added, prompt/decode throughput, and peak runtime
   memory at identical context, batch, KV-cache, offload, and hardware settings.
   The current `evaluate.py` does not capture peak GPU memory; add external
   instrumentation and record its measurement method before making that claim.
   Before **every remote GPU run**, inspect `nvidia-smi` for each card's active
   processes, memory use, and utilization. Use an idle card with sufficient free
   memory; set `CUDA_VISIBLE_DEVICES` to its physical index and record its GPU
   UUID in the run log (the selected card appears as `CUDA0` inside that
   process). Keep benchmark comparisons on the same GPU when possible. If no
   suitable card is idle, wait rather than starting on an occupied card. Warm
   up `llama-bench` and run each throughput configuration at least five times;
   repeated identical PPL runs are unnecessary. If uncertainty intervals are
   reported for PPL, first add per-document token-NLL capture and bootstrap
   those paired document losses; `evaluate.py` currently records aggregate
   PPL only.
6. Time the shared sensitivity extraction, cost-table construction, QUBO/MILP
   construction, solver execution, quantization, and evaluation separately.
   For the cloud hardware also record upload/network, queue, programming,
   physical evolution, readout, decoding, and repair. Show both solver-only and
   full allocation wall time; a physical-evolution-only speed figure must be
   labeled as such.

### Models, data, and measurements

| Role | Model | Planned use |
| --- | --- | --- |
| Pipeline smoke | Qwen3-0.6B or another shape-compatible small dense model | Check conversion, all four types, manifest, PPL command, and small-instance optimization. It is not the main quality result. |
| Primary | Qwen2.5-7B, 28 blocks x 7 projections = 196 allocation units | Five precommitted budgets; all main optimizer and GGUF quality results. |
| Independent architecture | Prefer Mistral-7B-v0.3, 32 x 7 = 224 units; Llama-3.1-8B is an alternative | Three budget anchors, after checking model license, converter support, four types, and full hardware encoding. Two-bit choice encoding uses 448 decision variables before budget slack. Hardware compatibility still depends on coefficient precision, not only the nominal 1000-variable limit. State the single-model limitation if this cannot be completed. |

Use three disjoint data roles: a fixed C4 calibration subset (with saved
document IDs and seed) for FI/proxy fitting and any `imatrix`; a separate C4
validation subset for G2 proxy choice; and untouched WikiText-2 test plus a
disjoint C4 test subset for final E3 PPL. Freeze the proxy and budgets before
the final test evaluations.
Record corpus hashes, token counts, tokenizer, context/stride, and treatment
of document boundaries. For G2, sample allocations within a narrow actual-byte
band (target `[0.99B, B]`) around each of three interior Qwen budgets; use
Kendall/Spearman and pairwise-selection accuracy **within each budget**.
If the band contains too few distinct plans, report that and widen it by a
predeclared rule rather than mixing budgets. Compare the proxy-selected plan
with the observed best and median on held-out PPL, without tuning to test data.

PPL and budget feasibility are mandatory. Add two downstream tasks such as
HellaSwag and ARC-Challenge only after implementing a reproducible
same-llama.cpp scoring harness; otherwise leave task claims out of the main
paper. Do not transfer the old PyTorch/AWQ/GPTQ PPL or old PagedAttention
throughput figures into the new GGUF result tables.

### Experiment matrix and paper evidence

| ID | Scope | Primary output for the article |
| --- | --- | --- |
| E0: correctness | Enumerate all `4^n` valid allocations for `n <= 8`; exhaustively inspect binary QUBO states only when small enough; Qwen full QUBO export/rounding/decoding | Encoding equivalence, constraint violations, spin/auxiliary/coupling counts and coefficient-precision table. |
| E1: proxy validity | Three interior Qwen budgets; 12-20 validation allocations each, expanded if uncertainty remains | Per-budget proxy-vs-validation-PPL scatter and rank/selection statistics; no pooled cross-budget correlation claim. |
| E2: optimizer comparison | Five Qwen and three independent-model budgets; MILP, Lagrangian+exchange, corrected greedy, plus photonic **only where the full QUBO maps** | Original-proxy objective, certified MILP bound/gap, raw and repaired feasibility, success distribution, full cloud and classical anytime timing where hardware runs. |
| E3: real quality | Quantize preselected outputs from E2 on one pinned llama.cpp build, plus uniform/preset references at measured sizes; include photonic only where mapped | Actual-GGUF-bytes vs untouched test PPL plot and **observed** Pareto frontier for each model. Clearly mark any model without hardware results. |
| E4: deployment | At least two representative Qwen budgets (near uniform Q4 and Q5); same idle GPU, context/batch/KV/offload settings | Prompt/decode throughput and measured peak GPU memory with run-to-run intervals. A separate end-device-memory claim needs its own feasibility test. |
| E5: ablations | Three Qwen budgets; historical FI versus recalibrated `c_it`, coefficient rounding/penalty, and any grouping/repair | Explain proxy and hardware effects without changing the baseline problem mid-comparison. |

For E3, use the predeclared minimum-proxy feasible raw photonic output for the
full-budget PPL curve. If none is raw-feasible, use the fixed repaired fallback
and label it; if repair cannot meet the actual-byte budget, omit that hardware
point and report the failure. Quantify PPL variation across distinct hardware
outputs at at least two representative budgets. Do not choose any output by
test-corpus quality. Quantizing identical deterministic allocations repeatedly
does not add statistical evidence. Deduplicate identical allocation JSONs and
retain the run-to-allocation mapping.

## 4. 1000-bit hardware checks and ablations

- For a 28-block, seven-projection model, two-bit encoding starts at
  `28 * 7 * 2 = 392` decision variables. Add only the bounded-binary slack
  needed by the exact global byte budget; verify its count for each budget.
  A nominal 1000-bit capacity alone does not establish that the complete
  global-budget QUBO fits the hardware's coupling graph or 8-bit coefficient
  precision.
- An exact byte-budget slack encoding may require more variables and much
  larger coefficients than the two-bit choice terms alone. Record any byte scaling,
  coefficient clipping, or 8-bit rounding error, then check decoded solutions
  against **unscaled actual bytes**; hardware acceptance is not a feasibility
  certificate.
- Keep the exact, unrounded two-bit QUBO as a mathematical reference. Submit
  only separately validated hardware matrices whose **QUBO and converted
  Ising coefficients** both fit the platform's signed 8-bit range. A smaller
  subproblem or decomposed run must be labeled as such; it is not evidence
  that the full 196-tensor global budget was solved on the device.
- Rebuild the QUBO from the **global** storage constraint. The historical base
  script sums separate per-tensor squared budget penalties, which is not a
  global allocation budget. Validate encoding, penalties, quantized
  coefficients, and decoding by enumerating all valid allocations on small
  instances, and by exhaustive binary-state checks only for tractable QUBOs,
  before running hardware.
- Ablate the proxy (historical Fisher score versus a calibrated tensor-type
  cost table), hardware coefficient rounding, and any grouping/repair strategy.
  If grouping changes the allocation units, rebuild the MILP and Lagrangian
  baselines on those exact same units and constraints.
- If MILP proves the shared additive model optimal quickly, report that result.
  The photonic method cannot claim better optimization quality on that model;
  frame any remaining contribution around verified hardware behavior or a
  genuinely harder, identically defined deployment problem.

## 5. Execution order and deliverables

| Priority | Work item | Completion evidence |
| --- | --- | --- |
| P0 | Validate GGUF four-type compatibility and byte accounting; define/falsify the proxy at fixed budgets. | Shared per-tensor cost table, source hashes, pilot GGUF manifests, held-out ranking plots. |
| P0 | Implement fresh MILP and Lagrangian + repair + exchange baselines, plus a corrected greedy control. | Versioned allocation JSON per method/budget, solver logs, certificates or bounds, unit tests on small instances. |
| P0 | Correct global-budget QUBO and test 1000-bit hardware encoding. | Small-instance equivalence tests and physical-run logs with decoded feasibility. |
| P1 | Run same-budget Qwen2.5-7B GGUF/PPL/throughput comparisons; add peak-memory instrumentation. | Main solver table plus observed actual-bytes vs PPL/frontier figures, repeated-run intervals, measured peak GPU memory. |
| P1 | Add independent dense model; add downstream tasks only after a same-engine scoring harness is validated. | External-validity table, plus reproducible task settings if tasks are reported. |
| P2 | Add DP/Pareto or CP-SAT where their respective assumptions/constraints apply; limited historical ablation. | Supplementary frontier, bound, and sensitivity analyses. |

The main paper should separate (a) surrogate optimization and its certificates,
(b) real GGUF quality at measured budgets, and (c) hardware timing. Old
unpacked AWQ/GPTQ-labeled results, nonmatching average-bit budgets, and
uncertified Cplex values must not be mixed into the new GGUF main table.
The existing Qwen2.5-7B `model_info` sensitivity/PPL files describe the old
4-bit experiment; they are not a four-type GGUF `c_it` table.

### Manuscript output map and claim rules

| Manuscript item | Evidence to prepare |
| --- | --- |
| Method/correctness table | Formal FI/proxy definition; candidate-to-binary map; global inequality and any slack/ancilla construction; binary-to-Ising substitution; variable, coupling, and coefficient counts before/after hardware scaling. |
| Fixed-budget proxy figure | E1 within-budget scatter, rank correlation with uncertainty, and examples where the proxy selects a better or worse actual GGUF. |
| Solver table | E2 incumbent/bound/gap, raw/repaired feasibility, repeated hardware success, and physical/core/full-cloud stage times. |
| Quality-storage figure | E3 actual full GGUF bytes versus held-out PPL, with each model and all comparable allocators identified. Label only the **observed** nondominated points. |
| Deployment table | E4 prompt/decode throughput, peak GPU memory and context settings on one engine/hardware configuration; no transfer of the old engine's PagedAttention or kernel claims. |
| Prior-work delta table | State exactly which Q4Q conference elements are inherited and which formulation, measurements, hardware experiments, and deployment evidence are genuinely new. |

Do not call a photonic output globally optimal unless an independent bound
certifies its original constrained objective. A measured cloud-vs-classical
wall-time comparison, even when favorable, does **not** by itself establish a
general quantum-computational speedup. Treat llama.cpp as
the common quantization/inference measurement engine unless new engine code is
implemented and separately ablated. If G2 fails, the central budget-aware
allocation claim needs a new proxy; if G3 fails, the 1000-bit full-model claim
needs a new encoding or a transparently smaller problem.
