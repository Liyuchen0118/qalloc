# Mixed-type GGUF with llama.cpp

These wrappers use an unmodified [llama.cpp](https://github.com/ggml-org/llama.cpp)
checkout. `quantize.py` converts a supported Hugging Face model to floating-point
GGUF, applies a QuBit allocation to seven projection matrices in each dense
decoder block, and checks the types in the output GGUF. `evaluate.py` invokes
llama.cpp for generation, throughput benchmarking, or corpus perplexity. Supply
your own licensed model weights and evaluation text; neither is in this repo.

## Setup

From the qalloc root, place llama.cpp alongside this repository and pin the
version used by the wrappers:

```bash
git clone https://github.com/ggml-org/llama.cpp.git ../llama.cpp
git -C ../llama.cpp checkout 493253fe5a5af7e800b3c92a2452b5684a99e961
cmake -S ../llama.cpp -B ../llama.cpp/build-qalloc -DCMAKE_BUILD_TYPE=Release
cmake --build ../llama.cpp/build-qalloc \
  --target llama-quantize llama-completion llama-bench llama-perplexity -j 4
```

The checkout currently used for local development is
`493253fe5a5af7e800b3c92a2452b5684a99e961`, built for CPU. For NVIDIA
measurements, configure a separate build with `-DGGML_CUDA=ON` on a CUDA host;
record the hardware, backend, and llama.cpp commit with each result.

Use Python 3.10 or newer. The GGUF reader is needed even when starting from an
existing F16/BF16 GGUF:

```bash
python -m pip install -e ../llama.cpp/gguf-py
```

Only when converting Hugging Face weights, also install llama.cpp's conversion
requirements (including PyTorch and Transformers):

```bash
python -m pip install -r ../llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

If the directory already exists, verify its Git revision and rebuild its
executables instead of cloning over it. Do not add llama.cpp or model files to
the qalloc Git repository.

## Allocation and quantization

Start with the *original* Hugging Face weights or an unquantized F16/BF16/F32
GGUF. Prequantized GGUF and the legacy AWQ/GPTQ-labeled floating-weight outputs
are not valid input to this quantizer. For conversion:

```bash
python llama_cpp/quantize.py convert \
  --llama-cpp-dir ../llama.cpp \
  --hf-model /path/to/hf-model \
  --output-gguf /path/to/model-f16.gguf
```

Create `/path/to/allocation.json` with a default type and any per-projection
overrides, for example:

```json
{
  "default": 4,
  "overrides": {
    "layers.0.self_attn.q_proj": 8,
    "layers.0.mlp.down_proj": 5
  }
}
```

The seven names per block are `self_attn.q_proj`, `self_attn.k_proj`,
`self_attn.v_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, and
`mlp.down_proj`. The wrapper also accepts a flat list in this order, repeated
for every block, an object keyed by `"0"`, `"1"`, etc. with all seven names per
block, or an object with every `layers.N.<name>` key. The source GGUF must have
all seven matrices in each consecutively numbered block; fused projections and
Mixture-of-Experts layouts need separate support.

Numeric choices `2, 3, 4, 5, 6, 8, 16` map respectively to `Q2_K`, `Q3_K`,
`Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `F16`. These are *tensor types*, not exact physical
bits per weight: block metadata, alignment, unallocated tensors, and the GGUF
container affect storage. `Q4_K_M` and `Q5_K_S` are whole-model presets, not
valid individual tensor choices. Compare actual file sizes and measured memory.
The shape check is material: SmolLM2-135M has 576-wide projection dimensions
that cannot use the K-quant block size in affected tensors, so its local smoke
run used Q8_0/F16 choices. Qwen3-0.6B is a K-quant-compatible small model for
testing the K-type path; use the actual target architecture for paper results.

Inspect the allocation plan before creating a model, then quantize:

```bash
python llama_cpp/quantize.py build \
  --llama-cpp-dir ../llama.cpp \
  --source-gguf /path/to/model-f16.gguf \
  --allocation /path/to/allocation.json \
  --output-gguf /path/to/model-mixed.gguf \
  --base-type Q4_K_M --plan-only

python llama_cpp/quantize.py build \
  --llama-cpp-dir ../llama.cpp \
  --source-gguf /path/to/model-f16.gguf \
  --allocation /path/to/allocation.json \
  --output-gguf /path/to/model-mixed.gguf \
  --base-type Q4_K_M --threads 4
```

The base preset controls tensors outside the seven-projection allocation.
Optionally pass `--imatrix /path/to/imatrix.dat` when comparing runs that use
the same calibration data. The build writes a tensor-rule file, quantizer log,
and `<output>.allocation.json` beside the model. Check that the manifest has
`"status": "validated"` and inspect `actual_tensor_types` before using the
model. The wrapper rejects missing tensors, shape changes, and type fallback by
default. A type mismatch leaves a failed status in the manifest.

## Inference, throughput, and perplexity

Run these from the qalloc root after the mixed GGUF exists:

```bash
python llama_cpp/evaluate.py infer \
  --bin-dir ../llama.cpp/build-qalloc/bin \
  --model /path/to/model-mixed.gguf \
  --prompt 'What is mixed-precision quantization?' \
  --threads 4 --gpu-layers 0 --predict 128

python llama_cpp/evaluate.py bench \
  --bin-dir ../llama.cpp/build-qalloc/bin \
  --model /path/to/model-mixed.gguf \
  --threads 4 --gpu-layers 0 \
  --prompt-tokens 512 --gen-tokens 128 --repetitions 5

python llama_cpp/evaluate.py ppl \
  --bin-dir ../llama.cpp/build-qalloc/bin \
  --model /path/to/model-mixed.gguf \
  --corpus /path/to/evaluation.txt \
  --threads 4 --gpu-layers 0 --ctx-size 2048
```

The PPL corpus must be plain text suitable for the model's tokenizer; use the
same corpus and context/stride settings across candidates. For GPU runs, pass
the relevant `--gpu-layers` and, when needed, `--device`; use a GPU-enabled
llama.cpp build. For a quick PPL smoke test, add `--max-chunks 1`; omit it for
full-corpus PPL and never report the limited run as a full evaluation. `bench`
records prompt processing and generation throughput.
All three commands write timestamped JSON to `results/experiment_logs/` by
default, or to `--output /path/to/result.json`; the JSON includes command
settings and executable version. Compare candidates with identical model,
corpus, calibration, hardware, backend, and runtime parameters. GGUF file size
and `llama-bench` throughput alone do not establish end-to-end application
latency or deployment memory use.
