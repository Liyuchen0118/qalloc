#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 MODEL_PATH OUTPUT_DIR [NSAMPLES] [SEQLEN]" >&2
    exit 2
fi

MODEL_PATH=$1
OUTPUT_DIR=$2
NSAMPLES=${3:-32}
SEQLEN=${4:-2048}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting evaluation for model at $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"

# 1. Extract Layer Params and Perplexities
echo "----------------------------------------------------------------"
echo "Step 1: Extracting Layer Parameters and Perplexities..."
echo "----------------------------------------------------------------"
python "$SCRIPT_DIR/extract_info.py" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_samples "$NSAMPLES" \
    --seq_len "$SEQLEN"

# 2. Extract Fisher Information (Differential)
echo "----------------------------------------------------------------"
echo "Step 2: Extracting Fisher Information (Differential)..."
echo "----------------------------------------------------------------"
python "$SCRIPT_DIR/standalone_fisher_info.py" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --test_bit "4" \
    --nsamples "$NSAMPLES" \
    --seqlen "$SEQLEN"

mv "$OUTPUT_DIR/fisher_info.json" "$OUTPUT_DIR/fisher.json"

echo "----------------------------------------------------------------"
echo "Evaluation Complete!"
echo "Files generated in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"
