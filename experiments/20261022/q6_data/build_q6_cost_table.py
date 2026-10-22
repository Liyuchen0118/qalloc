"""Measure Q6_K projection errors and extend the prior GGUF cost table."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

__author__ = "Yuchen Li"
TYPE_ORDER = ("Q3_K", "Q4_K", "Q5_K", "Q6_K")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(old_table: Path, q6_gguf: Path, llama_cpp_dir: Path) -> dict:
    sys.path.insert(0, str(llama_cpp_dir / "gguf-py"))
    import gguf

    old = json.loads(old_table.read_text(encoding="utf-8"))
    table = copy.deepcopy(old)
    source = Path(old["metadata"]["source_gguf"])
    if source.stat().st_size != old["metadata"]["source_gguf_bytes"]:
        raise ValueError("Source GGUF size no longer matches the prior experiment")

    source_reader = gguf.GGUFReader(str(source))
    q6_reader = gguf.GGUFReader(str(q6_gguf))
    source_tensors = {tensor.name: tensor for tensor in source_reader.tensors}
    q6_tensors = {tensor.name: tensor for tensor in q6_reader.tensors}
    if set(source_tensors) != set(q6_tensors):
        raise ValueError("Q6_K GGUF tensor inventory differs from F16 source")

    details = table["tensor_details"]
    if len(details) != old["metadata"]["tensor_count"]:
        raise ValueError("Prior table tensor count is inconsistent")
    target_bytes = 0
    for row in details:
        name = row["gguf_name"]
        source_tensor = source_tensors[name]
        quantized = q6_tensors[name]
        if quantized.tensor_type.name != "Q6_K":
            raise ValueError(f"{name} is {quantized.tensor_type.name}, expected Q6_K")
        if quantized.shape.tolist() != source_tensor.shape.tolist():
            raise ValueError(f"Shape mismatch for {name}")
        restored = gguf.quants.dequantize(quantized.data, quantized.tensor_type)
        source_values = np.asarray(source_tensor.data, dtype=np.float32)
        if restored.shape != source_values.shape:
            raise ValueError(f"Dequantized shape mismatch for {name}")
        difference = source_values - restored.astype(np.float32, copy=False)
        squared_error = float(np.einsum("i,i->", difference.ravel(), difference.ravel(), dtype=np.float64))
        source_norm_sq = float(np.einsum("i,i->", source_values.ravel(), source_values.ravel(), dtype=np.float64))
        q6_choice = {
            "bytes": int(quantized.n_bytes),
            "loss": squared_error,
            "squared_weight_error": squared_error,
            "relative_squared_error": squared_error / source_norm_sq if source_norm_sq else 0.0,
        }
        target_bytes += q6_choice["bytes"]
        old_choices = row["choices"]
        if set(old_choices) != {"Q3_K", "Q4_K", "Q5_K", "Q8_0"}:
            raise ValueError(f"Prior choices differ for {name}")
        if old_choices["Q3_K"]["bytes"] + q6_choice["bytes"] != (
            old_choices["Q4_K"]["bytes"] + old_choices["Q5_K"]["bytes"]
        ):
            raise ValueError(f"Two-bit byte identity fails for {name}")
        row["choices"] = {qtype: old_choices[qtype] for qtype in TYPE_ORDER[:-1]}
        row["choices"]["Q6_K"] = q6_choice
        del restored, source_values, difference

    fixed_bytes = q6_gguf.stat().st_size - target_bytes
    if fixed_bytes != old["metadata"]["fixed_non_target_bytes"]:
        raise ValueError(f"Untargeted bytes changed: {fixed_bytes}")
    metadata = table["metadata"]
    metadata["author"] = __author__
    metadata["candidate_gguf_bytes"] = {
        qtype: old["metadata"]["candidate_gguf_bytes"][qtype] for qtype in TYPE_ORDER[:-1]
    }
    metadata["candidate_gguf_bytes"]["Q6_K"] = q6_gguf.stat().st_size
    metadata["candidate_types"] = list(TYPE_ORDER)
    metadata["source_cost_table"] = str(old_table)
    metadata["source_cost_table_sha256"] = sha256(old_table)
    metadata["q6_candidate_gguf"] = str(q6_gguf)
    metadata["q6_candidate_sha256"] = sha256(q6_gguf)
    metadata["q6_loss_provenance"] = "measured squared weight error after dequantizing the actual Q6_K GGUF"
    metadata["two_bit_byte_identity_verified_tensors"] = len(details)
    table["tensors"] = [{"name": row["name"], "choices": row["choices"]} for row in details]
    return table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-table", required=True, type=Path)
    parser.add_argument("--q6-gguf", required=True, type=Path)
    parser.add_argument("--llama-cpp-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    table = build(args.old_table, args.q6_gguf, args.llama_cpp_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(table["metadata"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
