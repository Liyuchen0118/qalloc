"""Build measured storage and quantization-error costs from validated GGUF files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

__author__ = "Yuchen Li"
TYPE_ORDER = ("Q3_K", "Q4_K", "Q5_K", "Q8_0")
TARGET = re.compile(r"^blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$")
ROLE = {
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def _read_tensors(gguf_dir: Path, path: Path) -> tuple[Any, dict[str, Any]]:
    sys.path.insert(0, str(gguf_dir / "gguf-py"))
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    return reader, {tensor.name: tensor for tensor in reader.tensors}


def _candidate(value: str) -> tuple[str, Path]:
    tensor_type, separator, path = value.partition("=")
    if not separator or tensor_type not in TYPE_ORDER or not path:
        raise argparse.ArgumentTypeError("candidate must have the form Q3_K=/path/to/model.gguf")
    return tensor_type, Path(path)


def build_table(source: Path, candidates: dict[str, Path], gguf_dir: Path) -> dict[str, Any]:
    import gguf

    source_reader, source_tensors = _read_tensors(gguf_dir, source)
    target_names = sorted(name for name in source_tensors if TARGET.fullmatch(name))
    if len(target_names) != 196:
        raise ValueError(f"Expected 196 Qwen2.5-7B projection tensors, found {len(target_names)}")

    candidate_readers: dict[str, Any] = {}
    tensors_by_type: dict[str, dict[str, Any]] = {}
    for tensor_type in TYPE_ORDER:
        candidate_path = candidates[tensor_type]
        reader, tensors = _read_tensors(gguf_dir, candidate_path)
        candidate_readers[tensor_type] = reader
        tensors_by_type[tensor_type] = tensors
        if set(tensors) != set(source_tensors):
            raise ValueError(f"Tensor inventory differs in {tensor_type}: {candidate_path}")

    fixed_bytes_by_type: dict[str, int] = {}
    for tensor_type in TYPE_ORDER:
        tensors = tensors_by_type[tensor_type]
        for name in target_names:
            if tensors[name].tensor_type.name != tensor_type:
                raise ValueError(f"{tensor_type} candidate has {tensors[name].tensor_type.name} for {name}")
            if tensors[name].shape.tolist() != source_tensors[name].shape.tolist():
                raise ValueError(f"Shape mismatch for {tensor_type}/{name}")
        target_bytes = sum(int(tensors[name].n_bytes) for name in target_names)
        fixed_bytes_by_type[tensor_type] = candidates[tensor_type].stat().st_size - target_bytes
    if len(set(fixed_bytes_by_type.values())) != 1:
        raise ValueError(f"Untargeted/full-file overhead differs across uniform candidates: {fixed_bytes_by_type}")

    rows = []
    for name in target_names:
        match = TARGET.fullmatch(name)
        assert match is not None
        source_tensor = source_tensors[name]
        source_values = np.asarray(source_tensor.data, dtype=np.float32)
        source_norm_sq = float(np.einsum("i,i->", source_values.ravel(), source_values.ravel(), dtype=np.float64))
        choices: dict[str, dict[str, Any]] = {}
        for tensor_type in TYPE_ORDER:
            quantized_tensor = tensors_by_type[tensor_type][name]
            restored = gguf.quants.dequantize(quantized_tensor.data, quantized_tensor.tensor_type)
            if restored.shape != source_tensor.data.shape:
                raise ValueError(f"Dequantized shape mismatch for {tensor_type}/{name}")
            difference = source_values - restored.astype(np.float32, copy=False)
            squared_error = float(np.einsum("i,i->", difference.ravel(), difference.ravel(), dtype=np.float64))
            choices[tensor_type] = {
                "bytes": int(quantized_tensor.n_bytes),
                "loss": squared_error,
                "squared_weight_error": squared_error,
                "relative_squared_error": squared_error / source_norm_sq if source_norm_sq else 0.0,
            }
            del restored, difference
        rows.append({
            "name": f"layers.{match.group(1)}.{ROLE[match.group(2)]}",
            "gguf_name": name,
            "shape": [int(dim) for dim in source_tensor.shape],
            "source_dtype": source_tensor.tensor_type.name,
            "choices": choices,
        })

    uniform_full_bytes = {tensor_type: candidates[tensor_type].stat().st_size for tensor_type in TYPE_ORDER}
    anchor_types = ("Q3_K", "Q4_K", "Q5_K")
    full_anchors = [uniform_full_bytes[tensor_type] for tensor_type in anchor_types]
    full_budgets = sorted(set([
        *full_anchors,
        (full_anchors[0] + full_anchors[1]) // 2,
        (full_anchors[1] + full_anchors[2]) // 2,
    ]))
    fixed_bytes = next(iter(fixed_bytes_by_type.values()))
    return {
        "metadata": {
            "author": __author__,
            "model": "Qwen2.5-7B",
            "source_gguf": str(source),
            "source_gguf_bytes": source.stat().st_size,
            "candidate_gguf_bytes": uniform_full_bytes,
            "uniform_full_file_budget_bytes": full_budgets,
            "fixed_non_target_bytes": fixed_bytes,
            "loss_definition": "sum of squared differences between original F16 GGUF weights and actual dequantized candidate GGUF weights",
            "loss_warning": "weight-space proxy only; not an end-to-end PPL estimate and must be validated at fixed budgets",
            "tensor_count": len(rows),
        },
        "budgets": [budget - fixed_bytes for budget in full_budgets],
        "tensors": [{"name": row["name"], "choices": row["choices"]} for row in rows],
        "tensor_details": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True, action="append", type=_candidate)
    parser.add_argument("--llama-cpp-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    candidates = dict(args.candidate)
    if set(candidates) != set(TYPE_ORDER):
        parser.error(f"provide each candidate type once: {', '.join(TYPE_ORDER)}")
    table = build_table(args.source, candidates, args.llama_cpp_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(table["metadata"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
