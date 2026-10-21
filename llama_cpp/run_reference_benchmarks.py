"""Evaluate uniform, native-preset, and F16 reference GGUF models."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_baseline_benchmarks import _evaluate, _require_idle, _sha256, _write_json

__author__ = "Yuchen Li"


def _reference(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("reference must be LABEL=/path/to/model.gguf")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=_reference, action="append", required=True)
    parser.add_argument("--llama-cpp-dir", required=True, type=Path)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--evaluate-script", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    args.smoke = False
    sys.path.insert(0, str(args.llama_cpp_dir / "gguf-py"))
    from gguf import GGUFReader

    labels = [label for label, _ in args.reference]
    if len(labels) != len(set(labels)):
        raise ValueError("Reference labels must be distinct")
    for label, model in args.reference:
        if not model.is_file():
            raise FileNotFoundError(model)
        reader = GGUFReader(str(model))
        if reader.fields["general.author"].contents() != __author__:
            raise ValueError(f"GGUF author differs for {label}")
        for suffix in (".allocation.json", ".preset.json"):
            sidecar = Path(str(model) + suffix)
            if sidecar.exists() and json.loads(sidecar.read_text(encoding="utf-8"))["status"] != "validated":
                raise ValueError(f"GGUF validation did not finish: {sidecar}")
    card = _require_idle(args.gpu_index)
    report_path = args.output_dir / "reference_run.json"
    report: dict[str, Any] = {
        "author": __author__, "corpus_sha256": _sha256(args.corpus), "gpu": card,
        "llama_cpp_commit": subprocess.check_output([
            "git", "-C", str(args.llama_cpp_dir), "rev-parse", "HEAD"], text=True).strip(),
        "references": [],
    }
    _write_json(report_path, report)
    for label, model in args.reference:
        for kind in ("ppl", "prefill", "decode"):
            record = _evaluate(args, model, kind, card["uuid"])
            report["references"].append({
                "label": label, "model": str(model), "model_bytes": model.stat().st_size, **record,
            })
            _write_json(report_path, report)
            print(f"{label} {kind}: {record['status']}", flush=True)
    print(report_path)


if __name__ == "__main__":
    main()
