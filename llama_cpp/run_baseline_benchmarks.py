"""Validate allocation GGUFs and evaluate them on an idle CUDA device."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

__author__ = "Yuchen Li"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _gpu_state(index: int) -> dict[str, Any]:
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().splitlines()
    cards = {}
    for row in rows:
        fields = [field.strip() for field in row.split(",")]
        cards[int(fields[0])] = {
            "index": int(fields[0]), "uuid": fields[1],
            "memory_used_mib": int(fields[2]), "utilization_percent": int(fields[3]),
        }
    if index not in cards:
        raise ValueError(f"GPU {index} not present")
    apps = subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader",
    ], capture_output=True, text=True, check=False)
    if apps.returncode:
        raise RuntimeError(f"Cannot inspect active GPU processes: {apps.stderr}")
    card = cards[index]
    card["compute_processes"] = [
        row.strip() for row in apps.stdout.splitlines() if row.strip().split(",")[0].strip() == card["uuid"]
    ]
    return card


def _require_idle(index: int, settle_seconds: int = 60) -> dict[str, Any]:
    deadline = time.monotonic() + settle_seconds
    while True:
        card = _gpu_state(index)
        if card["memory_used_mib"] <= 64 and card["utilization_percent"] == 0 and not card["compute_processes"]:
            return card
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU {index} stayed busy: {card}")
        time.sleep(2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model(path: Path, entry: dict[str, Any], llama_cpp_dir: Path) -> None:
    sidecar = Path(str(path) + ".allocation.json")
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    if info["status"] != "validated" or info["type_mismatches"]:
        raise ValueError(f"Quantization manifest did not validate: {sidecar}")
    if len(info["actual_tensor_types"]) != 196:
        raise ValueError(f"Expected 196 validated projection types: {path}")
    if path.stat().st_size != entry["expected_full_file_bytes"]:
        raise ValueError(f"Measured model bytes differ from the solver table: {path}")
    sys.path.insert(0, str(llama_cpp_dir / "gguf-py"))
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    if reader.fields["general.author"].contents() != __author__:
        raise ValueError(f"GGUF author metadata differs: {path}")


def _evaluate(args: argparse.Namespace, model: Path, kind: str, uuid: str) -> dict[str, Any]:
    output = args.output_dir / f"{model.stem}.{kind}.json"
    common = [
        "--model", str(model), "--bin-dir", str(args.bin_dir),
        "--threads", str(args.threads), "--gpu-layers", "999", "--device", "CUDA0",
        "--batch-size", "2048", "--ubatch-size", "512", "--flash-attn", "on",
        "--output", str(output), "--timeout", str(args.timeout),
    ]
    if kind == "ppl":
        mode = ["ppl", *common, "--corpus", str(args.corpus), "--ctx-size", "2048"]
        if args.smoke:
            mode += ["--max-chunks", "2"]
    elif kind == "prefill":
        mode = ["bench", *common, "--prompt-tokens", "512", "--gen-tokens", "0",
                "--repetitions", "1" if args.smoke else "5"]
    else:
        mode = ["bench", *common, "--prompt-tokens", "0", "--gen-tokens", "128",
                "--repetitions", "1" if args.smoke else "5"]
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if (existing.get("status") == "ok"
                and existing.get("run_environment", {}).get("gpu_uuid") == uuid
                and existing.get("model", {}).get("path") == str(model)
                and existing.get("model", {}).get("size_bytes") == model.stat().st_size
                and existing.get("parameters", {}).get("corpus_sha256") == (_sha256(args.corpus) if kind == "ppl" else None)
                and existing.get("parameters", {}).get("max_chunks") == (2 if args.smoke and kind == "ppl" else None)
                and existing.get("parameters", {}).get("prompt_tokens") == (512 if kind == "prefill" else 0 if kind == "decode" else None)
                and existing.get("parameters", {}).get("gen_tokens") == (128 if kind == "decode" else 0 if kind == "prefill" else None)
                and existing.get("parameters", {}).get("repetitions") == (1 if args.smoke and kind != "ppl" else 5 if kind != "ppl" else None)):
            return {"kind": kind, "result_file": str(output), "status": "ok", "reused": True}
        raise FileExistsError(f"Existing result does not match current inputs: {output}")
    card = _require_idle(args.gpu_index)
    if card["uuid"] != uuid:
        raise RuntimeError("GPU UUID changed during this experiment")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, QALLOC_GPU_UUID=uuid)
    process = subprocess.Popen([sys.executable, str(args.evaluate_script), *mode],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    peak_mib = card["memory_used_mib"]
    while process.poll() is None:
        peak_mib = max(peak_mib, _gpu_state(args.gpu_index)["memory_used_mib"])
        time.sleep(0.5)
    stdout, stderr = process.communicate()
    record = {
        "kind": kind, "result_file": str(output), "status": "ok" if process.returncode == 0 else "failed",
        "exit_code": process.returncode, "gpu_before": card, "peak_gpu_memory_mib": peak_mib,
        "stdout_tail": stdout[-2000:], "stderr_tail": stderr[-2000:],
    }
    if process.returncode:
        raise RuntimeError(f"{model.name}/{kind} failed: {record}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--llama-cpp-dir", required=True, type=Path)
    parser.add_argument("--bin-dir", required=True, type=Path)
    parser.add_argument("--evaluate-script", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["distinct_allocations"][:1] if args.smoke else manifest["distinct_allocations"]
    for entry in entries:
        path = args.model_dir / entry["model_file"]
        _verify_model(path, entry, args.llama_cpp_dir)
        if any(path.stat().st_size > manifest["runs"][i]["budget_full_file_bytes"]
               for i, run in enumerate(manifest["runs"]) if run["allocation_id"] == entry["id"]):
            raise ValueError(f"Model exceeds one of its assigned budgets: {path}")
    card = _require_idle(args.gpu_index)
    report_path = args.output_dir / ("smoke_run.json" if args.smoke else "full_run.json")
    report: dict[str, Any] = {
        "author": __author__, "experiment": "Qwen2.5-7B mixed-type classical baselines",
        "kind": "smoke" if args.smoke else "full", "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus), "gpu": card,
        "llama_cpp_commit": subprocess.check_output([
            "git", "-C", str(args.llama_cpp_dir), "rev-parse", "HEAD"], text=True).strip(),
        "runs": [],
    }
    _write_json(report_path, report)
    for entry in entries:
        model = args.model_dir / entry["model_file"]
        for kind in ("ppl", "prefill", "decode"):
            record = _evaluate(args, model, kind, card["uuid"])
            report["runs"].append({"allocation_id": entry["id"], **record})
            _write_json(report_path, report)
            print(f"{entry['id']} {kind}: {record['status']}", flush=True)
    print(report_path)


if __name__ == "__main__":
    main()
