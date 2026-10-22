"""Run matched old/new budget-5 GGUF evaluations on an idle cloud GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

__author__ = "Yuchen Li"


def idle_gpu(index: int) -> str:
    uuid, used, busy = gpu_state(index)
    if used > 1024 or busy > 10:
        raise RuntimeError(f"GPU {index} is busy: {used} MiB, {busy}%")
    return uuid


def gpu_state(index: int) -> tuple[str, int, int]:
    query = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True,
    )
    uuid, used, busy = [field.strip() for field in query.stdout.strip().split(",")]
    return uuid, int(used), int(busy)


def validate_cached_result(result: dict, mode: str, model: Path, corpus: Path,
                           gpu_index: int, expected_uuid: str, commit: str,
                           mode_args: list[str]) -> None:
    """Reject stale results when model or benchmark inputs have changed."""
    stat = model.stat()
    recorded_model = result.get("model", {})
    parameters = result.get("parameters", {})
    environment = result.get("run_environment", {})
    expected_parameters = {
        "mode": mode, "threads": 8, "gpu_layers": 999,
        "device": "CUDA0", "batch_size": 2048, "ubatch_size": 512,
        "flash_attn": "on", "timeout": 900,
    }
    if mode == "ppl":
        expected_parameters.update({
            "corpus": str(corpus), "ctx_size": 2048,
            "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        })
    else:
        expected_parameters.update({
            "prompt_tokens": int(mode_args[1]), "gen_tokens": int(mode_args[3]),
            "repetitions": int(mode_args[5]),
        })
    mismatches = [
        result.get("status") != "ok",
        result.get("mode") != mode,
        recorded_model.get("path") != str(model),
        recorded_model.get("size_bytes") != stat.st_size,
        recorded_model.get("mtime_ns") != stat.st_mtime_ns,
        environment.get("cuda_visible_devices") != str(gpu_index),
        environment.get("gpu_uuid") != expected_uuid,
        commit[:9] not in result.get("executable_version", ""),
        any(parameters.get(key) != value for key, value in expected_parameters.items()),
    ]
    if any(mismatches):
        raise RuntimeError("Cached evaluation differs from the current model, corpus, GPU, build, or settings")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = repo / "experiments/20261022/q6_data"
    old = repo / "experiments/20261021/models/allocation_a0f124bfa1e1.gguf"
    new = root / "models/budget_05_milp.gguf"
    corpus = repo / "experiments/20261021/corpus/wiki.test.raw"
    bindir = args.llama_cpp_dir / "build-qalloc-cuda/bin"
    output_dir = root / "eval"
    output_dir.mkdir(exist_ok=True)
    common = [
        "--bin-dir", str(bindir), "--threads", "8", "--gpu-layers", "999",
        "--device", "CUDA0", "--batch-size", "2048", "--ubatch-size", "512",
        "--flash-attn", "on", "--timeout", "900",
    ]
    scenarios = []
    for label, model in (("old_b5", old), ("new_b5", new)):
        scenarios.extend([
            (f"{label}_ppl", "ppl", model, ["--corpus", str(corpus), "--ctx-size", "2048"]),
            (f"{label}_prefill", "bench", model, ["--prompt-tokens", "512", "--gen-tokens", "0", "--repetitions", "5"]),
            (f"{label}_decode", "bench", model, ["--prompt-tokens", "0", "--gen-tokens", "128", "--repetitions", "5"]),
        ])
    plan = {
        "author": __author__,
        "gpu_index": args.gpu,
        "gpu_uuid": gpu_state(args.gpu)[0],
        "llama_cpp_commit": subprocess.check_output(["git", "-C", str(args.llama_cpp_dir), "rev-parse", "HEAD"], text=True).strip(),
        "scenarios": [name for name, _, _, _ in scenarios],
        "results": [],
    }
    manifest_path = root / "b5_matched_eval_manifest.json"
    for name, mode, model, mode_args in scenarios:
        target = output_dir / f"{name}.json"
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            validate_cached_result(existing, mode, model, corpus, args.gpu,
                                   plan["gpu_uuid"], plan["llama_cpp_commit"], mode_args)
            plan["results"].append({"scenario": name, "status": "reused", "path": str(target)})
            continue
        gpu_uuid = idle_gpu(args.gpu)
        if plan["gpu_uuid"] != gpu_uuid:
            raise RuntimeError("GPU UUID changed during matched evaluation")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["QALLOC_GPU_UUID"] = gpu_uuid
        command = [
            sys.executable, str(repo / "llama_cpp/evaluate.py"), mode,
            "--model", str(model), *common, *mode_args, "--output", str(target),
        ]
        completed = subprocess.run(command, env=env, text=True, capture_output=True)
        plan["results"].append({
            "scenario": name, "status": "ok" if completed.returncode == 0 else "failed",
            "path": str(target), "stdout_tail": completed.stdout[-1000:], "stderr_tail": completed.stderr[-1000:],
        })
        manifest_path.write_text(json.dumps(plan, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        print(name, completed.stdout[-300:], flush=True)
        if completed.returncode:
            raise RuntimeError(f"Evaluation failed for {name}: {completed.stderr[-1000:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
