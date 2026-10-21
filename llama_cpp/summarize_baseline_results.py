"""Join measured GGUF evaluations to each budgeted solver comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

__author__ = "Yuchen Li"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(
    manifest: dict[str, Any], full_run: dict[str, Any], result_dir: Path,
    reference_run: dict[str, Any] | None = None, reference_dir: Path | None = None,
) -> dict[str, Any]:
    by_id: dict[str, dict[str, Any]] = {}
    for run in full_run["runs"]:
        if run["status"] != "ok":
            raise ValueError(f"Incomplete evaluation: {run}")
        by_id.setdefault(run["allocation_id"], {})[run["kind"]] = run
    if len(by_id) != len(manifest["distinct_allocations"]):
        raise ValueError("One or more distinct allocation models are missing evaluations")

    measured: dict[str, dict[str, Any]] = {}
    for entry in manifest["distinct_allocations"]:
        allocation_id = entry["id"]
        records = by_id[allocation_id]
        if set(records) != {"ppl", "prefill", "decode"}:
            raise ValueError(f"Incomplete metrics for {allocation_id}")
        values = {kind: _load(result_dir / f"{allocation_id}.{kind}.json") for kind in records}
        if any(value["status"] != "ok" for value in values.values()):
            raise ValueError(f"A raw evaluation failed for {allocation_id}")
        if any(value["model"]["size_bytes"] != entry["expected_full_file_bytes"] for value in values.values()):
            raise ValueError(f"Measured model size differs for {allocation_id}")
        if any(Path(value["model"]["path"]).name != entry["model_file"] for value in values.values()):
            raise ValueError(f"Measured model identity differs for {allocation_id}")
        if any(value["run_environment"]["gpu_uuid"] != full_run["gpu"]["uuid"] for value in values.values()):
            raise ValueError(f"GPU differs for {allocation_id}")
        if any(full_run["llama_cpp_commit"][:9] not in value["executable_version"] for value in values.values()):
            raise ValueError(f"llama.cpp commit differs for {allocation_id}")
        ppl_params = values["ppl"]["parameters"]
        if (ppl_params["corpus_sha256"] != full_run["corpus_sha256"]
                or ppl_params["max_chunks"] is not None or ppl_params["ctx_size"] != 2048):
            raise ValueError(f"PPL corpus or full-test settings differ for {allocation_id}")
        for kind, prompt, generated in (("prefill", 512, 0), ("decode", 0, 128)):
            params = values[kind]["parameters"]
            if (params["prompt_tokens"] != prompt or params["gen_tokens"] != generated
                    or params["repetitions"] != 5):
                raise ValueError(f"Benchmark settings differ for {allocation_id}/{kind}")
            if values[kind]["metrics"]["benchmarks"][0]["backends"] != "CUDA":
                raise ValueError(f"CUDA backend missing for {allocation_id}/{kind}")
        if any(value["parameters"]["gpu_layers"] != 999 or
               value["parameters"]["device"] != "CUDA0" or
               value["parameters"]["batch_size"] != 2048 or
               value["parameters"]["ubatch_size"] != 512 or
               value["parameters"]["flash_attn"] != "on" for value in values.values()):
            raise ValueError(f"Common execution settings differ for {allocation_id}")
        measured[allocation_id] = {
            "actual_file_bytes": entry["expected_full_file_bytes"],
            "wikitext2_test_ppl": values["ppl"]["metrics"]["perplexity"],
            "prefill_tokens_per_second": values["prefill"]["metrics"]["benchmarks"][0]["avg_ts"],
            "prefill_stddev_tokens_per_second": values["prefill"]["metrics"]["benchmarks"][0]["stddev_ts"],
            "decode_tokens_per_second": values["decode"]["metrics"]["benchmarks"][0]["avg_ts"],
            "decode_stddev_tokens_per_second": values["decode"]["metrics"]["benchmarks"][0]["stddev_ts"],
            "peak_gpu_memory_mib": max(record["peak_gpu_memory_mib"] for record in records.values()),
        }

    rows = []
    for run in manifest["runs"]:
        metric = measured[run["allocation_id"]]
        if metric["actual_file_bytes"] > run["budget_full_file_bytes"]:
            raise ValueError(f"Observed budget exceeded for {run}")
        rows.append({**run, **metric})
    report = {
        "author": __author__,
        "model": manifest["model"],
        "loss_definition": manifest["loss_definition"],
        "corpus_sha256": full_run["corpus_sha256"],
        "gpu": full_run["gpu"],
        "llama_cpp_commit": full_run["llama_cpp_commit"],
        "distinct_models": len(measured),
        "comparisons": rows,
    }
    if reference_run is not None:
        if reference_dir is None:
            raise ValueError("Reference result directory is required")
        if (reference_run["corpus_sha256"] != full_run["corpus_sha256"]
                or reference_run["gpu"]["uuid"] != full_run["gpu"]["uuid"]
                or reference_run["llama_cpp_commit"] != full_run["llama_cpp_commit"]):
            raise ValueError("Reference experiments used different corpus, GPU, or llama.cpp version")
        reference_rows = []
        grouped: dict[str, dict[str, Any]] = {}
        for row in reference_run["references"]:
            grouped.setdefault(row["label"], {})[row["kind"]] = row
        for label, records in grouped.items():
            if set(records) != {"ppl", "prefill", "decode"}:
                raise ValueError(f"Reference {label} is incomplete")
            stem = Path(records["ppl"]["model"]).stem
            values = {kind: _load(reference_dir / f"{stem}.{kind}.json") for kind in records}
            if any(value["status"] != "ok" or
                   value["model"]["path"] != records[kind]["model"] or
                   value["model"]["size_bytes"] != records[kind]["model_bytes"] or
                   value["run_environment"]["gpu_uuid"] != full_run["gpu"]["uuid"] or
                   full_run["llama_cpp_commit"][:9] not in value["executable_version"]
                   for kind, value in values.items()):
                raise ValueError(f"Reference provenance differs for {label}")
            if (values["ppl"]["parameters"]["corpus_sha256"] != full_run["corpus_sha256"]
                    or values["ppl"]["parameters"]["max_chunks"] is not None
                    or values["ppl"]["parameters"]["ctx_size"] != 2048):
                raise ValueError(f"Reference PPL settings differ for {label}")
            for kind, prompt, generated in (("prefill", 512, 0), ("decode", 0, 128)):
                params = values[kind]["parameters"]
                if (params["prompt_tokens"] != prompt or params["gen_tokens"] != generated
                        or params["repetitions"] != 5 or
                        values[kind]["metrics"]["benchmarks"][0]["backends"] != "CUDA"):
                    raise ValueError(f"Reference benchmark settings differ for {label}/{kind}")
            reference_rows.append({
                "label": label,
                "actual_file_bytes": values["ppl"]["model"]["size_bytes"],
                "wikitext2_test_ppl": values["ppl"]["metrics"]["perplexity"],
                "prefill_tokens_per_second": values["prefill"]["metrics"]["benchmarks"][0]["avg_ts"],
                "decode_tokens_per_second": values["decode"]["metrics"]["benchmarks"][0]["avg_ts"],
                "peak_gpu_memory_mib": max(row["peak_gpu_memory_mib"] for row in records.values()),
            })
        report["references"] = reference_rows
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--full-run", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--reference-run", type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if bool(args.reference_run) != bool(args.reference_dir):
        parser.error("--reference-run and --reference-dir must be given together")
    report = summarize(
        _load(args.manifest), _load(args.full_run), args.result_dir,
        _load(args.reference_run) if args.reference_run else None, args.reference_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['comparisons'])} comparisons for {report['distinct_models']} GGUF models")


if __name__ == "__main__":
    main()
