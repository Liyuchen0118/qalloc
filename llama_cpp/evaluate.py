"""Run inference, throughput benchmarks, and perplexity with llama.cpp binaries.

This module invokes upstream executables directly. It does not import the
third-party llama-cpp-python package, and it never passes commands through a
shell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_PPL_RE = re.compile(
    r"Final estimate:\s*PPL\s*=\s*([0-9]+(?:\.[0-9]+)?)"
    r"(?:\s*\+/-\s*([0-9]+(?:\.[0-9]+)?))?"
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path: Path, *, include_hash: bool = False) -> dict[str, Any]:
    stat = path.stat()
    info: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if include_hash:
        info["sha256"] = _sha256_file(path)
    return info


def _find_binary(bin_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        candidate = bin_dir / name
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return candidate
    raise FileNotFoundError(
        f"No executable {' or '.join(names)} in {bin_dir}. "
        "Build llama.cpp and pass its build/bin directory."
    )


def _common_args(args: argparse.Namespace, *, bench: bool) -> list[str]:
    command = ["-m", str(args.model)]
    if args.threads is not None:
        command += ["-t", str(args.threads)]
    if args.gpu_layers is not None:
        command += ["-ngl", str(args.gpu_layers)]
    if args.device is not None:
        devices = args.device.replace(",", "/") if bench else args.device
        command += ["-dev", devices]
    if args.batch_size is not None:
        command += ["-b", str(args.batch_size)]
    if args.ubatch_size is not None:
        command += ["-ub", str(args.ubatch_size)]
    if args.cache_type_k is not None:
        command += ["-ctk", args.cache_type_k]
    if args.cache_type_v is not None:
        command += ["-ctv", args.cache_type_v]
    if args.flash_attn is not None:
        command += ["-fa", args.flash_attn]
    if args.main_gpu is not None:
        command += ["-mg", str(args.main_gpu)]
    if args.split_mode is not None:
        command += ["-sm", args.split_mode]
    return command


def _build_command(args: argparse.Namespace, executable: Path) -> list[str]:
    bench = args.mode == "bench"
    command = [str(executable), *_common_args(args, bench=bench)]
    if args.mode == "infer":
        command += ["-c", str(args.ctx_size), "-n", str(args.predict)]
        command += ["--seed", str(args.seed), "--temp", str(args.temperature)]
        command += ["--no-display-prompt", "--single-turn"]
        if executable.name.startswith("llama-completion"):
            command += ["--no-conversation"]
        if args.prompt_file:
            command += ["-f", str(args.prompt_file)]
        else:
            command += ["-p", args.prompt]
    elif args.mode == "bench":
        command += ["-p", str(args.prompt_tokens)]
        command += ["-n", str(args.gen_tokens)]
        command += ["-r", str(args.repetitions)]
        command += ["-d", str(args.context_depth), "-o", "json"]
    else:
        command += ["-f", str(args.corpus), "-c", str(args.ctx_size)]
        if args.ppl_stride is not None:
            command += ["--ppl-stride", str(args.ppl_stride)]
        if args.max_chunks is not None:
            command += ["--chunks", str(args.max_chunks)]
    return command


def _redact_command(command: list[str]) -> list[str]:
    redacted = command.copy()
    for i, value in enumerate(redacted[:-1]):
        if value in {"-p", "--prompt"}:
            # Bench uses -p for the public prompt-token count, not prompt text.
            if i > 0 and Path(redacted[0]).name == "llama-bench":
                continue
            redacted[i + 1] = "<redacted: prompt text>"
    return redacted


def _version(executable: Path) -> str | None:
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode:
        return None
    return (completed.stdout + completed.stderr).strip() or None


def _parse_metrics(mode: str, stdout: str, stderr: str) -> dict[str, Any]:
    if mode == "bench":
        parsed = json.loads(stdout)
        if not isinstance(parsed, list):
            raise ValueError("llama-bench JSON output must be an array")
        return {"benchmarks": parsed}
    if mode == "ppl":
        matches = _PPL_RE.findall(stdout + "\n" + stderr)
        if not matches:
            raise ValueError("llama-perplexity did not print a final PPL estimate")
        ppl, uncertainty = matches[-1]
        return {
            "perplexity": float(ppl),
            "perplexity_uncertainty": float(uncertainty) if uncertainty else None,
        }
    return {}


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, type=Path, help="Local GGUF model file.")
    parser.add_argument(
        "--bin-dir", required=True, type=Path, help="Directory with built llama.cpp executables."
    )
    parser.add_argument("--threads", type=_positive_int, help="CPU generation threads.")
    parser.add_argument("--gpu-layers", type=int, help="Layers to offload; use 0 for CPU only.")
    parser.add_argument("--device", help="llama.cpp device name(s), e.g. CUDA0 or CUDA0,CUDA1.")
    parser.add_argument("--batch-size", type=_positive_int, help="Logical batch size.")
    parser.add_argument("--ubatch-size", type=_positive_int, help="Physical microbatch size.")
    parser.add_argument("--cache-type-k", help="KV cache K type, e.g. f16 or q8_0.")
    parser.add_argument("--cache-type-v", help="KV cache V type, e.g. f16 or q8_0.")
    parser.add_argument("--flash-attn", choices=("on", "off", "auto"))
    parser.add_argument("--main-gpu", type=_nonnegative_int)
    parser.add_argument("--split-mode", choices=("none", "layer", "row", "tensor"))
    parser.add_argument(
        "--output", type=Path, help="Result JSON path (default: a timestamped file in results/experiment_logs)."
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing --output file.")
    parser.add_argument("--timeout", type=_positive_int, help="Terminate the executable after N seconds.")
    parser.add_argument(
        "--hash-model", action="store_true", help="Hash the full GGUF for exact artifact tracking."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a GGUF model with llama.cpp and save reproducible JSON results."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    infer = subparsers.add_parser("infer", help="Generate text with llama-completion (or llama-cli).")
    _add_common_options(infer)
    prompt = infer.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="Prompt text; redacted in the saved command.")
    prompt.add_argument("--prompt-file", type=Path, help="UTF-8 prompt file.")
    infer.add_argument("--ctx-size", type=_positive_int, default=2048)
    infer.add_argument("--predict", type=_positive_int, default=128, help="Maximum generated tokens.")
    infer.add_argument("--seed", type=_nonnegative_int, default=42)
    infer.add_argument("--temperature", type=float, default=0.0)

    bench = subparsers.add_parser("bench", help="Measure prompt processing and token generation.")
    _add_common_options(bench)
    bench.add_argument("--prompt-tokens", type=_positive_int, default=512)
    bench.add_argument("--gen-tokens", type=_positive_int, default=128)
    bench.add_argument("--repetitions", type=_positive_int, default=5)
    bench.add_argument("--context-depth", type=_nonnegative_int, default=0)

    ppl = subparsers.add_parser("ppl", help="Evaluate text-corpus perplexity.")
    _add_common_options(ppl)
    ppl.add_argument("--corpus", required=True, type=Path, help="Plain-text evaluation corpus.")
    ppl.add_argument("--ctx-size", type=_positive_int, default=2048)
    ppl.add_argument("--ppl-stride", type=_positive_int)
    ppl.add_argument("--max-chunks", type=_positive_int, help="Limit chunks for smoke tests; omit for a full-corpus PPL.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    args.model = args.model.expanduser().resolve()
    args.bin_dir = args.bin_dir.expanduser().resolve()
    if not args.model.is_file():
        parser.error(f"GGUF model does not exist: {args.model}")
    if args.model.suffix.lower() != ".gguf":
        parser.error("--model must point to a .gguf file")
    if args.mode == "infer":
        if not 0 <= args.temperature:
            parser.error("--temperature must be nonnegative")
        if args.prompt_file is not None:
            args.prompt_file = args.prompt_file.expanduser().resolve()
            if not args.prompt_file.is_file():
                parser.error(f"Prompt file does not exist: {args.prompt_file}")
    if args.mode == "ppl":
        args.corpus = args.corpus.expanduser().resolve()
        if not args.corpus.is_file():
            parser.error(f"Evaluation corpus does not exist: {args.corpus}")
        if args.ppl_stride is not None and args.ppl_stride >= args.ctx_size:
            parser.error("--ppl-stride must be smaller than --ctx-size")
    if args.mode == "bench" and args.prompt_tokens + args.gen_tokens + args.context_depth > 2**31 - 1:
        parser.error("benchmark context token count is too large")
    names = {
        "infer": ("llama-completion", "llama-cli"),
        "bench": ("llama-bench",),
        "ppl": ("llama-perplexity",),
    }[args.mode]
    try:
        executable = _find_binary(args.bin_dir, names)
    except FileNotFoundError as error:
        parser.error(str(error))

    command = _build_command(args, executable)
    started = datetime.now(timezone.utc)
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else Path(__file__).resolve().parents[1]
        / "results"
        / "experiment_logs"
        / f"{args.mode}_{started.strftime('%Y%m%dT%H%M%S_%fZ')}.json"
    )
    protected_inputs = {args.model}
    if args.mode == "infer" and args.prompt_file is not None:
        protected_inputs.add(args.prompt_file)
    if args.mode == "ppl":
        protected_inputs.add(args.corpus)
    if output_path in protected_inputs:
        parser.error("--output cannot overwrite a model, prompt, or corpus input file")
    if output_path.exists() and not args.overwrite:
        parser.error(f"Result already exists: {output_path}; pass --overwrite to replace it")

    parameters = {key: value for key, value in vars(args).items() if key not in {"prompt", "model", "bin_dir", "output", "overwrite"}}
    parameters = {key: str(value) if isinstance(value, Path) else value for key, value in parameters.items()}
    if args.mode == "infer":
        if args.prompt_file:
            parameters["prompt_file_sha256"] = _sha256_file(args.prompt_file)
        else:
            prompt_bytes = args.prompt.encode("utf-8")
            parameters["prompt_sha256"] = hashlib.sha256(prompt_bytes).hexdigest()
            parameters["prompt_length_bytes"] = len(prompt_bytes)
    if args.mode == "ppl":
        parameters["corpus_sha256"] = _sha256_file(args.corpus)

    result: dict[str, Any] = {
        "schema_version": 1,
        "mode": args.mode,
        "started_at_utc": started.isoformat(),
        "model": _file_info(args.model, include_hash=args.hash_model),
        "executable": str(executable),
        "executable_version": _version(executable),
        "command": _redact_command(command),
        "parameters": parameters,
    }

    begin = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=args.timeout,
        )
        result.update(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        if completed.returncode == 0:
            try:
                result["metrics"] = _parse_metrics(args.mode, completed.stdout, completed.stderr)
                result["status"] = "ok"
            except (ValueError, json.JSONDecodeError) as error:
                result["status"] = "parse_error"
                result["error"] = str(error)
        else:
            result["status"] = "process_error"
    except subprocess.TimeoutExpired as error:
        result.update(
            status="timeout",
            exit_code=None,
            stdout=(error.stdout or b"").decode("utf-8", errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or ""),
            stderr=(error.stderr or b"").decode("utf-8", errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or ""),
            error=f"Exceeded --timeout {args.timeout} seconds",
        )
    except OSError as error:
        result.update(status="launch_error", exit_code=None, stdout="", stderr="", error=str(error))
    result["duration_seconds"] = round(time.monotonic() - begin, 3)
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Result: {output_path}")
    if args.mode == "infer" and result["status"] == "ok":
        print(result["stdout"])
    elif args.mode == "ppl" and result["status"] == "ok":
        print(f"PPL: {result['metrics']['perplexity']}")
    elif args.mode == "bench" and result["status"] == "ok":
        for row in result["metrics"]["benchmarks"]:
            print(f"pp={row.get('n_prompt')} tg={row.get('n_gen')} avg_ts={row.get('avg_ts')}")
    if result["status"] != "ok":
        print(f"Evaluation failed: {result['status']}; see {output_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
