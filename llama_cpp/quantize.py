"""Convert a QuBit allocation into an executable mixed-type GGUF model."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROLE_TO_TENSOR = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
ROLE_ORDER = tuple(ROLE_TO_TENSOR)
BIT_TO_TYPE = {
    2: "Q2_K",
    3: "Q3_K",
    4: "Q4_K",
    5: "Q5_K",
    6: "Q6_K",
    8: "Q8_0",
    16: "F16",
}
BASE_TYPES = (
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M",
    "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
)
FLOAT_TYPES = {"F16", "BF16", "F32"}
TARGET_NAME = re.compile(
    r"^blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$"
)


def normalize_choice(choice: Any) -> str:
    if isinstance(choice, bool):
        raise ValueError("Boolean values are not valid bit-width choices")
    if isinstance(choice, int) or (isinstance(choice, str) and choice.isdecimal()):
        bit = int(choice)
        if bit not in BIT_TO_TYPE:
            raise ValueError(f"Unsupported bit width {bit}; choose {sorted(BIT_TO_TYPE)}")
        return BIT_TO_TYPE[bit]
    if isinstance(choice, str):
        qtype = choice.upper()
        if qtype in BIT_TO_TYPE.values():
            return qtype
    raise ValueError(
        f"Unsupported choice {choice!r}; use a bit width or a tensor type such as Q4_K. "
        "Q4_K_M and Q5_K_S are whole-model presets, not tensor types."
    )


def canonical_choices(allocation: Any, block_count: int) -> dict[str, str]:
    if isinstance(allocation, dict) and "allocation" in allocation:
        allocation = allocation["allocation"]

    names = [
        f"layers.{block}.{role}"
        for block in range(block_count)
        for role in ROLE_ORDER
    ]
    if isinstance(allocation, list):
        if len(allocation) != len(names):
            raise ValueError(
                f"Expected {len(names)} choices ({block_count} blocks x 7 tensors), "
                f"got {len(allocation)}"
            )
        return {name: normalize_choice(value) for name, value in zip(names, allocation)}

    if not isinstance(allocation, dict):
        raise ValueError("Allocation must be a flat list or a JSON object")

    if "default" in allocation:
        if set(allocation) != {"default", "overrides"}:
            raise ValueError("Default allocation must contain only default and overrides")
        overrides = allocation["overrides"]
        if not isinstance(overrides, dict):
            raise ValueError("Allocation overrides must be a JSON object")
        extra = sorted(set(overrides) - set(names))
        if extra:
            raise ValueError(f"Allocation overrides contain unknown projection names: {extra[:3]}")
        return {
            name: normalize_choice(overrides.get(name, allocation["default"]))
            for name in names
        }

    if all(str(block) in allocation for block in range(block_count)):
        flattened = {}
        for block in range(block_count):
            values = allocation[str(block)]
            if not isinstance(values, dict) or set(values) != set(ROLE_ORDER):
                raise ValueError(f"Block {block} must specify all seven projection names")
            for role in ROLE_ORDER:
                flattened[f"layers.{block}.{role}"] = values[role]
        if set(allocation) != {str(block) for block in range(block_count)}:
            raise ValueError("Allocation contains unexpected block indices")
        return {name: normalize_choice(value) for name, value in flattened.items()}

    if set(allocation) != set(names):
        missing = sorted(set(names) - set(allocation))
        extra = sorted(set(allocation) - set(names))
        raise ValueError(f"Allocation keys do not match the GGUF model: missing={missing[:3]}, extra={extra[:3]}")
    return {name: normalize_choice(allocation[name]) for name in names}


def load_gguf(llama_cpp_dir: Path, path: Path) -> dict[str, dict[str, Any]]:
    gguf_py = llama_cpp_dir / "gguf-py"
    if not gguf_py.is_dir():
        raise FileNotFoundError(f"Missing llama.cpp GGUF reader at {gguf_py}")
    sys.path.insert(0, str(gguf_py))
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import llama.cpp/gguf-py. Install its dependencies, "
            "for example: python -m pip install -e /path/to/llama.cpp/gguf-py"
        ) from exc

    reader = gguf.GGUFReader(str(path))
    return {
        tensor.name: {
            "type": tensor.tensor_type.name,
            "shape": [int(dim) for dim in tensor.shape],
            "n_bytes": int(tensor.n_bytes),
        }
        for tensor in reader.tensors
    }


def target_inventory(tensors: dict[str, dict[str, Any]]) -> tuple[int, dict[str, dict[str, Any]]]:
    targets = {}
    block_ids = set()
    for name, info in tensors.items():
        match = TARGET_NAME.fullmatch(name)
        if match:
            targets[name] = info
            block_ids.add(int(match.group(1)))
    if not block_ids:
        raise ValueError("No seven-projection decoder blocks found in the source GGUF")
    if block_ids != set(range(max(block_ids) + 1)):
        raise ValueError("Decoder block indices in GGUF are not contiguous from zero")
    block_count = len(block_ids)
    expected = {
        f"blk.{block}.{suffix}.weight"
        for block in range(block_count)
        for suffix in ROLE_TO_TENSOR.values()
    }
    if set(targets) != expected:
        missing = sorted(expected - set(targets))
        raise ValueError(f"Source GGUF is missing required projection tensors: {missing[:7]}")
    return block_count, targets


def expected_tensor_types(choices: dict[str, str]) -> dict[str, str]:
    planned = {}
    for name, qtype in choices.items():
        match = re.fullmatch(r"layers\.(\d+)\.(.+)", name)
        if match is None:
            raise ValueError(f"Unexpected allocation key: {name}")
        role = match.group(2)
        planned[f"blk.{match.group(1)}.{ROLE_TO_TENSOR[role]}.weight"] = qtype
    return planned


def validate_source_types(tensors: dict[str, dict[str, Any]]) -> None:
    quantized = {name: info["type"] for name, info in tensors.items() if info["type"] not in FLOAT_TYPES}
    if quantized:
        raise ValueError(
            f"Source GGUF already has quantized tensors: {list(quantized.items())[:3]}. "
            "Start from F16/BF16/F32 to avoid requantization."
        )


def validate_shapes(targets: dict[str, dict[str, Any]], planned: dict[str, str], llama_cpp_dir: Path) -> None:
    sys.path.insert(0, str(llama_cpp_dir / "gguf-py"))
    import gguf

    for name, qtype in planned.items():
        shape = targets[name]["shape"]
        if len(shape) < 2:
            raise ValueError(f"Target tensor is not a matrix: {name} {shape}")
        block_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[qtype]][0]
        if shape[0] % block_size:
            raise ValueError(
                f"{name} has {shape[0]} columns, incompatible with {qtype} "
                f"block size {block_size}"
            )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def git_revision(llama_cpp_dir: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(llama_cpp_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def convert(args: argparse.Namespace) -> None:
    script = args.llama_cpp_dir / "convert_hf_to_gguf.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    if not args.hf_model.is_dir():
        raise FileNotFoundError(args.hf_model)
    if args.output_gguf.exists():
        raise FileExistsError(args.output_gguf)
    args.output_gguf.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(script), str(args.hf_model),
        "--outfile", str(args.output_gguf), "--outtype", args.outtype,
    ]
    subprocess.run(command, check=True)
    print(args.output_gguf)


def build(args: argparse.Namespace) -> None:
    if not args.source_gguf.is_file():
        raise FileNotFoundError(args.source_gguf)
    if args.output_gguf == args.source_gguf:
        raise ValueError("Output GGUF must differ from the source GGUF")
    if args.output_gguf.exists():
        raise FileExistsError(args.output_gguf)
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    bin_path = args.quantize_bin or args.llama_cpp_dir / "build-qalloc/bin/llama-quantize"
    if not args.plan_only and not bin_path.is_file():
        raise FileNotFoundError(f"llama-quantize binary not found: {bin_path}")
    if args.imatrix is not None and not args.imatrix.is_file():
        raise FileNotFoundError(args.imatrix)

    inventory = load_gguf(args.llama_cpp_dir, args.source_gguf)
    block_count, targets = target_inventory(inventory)
    validate_source_types(inventory)
    allocation = json.loads(args.allocation.read_text(encoding="utf-8"))
    choices = canonical_choices(allocation, block_count)
    planned = expected_tensor_types(choices)
    validate_shapes(targets, planned, args.llama_cpp_dir)

    rules_path = Path(str(args.output_gguf) + ".tensor-types.txt")
    manifest_path = Path(str(args.output_gguf) + ".allocation.json")
    log_path = Path(str(args.output_gguf) + ".quantize.log")
    manifest = {
        "llama_cpp_commit": git_revision(args.llama_cpp_dir),
        "source_gguf": str(args.source_gguf),
        "source_bytes": args.source_gguf.stat().st_size,
        "output_gguf": str(args.output_gguf),
        "allocation_file": str(args.allocation),
        "block_count": block_count,
        "base_type": args.base_type,
        "imatrix": str(args.imatrix) if args.imatrix else None,
        "rules_file": str(rules_path),
        "planned_tensor_types": planned,
        "status": "planned",
    }
    if args.plan_only:
        print(json.dumps(manifest, indent=2, ensure_ascii=True))
        return

    if rules_path.exists() or manifest_path.exists() or log_path.exists():
        raise FileExistsError("Quantization plan or log already exists for this output path")
    args.output_gguf.parent.mkdir(parents=True, exist_ok=True)
    rules_path.write_text(
        "\n".join(f"^{re.escape(name)}$={qtype}" for name, qtype in planned.items()) + "\n",
        encoding="utf-8",
    )
    write_json(manifest_path, manifest)

    command = [str(bin_path), "--tensor-type-file", str(rules_path)]
    if args.imatrix:
        command += ["--imatrix", str(args.imatrix)]
    command += [str(args.source_gguf), str(args.output_gguf), args.base_type, str(args.threads)]
    manifest["command"] = command
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    manifest["quantizer_exit_code"] = result.returncode
    manifest["quantizer_log"] = str(log_path)
    if result.returncode:
        manifest["status"] = "quantizer_failed"
        write_json(manifest_path, manifest)
        raise RuntimeError(f"llama-quantize failed; see {log_path}")

    try:
        output_inventory = load_gguf(args.llama_cpp_dir, args.output_gguf)
    except (OSError, ValueError, KeyError, IndexError, AssertionError) as exc:
        manifest["status"] = "validation_failed"
        manifest["validation_error"] = str(exc)
        write_json(manifest_path, manifest)
        raise RuntimeError(f"Cannot validate output GGUF; see {manifest_path}") from exc

    source_names = set(inventory)
    output_names = set(output_inventory)
    missing = sorted(source_names - output_names)
    extra = sorted(output_names - source_names)
    shape_mismatches = {
        name: {"source": inventory[name]["shape"], "actual": output_inventory[name]["shape"]}
        for name in sorted(source_names & output_names)
        if inventory[name]["shape"] != output_inventory[name]["shape"]
    }
    mismatches = {
        name: {"requested": qtype, "actual": output_inventory.get(name, {}).get("type")}
        for name, qtype in planned.items()
        if output_inventory.get(name, {}).get("type") != qtype
    }
    manifest["output_bytes"] = args.output_gguf.stat().st_size
    manifest["actual_tensor_types"] = {
        name: output_inventory[name]["type"] for name in planned if name in output_inventory
    }
    manifest["type_mismatches"] = mismatches
    manifest["missing_tensors"] = missing
    manifest["extra_tensors"] = extra
    manifest["shape_mismatches"] = shape_mismatches
    valid = not (mismatches or missing or extra or shape_mismatches)
    manifest["status"] = "validated" if valid else "validation_failed"
    write_json(manifest_path, manifest)
    if missing or extra or shape_mismatches or mismatches:
        raise RuntimeError(f"Output GGUF differs from the planned tensor inventory; see {manifest_path}")
    print(manifest_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert original HF weights to F16/BF16 GGUF")
    convert_parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    convert_parser.add_argument("--hf-model", type=Path, required=True)
    convert_parser.add_argument("--output-gguf", type=Path, required=True)
    convert_parser.add_argument("--outtype", choices=("f16", "bf16"), default="f16")
    convert_parser.set_defaults(func=convert)

    build_parser = subparsers.add_parser("build", help="Apply a seven-tensor-per-block allocation to GGUF")
    build_parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    build_parser.add_argument("--source-gguf", type=Path, required=True)
    build_parser.add_argument("--allocation", type=Path, required=True)
    build_parser.add_argument("--output-gguf", type=Path, required=True)
    build_parser.add_argument("--quantize-bin", type=Path)
    build_parser.add_argument("--base-type", choices=BASE_TYPES, default="Q4_K_M", help="Quantized whole-model preset for untargeted tensors")
    build_parser.add_argument("--imatrix", type=Path)
    build_parser.add_argument("--threads", type=int, default=4)
    build_parser.add_argument("--plan-only", action="store_true")
    build_parser.set_defaults(func=build)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
