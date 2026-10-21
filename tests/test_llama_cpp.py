"""Focused, offline checks for the llama.cpp allocation and evaluation adapters."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from llama_cpp import evaluate, quantize


class AllocationTests(unittest.TestCase):
    def test_flat_allocation_maps_all_seven_projections_in_block_order(self) -> None:
        bits = [2, 3, 4, 5, 6, 8, 16, 16, 8, 6, 5, 4, 3, 2]
        choices = quantize.canonical_choices(bits, block_count=2)
        planned = quantize.expected_tensor_types(choices)

        self.assertEqual(len(planned), 14)
        self.assertEqual(
            list(planned.items())[:7],
            [
                ("blk.0.attn_q.weight", "Q2_K"),
                ("blk.0.attn_k.weight", "Q3_K"),
                ("blk.0.attn_v.weight", "Q4_K"),
                ("blk.0.attn_output.weight", "Q5_K"),
                ("blk.0.ffn_gate.weight", "Q6_K"),
                ("blk.0.ffn_up.weight", "Q8_0"),
                ("blk.0.ffn_down.weight", "F16"),
            ],
        )
        self.assertEqual(planned["blk.1.ffn_down.weight"], "Q2_K")

    def test_default_and_override_preserve_projection_scope(self) -> None:
        allocation = {
            "allocation": {
                "default": 4,
                "overrides": {
                    "layers.1.self_attn.k_proj": 8,
                    "layers.0.mlp.down_proj": "f16",
                },
            }
        }
        planned = quantize.expected_tensor_types(quantize.canonical_choices(allocation, 2))

        self.assertEqual(planned["blk.1.attn_k.weight"], "Q8_0")
        self.assertEqual(planned["blk.0.ffn_down.weight"], "F16")
        self.assertEqual(planned["blk.0.attn_k.weight"], "Q4_K")
        self.assertEqual(len(planned), 14)

    def test_rejects_incomplete_or_unknown_allocations(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 14 choices"):
            quantize.canonical_choices([4] * 13, 2)
        with self.assertRaisesRegex(ValueError, "unknown projection names"):
            quantize.canonical_choices(
                {"default": 4, "overrides": {"layers.2.self_attn.q_proj": 2}}, 2
            )
        with self.assertRaisesRegex(ValueError, "all seven projection names"):
            quantize.canonical_choices({"0": {"self_attn.q_proj": 4}}, 1)
        with self.assertRaisesRegex(ValueError, "Boolean values"):
            quantize.normalize_choice(True)
        with self.assertRaisesRegex(ValueError, "whole-model presets"):
            quantize.normalize_choice("Q4_K_M")

    def test_inventory_requires_contiguous_complete_decoder_blocks(self) -> None:
        names = quantize.expected_tensor_types(quantize.canonical_choices([4] * 14, 2))
        inventory = {name: {"type": "F16", "shape": [256, 256]} for name in names}
        inventory["token_embd.weight"] = {"type": "F16", "shape": [256, 32000]}

        count, targets = quantize.target_inventory(inventory)
        self.assertEqual(count, 2)
        self.assertEqual(set(targets), set(names))

        incomplete = inventory.copy()
        del incomplete["blk.1.attn_v.weight"]
        with self.assertRaisesRegex(ValueError, "missing required projection"):
            quantize.target_inventory(incomplete)
        noncontiguous = {
            name.replace("blk.1.", "blk.2."): value for name, value in inventory.items()
        }
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            quantize.target_inventory(noncontiguous)

    def test_shape_preflight_checks_quantization_block_alignment(self) -> None:
        fake_gguf = types.SimpleNamespace(
            GGMLQuantizationType={"Q4_K": "Q4_K", "F16": "F16"},
            GGML_QUANT_SIZES={"Q4_K": (256, 144), "F16": (1, 2)},
        )
        planned = {"blk.0.attn_q.weight": "Q4_K"}
        with mock.patch.dict(sys.modules, {"gguf": fake_gguf}):
            quantize.validate_shapes(
                {"blk.0.attn_q.weight": {"shape": [256, 64]}}, planned, Path("/unused")
            )
            with self.assertRaisesRegex(ValueError, "incompatible with Q4_K"):
                quantize.validate_shapes(
                    {"blk.0.attn_q.weight": {"shape": [255, 64]}}, planned, Path("/unused")
                )
            with self.assertRaisesRegex(ValueError, "not a matrix"):
                quantize.validate_shapes(
                    {"blk.0.attn_q.weight": {"shape": [256]}}, planned, Path("/unused")
                )

    def test_cli_rejects_non_quantized_base_preset(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as exit_context:
            quantize.main(
                [
                    "build", "--llama-cpp-dir", "/unused", "--source-gguf", "/unused/in.gguf",
                    "--allocation", "/unused/allocation.json", "--output-gguf", "/unused/out.gguf",
                    "--base-type", "F16",
                ]
            )
        self.assertEqual(exit_context.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())


class EvaluationTests(unittest.TestCase):
    def test_inference_command_and_prompt_redaction(self) -> None:
        args = evaluate._parser().parse_args(
            [
                "infer", "--model", "/tmp/model.gguf", "--bin-dir", "/tmp/bin",
                "--prompt", "private test prompt", "--predict", "16", "--gpu-layers", "0",
                "--threads", "3", "--device", "CUDA0", "--temperature", "0",
            ]
        )
        command = evaluate._build_command(args, Path("/tmp/bin/llama-completion"))

        self.assertEqual(command[command.index("-p") + 1], "private test prompt")
        self.assertIn("--no-conversation", command)
        self.assertEqual(command[command.index("-ngl") + 1], "0")
        self.assertEqual(command[command.index("-n") + 1], "16")
        self.assertEqual(
            evaluate._redact_command(command)[command.index("-p") + 1],
            "<redacted: prompt text>",
        )

    def test_benchmark_command_and_json_parser(self) -> None:
        args = evaluate._parser().parse_args(
            [
                "bench", "--model", "/tmp/model.gguf", "--bin-dir", "/tmp/bin",
                "--device", "CUDA0,CUDA1", "--prompt-tokens", "128",
                "--gen-tokens", "32", "--repetitions", "3",
            ]
        )
        command = evaluate._build_command(args, Path("/tmp/bin/llama-bench"))

        self.assertEqual(command[command.index("-dev") + 1], "CUDA0/CUDA1")
        self.assertEqual(command[command.index("-p") + 1], "128")
        self.assertEqual(command[command.index("-o") + 1], "json")
        self.assertEqual(evaluate._redact_command(command), command)
        self.assertEqual(
            evaluate._parse_metrics("bench", '[{"n_prompt":128,"avg_ts":42.5}]', ""),
            {"benchmarks": [{"n_prompt": 128, "avg_ts": 42.5}]},
        )
        with self.assertRaisesRegex(ValueError, "must be an array"):
            evaluate._parse_metrics("bench", '{}', "")

    def test_perplexity_command_and_final_estimate_parser(self) -> None:
        args = evaluate._parser().parse_args(
            [
                "ppl", "--model", "/tmp/model.gguf", "--bin-dir", "/tmp/bin",
                "--corpus", "/tmp/corpus.txt", "--ctx-size", "512", "--ppl-stride", "256",
                "--max-chunks", "2",
            ]
        )
        command = evaluate._build_command(args, Path("/tmp/bin/llama-perplexity"))

        self.assertEqual(command[command.index("-f") + 1], "/tmp/corpus.txt")
        self.assertEqual(command[command.index("--ppl-stride") + 1], "256")
        self.assertEqual(command[command.index("--chunks") + 1], "2")
        self.assertEqual(
            evaluate._parse_metrics("ppl", "batch PPL = 99\n", "Final estimate: PPL = 6.25 +/- 0.11"),
            {"perplexity": 6.25, "perplexity_uncertainty": 0.11},
        )
        with self.assertRaisesRegex(ValueError, "final PPL estimate"):
            evaluate._parse_metrics("ppl", "batch PPL = 99", "")

    def test_main_records_benchmark_metrics_without_real_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            model = directory / "model.gguf"
            model.write_bytes(b"fixture")
            bin_dir = directory / "bin"
            bin_dir.mkdir()
            binary = bin_dir / "llama-bench"
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o755)
            output = directory / "benchmark.json"
            completions = [
                types.SimpleNamespace(returncode=0, stdout="llama.cpp test version", stderr=""),
                types.SimpleNamespace(
                    returncode=0,
                    stdout='[{"n_prompt": 64, "n_gen": 16, "avg_ts": 21.0}]',
                    stderr="",
                ),
            ]

            with mock.patch.object(evaluate.subprocess, "run", side_effect=completions) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    result_code = evaluate.main(
                        [
                            "bench", "--model", str(model), "--bin-dir", str(bin_dir),
                            "--prompt-tokens", "64", "--gen-tokens", "16",
                            "--output", str(output),
                        ]
                    )

            self.assertEqual(result_code, 0)
            self.assertEqual(run.call_count, 2)
            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "ok")
            self.assertEqual(record["metrics"]["benchmarks"][0]["avg_ts"], 21.0)
            self.assertEqual(record["model"]["size_bytes"], 7)


if __name__ == "__main__":
    unittest.main()
