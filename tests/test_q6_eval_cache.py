"""Checks that B5 evaluation results are reused only for identical inputs."""

from __future__ import annotations

import hashlib
from importlib import import_module
import tempfile
import unittest
from pathlib import Path

validate_cached_result = import_module("experiments.20261022.q6_data.run_b5_eval").validate_cached_result

__author__ = "Yuchen Li"


class Q6EvalCacheTests(unittest.TestCase):
    def test_rejects_changed_model_or_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            corpus = root / "corpus.txt"
            model.write_bytes(b"model")
            corpus.write_bytes(b"corpus")
            stat = model.stat()
            record = {
                "status": "ok", "mode": "ppl",
                "model": {"path": str(model), "size_bytes": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns},
                "parameters": {
                    "mode": "ppl", "threads": 8, "gpu_layers": 999,
                    "device": "CUDA0", "batch_size": 2048, "ubatch_size": 512,
                    "flash_attn": "on", "timeout": 900,
                    "corpus": str(corpus), "ctx_size": 2048,
                    "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                },
                "run_environment": {"cuda_visible_devices": "6", "gpu_uuid": "GPU-test"},
                "executable_version": "commit 493253fe5",
            }
            args = ["--corpus", str(corpus), "--ctx-size", "2048"]
            validate_cached_result(record, "ppl", model, corpus, 6, "GPU-test", "493253fe5abcdef", args)
            record["parameters"]["ctx_size"] = 1024
            with self.assertRaisesRegex(RuntimeError, "Cached evaluation differs"):
                validate_cached_result(record, "ppl", model, corpus, 6, "GPU-test", "493253fe5abcdef", args)
            record["parameters"]["ctx_size"] = 2048
            model.write_bytes(b"changed model")
            with self.assertRaisesRegex(RuntimeError, "Cached evaluation differs"):
                validate_cached_result(record, "ppl", model, corpus, 6, "GPU-test", "493253fe5abcdef", args)


if __name__ == "__main__":
    unittest.main()
