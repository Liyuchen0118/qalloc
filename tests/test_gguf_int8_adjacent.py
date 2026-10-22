"""Independent artifact checks for the adjacent-type signed-int8 QUBOs."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from qubo_formulation.gguf_int8_adjacent import CASES, DEFAULT_TABLE, decode_vector, generate_case

__author__ = "Yuchen Li"


class Int8AdjacentTests(unittest.TestCase):
    def test_generated_hardware_subproblems(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for case in CASES:
                with self.subTest(case=case[0]):
                    matrix_path, metadata_path = generate_case(DEFAULT_TABLE, Path(directory), case)
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    with matrix_path.open(newline="", encoding="utf-8") as handle:
                        matrix = [[int(value) for value in row] for row in csv.reader(handle)]
                    self.assertEqual(metadata["author"], __author__)
                    self.assertEqual(hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                                     metadata["matrix_sha256"])
                    self.assertEqual((len(matrix), {len(row) for row in matrix}), (196, {196}))
                    self.assertTrue(all(-128 <= value <= 127 for row in matrix for value in row))
                    self.assertTrue(all(matrix[i][j] == 0 for i in range(196) for j in range(i)))
                    self.assertTrue(metadata["ising_for_4x_qubo_energy"]["all_fields_and_couplings_signed_int8"])
                    self.assertTrue(all(block["all_int8_ground_states_satisfy_quota"]
                                        for block in metadata["blocks"]))
                    representative = metadata["exact_local_checks"]["representative_bits"]
                    decoded = decode_vector(metadata, representative)
                    self.assertTrue(decoded["satisfies_block_quotas"])
                    self.assertTrue(decoded["within_global_budget"])
                    self.assertEqual(decoded["target_bytes"], metadata["budget_target_bytes"])
                    self.assertEqual(decoded["full_file_bytes"], metadata["budget_full_file_bytes"])
                    self.assertEqual(Decimal(decoded["proxy_loss"]),
                                     Decimal(metadata["exact_local_checks"]["representative_proxy_loss"]))
                    self.assertGreaterEqual(
                        Decimal(metadata["exact_local_checks"]["restricted_exact_proxy_loss"]),
                        Decimal(metadata["exact_local_checks"]["unrestricted_two_choice_proxy_loss"]),
                    )

    def test_decoder_distinguishes_global_budget_and_block_quota(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _, metadata_path = generate_case(DEFAULT_TABLE, Path(directory), CASES[0])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            zero = decode_vector(metadata, [0] * 196)
            self.assertTrue(zero["within_global_budget"])
            self.assertFalse(zero["satisfies_block_quotas"])
            self.assertEqual(len(zero["block_violations"]), 42)
            with self.assertRaises(ValueError):
                decode_vector(metadata, [False] * 196)


if __name__ == "__main__":
    unittest.main()
