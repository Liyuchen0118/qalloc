"""Exactness tests for the two-bit, four-type GGUF QUBO."""

from __future__ import annotations

import csv
import itertools
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from qubo_formulation.gguf_two_bit_qubo import (
    Choice,
    Problem,
    Tensor,
    bounded_binary_weights,
    build_spec,
    direct_energy,
    encode_allocation,
    export_matrix,
    generate,
    inspect_vector,
    load_problem,
    matrix_energy,
)

__author__ = "Yuchen Li"


def synthetic_problem() -> Problem:
    return Problem(
        tensors=(
            Tensor("a", (
                Choice("Q3_K", 1, Decimal("10")),
                Choice("Q4_K", 2, Decimal("4")),
                Choice("Q5_K", 3, Decimal("2")),
                Choice("Q6_K", 4, Decimal("0")),
            )),
            Tensor("b", (
                Choice("Q3_K", 1, Decimal("9")),
                Choice("Q4_K", 3, Decimal("3")),
                Choice("Q5_K", 4, Decimal("1")),
                Choice("Q6_K", 6, Decimal("0")),
            )),
        ),
        budgets=(5,),
        metadata={"author": __author__, "model": "synthetic", "fixed_non_target_bytes": 7},
    )


class TwoBitQuboTests(unittest.TestCase):
    def test_slack_weights_cover_exact_interval(self) -> None:
        for capacity in range(64):
            weights = bounded_binary_weights(capacity)
            represented = {
                sum(weight for bit, weight in zip(bits, weights) if bit)
                for bits in itertools.product((0, 1), repeat=len(weights))
            }
            self.assertEqual(represented, set(range(capacity + 1)))

    def test_two_bit_mapping_and_actual_bytes_loss(self) -> None:
        spec = build_spec(synthetic_problem(), 0)
        self.assertEqual([variable.bit for variable in spec.variables[:4]], ["b0", "b1", "b0", "b1"])
        self.assertEqual(len(spec.variables), 2 * len(spec.problem.tensors) + 2)
        for type_a, type_b in itertools.product(("Q3_K", "Q4_K", "Q5_K", "Q6_K"), repeat=2):
            choices = [spec.problem.tensors[0].choice(type_a), spec.problem.tensors[1].choice(type_b)]
            if sum(choice.byte_count for choice in choices) > spec.budget_bytes:
                continue
            vector = encode_allocation(spec, {"a": type_a, "b": type_b})
            decoded = inspect_vector(spec, vector)
            self.assertEqual(decoded["allocation"], {"a": type_a, "b": type_b})
            self.assertEqual(decoded["target_bytes"], sum(choice.byte_count for choice in choices))
            self.assertEqual(Decimal(decoded["proxy_loss"]), sum((choice.loss for choice in choices), Decimal(0)))
            self.assertTrue(decoded["qubo_constraints_satisfied"])
            self.assertEqual(direct_energy(spec, vector), Decimal(decoded["proxy_loss"]))

    def test_matrix_matches_every_state_and_global_optimum(self) -> None:
        spec = build_spec(synthetic_problem(), 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.csv"
            export_matrix(spec, path)
            with path.open(newline="", encoding="utf-8") as handle:
                matrix = [[Decimal(entry) for entry in row] for row in csv.reader(handle)]
            self.assertEqual(len(matrix), len(spec.variables))
            self.assertTrue(all(matrix[i][j] == 0 for i in range(len(matrix)) for j in range(i)))
            best_energy = None
            for state in itertools.product((0, 1), repeat=len(spec.variables)):
                vector = list(state)
                selected = [index for index, bit in enumerate(vector) if bit]
                matrix_value = sum((matrix[i][j] for i in selected for j in selected if i <= j), Decimal(0))
                semantic_value = direct_energy(spec, vector)
                self.assertEqual(matrix_value + spec.offset, semantic_value)
                self.assertEqual(matrix_energy(path, vector) + spec.offset, semantic_value)
                best_energy = semantic_value if best_energy is None else min(best_energy, semantic_value)
            feasible_losses = [
                choice_a.loss + choice_b.loss
                for choice_a in spec.problem.tensors[0].choices
                for choice_b in spec.problem.tensors[1].choices
                if choice_a.byte_count + choice_b.byte_count <= spec.budget_bytes
            ]
            self.assertEqual(best_energy, min(feasible_losses))

    def test_rejects_non_linear_measured_byte_costs(self) -> None:
        problem = synthetic_problem()
        choices = list(problem.tensors[0].choices)
        choices[-1] = Choice("Q6_K", 5, Decimal("0"))
        bad = Problem((Tensor("a", tuple(choices)), *problem.tensors[1:]), problem.budgets, problem.metadata)
        with self.assertRaisesRegex(ValueError, "c3\\+c6=c4\\+c5"):
            build_spec(bad, 0)

    def test_json_round_trip_and_author_metadata(self) -> None:
        problem = synthetic_problem()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            table_path = directory / "synthetic_costs.json"
            table = {
                "metadata": problem.metadata,
                "budgets": list(problem.budgets),
                "tensors": [
                    {"name": tensor.name,
                     "choices": {choice.tensor_type: {"bytes": choice.byte_count, "loss": float(choice.loss)}
                                 for choice in tensor.choices}}
                    for tensor in problem.tensors
                ],
            }
            table_path.write_text(json.dumps(table), encoding="utf-8")
            loaded = load_problem(table_path)
            self.assertEqual(loaded.tensors, problem.tensors)
            paths = generate(table_path, directory / "output")
            self.assertEqual(len(paths), 2)
            metadata = json.loads(paths[1].read_text(encoding="utf-8"))
            self.assertEqual(metadata["author"], "Yuchen Li")
            self.assertEqual(metadata["decision_variable_count"], 4)
            self.assertEqual(metadata["slack_variable_count"], 2)
            self.assertEqual(len(metadata["source_cost_table_sha256"]), 64)
            self.assertIn("not certified", metadata["hardware_acceptance"])


if __name__ == "__main__":
    unittest.main()
