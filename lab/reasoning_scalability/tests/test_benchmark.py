"""Independent routing validation and benchmark bookkeeping checks."""

import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark.py"
spec = importlib.util.spec_from_file_location("scaling_benchmark", SCRIPT)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)


class BenchmarkTest(unittest.TestCase):
    def test_descriptors_against_reference(self):
        self.assertEqual(b.reference_validate()["logical_cases_checked"], 25)

    def test_permuted_layout_changes_positions(self):
        _, left, right = b.chain(8, "permuted")
        nodes = list(b.internal(left)) + list(b.internal(right))
        self.assertGreater(len({tuple(n.attrs[f][0] for f in b.FIELDS) for n in nodes}), 1)
        self.assertEqual(left.axes, right.axes)

    def test_matched_modes_and_size(self):
        for size in b.SIZES:
            for layout in ("canonical", "permuted"):
                _, explicit = b.dot_source(size, layout, "explicit")
                _, auto = b.dot_source(size, layout, "automatic")
                self.assertEqual(explicit.count("(dot["), 2 * (size - 1))
                self.assertEqual(auto.count("[auto]"), size - 1)
                self.assertEqual(explicit.split("    where {")[1].split("    }")[0],
                                 auto.split("    where {")[1].split("    }")[0])
                self.assertEqual(explicit.split("    =>")[0].split("\n", 4)[-1],
                                 auto.split("    =>")[0].split("\n", 4)[-1])

    def test_compact_product_source(self):
        for k in range(2, 11):
            _, source = b.product_source(k)
            self.assertEqual(source.count("(add "), 2 * k)
            self.assertEqual(source.count("(mul "), 2 * (k - 1))

    def test_failure_classification(self):
        self.assertEqual(b.classify(1, "", "budget exhausted: normalization term limit exceeded"), "budget_exhausted")
        self.assertEqual(b.classify(1, "", "cannot verify rule 'x': unsupported"), "unresolved")
        self.assertEqual(b.classify(-9, "", "", True), "timeout")
        self.assertEqual(b.classify(-11, "", ""), "error")
        self.assertEqual(b.classify(0, "no semantic proof", ""), "error")

    def test_constructor_completion_is_required(self):
        case = {"attribute_mode": "automatic", "size": 3}
        self.assertFalse(b.constructor_check("additional-guard: true\n", case)["complete"])


if __name__ == "__main__":
    unittest.main()
