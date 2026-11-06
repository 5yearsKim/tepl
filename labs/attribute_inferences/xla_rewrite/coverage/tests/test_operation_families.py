"""Check graph-based memberships and the overlapping reporting denominator."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from _support import HERE, TEPL
from scripts import hlo_support, operation_families as families


class OperationFamiliesTest(unittest.TestCase):
    def setUp(self):
        self.mapping = families.read(families.MAPPING)

    def classify_source(self, source):
        # Keep the temporary source inside the project for provenance paths.
        with tempfile.TemporaryDirectory(dir=TEPL) as directory:
            path = Path(directory) / "case.tepl"
            path.write_text(source)
            case = dict(
                case="example", catalog_id="example", control=False, in_scope=True,
                source_file=str(path),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                required_operations=hlo_support.required_operations(path),
            )
            return families.classify([case], self.mapping)[0]

    def test_rhs_only_family_and_repeated_operations_count_once(self):
        row = self.classify_source("""rule example {
            (hlo.add X (hlo.add Y Z)) => (hlo.transpose (hlo.add X Y))
        }""")
        self.assertEqual(row["families"], ["elementwise", "transpose"])
        self.assertEqual(row["operations"], ["add", "transpose"])

    def test_metadata_calls_do_not_add_indexing_membership(self):
        cases = families.read(families.CLASSIFICATION)["cases"]
        case = next(c for c in cases if c["case"] == "catalog_broadcast_transpose_broadcast")
        source = (HERE / case["source_file"]).read_text()
        self.assertIn("gather(", source)
        row = families.classify([case], self.mapping)[0]
        self.assertEqual(row["families"], ["broadcast", "transpose"])

    def test_transpose_and_reshape_are_separate_from_dot(self):
        row = self.classify_source("""rule example {
            (hlo.dot X Y) => (hlo.reshape (hlo.transpose (hlo.dot X Y)))
        }""")
        self.assertEqual(row["families"], ["dot", "transpose", "reshape"])

    def test_controls_are_excluded_but_out_of_scope_cases_are_included(self):
        rows = [
            dict(control=False, in_scope=False, families=["dot", "transpose"]),
            dict(control=False, in_scope=True, families=["dot"]),
            dict(control=True, in_scope=True, families=["elementwise"]),
        ]
        summary = {r["family"]: r for r in families.summarize(rows, self.mapping)}
        self.assertEqual(summary["dot"]["count"], 2)
        self.assertEqual(summary["dot"]["percentage"], 100.0)
        self.assertEqual(summary["dot"]["in_scope_count"], 1)
        self.assertEqual(summary["dot"]["in_scope_percentage"], 50.0)
        self.assertEqual(summary["transpose"]["percentage"], 50.0)
        self.assertEqual(summary["transpose"]["in_scope_count"], 0)
        self.assertEqual(summary["transpose"]["in_scope_percentage"], 0.0)
        self.assertEqual(summary["elementwise"]["count"], 0)
        self.assertIsNone(summary["elementwise"]["in_scope_percentage"])

    def test_mapping_rejects_missing_or_duplicate_operations(self):
        declared = hlo_support.inventory()
        missing = copy.deepcopy(self.mapping)
        missing["families"][0]["operations"].remove("add")
        with self.assertRaisesRegex(ValueError, "cover every declared"):
            families.operation_mapping(missing, declared)
        duplicate = copy.deepcopy(self.mapping)
        duplicate["families"][1]["operations"].append("add")
        with self.assertRaisesRegex(ValueError, "multiple families"):
            families.operation_mapping(duplicate, declared)

    def test_frozen_source_changes_are_rejected(self):
        case = copy.deepcopy(families.read(families.CLASSIFICATION)["cases"][0])
        case["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source differs"):
            families.classify([case], self.mapping)


if __name__ == "__main__":
    unittest.main()
