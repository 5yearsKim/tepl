"""Checks that catalog validation rejects changes that invalidate the counts."""

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_catalog as catalog


class CatalogValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = catalog.read_json("source_manifest.json")
        cls.rules = catalog.read_json("rules.json")
        cls.review = catalog.read_json("source_review.json")

    def test_snapshot_and_catalog_are_consistent(self):
        catalog.validate(self.manifest, self.rules, self.review)

    def test_duplicate_schema_id_is_rejected(self):
        rules = copy.deepcopy(self.rules)
        rules["rules"].append(copy.deepcopy(rules["rules"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate ID"):
            catalog.validate(self.manifest, rules, self.review)

    def test_missing_method_review_is_rejected(self):
        review = copy.deepcopy(self.review)
        review["methods"].pop()
        with self.assertRaisesRegex(ValueError, "Method inventory mismatch"):
            catalog.validate(self.manifest, self.rules, review)

    def test_nonexistent_test_reference_is_rejected(self):
        rules = copy.deepcopy(self.rules)
        linked = next(r for r in rules["rules"] if r["tests"])
        linked["tests"][0]["name"] = "AlgebraicSimplifierTest.DoesNotExist"
        with self.assertRaisesRegex(ValueError, "Missing test"):
            catalog.validate(self.manifest, rules, self.review)

    def test_changed_snapshot_is_rejected(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Changed hash"):
            catalog.validate(manifest, self.rules, self.review)

    def test_new_method_requires_review_even_with_updated_hash(self):
        manifest = copy.deepcopy(self.manifest)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for item in manifest["files"]:
                destination = root / item["local_path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = (catalog.ROOT / item["local_path"]).read_bytes()
                if destination.name == "algebraic_simplifier.cc":
                    data += b"\nvoid AlgebraicSimplifierVisitor::HandleNewOperation() {}\n"
                    item["bytes"] = len(data)
                    item["sha256"] = hashlib.sha256(data).hexdigest()
                destination.write_bytes(data)
            with patch.object(catalog, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Method inventory mismatch"):
                    catalog.validate(manifest, self.rules, self.review)


if __name__ == "__main__":
    unittest.main()
