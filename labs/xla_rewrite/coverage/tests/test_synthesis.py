"""Check migration, classification, and inference success accounting."""

import json
import unittest

from _support import HERE, TEPL
from scripts import hlo_support, migrate_syntax, synthesis_cases, synthesis_report


class SynthesisTest(unittest.TestCase):
    def test_migration_preserves_original_snapshots(self):
        record = migrate_syntax.validate()
        self.assertEqual(len(record["files"]), 272)
        for row in record["files"]:
            self.assertNotIn("@", (synthesis_cases.ROOT / row["file"]).read_text())

    def test_shared_descriptors_are_copied_from_the_same_source(self):
        source = "rule r {(add (transpose[@p] X) (transpose[@p] Y)) => (transpose[@p] (add X Y)) where {@p.permutation==[0]}}"
        converted = migrate_syntax.migrate(source)
        self.assertIn("(attr_p: transpose X)", converted)
        self.assertEqual(converted.count("transpose[attr_p.attrs]"), 2)
        self.assertIn("where {attr_p.attrs.permutation==[0]}", converted)

    def test_metadata_gather_is_not_a_graph_operation(self):
        path = TEPL / "rules/semantic/catalog_broadcast_transpose_broadcast.tepl"
        self.assertEqual(hlo_support.required_operations(path),
                         ["broadcast_in_dim", "transpose"])

    def test_descriptor_erasure_preserves_conditions_and_operator_structure(self):
        path = TEPL / "rules/semantic/dot_swap_operands.tepl"
        original = synthesis_cases.Rule(path.read_text())
        full = synthesis_cases.Rule(original.auto_source([0, 1]))
        partial = synthesis_cases.Rule(original.auto_source([1]))
        self.assertEqual([n.name for n in full.rhs.operations()],
                         [n.name for n in original.rhs.operations()])
        def conditions(rule):
            tokens = rule.values
            start = tokens.index("where")
            end = rule.closing(start + 1, "{", "}")
            return tokens[start:end + 1]
        self.assertEqual(conditions(original), conditions(full))
        self.assertEqual(conditions(original), conditions(partial))
        self.assertEqual(len(full.statements()), 0)
        self.assertEqual(len(partial.statements()), 1)

    def test_full_auto_drops_unused_explicit_metadata_formulas(self):
        rule = synthesis_cases.Rule((TEPL / "rules/semantic/catalog_broadcast_transpose_broadcast.tepl").read_text())
        source = rule.auto_source([0])
        self.assertNotIn("derive", source)
        self.assertNotIn("let inv", source)

    def test_theorem_filter_checks_structure_and_not_only_operation_names(self):
        _, aliases = synthesis_cases.operation_inventory()
        valid = synthesis_cases.Rule("rule r {(hlo.dot A B)=>(hlo.dot B A)}")
        repeated = synthesis_cases.Rule("rule r {(hlo.dot A A)=>(hlo.dot A A)}")
        duplicate_terms = synthesis_cases.Rule("rule r {(hlo.add A A)=>(hlo.add A A)}")
        self.assertTrue(synthesis_cases.eligibility(valid, aliases)[0])
        self.assertFalse(synthesis_cases.eligibility(repeated, aliases)[0])
        self.assertFalse(synthesis_cases.eligibility(duplicate_terms, aliases)[0])

    def test_measurement_and_frozen_classification_are_consistent(self):
        data = json.loads((HERE / "results/synthesis.json").read_text())
        classification = json.loads((HERE / "data/synthesis_classification.json").read_text())
        synthesis_report.validate(data, classification)
        self.assertEqual(len(classification["schemas"]), 239)
        self.assertEqual(len(classification["cases"]), 209)

    def test_unproved_attempt_cannot_be_relabelled_success(self):
        data = json.loads((HERE / "results/synthesis.json").read_text())
        classification = json.loads((HERE / "data/synthesis_classification.json").read_text())
        row = next(r for r in data["results"] if r["attempts"] and
                   r["outcome"] == "not_proved")
        row["outcome"] = "full_auto"
        with self.assertRaisesRegex(ValueError, "success lacks"):
            synthesis_report.validate(data, classification)


if __name__ == "__main__":
    unittest.main()
