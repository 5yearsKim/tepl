"""Axis migration must preserve supplied data and never hide the old answer."""

import copy
import json
import re
import unittest

from _support import HERE, TEPL
from scripts.paths import repo_path
from scripts import (
    run_synthesis,
    axis_fields,
    hlo_support,
    synthesis_cases,
    synthesis_report,
)


class SynthesisTest(unittest.TestCase):
    def task(self, name):
        source = (TEPL / "rules/semantic" / f"{name}.tepl").read_text()
        task = axis_fields.AxisTask(source)
        return task, task.generate()

    def test_metadata_gather_is_not_a_graph_operation(self):
        self.assertEqual(
            hlo_support.required_operations(
                TEPL / "rules/semantic/catalog_broadcast_transpose_broadcast.tepl"
            ),
            ["broadcast_in_dim", "transpose"],
        )

    def test_all_axis_fields_are_auto_without_whole_descriptor_erasure(self):
        task, source = self.task("dot_swap_operands")
        self.assertEqual(len(task.targets), 5)
        self.assertEqual(len(re.findall(r":\s*auto\b", source)), 5)
        self.assertNotIn("let ", source)
        before = task.rule
        after = synthesis_cases.Rule(source)
        self.assertEqual(
            [n.name for n in before.rhs.operations()],
            [n.name for n in after.rhs.operations()],
        )

        def conditions(rule):
            start = rule.values.index("where")
            return rule.values[start : rule.closing(start + 1, "{", "}") + 1]

        self.assertEqual(conditions(before), conditions(after))

    def test_mixed_descriptor_retains_supplied_extents(self):
        task, source = self.task("broadcast_multiply")
        self.assertEqual(len(task.targets), 1)
        self.assertIn("dimensions: auto", source)
        self.assertIn("new_extents: attr_left.attrs.new_extents", source)
        self.assertNotIn("new_extents: auto", source)

    def test_supplied_extent_program_keeps_required_helpers(self):
        _, source = self.task("catalog_broadcast_transpose_broadcast")
        self.assertIn("let oldshape", source)
        self.assertIn("let newpos", source)
        self.assertIn("new_extents: gather(gather(oldshape", source)
        self.assertNotIn("dimensions: d", source)

    def test_supplied_rhs_dependency_cannot_pin_old_axis_answer(self):
        source = """rule r {
          (s: hlo.broadcast X) => (t: hlo.broadcast X)
          derive { t.attrs={dimensions:s.attrs.dimensions,
                            new_extents:gather(s.attrs.new_extents,t.attrs.dimensions)}; }
        }"""
        with self.assertRaisesRegex(ValueError, "automatic RHS axis"):
            axis_fields.AxisTask(source).generate()

    def test_supplied_helper_cannot_hide_rhs_axis_dependency(self):
        source = """rule r {
          (s: hlo.broadcast X) => (t: hlo.broadcast X)
          derive { let e = gather(s.attrs.new_extents,t.attrs.dimensions);
                   t.attrs={dimensions:s.attrs.dimensions,new_extents:e}; }
        }"""
        with self.assertRaisesRegex(ValueError, "automatic RHS axis"):
            axis_fields.AxisTask(source).generate()

    def test_scalar_broadcast_choices_are_recorded_as_forced(self):
        for name in [
            "scalar_broadcast_reverse",
            "scalar_broadcast_transpose",
            "catalog_broadcast_divide_broadcast",
        ]:
            task, _ = self.task(name)
            self.assertTrue(all(field["forced"] for field in task.targets))

    def test_unit_normalization_does_not_create_extra_holes(self):
        task, source = self.task("catalog_indexing_reverse_inserted_unit_dims")
        self.assertEqual(len(task.targets), 1)
        self.assertEqual(len(task.normalizations), 2)
        self.assertEqual(len(re.findall(r":\s*auto\b", source)), 1)
        self.assertNotIn("hlo.reshape", source)
        self.assertIn("new_extents: [1]", source)
        self.assertIn(".attrs.dimensions == [1]", source)

    def test_unit_removal_preserves_types_without_numeric_guard(self):
        task, source = self.task("catalog_broadcast_drop_unit_dims")
        self.assertIn("axis_view.remove_unit_axes", source)
        self.assertNotIn("hlo.reduce_sum", source)
        self.assertEqual(len(task.targets), 1)
        view = (TEPL / "dialects/axis_views.tepl").read_text()
        self.assertIn("dtype(t) { yield t; }", view)
        self.assertNotIn("is_numeric", view)

    def test_reshape_after_auto_dot_must_not_use_handwritten_shape(self):
        # With M=N=4,K=2 a legal alternative batch-only dot has shape
        # [4,2,2]; reshaping it to [1,4,4] is not a unit-only view.
        with self.assertRaisesRegex(ValueError, "every automatic axis assignment"):
            self.task("catalog_dot_drop_unit_dims")

    def test_disjunction_or_negation_is_not_a_unit_view_premise(self):
        base = """rule r { X:[N]
          (s: hlo.reshape X) => X where { CONDITION }
        }"""
        for condition in ["s.attrs.shape==[1,N] || N==2", "!(s.attrs.shape==[1,N])"]:
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                axis_fields.AxisTask(base.replace("CONDITION", condition)).generate()

    def test_joint_linear_context_is_admitted_but_repeated_capture_is_not(self):
        _, aliases = synthesis_cases.operation_inventory()
        valid = synthesis_cases.Rule(
            "rule r {(hlo.add (hlo.broadcast X) (hlo.broadcast Y))=>(hlo.broadcast (hlo.add X Y))}"
        )
        repeated = synthesis_cases.Rule("rule r {(hlo.dot A A)=>(hlo.dot A A)}")
        product_sum = synthesis_cases.Rule(
            "rule r {(hlo.dot (hlo.add A B) C)=>(hlo.dot (hlo.add A B) C)}"
        )
        self.assertTrue(synthesis_cases.eligibility(valid, aliases)[0])
        self.assertFalse(synthesis_cases.eligibility(repeated, aliases)[0])
        self.assertFalse(synthesis_cases.eligibility(product_sum, aliases)[0])

    def test_reciprocal_admission_does_not_admit_general_division(self):
        _, aliases = synthesis_cases.operation_inventory()
        for source in [
            "rule r {(hlo.divide X Y)=>(hlo.divide X Y)}",
            "rule r {Y:[] (hlo.divide Y (hlo.broadcast Y))=>(hlo.multiply Y (hlo.broadcast (hlo.divide 1 Y)))}",
        ]:
            self.assertFalse(
                synthesis_cases.eligibility(synthesis_cases.Rule(source), aliases)[0]
            )
        source = (
            TEPL / "rules/semantic/catalog_broadcast_divide_broadcast.tepl"
        ).read_text()
        self.assertTrue(
            synthesis_cases.eligibility(synthesis_cases.Rule(source), aliases)[0]
        )

    def test_classifier_provenance_is_checked_before_execution(self):
        classification = json.loads(
            (HERE / "data/synthesis_classification.json").read_text()
        )
        run_synthesis.validate_classification(classification)
        dependency = next(iter(classification["dependencies"]))
        classification["dependencies"][dependency] = "old"
        with self.assertRaisesRegex(ValueError, "stale classification dependency"):
            run_synthesis.validate_classification(classification)

    def test_measurement_matches_frozen_classification(self):
        data = json.loads((HERE / "results/synthesis.json").read_text())
        classification = json.loads(
            (HERE / "data/synthesis_classification.json").read_text()
        )
        synthesis_report.validate(data, classification)
        self.assertEqual(
            len([r for r in classification["cases"] if not r["control"]]), 209
        )
        for row in classification["cases"]:
            if not row["in_scope"]:
                continue
            self.assertEqual(
                len(
                    re.findall(
                        r":\s*auto\b", (repo_path(row["file"])).read_text()
                    )
                ),
                len(row["axis_fields"]),
            )

    def test_success_cannot_omit_requested_fields_or_add_guards(self):
        original = json.loads((HERE / "results/synthesis.json").read_text())
        classification = json.loads(
            (HERE / "data/synthesis_classification.json").read_text()
        )
        for mutation in ["missing_field", "extra_guard", "not_compiled"]:
            data = copy.deepcopy(original)
            row = next(r for r in data["results"] if r["outcome"] == "inferred")
            if mutation == "missing_field":
                row["requested_axis_fields"] -= 1
            elif mutation == "extra_guard":
                row["check"]["stdout"] = row["check"]["stdout"].replace(
                    "additional-guard: true", "additional-guard: unresolved"
                )
            else:
                row["compile"]["returncode"] = 1
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                ValueError, "success requires"
            ):
                synthesis_report.validate(data, classification)

    def test_outside_scope_is_not_an_inference_failure(self):
        data = json.loads((HERE / "results/synthesis.json").read_text())
        classification = json.loads(
            (HERE / "data/synthesis_classification.json").read_text()
        )
        row = next(r for r in data["results"] if not r["in_scope"])
        row["check"] = {"returncode": 1}
        with self.assertRaisesRegex(ValueError, "outside-scope"):
            synthesis_report.validate(data, classification)


if __name__ == "__main__":
    unittest.main()
