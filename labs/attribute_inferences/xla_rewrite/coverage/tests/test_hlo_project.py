"""Exercise project-root imports and semantic-language boundaries."""

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from _support import hlo_support, TEPL

HERE = Path(__file__).resolve().parents[1]
COMPILER = HERE.parents[3] / "bazel-bin/tepl"


class HLOProjectTest(unittest.TestCase):
    def test_profiles_share_interfaces_and_support_inventory(self):
        hlo_support.validate_support()

    def test_imports_stay_inside_the_tepl_project(self):
        for directory in ("rules/semantic", "rules/structural", "graphs"):
            for path in (TEPL / directory).glob("*.tepl"):
                with self.subTest(path=path.name):
                    imports = re.findall(r'from "([^"]+)" import (\w+)', path.read_text())
                    self.assertEqual(len(imports), 1)
                    target, dialect = imports[0]
                    self.assertEqual(dialect, "HLO")
                    self.assertIn(target, ("dialects/hlo.tepl", "dialects/hlo_structural.tepl"))
                    self.assertTrue((TEPL / target).is_file())

    @unittest.skipUnless(COMPILER.is_file(), "build //:tepl")
    def test_root_imports_work_from_an_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [str(COMPILER), "check", str(TEPL / "rules/semantic/add_zero.tepl"),
                 "--semantic-policy=algebraic"],
                cwd=directory, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("status=proved", result.stdout)

    @unittest.skipUnless(COMPILER.is_file(), "build //:tepl")
    def test_reported_language_limits_are_rejected_by_the_compiler(self):
        fixtures = (
            ("unknown_scalar", """dialect HLO {
                op divide(x:tensor,y:tensor)->tensor {
                    dtype(a,b){assert a==b yield a }
                    axes {bind S {x:all_axes(x),y:all_axes(y)} yield S }
                    semantics {yield custom_scalar(x,y) }
                }}""", "unknown scalar semantic function"),
            ("coordinate_division", """dialect HLO {
                op reindex(x:tensor)->tensor {
                    dtype(a){yield a }
                    axes {bind S {x:all_axes(x)} yield S }
                    semantics {yield Read(x,IndexFloorDiv(Coords(S),Coords(S))) }
                }}""", "unsupported coordinate primitive"),
            ("attribute_constant", """dialect HLO {
                op fill()->tensor {
                    attrs {value: i64 shape: extent[] }
                    dtype(){yield i64 }
                    axes {bind S=group(attrs.shape) yield S }
                    semantics {yield Const(attrs.value) }
                }}""", "Const requires one signed integer literal"),
            ("dynamic_const", """dialect HLO {
                op scalar()->tensor {
                    attrs {value: u64[] }
                    dtype(){yield i64 }
                    axes {bind S=group([]) yield S }
                    semantics {yield Const(attrs.value) }
                }}""", "Const requires one signed integer literal"),
        )
        for name, source, diagnostic in fixtures:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "probe.tepl"
                path.write_text(source)
                result = subprocess.run([str(COMPILER), "check", str(path)],
                                        text=True, capture_output=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)
