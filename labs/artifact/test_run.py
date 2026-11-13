"""Evidence validation controls; uses an existing run when explicitly supplied."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("artifact_run", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ArtifactTests(unittest.TestCase):
    def test_legacy_archive_preserves_every_recorded_file(self):
        directory = runner.ROOT / "labs/archive"
        manifest = json.loads((directory / "transpose-folding.json").read_text())
        archive = directory / "transpose-folding.tar.gz"
        self.assertEqual(runner.sha(archive), manifest["archive_sha256"])
        with tarfile.open(archive) as stream:
            self.assertEqual(set(stream.getnames()), set(manifest["files"]))
            for name, digest in manifest["files"].items():
                self.assertEqual(hashlib.sha256(stream.extractfile(name).read()).hexdigest(), digest)

    def test_bundled_population_and_integrity(self):
        manifest, cases = runner.verify_inputs()
        self.assertEqual(sum(r["paper_accepted"] for r in cases), 98)
        self.assertEqual(manifest["targets"]["coverage"]["axis_accepted"], 31)

    def test_syntax_migration_preserves_strings_and_nested_expressions(self):
        source = ('// where { keep; }\nrule r { X => X where { '
                  'f({a: "where {;}"}) == true; } derive { x.attrs = {a: 1}; } }')
        migrated = runner.migrate(source)
        self.assertIn('// where { keep; }', migrated)
        self.assertIn('f({a: "where {;}"}) == true', migrated)
        self.assertIn('derive { x.attrs = {a: 1} }', migrated)
        self.assertNotIn('where { f', migrated)

    def test_unresolved_executable_holes(self):
        self.assertTrue(runner.has_hole({"nodes": [{"kind": "auto_field"}]}))
        self.assertFalse(runner.has_hole({"kind": "literal", "text": "auto"}))

    def test_missing_proof_cannot_be_success(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "inputs").mkdir()
            (out / "inputs/x.tepl").write_text("rule x { X => X }")
            with patch.object(runner, "command", return_value={"returncode": 0, "stdout": "status=proved\n"}):
                result = runner.attempt({"case": "x", "rule_name": "x"}, "tepl", out, "default", 1)
            self.assertFalse(result["accepted"])
            self.assertEqual(result["outcome"], "check_unresolved")

    def test_duplicate_population_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "bin").mkdir()
            (out / "bin/tepl").write_text("compiler")
            data = {"provenance": {"input_hashes": {}, "compiler_sha256": runner.sha(out / "bin/tepl")},
                    "records": [{"case": "add_zero"}] * 124}
            runner.save(out / "results.json", data)
            with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                runner.validate_run(out)


if __name__ == "__main__":
    unittest.main()
