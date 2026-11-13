import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("prepare", HERE / "prepare_xla.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class PreparationTests(unittest.TestCase):
    def test_branch_hooks_preserve_original_bodies(self):
        source = "absl::Status AlgebraicSimplifierVisitor::HandleTranspose(\n"
        source += "".join(f"    return ObserveAlgebraicRewrite({site}, __func__, [&]() {{ return original_body;\n"
                          for site in (10114, 10118))
        result = prepare.patch_source(source)
        self.assertEqual(result.count("original_body"), 2)
        self.assertEqual(result.count("tepl_algebraic_hook(transpose,"), 2)
        self.assertEqual(prepare.patch_source(result), result)

    def test_unknown_source_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unexpected HandleTranspose"):
            prepare.patch_source("unrelated source")

    def test_conflicting_checkout_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / prepare.SOURCE
            source.parent.mkdir(parents=True)
            source.write_text("local changes must survive\n")
            with patch("sys.argv", ["prepare_xla.py", str(root)]), patch.object(
                    prepare.subprocess, "check_output",
                    return_value="39c348196c834a7ee45a57f5b44688d53ae99a64\n"):
                with self.assertRaisesRegex(ValueError, "source differs"):
                    prepare.main()
            self.assertEqual(source.read_text(), "local changes must survive\n")
            self.assertFalse((root / "research/tepl_algebraic").exists())


if __name__ == "__main__":
    unittest.main()
