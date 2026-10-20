import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import transpose_scaling as t


class TransposeScalingTests(unittest.TestCase):
    def test_composition_order(self):
        inner, outer = (2, 0, 1), (1, 0, 2)
        self.assertEqual(t.compose((inner, outer)), [0, 2, 1])
        self.assertNotEqual(t.compose((inner, outer)), t.compose((outer, inner)))

    def test_whole_chain_and_explicit_composition(self):
        _, explicit = t.source(8, "explicit")
        _, automatic = t.source(8, "automatic")
        self.assertEqual(explicit.count("(transpose[@p"), 8)
        self.assertEqual(explicit.count("gather("), 7)
        for i in range(1, 9):
            self.assertEqual(explicit.count(f"@p{i}.permutation"), 1)
        self.assertIn("(transpose[auto] A)", automatic)
        self.assertNotIn("derive", automatic)
        self.assertNotIn(".permutation ==", explicit)

    def test_one_complete_constructor(self):
        stdout = ("      constructor auto#2:\n"
                  "        permutation = gather(@#1.permutation, @#0.permutation)\n"
                  "      additional-guard: true\n")
        case = {"attribute_mode": "automatic"}
        self.assertTrue(t.constructor_check(stdout, case)["complete"])
        self.assertFalse(t.constructor_check(stdout.replace("permutation =", "wrong ="), case)["complete"])
        self.assertFalse(t.constructor_check(stdout.replace("@#0.permutation", "auto#2.permutation"), case)["complete"])
        self.assertFalse(t.constructor_check(stdout.replace("additional-guard: true", "additional-guard: false"), case)["complete"])
        self.assertFalse(t.constructor_check(stdout + stdout, case)["complete"])

    def test_explicit_needs_no_auto_constructor(self):
        self.assertTrue(t.constructor_check("      additional-guard: true\n",
                                            {"attribute_mode": "explicit"})["complete"])
        self.assertFalse(t.constructor_check("      additional-guard: true\n",
                                             {"attribute_mode": "automatic"})["complete"])


if __name__ == "__main__":
    unittest.main()
