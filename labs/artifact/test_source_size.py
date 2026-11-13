import unittest

import source_size as size


class SourceSizeTests(unittest.TestCase):
    def test_comments_and_cpp_literals_are_single_lexical_units(self):
        source = '"//literal" /* omit */ R"x(a /*literal*/ b)x" <= 2 // omit\n'
        self.assertEqual([t[0] for t in size.tokens(source)],
                         ['"//literal"', 'R"x(a /*literal*/ b)x"', '<=', '2'])

    def test_published_small_example(self):
        explicit = '''rule transpose_compose {
          (t2: transpose (t1: transpose X)) => (tr: transpose X)
          derive { tr.attrs = {permutation: gather(t1.attrs.permutation, t2.attrs.permutation)}; }
        }'''
        inferred = '''rule transpose_compose {
          (t2: transpose (t1: transpose X)) => (tr: transpose X)
        }'''
        self.assertEqual(sum(size.sections(explicit).values()), 48)
        self.assertEqual(sum(size.sections(inferred).values()), 22)

    def test_recovered_cohorts_are_counted_separately(self):
        result, _ = size.measure()
        self.assertEqual(result["matched_cases"], 30)
        self.assertEqual(result["all_tepl_cases"], 31)
        self.assertEqual([sum(result["matched"][v].values()) for v in ("xla", "explicit", "inferred")],
                         [10022, 3205, 2374])
        self.assertEqual([sum(result["all_tepl"][v].values()) for v in ("explicit", "inferred")],
                         [3396, 2520])


if __name__ == "__main__":
    unittest.main()
