"""Check measurement conventions that affect the reported size reductions."""

import unittest

import _support  # noqa: F401
from scripts.measure_rule_specification import body, excerpt, metrics, span_lines, tokens


class SpecificationMetricsTest(unittest.TestCase):
    def test_comments_do_not_count_but_comment_markers_in_strings_do(self):
        source = '// heading\nvalue = "https://example.test/*literal*/"; // tail\n'
        self.assertEqual(metrics(source), {"lines": 1, "tokens": 4})
        self.assertIn('"https://example.test/*literal*/"', tokens(source))

    def test_multiline_comments_preserve_physical_line_count(self):
        source = 'a /* first\n second */ b\n/* comment only */\n\nc\n'
        self.assertEqual(metrics(source), {"lines": 3, "tokens": 3})

    def test_token_count_is_independent_of_line_wrapping(self):
        compact = 'x::f(a, b) => g[..attrs];'
        wrapped = 'x :: f (\n a ,\n b\n) =>\n g [ .. attrs ] ;\n'
        self.assertEqual(tokens(compact), tokens(wrapped))
        self.assertNotEqual(metrics(compact)["lines"], metrics(wrapped)["lines"])

    def test_union_does_not_double_count_shared_source(self):
        lines = ['a;', 'b;', 'c;', 'd;']
        first = span_lines([[1, 2]], len(lines))
        second = span_lines([[2, 3]], len(lines))
        self.assertEqual(metrics(excerpt(lines, first | second)), {"lines": 3, "tokens": 6})

    def test_rule_body_excludes_imports_and_includes_conditions(self):
        source = ('from "dialects/hlo.tepl" import HLO as hlo;\n'
                  'rule keep {\n X => X\n where { len(shape(X)) == 1 }\n}\n')
        self.assertEqual(metrics(body(source))["lines"], 4)
        self.assertNotIn('import', tokens(body(source)))
        self.assertIn('where', tokens(body(source)))

    def test_invalid_source_ranges_are_rejected(self):
        for invalid in ([[0, 2]], [[3, 2]], [[1, 5]]):
            with self.subTest(ranges=invalid), self.assertRaises(ValueError):
                span_lines(invalid, 4)


if __name__ == "__main__":
    unittest.main()
