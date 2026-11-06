"""Check the current corpus manifests and axis-only project layout."""

import json
import re
import unittest

from _support import HERE, TEPL


class LayoutTest(unittest.TestCase):
    def test_current_manifests_cover_the_same_catalog_cases(self):
        semantic = json.loads((HERE / "data/verification_cases.json").read_text())[
            "cases"
        ]
        structural = json.loads((HERE / "data/compilation_cases.json").read_text())[
            "cases"
        ]
        self.assertEqual(len(structural), 209)
        expected = {c["name"]: c["catalog_id"] for c in structural}
        actual = {
            c["expression_case"]: c["catalog_id"] for c in semantic if not c["control"]
        }
        self.assertEqual(actual, expected)
        for case in semantic + structural:
            self.assertTrue((HERE / case["file"]).is_file(), case["name"])

    def test_tepl_sources_share_one_project_root(self):
        self.assertTrue((TEPL / "tepl.toml").is_file())
        sources = list(HERE.parent.rglob("*.tepl"))
        self.assertTrue(sources)
        self.assertTrue(all(path.is_relative_to(TEPL) for path in sources))

    def test_local_documentation_links_resolve(self):
        documents = [
            HERE.parent / "README.md",
            HERE.parent / "catalog/README.md",
            HERE / "README.md",
            TEPL / "README.md",
            *(HERE / "reports").glob("*.md"),
        ]
        for document in documents:
            for target in re.findall(r"\]\(([^)\s]+)\)", document.read_text()):
                if "://" in target or target.startswith("#"):
                    continue
                with self.subTest(document=document.name, target=target):
                    self.assertTrue(
                        (document.parent / target.split("#", 1)[0]).exists()
                    )


if __name__ == "__main__":
    unittest.main()
