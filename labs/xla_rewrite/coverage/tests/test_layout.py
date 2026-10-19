"""Keep the directory migration separate from changes to the experiment."""

import hashlib
import json
import re
import unittest

from _support import HERE, TEPL, paths


class LayoutTest(unittest.TestCase):
    def test_frozen_baseline_and_manifest_contents_survive_relocation(self):
        baseline = json.loads((HERE / "baselines/reasoning_baseline.json").read_text())
        paths.validate_frozen_corpus(baseline)
        migration = json.loads(paths.MIGRATION.read_text())
        for original, snapshot in migration["manifest_snapshots"].items():
            with self.subTest(manifest=original):
                snapshot_path = paths.ROOT / snapshot
                self.assertEqual(hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
                                 baseline["corpus_hashes"][original])
                current = paths.ROOT / paths.relocate(original, migration)
                self.assertEqual(json.loads(current.read_text()),
                                 paths.relocated_manifest(json.loads(snapshot_path.read_text()), migration))

    def test_tepl_sources_share_one_project_root(self):
        self.assertTrue((TEPL / "tepl.toml").is_file())
        sources = list(HERE.parent.rglob("*.tepl"))
        self.assertTrue(sources)
        self.assertTrue(all(path.is_relative_to(TEPL) for path in sources))

    def test_local_documentation_links_resolve(self):
        documents = [HERE.parent / "README.md", HERE.parent / "evaluation_plan.md",
                     HERE / "README.md", TEPL / "README.md", *(HERE / "reports").glob("*.md")]
        for document in documents:
            for target in re.findall(r"\]\(([^)\s]+)\)", document.read_text()):
                if "://" in target or target.startswith("#"):
                    continue
                with self.subTest(document=document.name, target=target):
                    self.assertTrue((document.parent / target.split("#", 1)[0]).exists())


if __name__ == "__main__":
    unittest.main()
