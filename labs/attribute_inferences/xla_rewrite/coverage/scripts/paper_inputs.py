"""Resolve bundled paper evidence, with an optional external paper checkout."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
BUNDLED = ROOT / "labs/artifact/fixtures/paper"


def audit_path(paper=None):
    return paper / "evaluation/theorem-scope-audit.json" if paper else BUNDLED / "theorem-scope-audit.json"


def source_path(name, paper=None):
    return paper / name if paper else BUNDLED / "inputs" / Path(name).name
