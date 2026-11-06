"""Paths shared by the current XLA evaluation tools."""

from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[3]
TEPL = HERE.parent / "tepl"


def repo_path(path):
    """Resolve current paths and paths retained in frozen pre-move records."""
    path = Path(path)
    old_lab = Path("labs/xla_rewrite")
    if path.is_relative_to(old_lab):
        return HERE.parent / path.relative_to(old_lab)
    return ROOT / path


def implementation_dependencies(dependencies):
    """Exclude historical research-document hashes from engineering checks."""
    return {
        path: digest
        for path, digest in dependencies.items()
        if not repo_path(path).is_relative_to(ROOT / "research")
    }
