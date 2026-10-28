"""Paths shared by the current XLA evaluation tools."""

from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
TEPL = HERE.parent / "tepl"
