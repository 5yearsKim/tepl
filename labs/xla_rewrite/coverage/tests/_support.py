"""Make coverage helpers available during unittest discovery from any directory."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from scripts import expression_coverage, hlo_support, summarize_coverage, paths

TEPL = paths.TEPL
