from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ruscuav_benchmark

REPO = Path(__file__).resolve().parents[1]


def test_single_version_source():
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert 'version = {attr = "ruscuav_benchmark.__version__"}' in pyproject
    assert (REPO / "VERSION").read_text(encoding="utf-8").strip() == ruscuav_benchmark.__version__
