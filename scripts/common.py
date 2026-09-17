"""Shared paths and configuration; no third-party dependencies."""
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party" / "sglang"
VENV = ROOT / ".venv"
PYTHON = VENV / "bin" / "python"


def config():
    result = json.loads((ROOT / "config/runtime.json").read_text(encoding="utf-8"))
    override = ROOT / "config/runtime.local.json"
    if override.exists():
        result.update(json.loads(override.read_text(encoding="utf-8")))
    return result


def isolated_env(extra=None):
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", VIRTUAL_ENV=str(VENV))
    env["PATH"] = str(VENV / "bin") + os.pathsep + env.get("PATH", "")
    env.update(extra or {})
    return env


def run_dir(name):
    path = ROOT / "outputs" / name / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    path.mkdir(parents=True)
    return path
