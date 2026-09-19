"""Shared paths and configuration; no third-party dependencies."""
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party" / "sglang"
VENV = ROOT / ".venv"
PYTHON = VENV / "bin" / "python"


def dllm_algorithm_config():
    """Prefer an ignored server-local threshold config when it exists."""
    local = ROOT / "config" / "joint_threshold_t2t.local.yaml"
    return local if local.exists() else ROOT / "config" / "joint_threshold_t2t.yaml"


def yaml_float(path, key):
    """Read one top-level numeric scalar without adding a YAML dependency."""
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line.startswith(key + ":"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError as exc:
                raise ValueError(f"{path}: {key} must be numeric") from exc
    raise ValueError(f"{path}: missing {key}")


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
