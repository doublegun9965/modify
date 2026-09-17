"""Collect server facts without installing or changing anything."""
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from common import run_dir


def main():
    report = {"platform": platform.platform(), "python": sys.version,
              "executable": sys.executable}
    for name, args in (("rocm-smi", []), ("amd-smi", ["static"]),
                       ("hipcc", ["--version"]), ("rocminfo", []),
                       ("cargo", ["--version"])):
        executable = shutil.which(name)
        if not executable and name == "hipcc":
            candidate = Path("/opt/rocm/bin/hipcc")
            executable = str(candidate) if candidate.exists() else None
        if not executable:
            report[name] = "not found"
            continue
        try:
            result = subprocess.run([executable, *args], capture_output=True,
                                    text=True, errors="replace", timeout=30)
            report[name] = {"returncode": result.returncode,
                            "output": result.stdout + result.stderr}
        except subprocess.TimeoutExpired:
            report[name] = "timed out after 30 seconds"
    output = run_dir("probe") / "server.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
