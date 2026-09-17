"""Collect server facts without installing or changing anything."""
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from common import run_dir


def capture(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                errors="replace", timeout=30)
        return {"command": command, "returncode": result.returncode,
                "output": result.stdout + result.stderr}
    except subprocess.TimeoutExpired:
        return {"command": command, "error": "timed out after 30 seconds"}
    except OSError as exc:
        return {"command": command, "error": str(exc)}


def main():
    report = {"platform": platform.platform(), "python": sys.version,
              "executable": sys.executable}
    for name, args in (("rocm-smi", []), ("amd-smi", ["static"]),
                       ("hipcc", ["--version"]), ("rocminfo", []),
                       ("cargo", ["--version"]), ("nvidia-smi", []),
                       ("hy-smi", []), ("lspci", ["-nn"])):
        executable = shutil.which(name)
        if not executable:
            candidates = [Path("/opt/rocm/bin") / name,
                          Path("/opt/rocm/hip/bin") / name,
                          Path("/opt/rocm/rocm_smi/bin") / name,
                          Path("/opt/dtk/bin") / name,
                          Path.home() / ".cargo/bin" / name]
            executable = next((str(p) for p in candidates if p.is_file()), None)
        if not executable:
            report[name] = "not found"
            continue
        report[name] = capture([executable, *args])
    # Inspect the existing interpreter in a subprocess, without installing packages.
    report["existing_torch"] = capture([sys.executable, "-c", """
import json
import torch
available = torch.cuda.is_available()
print(json.dumps({
    'version': torch.__version__, 'path': torch.__file__,
    'hip': getattr(torch.version, 'hip', None),
    'cuda': getattr(torch.version, 'cuda', None),
    'gpu_available': available,
    'devices': [{'name': torch.cuda.get_device_name(i),
                 'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                for i in range(torch.cuda.device_count())] if available else []
}, indent=2))
"""])
    report["device_paths"] = [str(p) for pattern in ("kfd", "dri/*", "nvidia*", "hygon*")
                              for p in Path("/dev").glob(pattern)]
    report["toolkit_paths"] = [str(p) for pattern in ("rocm*", "dtk*", "cuda*")
                               for p in Path("/opt").glob(pattern)]
    report["os_release"] = (Path("/etc/os-release").read_text(errors="replace")
                            if Path("/etc/os-release").exists() else "not found")
    output = run_dir("probe") / "server.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
