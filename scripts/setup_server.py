"""Create a server-only venv and install a pinned, private SGLang checkout."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import venv

from common import ROOT, SOURCE, VENV, PYTHON, config, isolated_env, run_dir


def run(args, **kwargs):
    print("+ " + " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Create venv and fetch source, without installing packages")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run on the Linux AMD server. No local venv is needed.")
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required")
    cfg = config()
    if not args.prepare_only and not cfg["torch_pip_args"]:
        parser.error("Set torch_pip_args in config/runtime.local.json after checking GPU/ROCm; see README")
    if SOURCE.is_symlink() or VENV.is_symlink():
        parser.error("Source and venv must be project-local directories, not symlinks")
    if not VENV.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(VENV)
    if not PYTHON.exists():
        parser.error("Existing .venv is not a Linux venv; it was left untouched")
    env = isolated_env(cfg["build_env"])
    run([PYTHON, "-c", "import sys; from pathlib import Path; "
         f"assert Path(sys.prefix).resolve() == Path({str(VENV)!r}).resolve(); "
         "assert 'include-system-site-packages = false' in "
         "(Path(sys.prefix) / 'pyvenv.cfg').read_text().lower()"], env=env)
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--branch", cfg["sglang_tag"],
             cfg["sglang_repository"], SOURCE])
    head = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if head != cfg["sglang_commit"]:
        parser.error(f"Existing source commit {head} differs from configured pin; source left untouched")
    print(f"Source: {SOURCE}\nCommit: {head}\nVenv: {VENV}")
    if args.prepare_only:
        return
    if not shutil.which("hipcc") and not (os.path.exists("/opt/rocm/bin/hipcc")):
        parser.error("ROCm development tools (hipcc) are required to build kernels")
    if not shutil.which("cargo"):
        parser.error("Rust cargo is required by this SGLang version; install the server toolchain first")
    output = run_dir("setup")
    (output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    pip = [PYTHON, "-m", "pip"]
    run([*pip, "install", "--upgrade", "pip", "setuptools", "wheel", "ninja", "packaging", "pybind11"], env=env)
    run([*pip, "install", *cfg["torch_pip_args"]], env=env)
    run([PYTHON, "-c", "import torch; assert torch.version.hip, 'Not ROCm torch'; "
         "assert torch.cuda.is_available(), 'No accessible AMD GPU'; "
         "print(torch.__version__, torch.version.hip)"], env=env)
    # Prevent dependency resolution from replacing the installed ROCm torch.
    torch_version = subprocess.check_output([PYTHON, "-c", "import torch; print(torch.__version__)"],
                                            text=True, env=env).strip()
    constraints = output / "constraints.txt"
    constraints.write_text(f"torch=={torch_version}\n", encoding="utf-8")
    env["PIP_CONSTRAINT"] = str(constraints)
    # Select upstream's AMD packaging metadata, preserving the original files.
    replacements = [
        (SOURCE / "python/pyproject.toml", SOURCE / "python/pyproject_other.toml"),
        (SOURCE / "python/sglang/kernels/aot/pyproject.toml",
         SOURCE / "python/sglang/kernels/aot/pyproject_rocm.toml"),
    ]
    for target, template in replacements:
        if not template.exists():
            raise RuntimeError(f"Missing upstream AMD packaging file: {template}")
        if target.exists() and target.read_bytes() != template.read_bytes():
            backup = target.with_name(target.name + ".pre-amd")
            if backup.exists():
                raise RuntimeError(f"Metadata differs after earlier setup; inspect {target} before retrying")
            shutil.copy2(target, backup)
        shutil.copy2(template, target)
    run([PYTHON, "setup_rocm.py", "install"],
        cwd=SOURCE / "python/sglang/kernels/aot", env=env)
    # Text serving needs srt_hip, not the image/video diffusion extras in all_hip.
    run([*pip, "install", "-e", str(SOURCE / "python") + "[srt_hip]"], env=env)
    run([*pip, "check"], env=env)
    run([PYTHON, ROOT / "scripts/check_runtime.py"], env=env)
    with (output / "requirements.freeze.txt").open("w", encoding="utf-8") as stream:
        run([*pip, "freeze"], env=env, stdout=stream)
    print(f"Setup checks passed. Installation record: {output}")


if __name__ == "__main__":
    main()
