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
    reuse_system_torch = cfg.get("reuse_system_torch", False)
    if not isinstance(reuse_system_torch, bool):
        parser.error("reuse_system_torch must be true or false")
    if not args.prepare_only and not (cfg["torch_pip_args"] or reuse_system_torch):
        parser.error("Set torch_pip_args or reuse_system_torch in config/runtime.local.json; see README")
    if reuse_system_torch and cfg["torch_pip_args"]:
        parser.error("Choose either reuse_system_torch or torch_pip_args, not both")
    if SOURCE.is_symlink() or VENV.is_symlink():
        parser.error("Source and venv must be project-local directories, not symlinks")
    if not VENV.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=reuse_system_torch).create(VENV)
    if not PYTHON.exists():
        parser.error("Existing .venv is not a Linux venv; it was left untouched")
    env = isolated_env(cfg["build_env"])
    run([PYTHON, "-c", "import sys; from pathlib import Path; "
         f"assert Path(sys.prefix).resolve() == Path({str(VENV)!r}).resolve(); "
         f"assert 'include-system-site-packages = {str(reuse_system_torch).lower()}' in "
         "(Path(sys.prefix) / 'pyvenv.cfg').read_text().lower()"], env=env)
    if reuse_system_torch:
        run([PYTHON, "-c", "import torch; "
             "assert torch.version.hip, 'System torch is not ROCm'; "
             "assert torch.cuda.is_available(), 'System ROCm GPU not accessible'; "
             "print('System ROCm torch:', torch.__version__, torch.__file__)"], env=env)
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
    if env.get("SGLANG_BUILD_RUST_EXTS", "").strip().lower() == "none":
        parser.error("This setup expects SGLang's Rust extensions; remove "
                     "SGLANG_BUILD_RUST_EXTS=none from build_env")
    cargo = shutil.which("cargo") or str(os.path.expanduser("~/.cargo/bin/cargo"))
    if not os.path.isfile(cargo):
        parser.error("Rust cargo is required to build SGLang Rust extensions; install Rust first")
    env["PATH"] = os.path.dirname(cargo) + os.pathsep + env["PATH"]
    output = run_dir("setup")
    (output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    pip = [PYTHON, "-m", "pip"]
    run([*pip, "install", "--upgrade", "pip", "setuptools", "wheel", "ninja", "packaging", "pybind11"], env=env)
    if cfg["torch_pip_args"]:
        run([*pip, "install", *cfg["torch_pip_args"]], env=env)
    run([PYTHON, "-c", "import torch; assert torch.version.hip, 'Not ROCm torch'; "
         "assert torch.cuda.is_available(), 'No accessible AMD GPU'; "
         "print(torch.__version__, torch.version.hip)"], env=env)
    # Prevent dependency resolution from replacing the installed ROCm torch.
    torch_version = subprocess.check_output([PYTHON, "-c", "import torch; print(torch.__version__)"],
                                            text=True, env=env).strip()
    if reuse_system_torch and torch_version.startswith("2.12.") and not cfg.get("compressed_tensors_version"):
        cfg["compressed_tensors_version"] = "0.16.0"
        print("Using compressed-tensors 0.16.0 for system torch 2.12", flush=True)
        (output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if reuse_system_torch:
        run([PYTHON, "-c", "import torch, sys; "
             "print('Reusing system ROCm torch:', torch.__file__); "
             "assert not torch.__file__.startswith(sys.prefix + '/'), "
             "'Expected system torch but found a venv copy'"], env=env)
    constraints = output / "constraints.txt"
    constraints.write_text(f"torch=={torch_version}\n", encoding="utf-8")
    env["PIP_CONSTRAINT"] = str(constraints)
    # Select upstream's AMD packaging metadata, preserving the original files.
    # The upstream 0.15.0 pin targets older ROCm torch (<2.11). The MI308X
    # image instead ships torch 2.12, for which compressed-tensors 0.16.0
    # declares a compatible lower bound (torch>=2.10).
    compressed_version = cfg.get("compressed_tensors_version")
    if compressed_version and not reuse_system_torch:
        parser.error("compressed_tensors_version is only for a vetted system torch")
    replacements = [
        (SOURCE / "python/pyproject.toml", SOURCE / "python/pyproject_other.toml"),
        (SOURCE / "python/sglang/kernels/aot/pyproject.toml",
         SOURCE / "python/sglang/kernels/aot/pyproject_rocm.toml"),
    ]
    for target, template in replacements:
        if not template.exists():
            raise RuntimeError(f"Missing upstream AMD packaging file: {template}")
        original = template.read_bytes()
        desired = original
        if target == SOURCE / "python/pyproject.toml" and compressed_version:
            old_pin = b'"compressed-tensors==0.15.0"'
            if original.count(old_pin) != 1:
                raise RuntimeError("Expected upstream compressed-tensors pin changed; inspect source")
            if compressed_version != "0.16.0":
                parser.error("Only compressed-tensors 0.16.0 has been reviewed for this server")
            desired = original.replace(old_pin, b'"compressed-tensors==0.16.0"')
        if target.exists() and target.read_bytes() not in (original, desired):
            backup = target.with_name(target.name + ".pre-amd")
            if backup.exists():
                raise RuntimeError(f"Metadata differs after earlier setup; inspect {target} before retrying")
            shutil.copy2(target, backup)
        target.write_bytes(desired)
    run([PYTHON, "setup_rocm.py", "install"],
        cwd=SOURCE / "python/sglang/kernels/aot", env=env)
    # Text serving needs srt_hip, not the image/video diffusion extras in all_hip.
    run([*pip, "install", "-e", str(SOURCE / "python") + "[srt_hip]"], env=env)
    run([PYTHON, "-c", "import torch; "
         f"assert torch.__version__ == {torch_version!r}, "
         "'SGLang install replaced ROCm torch'"], env=env)
    run([*pip, "check"], env=env)
    run([PYTHON, ROOT / "scripts/check_runtime.py"], env=env)
    with (output / "requirements.freeze.txt").open("w", encoding="utf-8") as stream:
        run([*pip, "freeze"], env=env, stdout=stream)
    print(f"Setup checks passed. Installation record: {output}")


if __name__ == "__main__":
    main()
