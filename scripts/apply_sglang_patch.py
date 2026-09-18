"""Apply a recorded SGLang patch to this project's dedicated source checkout."""
import argparse
import subprocess
from pathlib import Path

from common import ROOT, SOURCE


def run(args, **kwargs):
    print("+ " + " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE,
                        help="Dedicated SGLang checkout (default: project third_party/sglang)")
    parser.add_argument("patch", nargs="?", default="disable_vllm_rmsnorm.patch",
                        help="Patch filename under sglang_patches/")
    args = parser.parse_args()
    source = args.source.resolve()
    patch = ROOT / "sglang_patches" / args.patch
    if not (source / ".git").is_dir():
        parser.error(f"Not a Git checkout: {source}")
    if not patch.is_file():
        parser.error(f"Missing patch: {patch}")
    applied = subprocess.run(
        ["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if applied.returncode == 0:
        print(f"Already applied: {patch.name}")
        return
    run(["git", "-C", str(source), "apply", "--check", str(patch)])
    run(["git", "-C", str(source), "apply", str(patch)])
    print(f"Applied: {patch.name}")


if __name__ == "__main__":
    main()
