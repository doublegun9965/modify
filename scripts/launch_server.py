"""Start LLaDA using this project's venv; --dry-run works without GPU packages."""
import argparse
import json
import os
import shlex

from common import ROOT, PYTHON, config, isolated_env, run_dir


def command(cfg):
    return [str(PYTHON), "-m", "sglang.launch_server",
            "--model-path", cfg["model_path"],
            "--dllm-algorithm", "JointThreshold",
            "--dllm-algorithm-config", str(ROOT / "config/joint_threshold_t2t.yaml"),
            "--dllm-fdfo", "--trust-remote-code",
            "--tp", str(cfg["tp"]), "--host", cfg["host"],
            "--port", str(cfg["port"]),
            "--mem-fraction-static", str(cfg["mem_fraction_static"]),
            "--max-running-requests", "1", *cfg["extra_server_args"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = config()
    cmd = command(cfg)
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    # Re-exec under the isolated environment before importing GPU packages.
    from pathlib import Path
    import sys
    from common import VENV
    if Path(sys.prefix).resolve() != VENV.resolve() or os.environ.get("MODIFY_RUNTIME_CLEAN") != "1":
        env = isolated_env(cfg["server_env"])
        env["MODIFY_RUNTIME_CLEAN"] = "1"
        os.execve(str(PYTHON), [str(PYTHON), str(ROOT / "scripts/launch_server.py")], env)
    from check_runtime import check
    runtime = check()
    output = run_dir("server")
    (output / "launch.json").write_text(json.dumps({"config": cfg, "runtime": runtime,
                                                   "command": cmd}, indent=2), encoding="utf-8")
    print(f"Launch record: {output}\nServer output remains in this terminal.", flush=True)
    os.execve(str(PYTHON), cmd, os.environ.copy())


if __name__ == "__main__":
    main()
