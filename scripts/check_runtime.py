"""Fail if this process uses another project's runtime or a non-ROCm torch."""
import json
import sys
from pathlib import Path

from common import SOURCE, VENV


def check():
    if Path(sys.prefix).resolve() != VENV.resolve():
        raise RuntimeError(f"Use {VENV}/bin/python; current environment: {sys.prefix}")
    import torch
    import sglang

    actual = Path(sglang.__file__).resolve()
    expected = (SOURCE / "python/sglang").resolve()
    if not actual.is_relative_to(expected):
        raise RuntimeError(f"Wrong SGLang source: {actual}; expected {expected}")
    if not torch.version.hip:
        raise RuntimeError(f"ROCm PyTorch required, found {torch.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch cannot access a GPU. Check driver/device access.")
    result = {"python": sys.executable, "sglang": str(actual),
              "torch_path": str(Path(torch.__file__).resolve()),
              "torch": torch.__version__, "hip": torch.version.hip,
              "gpus": [torch.cuda.get_device_name(i)
                       for i in range(torch.cuda.device_count())]}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    check()
