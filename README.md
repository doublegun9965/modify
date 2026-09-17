# LLaDA text correction experiments

Evaluate LLaDA 2.1 as a text correction model, including refinement of handwritten
OCR output. The first milestone is serving an unpatched LLaDA2.1-mini with a new
SGLang version on the AMD server. A successful smoke request does not establish
correction quality or in-place editing support.

## Isolation and version

- Local development does not require a venv or GPU packages.
- The Linux server creates `.venv/` without shared system packages.
- SGLang source lives in `third_party/sglang/`, an ignored, independent checkout.
- Initial version: `v0.5.18`, commit `71de97b264b04dcd514cf904003028aefe9775c8`.
- No previous experimental patches are applied. AMD packaging files supplied by
  upstream are selected during installation; originals are backed up in place.
- Source and venv directories must not be symlinks to shared installations.
- Source modifications are not uploaded by pushing this repository. Before making
  experimental SGLang changes, select a fork/commit or patch synchronization workflow.

## 1. Inspect the server

```bash
git clone https://github.com/doublegun9965/modify.git
cd modify
python3 scripts/probe_server.py
```

Share the generated `outputs/probe/run_*/server.json` to select the correct ROCm
PyTorch packages. GPU model, ROCm development toolkit, Python ABI and wheel
availability must match. The installer deliberately has no guessed wheel URL.
An existing working environment's PyTorch is not inherited by the new venv.

## 2. Prepare project source and venv

Choose the server Python interpreter compatible with the selected ROCm wheels
(Python 3.10+; using a supported Python 3.12 installation is a reasonable starting
point, but wheel availability must be checked).

```bash
python3 scripts/setup_server.py --prepare-only
cp config/runtime.local.example.json config/runtime.local.json
```

If `venv`/`ensurepip` is missing, install the OS's matching Python venv package.
The setup script never changes system drivers or installs system packages.

Edit the ignored `config/runtime.local.json` with your model path and
`torch_pip_args`. This list is passed as individual arguments to `pip install`.
For example, the shape is
`["torch==VERSION", "--index-url", "https://download.pytorch.org/whl/ROCM_CHANNEL"]`.
`VERSION` and `ROCM_CHANNEL` are placeholders, not usable settings. Use the exact
ROCm package selection confirmed for the server; URLs to official compatible
wheels can also be supplied. Do not use default PyPI CUDA torch packages.

The local JSON overrides top-level keys in `config/runtime.json`. Set optional
build variables (for example the confirmed GPU architecture) in `build_env` and
runtime variables in `server_env`. No old RMSNorm workaround is enabled by default.

## 3. Install on the AMD server

Prerequisites: Git, a compatible Python with venv, ROCm development toolkit
including `hipcc`, C/C++ build tools, and Rust/Cargo on PATH. The latest source
build has more dependencies than an ordinary pure-Python package.

```bash
python3 scripts/setup_server.py
```

This installs ROCm PyTorch first, verifies GPU access, constrains that torch
version during dependency resolution, builds upstream ROCm kernels and installs
the text-serving `srt_hip` extra in editable mode. It then runs dependency and
runtime checks and records the package versions under `outputs/setup/`.
The larger `all_hip` extra also installs image/video diffusion components, which
this text-serving experiment does not need.

This is an initial source-build attempt, not a claim of a tested AMD dependency
combination. If installation or startup fails, preserve the terminal error and
probe report. Some GPU/backend combinations may need additional AMD components
such as AITER; select those after examining the actual server rather than applying
old patches or replacing the shared environment.

## 4. Launch LLaDA2.1-mini

```bash
.venv/bin/python scripts/launch_server.py
```

The script checks that SGLang resolves inside this project's checkout and that
PyTorch sees AMD GPUs. Default address: `127.0.0.1:30001`, TP=1, one concurrent
request. GPU capacity and multi-GPU settings must fit the chosen model.
Attention backend selection is left to SGLang; the NVIDIA FlashInfer example is
not hardcoded for AMD. Additional launch options belong in `extra_server_args`.
Launch records are saved under `outputs/server/`; logs remain in the terminal.
Wait for server readiness, then open another terminal in the repository.

## 5. Send a text correction request

```bash
.venv/bin/python scripts/smoke_test.py
```

The script saves its input, response, timing and any error under a new
`outputs/smoke/run_*/` directory. It fails on request errors or empty output.
The prompt asks for a corrected copy of synthetic OCR-like text; this is not an
OCR dataset evaluation and does not edit prompt tokens in place.

## Local checks (no venv)

```bash
python -m compileall -q scripts
python scripts/setup_server.py --help
python scripts/launch_server.py --dry-run
python scripts/smoke_test.py --help
```

## Upstream references

- [SGLang AMD installation](https://docs.sglang.io/docs/hardware-platforms/amd_gpu)
- [Pinned AMD package metadata](https://github.com/sgl-project/sglang/blob/v0.5.18/python/pyproject_other.toml)
- [Pinned ROCm build recipe](https://github.com/sgl-project/sglang/blob/v0.5.18/docker/rocm.Dockerfile)
- [LLaDA 2.1 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/InclusionAI/LLaDA-2.1.mdx)
