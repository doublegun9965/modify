# LLaDA text correction experiments

Evaluate LLaDA 2.1 as a text correction model, including refinement of handwritten
OCR output. The first milestone is serving an unpatched LLaDA2.1-mini with a new
SGLang version on the AMD server. A successful smoke request does not establish
correction quality or in-place editing support.

## Isolation and version

- Local development does not require a venv or GPU packages.
- The Linux server creates `.venv/`. By default it does not inherit system
  packages. On the identified MI308X server, `reuse_system_torch: true` can
  expose its working ROCm PyTorch installation while keeping the project SGLang
  source and new Python packages in the venv. This intentionally shares the
  system PyTorch and its dependencies; SGLang is still installed separately.
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

The identified server has an MI308X (gfx942), ROCm 7.2.3, Python 3.12 and a
working system PyTorch `2.12.0+git6bbd260` with HIP 7.2.53211. It does not
have Cargo. The source build uses Rust extensions, so install Rust/Cargo before
the full setup. If moving to a
different server, probe it again and select matching ROCm packages there.

## 2. Prepare project source and venv

On the identified MI308X server, use the probed `/usr/bin/python3`, which
already sees its ROCm PyTorch installation. Copy the matching example first:

```bash
cp config/runtime.mi308x.example.json config/runtime.local.json
/usr/bin/python3 scripts/setup_server.py --prepare-only
```

The example sets `reuse_system_torch: true` and keeps `torch_pip_args` empty.
The venv inherits system packages only when created.
It also selects `compressed-tensors==0.16.0` for this server's PyTorch 2.12.
An existing local config copied before this change gets the same selection
automatically when setup detects system PyTorch 2.12; no file reset is needed.
Upstream SGLang v0.5.18 pins 0.15.0 for its older ROCm PyTorch 2.9.1 base,
but that package requires torch<2.11 and cannot resolve here. The setup script
changes only this dependency pin in the ignored SGLang checkout; it records the
effective config under `outputs/setup/`. This pairing still needs a server smoke
test to confirm runtime API compatibility.
If you copied the earlier MI308X example containing
`"SGLANG_BUILD_RUST_EXTS": "none"`, remove that key from `build_env` before
full installation. Keep any custom model path or other local settings.
If `.venv` already exists, the installer checks its setting and stops on a
mismatch rather than silently changing it. No project environment was present
when the initial probe was run.

If `venv`/`ensurepip` is missing, install the OS's matching Python venv package.
The setup script never changes system drivers or installs system packages.

For another server without suitable system PyTorch, set `torch_pip_args` in
the ignored `config/runtime.local.json`. This list is passed as individual
arguments to `pip install`.
For example, the shape is
`["torch==VERSION", "--index-url", "https://download.pytorch.org/whl/ROCM_CHANNEL"]`.
`VERSION` and `ROCM_CHANNEL` are placeholders, not usable settings. Use the exact
ROCm package selection confirmed for the server; URLs to official compatible
wheels can also be supplied. Do not use default PyPI CUDA torch packages.

The local JSON overrides top-level keys in `config/runtime.json`. Set optional
build variables (for example the confirmed GPU architecture) in `build_env` and
runtime variables in `server_env`. No old RMSNorm workaround is enabled by default.

## 3. Install on the AMD server

If Cargo downloads take time, prefetch with visible progress before the full
installation. The command exits successfully only after Cargo has resolved the
Rust dependencies; returning to a prompt after `timeout` does not prove success.
The command leaves successfully downloaded crates in the Cargo cache.

```bash
/usr/bin/python3 scripts/setup_server.py --prefetch-rust-only
```

The full setup also performs this visible Cargo check before pip's editable
build. If a previous installation is still running, stop it before prefetching
so the two Cargo processes do not contend for the same cache or lock.

Prerequisites: Git, a compatible Python with venv, ROCm development toolkit
including `hipcc`, C/C++ build tools, and Rust/Cargo. The pinned SGLang source
build includes a Rust extension. On the probed machine, install Rust using the
[official rustup instructions](https://rust-lang.org/tools/install/), then check
`~/.cargo/bin/cargo --version`. The source build has more dependencies than an
ordinary pure-Python package.


```bash
/usr/bin/python3 scripts/setup_server.py
```

This uses the selected ROCm PyTorch, verifies GPU access, constrains that torch
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
