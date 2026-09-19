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
runtime variables in `server_env`. The MI308X example sets
`SGLANG_DISABLE_VLLM_RMSNORM=1`: it avoids the incompatible vLLM RMSNorm path
on this server's ROCm stack. Apply the project patches once after the source
checkout exists. This installs both the ROCm compatibility change and the
GSM8K fixed-length T2T editing extension:

```bash
/usr/bin/python3 scripts/apply_sglang_patch.py
```

`launch_server.py` then loads the variable automatically for each server start.
The command is idempotent: patches already present are reported and skipped. It
changes only `third_party/sglang/`, which is this project's ignored, dedicated
source checkout.

## 3. Install on the AMD server

Cargo crate downloads use the project-local `.cargo/config.toml` with the USTC
sparse mirror. Cargo commands under this checkout, including the nested SGLang
source build, inherit it. The mirror does not alter other projects or the
system-wide Cargo configuration. If the mirror is unavailable from a server,
check `https://mirrors.ustc.edu.cn/crates.io-index/config.json` there before
trying another source. Switching registries may redownload crates already cached
from crates.io.

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
When borrowing system ROCm PyTorch, dependency checks cover packages installed
in the project venv. Global packages such as the server image's vLLM may
require conflicting versions, so a whole-environment `pip check` can report
unrelated conflicts. If SGLang already finished installing and the old setup
script stopped at `pip check`, pull the updated code and verify the existing
installation without rebuilding:

```bash
cd /mnt/workspace/modify
git pull
/usr/bin/python3 scripts/setup_server.py --verify-only
```

Only proceed to launch if this verification passes.
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

## 6. Measure preservation of correct GSM8K answers

This experiment keeps the chat-formatted question fixed and treats the existing
gold answer tokens as the editable region. The client sends those regions to the
project-patched SGLang server, which runs JointThreshold same-position
token-to-token (T2T) editing. It does not ask the model to generate a corrected
copy. There are no mask tokens and no M2T decisions; the server also excludes
the mask token from replacement candidates.

The tracked default config reads `/mnt/workspace/data/gsm8k_test.jsonl`, uses
the `question` and `answer` fields, and sets the T2T confidence threshold to
`0.0`. The matching server settings are tracked in
`config/joint_threshold_t2t.yaml`. Answer length is fixed: the experiment cannot
insert or delete tokens. Blocks follow the model's absolute token positions; an
answer may start partway through the block containing the fixed prompt.

Threshold 0 is maximally aggressive. For threshold experiments, create the
ignored server-local configuration and edit its `edit_threshold` value:

```bash
cp config/joint_threshold_t2t.yaml config/joint_threshold_t2t.local.yaml
sed -i 's/^edit_threshold:.*/edit_threshold: 0.9/' config/joint_threshold_t2t.local.yaml
```

Both the launcher and evaluation client automatically select this local file,
and each run records its path and effective threshold. Restart the server after
every threshold change so the new value takes effect.

After pulling this version, apply the patches and restart the server. An older
running server does not have the editing extension:

```bash
/usr/bin/python3 scripts/apply_sglang_patch.py
.venv/bin/python scripts/launch_server.py
```

Leave that terminal running. Run the experiment from a second terminal after
the server reports that it is ready.

Start with a small run to verify the server, tokenizer, data fields and output
files:

```bash
.venv/bin/python scripts/edit_gsm8k.py --limit 20
```

Then run the full test set:

```bash
.venv/bin/python scripts/edit_gsm8k.py
```

Every invocation creates `outputs/gsm8k_preservation/run_<timestamp>/`. The
main files are `summary.json`, all per-example data in `records.jsonl`, modified
examples in `changed_records.jsonl`, a changed-first `review.html`, and
`trace_report.html`. Open `trace_report.html` for manual inspection: it presents
each editing round as a timeline with answer-relative positions, old and new
tokens, selected-token confidence, local before/after context, and the complete
answer after that round. The JSONL files retain the same trace as structured
data for later analysis.

Tracing is requested only by the project editing client. It adds synchronization
and response-size overhead, so use it for diagnostic experiments rather than
throughput benchmarking. The client replays every trace locally and fails the
example if the replay does not exactly reproduce the server's final token IDs.
The summary reports both the final token difference percentage and the number of
intermediate edit rounds and replacement events, including edits that later
revert.
`final_answer_changed` compares the text after GSM8K's `####` marker; it is a
triage signal for selecting manual-review cases, not a complete mathematical
correctness judgment.

For a server-specific model path or other settings, copy the tracked config to
the ignored local override and edit only the required keys:

```bash
cp config/edit_gsm8k.json config/edit_gsm8k.local.json
```

## Local checks (no venv)

```bash
python -m compileall -q scripts
python scripts/setup_server.py --help
python scripts/launch_server.py --dry-run
python scripts/smoke_test.py --help
python scripts/edit_gsm8k.py --help
python -m pytest -q tests
```

## Upstream references

- [SGLang AMD installation](https://docs.sglang.io/docs/hardware-platforms/amd_gpu)
- [Pinned AMD package metadata](https://github.com/sgl-project/sglang/blob/v0.5.18/python/pyproject_other.toml)
- [Pinned ROCm build recipe](https://github.com/sgl-project/sglang/blob/v0.5.18/docker/rocm.Dockerfile)
- [LLaDA 2.1 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/InclusionAI/LLaDA-2.1.mdx)
