# Project Instructions for Codex Agents

This project studies the potential of LLaDA 2.1 and similar models as text editing and correction models. Experiments are developed locally and run on a remote server.

## Research Direction

- The current goal is to evaluate whether these models can repair erroneous text while preserving correct content and the intended meaning.
- One motivating use case is refining text recognized from handwritten documents by OCR. This is an example application, not a fixed dataset or exclusive task scope.
- Evaluate text correction directly rather than assuming that ordinary continuation quality demonstrates editing ability.
- Distinguish correction of erroneous spans from unnecessary changes to correct spans. When clean references are available, assess both repair quality and preservation of correct content.
- Keep task definitions, datasets, error types, metrics, and inference strategies explicit. Do not assume that choices from earlier GSM8K experiments apply to this project.
- Threshold sweeps and error-trace collection from previous experiments are outside the current scope unless the user requests them again.

## Operating Model

- Develop and verify code locally, then synchronize it to the remote server for experiments.
- When this directory is a Git repository with a configured remote, commit and push useful code changes unless the user says not to. Do not invent a remote or initialize a repository solely to satisfy this rule.
- Do not assume that model experiments can run locally. Local verification should focus on syntax, CLI help, config parsing, and small unit-style checks.
- Server commands in README and scripts should be copy-paste friendly for Linux shells.

## Server-Local Configuration

- Prefer committed default or example configuration files plus ignored local override files.
- Use tracked `*.json` or `*.example.json` files for defaults and templates, and ignored `*.local.json` files for server-specific values where practical.
- Scripts should prefer local override files when present and otherwise fall back to tracked defaults.
- For server runtime environment variables, use an ignored local env file and a tracked example rather than hardcoding server-specific values.
- Do not commit credentials or server-specific configuration.

## Model Execution and Editing

- Choose the execution backend according to the editing experiment; SGLang is an option, not a mandatory backend for every experiment.
- Standard SGLang chat/completion APIs treat the prompt as fixed context and generate a continuation. They do not replace mask tokens inside the prompt.
- If an experiment requires in-place reconstruction or editing of an existing token span, use a suitable direct model forward/denoising implementation or a verified backend extension.
- Distinguish prompting the model to generate a corrected copy from directly editing tokens in the supplied text. Document which behavior each experiment measures.
- Verify the selected model and backend's editing behavior before relying on it in an experiment.

## Project-Specific SGLang Isolation

- If SGLang is used, keep a dedicated source checkout under this project, for example `third_party/sglang/`, on each machine that needs it.
- Do not modify or apply patches to the shared SGLang installation used by other projects.
- Create a project-specific `.venv` on the Linux server for SGLang and its dependencies. Local development does not need a venv. A separate source directory alone does not isolate an installed Python package.
- Install and launch SGLang through that environment, and verify that the imported package resolves to the intended project-specific source checkout.
- Record the upstream version or commit and the project's changes so the server environment can be reproduced.
- Experiments requiring incompatible SGLang changes should use separate checkouts and environments. Do not accumulate incompatible patches in one working tree.
- If multiple servers run concurrently, give them separate ports and runtime output paths.
- The initial unpatched trial uses an ignored checkout pinned by `config/runtime.json`. It currently targets SGLang v0.5.18. The long-term approach for synchronizing experimental source modifications remains to be decided: a dedicated fork with commits, a pinned submodule, or project-specific patches.
- Patch files are not a mandatory storage format. Do not automatically carry over or apply patches from previous experiments.
- Before adding the source checkout, decide how it will be tracked or ignored and how the local and server copies will be synchronized. Do not accidentally commit a full third-party source tree into the main repository.

## Experiment Outputs

- Write experiment results under `outputs/` by default.
- Every experiment run must use a fresh timestamped directory, such as `outputs/<experiment>/run_YYYYmmdd_HHMMSS/`, to avoid overwriting previous runs.
- Treat `--output-dir` as the base directory and create a timestamped run subdirectory unless the user explicitly requests a fixed output path.
- Save the effective experiment configuration and relevant model/backend version information with results.
- When useful, save aggregate metrics and per-example records containing the erroneous input, corrected output, clean reference if available, and evaluation results.
- Per-example evaluation records do not require internal denoising traces or automatic reruns of failed examples.
- Do not commit generated outputs, logs, datasets, model weights, or server-specific local configuration.

## Coding Conventions

- Keep scripts runnable from a fresh checkout without requiring an editable project install when practical.
- Favor explicit CLI arguments and configuration files over hardcoded constants.
- When adding an experiment script, add a copy-paste-friendly server command to the README.
- Use ASCII in code and docs unless there is a clear reason otherwise.
- Keep changes narrowly scoped to the requested experiment or workflow.
