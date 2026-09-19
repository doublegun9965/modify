"""Edit existing GSM8K gold tokens directly with LLaDA2.1 forward passes."""
import argparse
from collections.abc import Mapping
from datetime import datetime
import hashlib
import html
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import subprocess
import time

from common import ROOT, config as runtime_config
from edit_core import aggregate, edit_tokens, extract_gsm8k_final_answer


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_id_list(value):
    """Normalize tokenizer output to one unbatched Python token-ID list."""
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one tokenized sequence")
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise TypeError("Tokenizer did not return a token-ID list")
    return value


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Explicit JSON override (otherwise *.local.json)")
    for name in ("dataset", "model-path", "question-field", "answer-field", "output-dir", "device", "revision"):
        parser.add_argument("--" + name)
    for name in ("block-length", "max-edit-steps", "limit", "offset", "max-sequence-tokens"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--t2t-threshold", type=float)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data without GPU packages")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config/edit_gsm8k.json").read_text(encoding="utf-8"))
    cfg["model_path"] = runtime_config()["model_path"]
    override = args.config or ROOT / "config/edit_gsm8k.local.json"
    if args.config or override.exists():
        extra = json.loads(override.read_text(encoding="utf-8"))
        unknown = extra.keys() - cfg.keys()
        if unknown:
            parser.error("Unknown configuration keys: " + ", ".join(sorted(unknown)))
        cfg.update(extra)
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k not in ("config", "dry_run")})
    if not 0 <= cfg["t2t_threshold"] <= 1:
        parser.error("t2t_threshold must be in [0, 1]")
    for name in ("block_length", "max_edit_steps", "max_sequence_tokens"):
        if not isinstance(cfg[name], int) or cfg[name] < 1:
            parser.error(name + " must be a positive integer")
    for name in ("limit", "offset"):
        if not isinstance(cfg[name], int) or cfg[name] < 0:
            parser.error(name + " must be a nonnegative integer")
    if cfg["dtype"] not in ("bfloat16", "float16", "float32"):
        parser.error("Invalid dtype")
    for name in ("dataset", "output_dir"):
        path = Path(cfg[name])
        cfg[name] = str(path if path.is_absolute() else ROOT / path)
    return args, cfg


def read_examples(cfg):
    examples = []
    with Path(cfg["dataset"]).open(encoding="utf-8-sig") as stream:
        nonempty = 0
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            index = nonempty
            nonempty += 1
            if index < cfg["offset"]:
                continue
            row = json.loads(line)
            question, answer = row[cfg["question_field"]], row[cfg["answer_field"]]
            if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Line {line_number}: question/answer must be nonempty strings")
            examples.append({"index": index, "source_line": line_number,
                             "question": question, "original_answer": answer})
            if cfg["limit"] and len(examples) >= cfg["limit"]:
                break
    if not examples:
        raise ValueError("No selected examples")
    return examples


def render_review(path, records):
    sections = ["<!doctype html><html><meta charset='utf-8'><title>GSM8K editing review</title>",
                "<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px}"
                "pre{white-space:pre-wrap;overflow-wrap:anywhere} .pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}"
                "mark{background:#ffdb8c}article{border-top:1px solid #aaa;padding:20px 0}td,th{padding:4px 12px;text-align:left}</style>",
                "<h1>GSM8K: original vs edited</h1><p>Changed tokens are highlighted. "
                "Changes are not correctness judgments. Changed examples appear first. "
                "Percentages compare token IDs at the same positions; no text re-tokenization.</p>"]
    ordered = sorted(records, key=lambda r: (r.get("unchanged", True), r["index"]))
    for r in ordered:
        sections.append(f"<article><h2>Example {r['index']} / line {r['source_line']}</h2><pre>{html.escape(r['question'])}</pre>")
        if r["status"] != "ok":
            sections.append(f"<p>{html.escape(r['status'] + ': ' + r['error'])}</p></article>")
            continue
        sections.append(f"<p>Final changes: {r['changed_tokens']}/{r['answer_tokens']} ({r['changed_token_pct']:.3f}%). "
                        f"Ever changed: {r['ever_changed_token_pct']:.3f}%. Replacement events: {r['replacement_events']}. "
                        f"Final answer changed: {r['final_answer_changed']}.</p>")
        sections.append("<div class='pair'><div><h3>Original</h3><pre>" + html.escape(r["original_answer"]) +
                        "</pre></div><div><h3>Edited</h3><pre>" + html.escape(r["edited_answer"]) + "</pre></div></div>")
        sections.append("<details><summary>Token replacements (zero-based answer positions)</summary><table>"
                        "<tr><th>Position</th><th>Original token</th><th>Edited token</th><th>Last confidence</th></tr>")
        for c in r["token_changes"]:
            sections.append(f"<tr><td>{c['position']}</td><td><mark>{html.escape(repr(c['old_text']))}</mark> ({c['old_id']})</td>"
                            f"<td><mark>{html.escape(repr(c['new_text']))}</mark> ({c['new_id']})</td>"
                            f"<td>{c['last_replacement_confidence']:.6f}</td></tr>")
        sections.append("</table></details></article>")
    path.write_text("\n".join(sections) + "</html>", encoding="utf-8")


def main():
    args, cfg = arguments()
    examples = read_examples(cfg)
    if args.dry_run:
        print(json.dumps({"config": cfg, "selected_examples": len(examples)}, indent=2))
        return
    run = Path(cfg["output_dir"]) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    cfg.update(backend="transformers_direct_forward", temperature=0.0,
               attention="absolute_block_causal", token_alignment="same_position",
               prompt="tokenizer chat template: user question, assistant generation prefix",
               answer_processing="verbatim, no appended EOS, no truncation", m2t_enabled=False,
               mask_token_candidate_enabled=False,
               dataset_sha256=sha256(cfg["dataset"]))
    write_json(run / "effective_config.json", cfg)
    print(f"Run directory: {run}", flush=True)
    records = []
    state = "initializing"
    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], revision=cfg["revision"], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_path"], revision=cfg["revision"], trust_remote_code=True,
            torch_dtype=getattr(torch, cfg["dtype"]), device_map=cfg["device"],
            attn_implementation="eager").eval()
        if model.__class__.__name__ != "LLaDA2MoeModelLM":
            raise ValueError(f"Unverified model class: {model.__class__.__name__}")
        source = inspect.getsourcefile(type(model))
        mask_id = inspect.signature(model.generate).parameters["mask_id"].default
        if not isinstance(mask_id, int):
            raise ValueError("Cannot resolve model mask ID")
        versions = {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "hip": torch.version.hip,
                    "cuda": torch.version.cuda, "model_class": type(model).__name__,
                    "model_commit": getattr(model.config, "_commit_hash", None),
                    "model_source": source, "model_source_sha256": sha256(source), "mask_id": mask_id,
                    "chat_template": tokenizer.chat_template,
                    "tokenizer_class": type(tokenizer).__name__}
        for package in ("accelerate", "tokenizers", "safetensors"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        versions["project_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
        write_json(run / "versions.json", versions)
        write_json(run / "model_config.json", model.config.to_dict())
        device = model.get_input_embeddings().weight.device
        state = "running"
        with (run / "records.jsonl").open("w", encoding="utf-8") as output:
            for example in examples:
                record = dict(example)
                started = time.monotonic()
                try:
                    prefix = token_id_list(tokenizer.apply_chat_template(
                        [{"role": "user", "content": example["question"]}],
                        tokenize=True, add_generation_prompt=True))
                    answer = token_id_list(tokenizer.encode(
                        example["original_answer"], add_special_tokens=False))
                    max_length = min(cfg["max_sequence_tokens"], model.config.max_position_embeddings)
                    if len(prefix) + len(answer) > max_length:
                        record.update(status="skipped", error=f"Sequence exceeds {max_length} tokens; no truncation")
                    else:
                        def predict(sequence, start, end):
                            ids = torch.tensor([sequence], dtype=torch.long, device=device)
                            positions = torch.arange(end, device=device)
                            blocks = positions // cfg["block_length"]
                            allowed = blocks[:, None] >= blocks[None, :]
                            # Explicit additive 4D mask: zero permits, -inf blocks.
                            # A float 0/1 matrix would add bias rather than block in eager attention.
                            attention = torch.zeros((end, end), dtype=model.dtype, device=device)
                            attention.masked_fill_(~allowed, float("-inf"))
                            with torch.inference_mode():
                                result = model(input_ids=ids, attention_mask=attention[None, None],
                                               position_ids=positions[None], use_cache=False,
                                               return_dict=True, output_attentions=False,
                                               output_router_logits=False)
                                logits = result.logits[0, start:end].float()
                                if not torch.isfinite(logits).all().item():
                                    raise ValueError("Non-finite model logits")
                                # Restrict replacement candidates to ordinary tokens. Accepting the
                                # mask token would turn this into a token-to-mask transition.
                                logits[:, mask_id] = float("-inf")
                                values, candidates = logits.max(dim=-1)
                                probabilities = (values - torch.logsumexp(logits, dim=-1)).exp()
                            return candidates.tolist(), probabilities.tolist()

                        edited, metrics = edit_tokens(prefix, answer, predict,
                            cfg["t2t_threshold"], cfg["block_length"], cfg["max_edit_steps"], mask_id)
                        decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False,
                                                              clean_up_tokenization_spaces=False)
                        changes = [{"position": i, "old_id": answer[i], "new_id": edited[i],
                                    "old_text": decode([answer[i]]), "new_text": decode([edited[i]]),
                                    "last_replacement_confidence": metrics["last_replacement_confidences"][i]}
                                   for i in metrics["changed_positions"]]
                        edited_text = decode(edited)
                        original_decoded = decode(answer)
                        original_final = extract_gsm8k_final_answer(original_decoded)
                        edited_final = extract_gsm8k_final_answer(edited_text)
                        record.update(metrics, status="ok", edited_answer=edited_text,
                                      original_token_ids=answer, edited_token_ids=edited,
                                      fixed_prefix_token_ids=prefix, token_changes=changes,
                                      original_decoded=original_decoded,
                                      tokenizer_roundtrip_exact=original_decoded == example["original_answer"],
                                      original_final_answer=original_final,
                                      edited_final_answer=edited_final,
                                      final_answer_changed=original_final != edited_final,
                                      introduced_special_token_positions=[i for i in metrics["changed_positions"]
                                                                          if edited[i] in tokenizer.all_special_ids])
                except Exception as exc:
                    record.update(status="error", error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    record["elapsed_seconds"] = time.monotonic() - started
                    records.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    print(f"[{len(records)}/{len(examples)}] {record['status']} "
                          f"changed={record.get('changed_token_pct', 'n/a')}%", flush=True)
        state = "completed"
    except BaseException as exc:
        state = "failed"
        write_json(run / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        summary = aggregate(records)
        summary.update(run_status=state, requested_examples=len(examples),
                       unprocessed_examples=len(examples) - len(records))
        write_json(run / "summary.json", summary)
        with (run / "changed_records.jsonl").open("w", encoding="utf-8") as changed_output:
            for record in records:
                if record.get("status") == "ok" and not record["unchanged"]:
                    changed_output.write(json.dumps(record, ensure_ascii=False) + "\n")
        render_review(run / "review.html", records)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
