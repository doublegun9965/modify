"""Measure preservation when a patched SGLang server edits GSM8K gold tokens."""
import argparse
from collections.abc import Mapping
from datetime import datetime
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from common import ROOT, config as runtime_config, dllm_algorithm_config, yaml_float
from edit_core import aggregate, extract_gsm8k_final_answer


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_id_list(value):
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
    for name in ("dataset", "model-path", "question-field", "answer-field", "output-dir", "revision", "server-url"):
        parser.add_argument("--" + name)
    for name in ("limit", "offset", "max-sequence-tokens", "request-timeout", "request-retries"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data without contacting the server")
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config/edit_gsm8k.json").read_text(encoding="utf-8"))
    runtime = runtime_config()
    cfg["model_path"] = runtime["model_path"]
    cfg["server_url"] = f"http://{runtime['host']}:{runtime['port']}"
    algorithm_config = dllm_algorithm_config()
    cfg["t2t_threshold"] = yaml_float(algorithm_config, "edit_threshold")
    cfg["algorithm_config"] = str(algorithm_config)
    override = args.config or ROOT / "config/edit_gsm8k.local.json"
    if args.config or override.exists():
        extra = json.loads(override.read_text(encoding="utf-8"))
        unknown = extra.keys() - cfg.keys()
        if unknown:
            parser.error("Unknown configuration keys: " + ", ".join(sorted(unknown)))
        cfg.update(extra)
    cfg.update({k: v for k, v in vars(args).items() if v is not None and k not in ("config", "dry_run")})
    if not 0 <= cfg["t2t_threshold"] <= 1:
        parser.error("edit_threshold in the algorithm config must be in [0, 1]")
    for name in ("max_sequence_tokens", "request_timeout"):
        if not isinstance(cfg[name], int) or cfg[name] < 1:
            parser.error(name + " must be a positive integer")
    for name in ("limit", "offset", "request_retries"):
        if not isinstance(cfg[name], int) or cfg[name] < 0:
            parser.error(name + " must be a nonnegative integer")
    for name in ("dataset", "output_dir"):
        path = Path(cfg[name])
        cfg[name] = str(path if path.is_absolute() else ROOT / path)
    cfg["server_url"] = cfg["server_url"].rstrip("/")
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


def http_json(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc.reason}") from exc


def generate(cfg, prefix_ids, answer_ids, pad_id):
    payload = {
        "input_ids": prefix_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": len(answer_ids),
            "ignore_eos": True,
            "custom_params": {
                "dllm_edit_ids": answer_ids,
                "dllm_edit_pad_id": pad_id,
                "dllm_edit_trace": True,
            },
        },
    }
    last_error = None
    for attempt in range(cfg["request_retries"] + 1):
        try:
            return http_json(cfg["server_url"] + "/generate", payload, cfg["request_timeout"])
        except RuntimeError as exc:
            last_error = exc
            if attempt == cfg["request_retries"]:
                raise
            time.sleep(min(2 ** attempt, 10))
    raise last_error


def compare_tokens(original, edited):
    if len(edited) != len(original):
        raise ValueError(f"Server returned {len(edited)} tokens for a {len(original)}-token answer")
    changed = [i for i, pair in enumerate(zip(original, edited)) if pair[0] != pair[1]]
    return {
        "answer_tokens": len(original), "changed_tokens": len(changed),
        "changed_token_pct": 100 * len(changed) / len(original),
        "unchanged": not changed, "changed_positions": changed,
    }


def extract_edit_trace(meta_info):
    """Extract the one request-level trace stored in token-aligned metadata."""
    slots = meta_info.get("dllm_edit_trace", []) if isinstance(meta_info, dict) else []
    if not isinstance(slots, list):
        raise ValueError("Server returned a malformed dllm_edit_trace")
    for value in slots:
        if value is not None:
            if not isinstance(value, list):
                raise ValueError("Server returned a malformed trace payload")
            return value
    return []


def enrich_edit_trace(raw_trace, prefix_length, original, edited, decode):
    """Replay server events and add decoded, answer-relative views for review."""
    current = list(original)
    rounds = []
    for sequence_number, raw_round in enumerate(raw_trace, 1):
        before = list(current)
        changes = []
        for raw_change in raw_round.get("changes", []):
            answer_position = raw_change["absolute_position"] - prefix_length
            if not 0 <= answer_position < len(current):
                raise ValueError(f"Trace position {answer_position} is outside the answer")
            if current[answer_position] != raw_change["old_id"]:
                raise ValueError(
                    f"Trace old token mismatch at answer position {answer_position}"
                )
            change = dict(raw_change, answer_position=answer_position,
                          old_text=decode([raw_change["old_id"]]),
                          new_text=decode([raw_change["new_id"]]))
            changes.append(change)
        for change in changes:
            current[change["answer_position"]] = change["new_id"]
        changed_positions = [change["answer_position"] for change in changes]
        left = max(0, min(changed_positions) - 6)
        right = min(len(current), max(changed_positions) + 7)
        rounds.append({
            "round": sequence_number,
            "block_start_absolute": raw_round["block_start"],
            "answer_block_start": max(0, raw_round["block_start"] - prefix_length),
            "answer_block_end": min(
                len(current), raw_round["block_end"] - prefix_length
            ),
            "block_step": raw_round["step"],
            "changes": changes,
            "context_before": decode(before[left:right]),
            "context_after": decode(current[left:right]),
            "answer_after": decode(current),
        })
    return rounds, current == list(edited)


def render_review(path, records):
    sections = ["<!doctype html><html><meta charset='utf-8'><title>GSM8K editing review</title>",
                "<style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px}"
                "pre{white-space:pre-wrap;overflow-wrap:anywhere}.pair{display:grid;grid-template-columns:1fr 1fr;gap:24px}"
                "mark{background:#ffdb8c}article{border-top:1px solid #aaa;padding:20px 0}td,th{padding:4px 12px;text-align:left}</style>",
                "<h1>GSM8K: original vs edited</h1><p>Final changed tokens are highlighted. "
                "Changed examples appear first. Percentages compare exact server output token IDs at the same positions.</p>"]
    for record in sorted(records, key=lambda item: (item.get("unchanged", True), item["index"])):
        sections.append(f"<article><h2>Example {record['index']} / line {record['source_line']}</h2><pre>{html.escape(record['question'])}</pre>")
        if record["status"] != "ok":
            sections.append(f"<p>{html.escape(record['status'] + ': ' + record['error'])}</p></article>")
            continue
        sections.append(f"<p>Final changes: {record['changed_tokens']}/{record['answer_tokens']} ({record['changed_token_pct']:.3f}%). "
                        f"Final answer changed: {record['final_answer_changed']}.</p>")
        sections.append("<div class='pair'><div><h3>Original</h3><pre>" + html.escape(record["original_answer"]) +
                        "</pre></div><div><h3>Edited</h3><pre>" + html.escape(record["edited_answer"]) + "</pre></div></div>")
        sections.append("<details><summary>Token replacements (zero-based answer positions)</summary><table>"
                        "<tr><th>Position</th><th>Original token</th><th>Edited token</th></tr>")
        for change in record["token_changes"]:
            sections.append(f"<tr><td>{change['position']}</td><td><mark>{html.escape(repr(change['old_text']))}</mark> ({change['old_id']})</td>"
                            f"<td><mark>{html.escape(repr(change['new_text']))}</mark> ({change['new_id']})</td></tr>")
        sections.append("</table></details></article>")
    path.write_text("\n".join(sections) + "</html>", encoding="utf-8")


def trace_report_html(records):
    ok = [record for record in records if record.get("status") == "ok"]
    changed = [record for record in ok if not record["unchanged"]]
    rounds = sum(len(record.get("edit_trace", [])) for record in ok)
    events = sum(len(item["changes"]) for record in ok
                 for item in record.get("edit_trace", []))
    sections = ["<!doctype html><html><head><meta charset='utf-8'>",
                "<title>GSM8K T2T edit trace</title>",
                "<style>:root{color-scheme:light}body{font:15px/1.55 system-ui;max-width:1500px;margin:auto;padding:32px;color:#172033}"
                ".summary,.round{border:1px solid #d8deea;border-radius:12px;padding:16px;margin:14px 0;background:#fff}"
                ".pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}.muted{color:#657086}"
                "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fb;padding:14px;border-radius:8px}"
                "table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #e4e8f0;padding:8px;text-align:left;vertical-align:top}"
                "code{background:#eef2f8;padding:2px 5px;border-radius:4px}.bad{color:#a12622}.good{color:#176b3a}"
                "details.example{border-top:2px solid #d8deea;padding:14px 0}summary{cursor:pointer;font-weight:650}"
                "@media(max-width:850px){.pair{grid-template-columns:1fr}}</style></head><body>",
                "<h1>GSM8K token-edit timeline</h1>",
                f"<div class='summary'><b>{len(ok)}</b> completed examples · <b>{len(changed)}</b> changed · "
                f"<b>{rounds}</b> edit rounds · <b>{events}</b> replacement events"
                "<p class='muted'>Positions are zero-based within the gold answer. Confidence is the selected token probability in that forward pass. "
                "Each round shows the text after all replacements from that pass were applied simultaneously.</p></div>"]
    ordered = sorted(ok, key=lambda item: (item["unchanged"], item["index"]))
    for record in ordered:
        trace = record.get("edit_trace", [])
        state = "unchanged" if record["unchanged"] else f"{record['changed_tokens']} final token changes"
        sections.append(f"<details class='example' {'open' if trace else ''}><summary>Example {record['index']} — {state} — {len(trace)} edit rounds</summary>")
        sections.append("<h3>Question</h3><pre>" + html.escape(record["question"]) + "</pre>")
        sections.append("<div class='pair'><div><h3>Original gold answer</h3><pre>" +
                        html.escape(record["original_answer"]) +
                        "</pre></div><div><h3>Final edited answer</h3><pre>" +
                        html.escape(record["edited_answer"]) + "</pre></div></div>")
        if not trace:
            sections.append("<p class='good'>No token replacement occurred.</p>")
        for item in trace:
            sections.append(
                f"<div class='round'><h3>Round {item['round']} · answer block "
                f"[{item['answer_block_start']}, {item['answer_block_end']}) · "
                f"block step {item['block_step']}</h3>"
            )
            sections.append("<table><tr><th>Answer position</th><th>Old token</th><th>New token</th><th>Confidence</th></tr>")
            for change in item["changes"]:
                sections.append(
                    f"<tr><td>{change['answer_position']}</td>"
                    f"<td><code>{html.escape(repr(change['old_text']))}</code> ({change['old_id']})</td>"
                    f"<td><code>{html.escape(repr(change['new_text']))}</code> ({change['new_id']})</td>"
                    f"<td>{change['confidence']:.6f}</td></tr>"
                )
            sections.append("</table><div class='pair'><div><h4>Local context before</h4><pre>" +
                            html.escape(item["context_before"]) +
                            "</pre></div><div><h4>Local context after</h4><pre>" +
                            html.escape(item["context_after"]) + "</pre></div></div>")
            sections.append("<details><summary>Full answer after this round</summary><pre>" +
                            html.escape(item["answer_after"]) + "</pre></details></div>")
        sections.append("</details>")
    sections.append("</body></html>")
    return "\n".join(sections)


def render_trace_report(path, records):
    path.write_text(trace_report_html(records), encoding="utf-8")


def main():
    args, cfg = arguments()
    examples = read_examples(cfg)
    if args.dry_run:
        print(json.dumps({"config": cfg, "selected_examples": len(examples)}, indent=2))
        return
    run = Path(cfg["output_dir"]) / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    cfg.update(backend="sglang_patched_joint_threshold", temperature=0.0,
               token_alignment="same_position", answer_processing="verbatim, fixed length",
               m2t_enabled=False, mask_token_candidate_enabled=False,
               trace_metrics_available=True, trace_report="trace_report.html",
               dataset_sha256=sha256(cfg["dataset"]))
    write_json(run / "effective_config.json", cfg)
    print(f"Run directory: {run}", flush=True)
    records, state = [], "initializing"
    try:
        import transformers
        from transformers import AutoTokenizer
        server_info = http_json(cfg["server_url"] + "/get_model_info", timeout=cfg["request_timeout"])
        if server_info.get("project_dllm_t2t_edit_api") != 1:
            raise RuntimeError(
                "SGLang server lacks the project T2T editing patch; apply patches and restart it"
            )
        if server_info.get("project_dllm_t2t_trace_api") != 1:
            raise RuntimeError(
                "SGLang server lacks the project T2T trace patch; apply patches and restart it"
            )
        server_runtime = http_json(cfg["server_url"] + "/server_info", timeout=cfg["request_timeout"])
        if server_runtime.get("dllm_algorithm") != "JointThreshold":
            raise RuntimeError("SGLang server is not running JointThreshold")
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], revision=cfg["revision"], trust_remote_code=True)
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
        if not isinstance(pad_id, int):
            raise ValueError("Tokenizer has neither an EOS nor a pad token ID")
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        write_json(run / "versions.json", {
            "python": platform.python_version(), "transformers": transformers.__version__,
            "tokenizers": importlib.metadata.version("tokenizers"), "server": server_info,
            "server_runtime": server_runtime,
            "project_commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "tokenizer_class": type(tokenizer).__name__, "chat_template": tokenizer.chat_template,
        })
        state = "running"
        with (run / "records.jsonl").open("w", encoding="utf-8") as output:
            for example in examples:
                record, started = dict(example), time.monotonic()
                try:
                    prefix = token_id_list(tokenizer.apply_chat_template(
                        [{"role": "user", "content": example["question"]}],
                        tokenize=True, add_generation_prompt=True))
                    answer = token_id_list(tokenizer.encode(example["original_answer"], add_special_tokens=False))
                    if len(prefix) + len(answer) > cfg["max_sequence_tokens"]:
                        record.update(status="skipped", error=f"Sequence exceeds {cfg['max_sequence_tokens']} tokens; no truncation")
                    else:
                        response = generate(cfg, prefix, answer, pad_id)
                        edited = token_id_list(response["output_ids"])
                        metrics = compare_tokens(answer, edited)
                        decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                        raw_trace = extract_edit_trace(response.get("meta_info", {}))
                        edit_trace, trace_replay_exact = enrich_edit_trace(
                            raw_trace, len(prefix), answer, edited, decode
                        )
                        if not trace_replay_exact:
                            raise ValueError("Replaying the server edit trace did not reproduce its output")
                        changes = [{"position": i, "old_id": answer[i], "new_id": edited[i],
                                    "old_text": decode([answer[i]]), "new_text": decode([edited[i]])}
                                   for i in metrics["changed_positions"]]
                        edited_text, original_decoded = decode(edited), decode(answer)
                        original_final = extract_gsm8k_final_answer(original_decoded)
                        edited_final = extract_gsm8k_final_answer(edited_text)
                        record.update(metrics, status="ok", trace_metrics_available=True,
                                      edited_answer=edited_text,
                                      original_token_ids=answer, edited_token_ids=edited,
                                      fixed_prefix_token_ids=prefix, token_changes=changes,
                                      original_decoded=original_decoded,
                                      tokenizer_roundtrip_exact=original_decoded == example["original_answer"],
                                      original_final_answer=original_final, edited_final_answer=edited_final,
                                      final_answer_changed=original_final != edited_final,
                                      server_meta_info={k: v for k, v in response.get("meta_info", {}).items()
                                                        if k != "dllm_edit_trace"},
                                      edit_trace=edit_trace,
                                      trace_rounds=len(edit_trace),
                                      trace_replacement_events=sum(len(item["changes"]) for item in edit_trace),
                                      trace_replay_exact=trace_replay_exact,
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
                    print(f"[{len(records)}/{len(examples)}] {record['status']} changed={record.get('changed_token_pct', 'n/a')}%", flush=True)
        state = "completed"
    except BaseException as exc:
        state = "failed"
        write_json(run / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        summary = aggregate(records)
        summary.update(run_status=state, requested_examples=len(examples),
                       unprocessed_examples=len(examples) - len(records), trace_metrics_available=True,
                       trace_rounds=sum(record.get("trace_rounds", 0) for record in records),
                       trace_replacement_events=sum(record.get("trace_replacement_events", 0)
                                                    for record in records))
        write_json(run / "summary.json", summary)
        with (run / "changed_records.jsonl").open("w", encoding="utf-8") as changed_output:
            for record in records:
                if record.get("status") == "ok" and not record["unchanged"]:
                    changed_output.write(json.dumps(record, ensure_ascii=False) + "\n")
        render_review(run / "review.html", records)
        render_trace_report(run / "trace_report.html", records)
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
