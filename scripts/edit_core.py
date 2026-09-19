"""Dependency-free, fixed-length T2T editing and preservation metrics."""
import math
import re


def edit_tokens(prefix, answer, predict, threshold=0.0, block_length=32,
                max_steps=16, mask_id=None):
    """predict(sequence, start, end) returns same-position IDs/probabilities.

    Blocks align to absolute sequence positions, as in upstream LLaDA2.1.
    Only the active answer block is replaced, simultaneously per forward pass.
    Future blocks are not visible. No masks, insertions, deletions or EOS stops.
    """
    if not prefix or not answer:
        raise ValueError("Both the fixed prefix and editable answer must be nonempty")
    if not 0 <= threshold <= 1 or block_length < 1 or max_steps < 1:
        raise ValueError("Invalid editing parameters")
    original = list(answer)
    tokens = list(prefix) + original
    if mask_id is not None and mask_id in tokens:
        raise ValueError("Input contains a mask token; this is a T2T-only experiment")
    boundary = len(prefix)
    ever = set()
    last_confidence = {}
    events = 0
    blocks = []
    for block_start in range(boundary // block_length * block_length,
                             len(tokens), block_length):
        start = max(boundary, block_start)
        end = min(block_start + block_length, len(tokens))
        stopped = "step_limit"
        for step in range(1, max_steps + 1):
            proposed, confidence = predict(list(tokens[:end]), start, end)
            if len(proposed) != end - start or len(confidence) != end - start:
                raise ValueError("Predictor returned the wrong number of positions")
            replacements = []
            for index, (new, prob) in enumerate(zip(proposed, confidence), start):
                if not math.isfinite(prob) or not 0 <= prob <= 1:
                    raise ValueError("Non-finite or invalid prediction probability")
                if new != tokens[index] and prob > threshold:
                    if new == mask_id:
                        raise ValueError("Model proposed an accepted mask token; refusing an M2T transition")
                    replacements.append((index, new))
            for index, new in replacements:
                relative = index - boundary
                tokens[index] = new
                ever.add(relative)
                last_confidence[relative] = confidence[index - start]
            events += len(replacements)
            if not replacements:
                stopped = "unchanged"
                break
        blocks.append({"answer_start": start - boundary, "answer_end": end - boundary,
                       "forward_passes": step, "stop_reason": stopped})
    if tokens[:boundary] != list(prefix):
        raise AssertionError("Fixed context was modified")
    edited = tokens[boundary:]
    changed = [i for i, (old, new) in enumerate(zip(original, edited)) if old != new]
    return edited, {
        "answer_tokens": len(original), "changed_tokens": len(changed),
        "changed_token_pct": 100 * len(changed) / len(original),
        "ever_changed_tokens": len(ever), "ever_changed_token_pct": 100 * len(ever) / len(original),
        "replacement_events": events, "unchanged": not changed,
        "changed_positions": changed, "ever_changed_positions": sorted(ever),
        "last_replacement_confidences": {i: last_confidence[i] for i in changed},
        "forward_passes": sum(b["forward_passes"] for b in blocks),
        "blocks": blocks,
    }


def aggregate(records):
    good = [r for r in records if r["status"] == "ok"]
    total = sum(r["answer_tokens"] for r in good)
    changed = sum(r["changed_tokens"] for r in good)
    ever = sum(r["ever_changed_tokens"] for r in good)
    return {
        "selected_examples": len(records), "successful_examples": len(good),
        "failed_examples": sum(r["status"] == "error" for r in records),
        "skipped_examples": sum(r["status"] == "skipped" for r in records),
        "answer_tokens": total, "changed_tokens": changed,
        "changed_token_pct_micro": 100 * changed / total if total else None,
        "changed_token_pct_macro": sum(r["changed_token_pct"] for r in good) / len(good) if good else None,
        "ever_changed_tokens": ever,
        "ever_changed_token_pct_micro": 100 * ever / total if total else None,
        "replacement_events": sum(r["replacement_events"] for r in good),
        "unchanged_examples": sum(r["unchanged"] for r in good),
        "unchanged_example_pct": 100 * sum(r["unchanged"] for r in good) / len(good) if good else None,
        "final_answer_changed_examples": sum(r["final_answer_changed"] for r in good),
        "final_answer_changed_example_pct": (
            100 * sum(r["final_answer_changed"] for r in good) / len(good) if good else None
        ),
        "blocks_at_step_limit": sum(b["stop_reason"] == "step_limit" for r in good for b in r["blocks"]),
        "correctness_evaluated": False,
    }


def extract_gsm8k_final_answer(text):
    """Return a comparison-friendly answer following GSM8K's final #### marker."""
    matches = re.findall(r"####\s*([^\r\n]+)", text)
    if not matches:
        return None
    return re.sub(r"[\s,]", "", matches[-1]).strip()
