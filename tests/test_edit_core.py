import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from edit_core import aggregate, edit_tokens, extract_gsm8k_final_answer

from edit_gsm8k import compare_tokens, token_id_list
from common import ROOT, yaml_float


def test_edits_only_answer_and_counts_final_and_transient_changes():
    calls = 0

    def predict(sequence, start, end):
        nonlocal calls
        calls += 1
        proposed = sequence[start:end]
        if calls == 1:
            proposed = [90, proposed[1]]
        elif calls == 2:
            proposed = [50, 91]
        return proposed, [0.8] * len(proposed)

    edited, metrics = edit_tokens(
        prefix=[10, 11, 12], answer=[50, 51], predict=predict,
        threshold=0.0, block_length=8, max_steps=4, mask_id=99,
    )

    assert edited == [50, 91]
    assert metrics["changed_positions"] == [1]
    assert metrics["ever_changed_positions"] == [0, 1]
    assert metrics["changed_token_pct"] == 50.0
    assert metrics["ever_changed_token_pct"] == 100.0
    assert metrics["replacement_events"] == 3
    assert metrics["forward_passes"] == 3
    assert metrics["last_replacement_confidences"] == {1: 0.8}


def test_threshold_is_strict_and_blocks_align_to_absolute_positions():
    windows = []

    def predict(sequence, start, end):
        windows.append((len(sequence), start, end))
        return [7] * (end - start), [0.5] * (end - start)

    edited, metrics = edit_tokens(
        prefix=[1, 2, 3], answer=[7, 8, 9], predict=predict,
        threshold=0.5, block_length=4, max_steps=2,
    )

    assert edited == [7, 8, 9]
    assert windows == [(4, 3, 4), (6, 4, 6)]
    assert metrics["unchanged"] is True


def test_rejects_mask_transition():
    def predict(sequence, start, end):
        return [99] * (end - start), [1.0] * (end - start)

    try:
        edit_tokens([1], [2], predict, mask_id=99)
    except ValueError as exc:
        assert "mask token" in str(exc)
    else:
        raise AssertionError("Expected token-to-mask transition to be rejected")


def test_extract_and_aggregate():
    assert extract_gsm8k_final_answer("work\n#### 1,234 ") == "1234"
    assert extract_gsm8k_final_answer("no marker") is None
    record = {
        "status": "ok", "answer_tokens": 4, "changed_tokens": 1,
        "changed_token_pct": 25.0, "ever_changed_tokens": 2,
        "replacement_events": 3, "unchanged": False,
        "final_answer_changed": True, "blocks": [],
    }
    summary = aggregate([record, {"status": "skipped"}])
    assert summary["changed_token_pct_micro"] == 25.0
    assert summary["final_answer_changed_examples"] == 1


def test_token_id_list_accepts_common_tokenizer_shapes():
    assert token_id_list([1, 2]) == [1, 2]
    assert token_id_list({"input_ids": [[3, 4]]}) == [3, 4]


def test_server_token_comparison_is_fixed_length():
    metrics = compare_tokens([10, 11, 12], [10, 21, 12])
    assert metrics["changed_positions"] == [1]
    assert metrics["changed_token_pct"] == 100 / 3

    try:
        compare_tokens([10, 11], [10])
    except ValueError as exc:
        assert "returned 1 tokens" in str(exc)
    else:
        raise AssertionError("Expected a fixed-length response check")


def test_aggregate_omits_unavailable_server_trace_metrics():
    record = {
        "status": "ok", "answer_tokens": 2, "changed_tokens": 1,
        "changed_token_pct": 50.0, "unchanged": False,
        "final_answer_changed": False,
        "trace_metrics_available": False,
    }
    summary = aggregate([record])
    assert "ever_changed_tokens" not in summary
    assert "replacement_events" not in summary


def test_reads_threshold_from_yaml():
    path = ROOT / "config" / "joint_threshold_t2t.yaml"
    assert yaml_float(path, "edit_threshold") == 0.0
