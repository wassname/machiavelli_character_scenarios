"""Unpaid blinding, option mapping and parser regressions. -- PI/Sol"""

import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from action_choice import MODEL_PATH, ROOT, build_packet, parse_receipt, render, select_scenes
from score_action_choice import uniform_baseline, cluster_interval


def main() -> None:
    config = json.loads(MODEL_PATH.read_text())
    model = next(iter(config["models"]))
    row = select_scenes(ROOT / "outputs/20261023_virtues_v2_local_release/parquet/train.parquet",
                        ROOT / "outputs/20261023_virtues_v2_contrast_2500_resumed/virtues.parquet", 1, 20261023)[0]
    forward = build_packet(row, "forward", model, config)
    reverse = build_packet(row, "reverse", model, config)
    contaminated = row | {"choice_virtue_labels": "SENTINEL_HIDDEN_VIRTUE_LABEL", "choice_labels": "SENTINEL_HIDDEN_OUTCOME"}
    assert build_packet(contaminated, "forward", model, config) == forward
    assert "SENTINEL_HIDDEN" not in json.dumps(forward["request"])
    assert forward["request"]["provider"]["allow_fallbacks"] is False
    assert set(reverse["metadata"]["displayed_to_original"].values()) == set(row["choice_text"])
    assert list(forward["metadata"]["displayed_to_original"].values()) == list(reverse["metadata"]["displayed_to_original"].values())[::-1]
    old = "Actions to choose:\n" + row["actions_text"] + "\n\n"
    prefix, suffix = row["text"].split(old)
    for order in ("forward", "reverse"):
        consumed, mapping = render(row, order)
        assert consumed.startswith(prefix) and consumed.endswith(suffix)
        assert all(consumed.count(f"{i}: {row['choice_text'][key]}") == 1 for i, key in mapping.items())
    response = {"model": model, "provider": "DeepInfra", "usage": {"cost": 0}, "choices": [{
        "finish_reason": "stop", "message": {"content": json.dumps({"choice": 0, "reasoning": "I choose this action. I accept its risk.", "setup_note": ""})}}]}
    record = {"metadata": reverse["metadata"], "request_key": "offline", "cache_hit": True, "http_status": 200, "response": response}
    parsed = parse_receipt(record, config)
    assert parsed["status"] == "ok" and parsed["original_choice_id"] == reverse["metadata"]["displayed_to_original"]["0"]
    length = copy.deepcopy(record)
    length["response"]["choices"][0]["finish_reason"] = "length"
    assert parse_receipt(length, config)["status"] == "length_error"
    refusal = copy.deepcopy(record)
    refusal["response"]["choices"][0]["message"]["content"] = "I cannot choose an action in this scene."
    assert parse_receipt(refusal, config)["status"] == "model_refusal"
    filtered = copy.deepcopy(record)
    filtered["response"]["choices"][0]["finish_reason"] = "content_filter"
    assert parse_receipt(filtered, config)["status"] == "api_refusal"
    invalid = copy.deepcopy(record)
    invalid["response"]["choices"][0]["message"]["content"] = json.dumps({"choice": True, "reasoning": "Text.", "setup_note": ""})
    assert parse_receipt(invalid, config)["status"] == "parse_error"
    values = {"a": [-1., .3], "b": [1., .5], "c": [.6, -.5]}
    baseline = uniform_baseline(values)
    assert all(abs(sum(vector[i] - baseline[i] for vector in values.values())) < 1e-12 for i in range(2))
    assert cluster_interval({"a": [0., 0.], "b": [0., 0.]}, {"a": "game1", "b": "game2"}, seed=1, replicates=100) == [(0., 0.), (0., 0.)]
    print("PASS: hidden labels excluded; only action block reordered; reverse IDs map to original choices; no provider fallback; refusals/length/malformed choices separated; uniform-null deltas and zero bootstrap control hold. No API calls. -- PI/Sol")


if __name__ == "__main__":
    main()
