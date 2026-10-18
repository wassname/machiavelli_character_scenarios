"""Verify saved labels, model inputs, source preservation and billing. PI/OpenAI."""

import argparse
import gzip
import hashlib
import json
import math
import sys
import traceback
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from loguru import logger
from tabulate import tabulate

from dataset_io import ROOT, input_rows, published_rows, select_rows, source_rows
from prepare_inputs import verify_inputs
from judge_questions import RUBRIC, TASK, identity, interpret, request
from label_actions import validate


def verify_request(payload: dict, parent: dict, choice: str, config: dict) -> None:
    expected = request(parent, choice, Path(config["rubrics"]), virtues=not config["no_virtues"])
    assert payload == expected, "Saved request differs from the complete configured questions or visible state"


def verify(run: Path) -> dict:
    metadata = json.loads((run / "metadata.json").read_text())
    config = metadata["config"]
    assert config["rubric_version"] == RUBRIC["version"], "Use the original rubric version to verify older runs"
    path = Path(config["inputs"]) if config.get("inputs") else None
    original = select_rows(input_rows(path), config["limit"], config["seed"])
    if path is not None:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config["inputs_sha256"]
        by_id = {row["source_id"]: row for row in published_rows()}
        published = [by_id[row["source_id"]] for row in original]
        verify_inputs(original, published, source_rows(published))
    source = {row["source_id"]: row for row in original}
    expected = {(row["source_id"], key) for row in original for key, text in row["choice_text"].items() if text}
    seen = set()
    label_hashes = {}
    cache_keys = set()
    words = Counter()
    raw_log = run / "requests.jsonl"
    receipt_offsets = {}
    api_errors = Counter()
    if raw_log.exists():
        with raw_log.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                event = json.loads(line)
                if "error" in event:
                    api_errors["transport"] += 1
                    continue
                if event["http_status"] != 200:
                    api_errors[str(event["http_status"])] += 1
                if event["http_status"] == 200:
                    pair = (event["source_id"], event["choice_id"], event["request_key"])
                    assert pair not in receipt_offsets, "Multiple successful receipts for one action"
                    receipt_offsets[pair] = offset
    for line in (run / "labels.jsonl").open():
        row = json.loads(line)
        pair = (row["source_id"], row["choice_id"])
        assert pair in expected and pair not in seen, pair
        seen.add(pair)
        label_hashes[pair] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        parent = source[row["source_id"]]
        assert row["action"] == parent["choice_text"][row["choice_id"]]
        if row["cache_hit"]:
            cache = ROOT / "data/interim/decision_cache" / (row["request_key"] + ".json.gz")
            record = json.loads(gzip.decompress(cache.read_bytes()))
            payload, body = record["request"], record["response"]
            assert record["request_key"] == row["request_key"]
        else:
            offset = receipt_offsets[(row["source_id"], row["choice_id"], row["request_key"])]
            with raw_log.open("rb") as stream:
                stream.seek(offset)
                event = json.loads(stream.readline())
            payload, body = event["request"], json.loads(event["response_text"])
        verify_request(payload, parent, row["choice_id"], config)
        assert set(row["visible_action_values"]) == set(RUBRIC["dimensions"])
        assert identity(payload) == row["request_key"]
        assert payload["state"] == {"task": TASK, "scenario_text": parent["text"], "chosen_action": row["action"]}, pair
        validate(body, payload)
        assert row["served_model"] == body["model"] and row["usage"] == body["usage"]
        for field, value in interpret(body["answers"]).items():
            assert row[field] == value, (pair, field)
        cache_keys.add(row["request_key"])
        words[row["judge_feedback"]["choice"]] += 1
    assert seen == expected, {"missing_actions": len(expected - seen)}
    enriched = [json.loads(line) for line in (run / "enriched.jsonl").open()]
    assert [row["source_id"] for row in enriched] == [row["source_id"] for row in original]
    for row, parent in zip(enriched, original):
        assert set(row) == set(parent) | {"choice_labels_visible"}
        assert {key: row[key] for key in parent} == parent, row["source_id"]
        assert set(row["choice_labels_visible"]) == {key for key, text in parent["choice_text"].items() if text}
        for choice, value in row["choice_labels_visible"].items():
            assert hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() == label_hashes[(row["source_id"], choice)]
    exported = pq.read_table(run / "parquet/train.parquet").to_pylist()
    assert [row["source_id"] for row in exported] == [row["source_id"] for row in original]
    for row, parent in zip(exported, original):
        assert set(row) == set(parent) | {"choice_labels_visible"}
        for key, value in parent.items():
            observed = json.loads(row[key]) if isinstance(value, (dict, list)) else row[key]
            assert observed == value, (parent["source_id"], key)
        visible = json.loads(row["choice_labels_visible"])
        assert set(visible) == {key for key, text in parent["choice_text"].items() if text}
        for choice, value in visible.items():
            assert hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() == label_hashes[(row["source_id"], choice)]
    ledger = [json.loads(line) for line in (ROOT / ".local/billing.jsonl").open()]
    project_spend = sum(row["cost_usd"] for row in ledger)
    run_spend = sum(row["cost_usd"] for row in ledger if (ROOT / row["run"]).resolve() == run.resolve())
    summary = json.loads((run / "summary.json").read_text())
    assert project_spend <= 40
    assert math.isclose(run_spend, summary["charged_usd"], abs_tol=1e-8)
    assert summary["complete_actions"] == len(expected) == summary["actions"]
    result = {"status": "PASS", "scenarios": len(original), "complete_actions": len(seen), "distinct_request_keys": len(cache_keys), "input_fields_preserved_in_exports": True, "original_fields_unchanged": path is None, "original_annotations_unchanged": True, "original_prompts_retained": True, "input_revision": original[0].get("input_revision", "original-published"), "only_visible_prompt_and_choice_sent": True, "every_saved_score_matches_raw_response": True, "complete_configured_questions_checked": True, "parquet_source_fields_preserved": True, "run_charged_usd": run_spend, "project_total_usd_at_verification": project_spend, "feedback_counts": dict(words), "raw_http_or_transport_errors": dict(api_errors), "caveat": "Artifact integrity and input isolation, not semantic accuracy or calibration."}
    (run / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run_log = args.run / "run.md"
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{message}\n", colorize=False)
    logger.add(run_log, level="DEBUG", format="{message}\n", colorize=False)
    logger.info("## Artifact verification\n\nChecking every expected action, cached model request, saved score, original field, export and project billing. No API calls.")
    try:
        result = verify(args.run)
    except Exception:
        logger.error("## Verification failed\n\n```text\n" + traceback.format_exc() + f"\n```\n\n{run_log}")
        raise
    logger.debug("## Verification details\n\n```json\n" + json.dumps(result, indent=2) + "\n```")
    table = tabulate([["Scenarios preserved", result["scenarios"]], ["Actions verified", result["complete_actions"]], ["Distinct raw requests checked", result["distinct_request_keys"]], ["Run charge, US$", f"{result['run_charged_usd']:.6f}"], ["Project charge / cap, US$", f"{result['project_total_usd_at_verification']:.6f} / 40"]], headers=["measure", "result"], tablefmt="pipe")
    logger.info(f"## Verified label output\n\n{table}\n\nEvery expected action is present. Input fields and original annotations are preserved; any input repairs were checked against the source observation. Each model state contains only the task, visible prompt and choice. Saved values match their raw responses. This verifies integrity, not semantic accuracy or calibration.\n\nMachine evidence: `{args.run / 'verification.json'}`. Labels: `{args.run / 'labels.jsonl'}`. Export: `{args.run / 'parquet/train.parquet'}`.\n\n{run_log}")


if __name__ == "__main__":
    main()
