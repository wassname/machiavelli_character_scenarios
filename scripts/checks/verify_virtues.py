"""Rebuild native virtue judgements and verify their choice-level parquet and receipts. -- PI/Sol"""

import argparse
from collections import Counter
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from label_virtues import packets
from virtue_questions import ROOT, choices, identity, load_rubric, parse_answers, visible_identity


def verify(run: Path) -> dict:
    metadata = json.loads((run / "metadata.json").read_text())
    config = metadata["config"]
    rubric = load_rubric(Path(config["rubric"]))
    assert identity(rubric) == metadata["rubric_sha256"], "Frozen teaching changed"
    selected = metadata["source_ids"]
    wanted = set(selected)
    sources = {row["source_id"]: row for row in map(json.loads, Path(config["inputs"]).open()) if row["source_id"] in wanted}
    assert set(sources) == wanted and len(selected) == len(wanted)
    rows = [sources[key] for key in selected]
    assert identity({"rows": rows}) == metadata["input_sha256"], "Source inputs changed"
    actions = choices(rows)
    exported = {(row["source_id"], row["choice_id"]): row for row in pq.read_table(run / "virtues.parquet").to_pylist()}
    assert set(exported) == {(row["source_id"], row["choice_id"]) for row in actions}
    items = rubric["items"]
    assert config["item_id"] is None, "Verify the complete 80-item overlay, not an item subset"
    contraries = {pair["virtue_id"]: pair["vice_id"] for pair in rubric["contraries"]}
    expected_items = {item["item_id"] for item in items} | set(contraries.values())
    assert all(row["virtues"].keys() == expected_items for row in exported.values())
    costs, feedback, usage = {}, {"full": Counter(), "summary": Counter()}, Counter()
    direct = derived = 0
    for item, batch, full, payload in packets(rubric, items, actions, config["batch_size"]):
        key = identity(payload)
        path = ROOT / "data/interim/virtues_v2_cache" / (key + ".json.gz")
        native = json.loads(gzip.decompress(path.read_bytes()))
        assert native["request_key"] == key and native["request"] == payload
        body = native["response"]
        assert body["model"] == rubric["served_model"] and body["provider"] == "TypeSafe"
        assert body["id"] not in costs, "Duplicate native response"
        costs[body["id"]] = (key, Decimal(str(body["usage"]["cost"])))
        usage["input_tokens"] += body["usage"]["input_tokens"]
        usage["output_tokens"] += body["usage"]["output_tokens"]
        feedback["full" if full else "summary"][body["answers"]["debug_word"]["choice"]] += 1
        for score in parse_answers(body, payload, item, batch):
            actual = exported[score["source_id"], score["choice_id"]]["virtues"][score["item_id"]]
            vector = [score["raw_probabilities"][str(i)] for i in range(5)]
            assert actual == {"signed": score["signed"], "raw_probabilities": vector, "derived_from": None}
            direct += 1
            if score["item_id"] in contraries:
                contrary = exported[score["source_id"], score["choice_id"]]["virtues"][contraries[score["item_id"]]]
                assert contrary == {"signed": score["signed"], "raw_probabilities": vector, "derived_from": score["item_id"]}
                derived += 1
    assert direct == len(actions) * len(items) and derived == len(actions) * len(contraries)
    ledger = [json.loads(line) for line in (ROOT / ".local/billing.jsonl").open()]
    booked = [entry for entry in ledger if entry.get("response_id") in costs]
    assert len(booked) == len(costs)
    for entry in booked:
        assert costs[entry["response_id"]] == (entry["request_key"], Decimal(str(entry["cost_usd"])))
    total = sum((cost for _, cost in costs.values()), Decimal(0))
    assert total == sum((Decimal(str(entry["cost_usd"])) for entry in booked), Decimal(0))
    export_check = json.loads((run / "export_verification.json").read_text())
    assert export_check["sha256"] == hashlib.sha256((run / "virtues.parquet").read_bytes()).hexdigest()
    result = {"status": "PASS", "author": "PI/Sol", "scenes": len(rows),
              "games": len({row["game_id"] for row in rows if "game_id" in row}),
              "topic_groups": len({row["topic_group"] for row in rows if "topic_group" in row}),
              "source_counts": dict(Counter(row["source"] for row in rows)), "choices": len(actions),
              "direct_cells": direct, "derived_cells": derived, "unique_successful_requests": len(costs),
              "subset_cost_usd": str(total), "usage": dict(usage),
              "project_ledger_usd": str(sum((Decimal(str(entry["cost_usd"])) for entry in ledger), Decimal(0))),
              "feedback_by_mode": {key: dict(value) for key, value in feedback.items()},
              "parquet_sha256": export_check["sha256"], "frozen_rubric_sha256": identity(rubric),
              "visible_inputs_sha256": visible_identity(rows),
              "evidence": "Every exact request rebuilt from frozen teachings and visible source inputs; every raw scored reply and derived cell matched to parquet; every actual charge matched by response ID and request key.",
              "limits": "Native artifact integrity, not human semantic accuracy or probability calibration. Full-copy none is not validation; summary-only diagnostics omit the story by design."}
    (run / "native_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run), indent=2))
