"""Price and label only the two rubric-based numerical D&D axes. -- PI/OpenAI"""

import argparse
import asyncio
from collections import Counter
import fcntl
import hashlib
import gzip
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys

from loguru import logger
import pyarrow as pa
import pyarrow.parquet as pq

from alignment_questions import ROOT, RUBRIC_PATH, load_rubric, request, parse_answers
from label_virtues import paid, preflight, packets
from virtue_questions import choices, visible_identity, identity


def verify(run: Path, rows: list[dict], rubric: dict, items: list[dict], actions: list[dict], batch_size: int) -> dict:
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["visible_inputs_sha256"] == visible_identity(rows)
    assert metadata["rubric_sha256"] == hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest()
    saved = [json.loads(line) for line in (run / "labels.jsonl").open()]
    indexed = {(row["source_id"], row["choice_id"], row["axis"]): row for row in saved}
    assert len(indexed) == len(saved) == len(actions) * 2
    ledger = {row["response_id"]: row for row in map(json.loads, (ROOT / ".local/billing.jsonl").read_text().splitlines()) if "response_id" in row}
    verified, receipts = set(), set()
    for item, batch, full, payload in packets(rubric, items, actions, batch_size, request_fn=request):
        key = identity(payload)
        record = json.loads(gzip.decompress((ROOT / "data/interim/dnd_alignment_v1_cache" / (key + ".json.gz")).read_bytes()))
        assert record["request"] == payload and record["request_key"] == key
        body = record["response"]
        assert body["model"] == "typesafe/jev-1.13-20260917" and body["provider"] == "TypeSafe"
        assert ledger[body["id"]]["cost_usd"] == body["usage"]["cost"]
        receipts.add(body["id"])
        for observed in parse_answers(body, payload, item, batch):
            index = observed["source_id"], observed["choice_id"], observed["axis"]
            actual = indexed[index]
            assert actual["request_key"] == key
            assert all(actual[name] == value for name, value in observed.items())
            verified.add(index)
    assert verified == indexed.keys()
    exported = [{"source_id": action["source_id"], "choice_id": action["choice_id"], "dnd_action_alignment": {
        axis: indexed[action["source_id"], action["choice_id"], axis] for axis in ("order", "morality")}} for action in actions]
    target = run / "alignment.parquet"
    table = pa.Table.from_pylist(exported)
    pq.write_table(table, target, compression="zstd")
    assert pq.read_table(target).equals(table)
    proof = {"status": "PASS", "author": "PI/OpenAI", "scenes": len(rows), "choices": len(actions), "direct_cells": len(saved),
        "native_receipts": len(receipts), "source_counts": dict(Counter(row["source"] for row in rows)),
        "every_label_reconstructed_from_native_response": True, "every_receipt_matches_ledger": True,
        "visible_inputs_sha256": visible_identity(rows), "rubric_sha256": metadata["rubric_sha256"],
        "rubric_version": rubric["version"], "served_model": rubric["served_model"],
        "null_axis_scores": dict(Counter(row["axis"] for row in saved if row["score"] is None)),
        "parquet_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "validation_scope": "Source, probability, numerical mapping and receipt integrity; not human accuracy or calibration."}
    (run / "verification.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/processed/parquet/train.parquet")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--allow-paid", action="store_true")
    parser.add_argument("--key-source", choices=["pi", "env"], default="env")
    parser.add_argument("--budget-usd", type=float, default=70)
    parser.add_argument("--max-new-usd", type=float, default=4)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert 0 < args.budget_usd <= 70 and 0 < args.max_new_usd <= 5 and 1 <= args.batch_size <= 40
    args.out.mkdir(parents=True, exist_ok=False)
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")
    logger.add(args.out / "run.md", format="{message}", level="DEBUG")
    logger.info("# Graded D&D axes\n\n-- PI/OpenAI\n\nCommand: `" + shlex.join(sys.argv) + "`\n")
    columns = ["source_id", "source", "player_character", "text", "choice_text", "game_id", "topic_group"]
    rows = pq.read_table(args.inputs, columns=columns).to_pylist()
    for row in rows:
        row["choice_text"] = json.loads(row["choice_text"])
    if args.limit is not None:
        assert args.limit > 0 and args.limit % 2 == 0
        rng = random.Random(20261024)
        rows = [row for source in ("machiavelli", "daily_dilemmas")
                for row in rng.sample([r for r in rows if r["source"] == source], args.limit // 2)]
    if args.fixtures is not None:
        for fixture in json.loads(args.fixtures.read_text())["fixtures"]:
            rows.append({"source_id": "fixture_" + fixture["id"], "source": "fixture", "player_character": "the player",
                "text": fixture["scene"], "choice_text": {"0": fixture["action"]}, "game_id": None, "topic_group": None})
    assert len({r["source_id"] for r in rows}) == len(rows)
    (args.out / "inputs.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    rubric = load_rubric()
    rubric.update(model="typesafe/jev-1.13", served_model="typesafe/jev-1.13-20260917", contraries=[])
    rubric["pricing_observation"] = json.loads(args.calibration.read_text()) if args.calibration else {
        "token_character_ratios": {"low": .25892706833018464, "central": .26677644985521454, "high": .31720105500651435},
        "method": "Initial five-level same-family reference, not a quote. Recalibrate on this protocol's pilot before full run."}
    items = [item | {"item_id": item["id"]} for item in rubric["items"]]
    actions = choices(rows)
    metadata = {"rubric_sha256": hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest(),
        "visible_inputs_sha256": visible_identity(rows), "rubric_version": rubric["version"],
        "inputs": str(args.inputs), "choices": len(actions), "scenes": len(rows),
        "source_counts": dict(Counter(r["source"] for r in rows)),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
            ("scripts/data/label_alignment.py", "scripts/data/alignment_questions.py", "scripts/data/label_virtues.py")},
        "method": "11 rubric grades plus unclear via Choice because Score accepts at most10levels; numerical expected grade, not a projection of old categories. Per-action grade order is deterministically randomized. No within-action counterbalancing."}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    estimate = preflight(args, rubric, items, actions, request_fn=request)
    estimate["feedback_policy"] = "Only a deterministic half-sample has full visible context copied into first-axis diagnostics; other actions are not inspected by that feedback. Every graded question still has its exact full story."
    estimate["token_count_method"] = rubric["pricing_observation"]["method"]
    (args.out / "preflight.json").write_text(json.dumps(estimate, indent=2) + "\n")
    logger.info("## Preflight\n\n" + json.dumps(estimate, indent=2))
    if args.allow_paid:
        with (ROOT / ".local/labelling.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = asyncio.run(paid(args, rubric, items, actions, request_fn=request,
                parse_fn=parse_answers, cache_name="dnd_alignment_v1_cache"))
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        proof = verify(args.out, rows, rubric, items, actions, args.batch_size)
        logger.info("## Result\n\n" + json.dumps(result, indent=2) + "\n\n" + json.dumps(proof, indent=2))
    logger.info(str(args.out / "run.md"))


if __name__ == "__main__":
    main()
