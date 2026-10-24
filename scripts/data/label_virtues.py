"""Budgeted five-level virtue labels, or an unpaid request/cost preflight. -- PI/Sol"""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
import traceback

import httpx
from dotenv import load_dotenv
from loguru import logger

from virtue_questions import ROOT, RUBRIC_PATH, choices, derive_contrary, identity, load_rubric, parse_answers, request
from billing_reservations import outstanding, unknown_fees, noncash_reviews

RESERVATION_USD = .01
TARIFF = .042 / 1_000_000


def select_rows(args: argparse.Namespace) -> list[dict]:
    rows = [json.loads(line) for line in args.inputs.open()]
    assert len({row["source_id"] for row in rows}) == len(rows)
    if args.limit is not None:
        assert 0 < args.limit <= len(rows)
        rng = random.Random(args.seed)
        groups = {}
        for row in sorted(rows, key=lambda row: row["source_id"]):
            groups.setdefault(row[args.group_field], []).append(row)
        for group in groups.values():
            if args.prefer_contrast:
                group.sort(key=lambda row: (-row["selection_score"], row["source_id"]))
            else:
                rng.shuffle(group)
        games = sorted(groups)
        rng.shuffle(games)
        ordered = [groups[game][depth] for depth in range(max(map(len, groups.values())))
                   for game in games if depth < len(groups[game])]
        included = set()
        if args.include_cases is not None:
            included = {row["source_id"] for row in json.loads(args.include_cases.read_text())}
        forced = [row for row in rows if row["source_id"] in included]
        assert len(forced) == len(included) and len(forced) <= args.limit
        rows = forced + [row for row in ordered if row["source_id"] not in included][:args.limit - len(forced)]
    return rows


def packets(rubric: dict, items: list[dict], actions: list[dict], batch_size: int, *, request_fn=request):
    full_owner = items[0]["item_id"]
    for item in items:
        for start in range(0, len(actions), batch_size):
            batch = actions[start:start + batch_size]
            full = item["item_id"] == full_owner
            yield item, batch, full, request_fn(rubric, item, batch, full_feedback=full)


def preflight(args: argparse.Namespace, rubric: dict, items: list[dict], actions: list[dict], *, request_fn=request) -> dict:
    characters = full_characters = count = 0
    example = None
    for item, batch, full, payload in packets(rubric, items, actions, args.batch_size, request_fn=request_fn):
        size = len(json.dumps(payload, ensure_ascii=False))
        characters += size
        full_characters += size if full else len(json.dumps(request_fn(rubric, item, batch, full_feedback=True), ensure_ascii=False))
        if example is None:
            path = args.out / "example_request.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            example = {"path": str(path), "item_id": item["item_id"], "questions": len(batch),
                       "serialized_characters": size, "full_feedback": full, "request_key": identity(payload)}
        count += 1
    ratios = rubric["pricing_observation"]["token_character_ratios"]
    estimate = {"requests": count, "choices": len(actions), "direct_items": len(items),
                "direct_cells": len(actions) * len(items), "batch_size": args.batch_size,
                "serialized_characters": characters, "estimated_input_tokens": {k: math.ceil(characters * r) for k, r in ratios.items()},
                "estimated_usd": {k: characters * r * TARIFF for k, r in ratios.items()},
                "full_feedback_everywhere_estimated_usd": full_characters * ratios["central"] * TARIFF,
                "tariff_usd_per_million_input_tokens": TARIFF * 1_000_000,
                "token_count_method": "Estimate using observed tiny-virtues native five-level input-token/serialized-character ratios; not exact tokenization or a quote.",
                "ratio_observation": rubric["pricing_observation"], "cache_discount_assumed": False,
                "feedback_policy": "Full scored question copy for first fixed-order virtue only; ID, agent and selected act elsewhere.",
                "full_feedback_owner": items[0]["item_id"],
                "example_request": example, "api_calls": 0}
    example["estimated_input_tokens"] = {k: math.ceil(example["serialized_characters"] * r) for k, r in ratios.items()}
    (args.out / "preflight.json").write_text(json.dumps(estimate, indent=2) + "\n")
    return estimate


async def paid(args: argparse.Namespace, rubric: dict, items: list[dict], actions: list[dict], *, request_fn=request,
               parse_fn=parse_answers, cache_name="virtues_v2_cache") -> dict:
    ledger = ROOT / ".local/billing.jsonl"
    initial = sum(row["cost_usd"] for row in map(json.loads, ledger.read_text().splitlines()))
    spent = reserved = 0.0
    count = hits = cells = 0
    feedback = Counter()
    full_feedback = Counter()
    stopping = False
    contrary = {pair["virtue_id"]: pair["vice_id"] for pair in rubric["contraries"]}
    cache = ROOT / "data/interim" / cache_name
    cache.mkdir(parents=True, exist_ok=True)
    if args.key_source == "pi":
        auth = json.loads((Path.home() / ".pi/agent/auth.json").read_text())
        assert auth["openrouter"]["type"] == "api_key"
        key = auth["openrouter"]["key"]
    else:
        load_dotenv(ROOT / ".env")
        key = os.environ["OPENROUTER_API_KEY"]
    headers = {"Authorization": "Bearer " + key,
               "X-OpenRouter-Title": "machiavelli-character-scenarios/virtues-v2"}
    work = packets(rubric, items, actions, args.batch_size, request_fn=request_fn)
    async with httpx.AsyncClient(headers=headers, timeout=180) as client:
        with (args.out / "labels.jsonl").open("w") as labels:
            async def worker():
                nonlocal spent, reserved, count, hits, cells
                for item, batch, full, payload in work:
                    if stopping:
                        return
                    await one(item, batch, full, payload)

            async def one(item, batch, full, payload):
                nonlocal spent, reserved, count, hits, cells
                key = identity(payload)
                path = cache / (key + ".json.gz")
                cached = path.exists()
                if cached:
                    record = json.loads(gzip.decompress(path.read_bytes()))
                    assert record["request"] == payload
                    body = record["response"]
                    hits += 1
                else:
                    assert initial + spent + reserved + outstanding(ROOT) + RESERVATION_USD <= args.budget_usd, "Cumulative budget reservation reached"
                    assert spent + reserved + RESERVATION_USD <= args.max_new_usd, "This run's budget reservation reached"
                    reserved += RESERVATION_USD
                    try:
                        for attempt in range(4):
                            event = {"time": datetime.now(timezone.utc).isoformat(), "request_key": key,
                                     "item_id": item["item_id"], "attempt": attempt,
                                     "choices": [[row["source_id"], row["choice_id"]] for row in batch], "request": payload}
                            try:
                                response = await client.post("https://openrouter.ai/api/alpha/decisions", json=payload)
                            except httpx.TransportError as error:
                                event["error"] = repr(error)
                                with (args.out / "requests.jsonl").open("a") as stream:
                                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                                raise
                            event |= {"http_status": response.status_code, "response_text": response.text}
                            with (args.out / "requests.jsonl").open("a") as stream:
                                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                            if (response.status_code in {408, 429} or 500 <= response.status_code < 600) and attempt < 3:
                                logger.warning(f"Transient HTTP {response.status_code}; retry {attempt + 1}/3 for unchanged request {key}.")
                                await asyncio.sleep(2 ** attempt)
                                continue
                            response.raise_for_status()
                            body = response.json()
                            charge = body["usage"]["cost"]
                            assert type(charge) in (float, int) and math.isfinite(charge) and charge >= 0
                            spent += charge
                            with ledger.open("a") as stream:
                                stream.write(json.dumps({"time": event["time"], "request_key": key, "response_id": body["id"],
                                                         "cost_usd": charge, "model": body["model"], "run": str(args.out)}) + "\n")
                            assert charge <= RESERVATION_USD, "Actual charge exceeded request reservation; stop before another request"
                            count += 1
                            break
                    finally:
                        reserved -= RESERVATION_USD
                assert body["model"] == rubric["served_model"], body["model"]
                assert body["provider"] == "TypeSafe", body["provider"]
                scores = parse_fn(body, payload, item, batch)
                if not cached:
                    record = {"request_key": key, "request": payload, "response": body}
                    temp = path.with_suffix(".tmp")
                    temp.write_bytes(gzip.compress(json.dumps(record, ensure_ascii=False).encode()))
                    temp.replace(path)
                if cells == 0:
                    (args.out / "example_receipt.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
                    logger.debug("## First complete receipt\n\n```json\n" + json.dumps(record, ensure_ascii=False, indent=2) + "\n```\n")
                word = body["answers"]["debug_word"]["choice"]
                feedback[word] += 1
                if full:
                    full_feedback[word] += 1
                for score in scores:
                    result = score | {"request_key": key, "rubric_version": rubric["version"], "served_model": body["model"],
                                      "cache_hit": cached, "judge_feedback": body["answers"]["debug_word"],
                                      "feedback_mode": "full" if full else "summary"}
                    labels.write(json.dumps(result, ensure_ascii=False) + "\n")
                    cells += 1
                    if item["item_id"] in contrary:
                        derived = derive_contrary(score, contrary[item["item_id"]]) | {
                            "request_key": key, "rubric_version": rubric["version"], "served_model": body["model"]}
                        labels.write(json.dumps(derived, ensure_ascii=False) + "\n")
                labels.flush()
                if (count + hits) % 250 == 0:
                    logger.info(f"Completed {count + hits} requests, {cells} direct labels; run cost ${spent:.4f}; project total ${initial + spent:.4f}.")

            tasks = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                stopping = True
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
    assert cells == len(actions) * len(items), "Incomplete direct-label coverage"
    return {"new_requests": count, "cache_hits": hits, "direct_cells": cells, "new_cost_usd": spent,
            "project_total_usd": initial + spent, "unbilled_cost_reserved_usd": unknown_fees(ROOT),
            "noncash_review_estimate_usd": noncash_reviews(ROOT),
            "conservative_budget_total_usd": initial + spent + outstanding(ROOT), "feedback_counts": dict(feedback),
            "full_copy_feedback_counts": dict(full_feedback)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=ROOT / "outputs/20261018_rendered_inputs_v3/inputs.jsonl")
    parser.add_argument("--rubric", type=Path, default=RUBRIC_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--group-field", default="game_id", help="Sampling group; use topic_group for DailyDilemmas")
    parser.add_argument("--prefer-contrast", action="store_true", help="Within each game, rank scenes by existing selection_score instead of shuffling")
    parser.add_argument("--include-cases", type=Path)
    parser.add_argument("--item-id", action="append", help="Restrict direct items; repeat for several")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--allow-paid", action="store_true", help="Omit for unpaid preflight")
    parser.add_argument("--key-source", choices=["env", "pi"], default="env")
    parser.add_argument("--budget-usd", type=float, default=70)
    parser.add_argument("--max-new-usd", type=float, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.batch_size <= 40 and 0 < args.budget_usd <= 70 and args.max_new_usd > 0 and args.concurrency > 0
    assert args.include_cases is None or args.limit is not None
    args.out.mkdir(parents=True, exist_ok=False)
    run_log = args.out / "run.md"
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{message}\n", colorize=False)
    logger.add(run_log, level="DEBUG", format="{message}\n", colorize=False)
    started = time.monotonic()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    logger.info("# Proposed-choice virtue labels\n\n-- PI/Sol\n\n## Config\n\n```json\n" + json.dumps(config, indent=2) + "\n```\n\nCommand: `" + shlex.join(sys.argv) + "`\n")
    try:
        rubric = load_rubric(args.rubric)
        rows = select_rows(args)
        with (args.out / "selection.jsonl").open("w") as stream:
            for row in rows:
                selection = {"source_id": row["source_id"], "source": row["source"],
                    "selection_group": row[args.group_field], "group_field": args.group_field,
                    "selection_priority": row["selection_score"] if args.prefer_contrast else None,
                    "selection_rule": "balanced groups; descending existing selection_score within each group" if args.prefer_contrast else "seeded balanced groups",
                    "choices": len(row["choice_text"])}
                if "game_id" in row:
                    selection["game_id"] = row["game_id"]
                stream.write(json.dumps(selection, ensure_ascii=False) + "\n")
        actions = choices(rows)
        items = rubric["items"]
        if args.item_id is not None:
            assert set(args.item_id) <= {item["item_id"] for item in items}
            items = [item for item in items if item["item_id"] in args.item_id]
        metadata = {"config": config, "input_sha256": identity({"rows": rows}), "rubric_sha256": identity(rubric),
                    "tiny_virtues_commit": rubric["source"]["commit"],
                    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    "source_ids": [row["source_id"] for row in rows], "choices": len(actions)}
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        logger.info(f"## Input\n\n{len(rows)} scenes, {len(actions)} choices, {len(items)} direct teachings. Exact visible prompts and selected proposed choices only. Existing labels remain unchanged.\n\nSHOULD: every choice receives a full diagnostic copy once, under the first fixed-order virtue; all other diagnostics contain only IDs, agents and proposed acts.\n")
        if not args.allow_paid:
            result = preflight(args, rubric, items, actions)
        else:
            (ROOT / ".local").mkdir(exist_ok=True)
            with (ROOT / ".local/labelling.lock").open("w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = asyncio.run(paid(args, rubric, items, actions))
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        logger.info("## Result\n\n```json\n" + json.dumps(result, indent=2) + f"\n```\n\nElapsed: {time.monotonic() - started:.1f}s.\n\n{run_log}\n")
    except Exception:
        logger.error("## Failed\n\n```text\n" + traceback.format_exc() + f"\n```\n\nRaw replies and incurred charges remain saved. No missing judgement becomes a score.\n\n{run_log}\n")
        raise


if __name__ == "__main__":
    if os.environ.get("BEARTYPE"):
        from beartype import beartype
        request = beartype(request)
        parse_answers = beartype(parse_answers)
    main()
