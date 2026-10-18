"""Relabel visible choices with a resumable, budgeted decisions-API pass. PI/OpenAI."""

import argparse
import asyncio
import fcntl
import gzip
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from dotenv import load_dotenv
from loguru import logger
from tabulate import tabulate

from dataset_io import PROVENANCE, ROOT, input_rows, select_rows, write_jsonl
from judge_questions import MODEL, RUBRIC, identity, interpret, request

RESERVATION_USD = 0.10


def validate(body: dict, payload: dict) -> None:
    assert isinstance(body["model"], str) and body["model"]
    assert math.isfinite(body["usage"]["cost"]) and 0 <= body["usage"]["cost"] <= RESERVATION_USD
    assert set(body["answers"]) == set(payload["questions"])
    for key, question in payload["questions"].items():
        answer = body["answers"][key]
        assert answer["type"] == question["type"], (key, answer)
        if question["type"] == "noul":
            assert math.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1, (key, answer)
        else:
            probabilities = answer["probabilities"]
            assert set(probabilities) == set(question["criteria"]), key
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities.values()), key
            assert abs(sum(probabilities.values()) - 1) <= min(0.05, len(probabilities) * 0.005) + 1e-8, key
            assert answer["choice"] in probabilities, key


def api_key(source: str) -> str:
    if source == "pi":
        auth = json.loads((Path.home() / ".pi/agent/auth.json").read_text())
        assert auth["openrouter"]["type"] == "api_key"
        return auth["openrouter"]["key"]
    load_dotenv(ROOT / ".env")
    return os.environ["OPENROUTER_API_KEY"]


async def label(args: argparse.Namespace, rows: list[dict], run_log: Path) -> None:
    ledger = ROOT / ".local" / "billing.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    historical = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    spent = sum(row["cost_usd"] for row in historical)
    initial_spend = spent
    reserved = 0.0
    semaphore = asyncio.Semaphore(args.concurrency)
    request_locks = {}
    cache = ROOT / "data" / "interim" / "decision_cache"
    cache.mkdir(parents=True, exist_ok=True)
    raw_log = args.out / "requests.jsonl"
    answers_log = args.out / "labels.jsonl"
    actions = [(row, key) for row in rows for key, text in row["choice_text"].items() if text]
    done = []
    first_trace = False
    paid_requests = 0
    cache_hits = 0
    stopping = False
    key = api_key(args.key_source)
    headers = {"Authorization": "Bearer " + key, "HTTP-Referer": "https://huggingface.co/datasets/wassname/machiavelli_character_scenarios", "X-OpenRouter-Title": "machiavelli-character-scenarios/visible-actions"}

    async with httpx.AsyncClient(headers=headers, timeout=120) as client:
        async def one(row: dict, choice: str) -> None:
            nonlocal spent, reserved, first_trace, paid_requests, cache_hits
            async with semaphore, request_locks.setdefault((row["text"], row["choice_text"][choice]), asyncio.Lock()):
                if stopping:
                    return
                payload = request(row, choice, args.rubrics, virtues=not args.no_virtues)
                request_key = identity(payload)
                path = cache / (request_key + ".json.gz")
                if path.exists():
                    record = json.loads(gzip.decompress(path.read_bytes()))
                    assert record["request"] == payload
                    validate(record["response"], payload)
                    cache_hits += 1
                    cache_hit = True
                else:
                    assert spent + reserved + RESERVATION_USD <= args.budget_usd, f"Project budget reservation reached: spent=${spent:.5f}, reserved=${reserved:.2f}, cap=${args.budget_usd:.2f}; cached work retained"
                    reserved += RESERVATION_USD
                    try:
                        for attempt in range(4):
                            event = {"time": datetime.now(timezone.utc).isoformat(), "request_key": request_key, "source_id": row["source_id"], "choice_id": choice, "attempt": attempt, "request": payload}
                            try:
                                response = await client.post("https://openrouter.ai/api/alpha/decisions", json=payload)
                            except httpx.TransportError as error:
                                event["error"] = repr(error)
                                with raw_log.open("a") as stream:
                                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                                if attempt == 3:
                                    raise
                                await asyncio.sleep(2 ** attempt)
                                continue
                            event |= {"http_status": response.status_code, "response_text": response.text}
                            with raw_log.open("a") as stream:
                                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                            if response.status_code in {408, 429, 499, 500, 502, 503, 504}:
                                if attempt == 3:
                                    response.raise_for_status()
                                await asyncio.sleep(2 ** attempt)
                                continue
                            response.raise_for_status()
                            body = response.json()
                            charged = body["usage"]["cost"]
                            assert math.isfinite(charged) and charged >= 0
                            spent += charged
                            paid_requests += 1
                            with ledger.open("a") as stream:
                                stream.write(json.dumps({"time": event["time"], "request_key": request_key, "cost_usd": charged, "model": body["model"], "run": str(args.out)}) + "\n")
                            validate(body, payload)
                            record = {"request_key": request_key, "request": payload, "response": body}
                            temporary = path.with_suffix(".tmp")
                            temporary.write_bytes(gzip.compress(json.dumps(record, ensure_ascii=False).encode()))
                            temporary.replace(path)
                            cache_hit = False
                            break
                        else:
                            raise RuntimeError("No successful decisions response after retries")
                    finally:
                        reserved -= RESERVATION_USD
                if not first_trace:
                    logger.debug("## Labels -> first full real request and response\n\n```json\n" + json.dumps(record, ensure_ascii=False, indent=2) + f"\n```\n\nRaw events: `{raw_log}:1`. Cache: `{path.relative_to(ROOT)}`. Cache hit: {cache_hit}.")
                    first_trace = True
                result = {"source_id": row["source_id"], "choice_id": choice, "action": row["choice_text"][choice], "request_key": request_key, "rubric_version": RUBRIC["version"], "served_model": record["response"]["model"], "cache_hit": cache_hit, "usage": record["response"]["usage"], **interpret(record["response"]["answers"])}
                with answers_log.open("a") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                done.append(result)
                if len(done) % 250 == 0:
                    logger.info(f"Labelled {len(done)}/{len(actions)} actions; new spend ${spent - initial_spend:.4f}; project total ${spent:.4f}.")

        tasks = [asyncio.create_task(one(row, choice)) for row, choice in actions]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            stopping = True
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    assert len(done) == len(actions)
    grouped = {}
    for row in done:
        grouped.setdefault(row["source_id"], {})[row["choice_id"]] = row
    enriched = [{**row, "choice_labels_visible": grouped[row["source_id"]]} for row in rows]
    write_jsonl(args.out / "enriched.jsonl", enriched)
    from summarise_machiavelli import _write_parquet
    parquet = args.out / "parquet" / "train.parquet"
    _write_parquet(parquet, enriched)
    exported = pq.read_table(parquet, columns=["source_id", "text", "choice_labels"]).to_pylist()
    assert [row["source_id"] for row in exported] == [row["source_id"] for row in rows]
    assert [row["text"] for row in exported] == [row["text"] for row in rows]
    assert [json.loads(row["choice_labels"]) for row in exported] == [row["choice_labels"] for row in rows]
    feedback = Counter(row["judge_feedback"]["choice"] for row in done)
    summary = {"rows": len(rows), "actions": len(actions), "complete_actions": len(done), "new_api_requests": paid_requests, "cache_hits": cache_hits, "charged_usd": spent - initial_spend, "project_total_usd": spent, "budget_usd": args.budget_usd, "feedback_counts": dict(feedback), "rubric_version": RUBRIC["version"], "model": MODEL, "original_labels_sent_to_judge": False, "secret_action_outcomes_sent_to_judge": False, "quality_claim": "Uncalibrated model estimates. Review diagnostic feedback and source examples before interpreting them."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = tabulate([[word, count] for word, count in feedback.most_common()], headers=["unscored feedback", "actions"], tablefmt="pipe")
    logger.info(f"## Labels -> completed\n\nEvery proposed action was labelled from the visible prompt. Original labels remain unchanged in enriched.jsonl; any input repairs and original prompts are retained in separate fields. A low probability can mean little evidence; use the separate context issues rather than interpreting it as proof of absence.\n\n## Result\n\nCoverage: {len(done)}/{len(actions)} actions from {len(rows)} scenarios. Charged this run: US${spent - initial_spend:.6f}. Project inference spend: US${spent:.6f} / ${args.budget_usd:.2f}.\n\n{table}\n\nLabels: `{answers_log}`. Full HTTP traces: `{raw_log}`. Original prompts plus both label sets: `{args.out / 'enriched.jsonl'}`.\n\n{run_log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Game-balanced scenario sample size; omit to label all published scenarios")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--inputs", type=Path, help="Versioned inputs.jsonl from prepare_inputs.py; omit for original published prompts")
    parser.add_argument("--budget-usd", type=float, default=40, help="Cumulative project inference cap, including all earlier paid pilots")
    parser.add_argument("--key-source", choices=["env", "pi"], default="env")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rubrics", type=Path, default=Path("data/raw/virtue_rubrics"))
    parser.add_argument("--no-virtues", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_labels"))
    args = parser.parse_args()
    assert 0 < args.budget_usd <= 40 and args.concurrency > 0
    args.out.mkdir(parents=True, exist_ok=False)
    run_log = args.out / "run.md"
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{message}\n", colorize=False)
    logger.add(run_log, level="DEBUG", format="{message}\n", colorize=False)
    started = time.monotonic()
    config = vars(args) | {"out": str(args.out), "rubrics": str(args.rubrics), "inputs": str(args.inputs) if args.inputs else None, "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest() if args.inputs else None, "model": MODEL, "rubric_version": RUBRIC["version"]}
    logger.info("# Visible-action relabelling\n\nCommand:\n\n```text\n" + shlex.join(sys.argv) + "\n```\n\n## Resolved config\n\n```json\n" + json.dumps(config, indent=2) + "\n```")
    metadata = {"config": config, "provenance": PROVENANCE, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True))}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        rows = select_rows(input_rows(args.inputs), args.limit, args.seed)
        actions = sum(sum(bool(text) for text in row["choice_text"].values()) for row in rows)
        logger.info(f"## Labels -> input\n\n{len(rows)} scenarios; {actions} available actions. Only the exact supplied text and selected choice enter the judge's state. No source histories, original annotations, rewards or future outcomes are passed.")
        if args.dry_run:
            payload = request(rows[0], next(key for key, text in rows[0]["choice_text"].items() if text), args.rubrics, virtues=not args.no_virtues)
            (args.out / "example_request.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            logger.info(f"## Result\n\nDry run: {len(payload['questions'])} questions per action, {actions} actions. No API calls. Example exact request: `{args.out / 'example_request.json'}`.\n\n{run_log}")
            return
        (ROOT / ".local").mkdir(exist_ok=True)
        with (ROOT / ".local" / "labelling.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            asyncio.run(label(args, rows, run_log))
    except Exception:
        logger.error("## Relabelling failed\n\n```text\n" + traceback.format_exc() + f"\n```\n\nPaid responses and project billing remain saved. No missing judgement is filled with a score. Elapsed: {time.monotonic() - started:.1f}s.\n\n{run_log}")
        raise


if __name__ == "__main__":
    if os.environ.get("BEARTYPE"):
        from beartype import beartype
        validate = beartype(validate)
        request = beartype(request)
        interpret = beartype(interpret)
    main()
