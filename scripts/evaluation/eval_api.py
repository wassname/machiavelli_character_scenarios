"""Exact-request chat cache and one shared capped evaluation ledger scope. -- PI/Sol"""

import asyncio
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import httpx
from loguru import logger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/data"))
from billing_reservations import outstanding, unknown_fees, noncash_reviews

SCOPE = "action-choice-eval-v1"
RESERVATION = .01


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def transient_provider_error(body: dict) -> bool:
    if "error" not in body:
        return False
    code = body["error"]["code"]
    return code in {408, 429} or 500 <= code < 600


async def run_requests(packets: list[dict], out: Path, *, max_new_usd: float, concurrency: int = 8) -> dict:
    ledger = ROOT / ".local/billing.jsonl"
    history = [json.loads(line) for line in ledger.open()]
    project_before = sum(row["cost_usd"] for row in history)
    scope_before = sum(row["cost_usd"] for row in history if row.get("scope") == SCOPE)
    new = reserved = 0.0
    counts = Counter()
    stopping = False
    work = iter(packets)
    cache = ROOT / "data/interim/action_choice_cache"
    cache.mkdir(parents=True, exist_ok=True)
    auth = json.loads((Path.home() / ".pi/agent/auth.json").read_text())
    assert auth["openrouter"]["type"] == "api_key"
    headers = {"Authorization": "Bearer " + auth["openrouter"]["key"],
               "X-OpenRouter-Title": "machiavelli-character-scenarios/action-choice-eval"}
    async with httpx.AsyncClient(headers=headers, timeout=180) as client:
        with (out / "receipts.jsonl").open("w") as receipts:
            async def worker():
                nonlocal new, reserved
                for packet in work:
                    if stopping:
                        return
                    payload = packet["request"]
                    key = digest(payload)
                    path = cache / (key + ".json.gz")
                    cached = path.exists()
                    if cached:
                        record = json.loads(gzip.decompress(path.read_bytes()))
                        assert record["request"] == payload and record["request_key"] == key
                        if transient_provider_error(record["response"]):
                            cached = False
                            counts["cached_infrastructure_errors_retried"] += 1
                    if cached:
                        counts["cache_hits"] += 1
                    else:
                        assert project_before + new + reserved + outstanding(ROOT) + RESERVATION <= 58, "Cumulative evaluation cap $58 reached"
                        assert scope_before + new + reserved + outstanding(ROOT, SCOPE) + RESERVATION <= 2, "Evaluation scope cap $2 (including pilots/replays) reached"
                        assert new + reserved + RESERVATION <= max_new_usd, "This evaluation run's cap reached"
                        reserved += RESERVATION
                        try:
                            for attempt in range(4):
                                event = {"time": datetime.now(timezone.utc).isoformat(), "request_key": key,
                                         "metadata": packet["metadata"], "attempt": attempt, "request": payload}
                                counts["http_attempts"] += 1
                                try:
                                    response = await client.post("https://openrouter.ai/api/v1/chat/completions", json=payload)
                                except httpx.TransportError as error:
                                    event["error"] = repr(error)
                                    with (out / "events.jsonl").open("a") as stream:
                                        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                                    raise
                                event |= {"http_status": response.status_code, "response_text": response.text}
                                with (out / "events.jsonl").open("a") as stream:
                                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                                if (response.status_code in {408, 429} or 500 <= response.status_code < 600) and attempt < 3:
                                    await asyncio.sleep(2 ** attempt)
                                    continue
                                if response.status_code == 200 or response.headers.get("content-type", "").startswith("application/json"):
                                    body = response.json()
                                else:
                                    body = {"error": {"message": response.text}}
                                if "usage" in body and "cost" in body["usage"]:
                                    charge = body["usage"]["cost"]
                                    assert type(charge) in (int, float) and math.isfinite(charge) and charge >= 0
                                    new += charge
                                    with ledger.open("a") as stream:
                                        stream.write(json.dumps({"time": event["time"], "request_key": key,
                                            "response_id": body["id"], "cost_usd": charge, "model": body["model"],
                                            "run": str(out), "scope": SCOPE}) + "\n")
                                    assert charge <= RESERVATION, "Charge exceeded reservation; stopping before more work"
                                elif response.status_code == 200 and "error" not in body:
                                    raise ValueError("Successful chat response lacks an actual cost receipt")
                                provider_timeout = response.status_code == 200 and transient_provider_error(body)
                                if provider_timeout and attempt < 3:
                                    await asyncio.sleep(2 ** attempt)
                                    continue
                                record = {"request_key": key, "request": payload, "http_status": response.status_code, "response": body}
                                counts["new_requests"] += 1
                                if response.status_code == 200 and not provider_timeout:
                                    temporary = path.with_suffix(".tmp")
                                    temporary.write_bytes(gzip.compress(json.dumps(record, ensure_ascii=False).encode()))
                                    temporary.replace(path)
                                break
                        finally:
                            reserved -= RESERVATION
                    receipts.write(json.dumps(record | {"metadata": packet["metadata"], "cache_hit": cached}, ensure_ascii=False) + "\n")
                    receipts.flush()
                    counts["completed"] += 1
                    if counts["completed"] == 1:
                        logger.debug("## First complete request/reply\n\n```json\n" + json.dumps(record, ensure_ascii=False, indent=2) + "\n```\n")
                    if counts["completed"] % 100 == 0:
                        logger.info(f"Completed {counts['completed']}/{len(packets)} requests; new cost ${new:.5f}; cumulative ${project_before + new:.5f}.")
            tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                stopping = True
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
    assert counts["completed"] == len(packets)
    return dict(counts) | {"new_cost_usd": new, "project_total_usd": project_before + new,
                           "evaluation_scope_usd": scope_before + new, "unbilled_cost_reserved_usd": unknown_fees(ROOT),
                           "noncash_review_estimate_usd": noncash_reviews(ROOT),
                           "conservative_budget_total_usd": project_before + new + outstanding(ROOT), "scope": SCOPE}
