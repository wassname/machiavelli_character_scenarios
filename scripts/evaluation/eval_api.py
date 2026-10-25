"""Exact-request chat cache and one shared capped evaluation ledger scope. -- PI/Sol"""

import asyncio
import base64
from collections import Counter
from datetime import datetime, timezone
import functools
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
from billing_reservations import outstanding, unknown_fees, noncash_reviews, PROJECT_CAP_USD, EVALUATION_CAP_USD, EVALUATION_SCOPE
from label_actions import api_key

SCOPE = EVALUATION_SCOPE
RESERVATION = .01


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def transient_provider_error(body: dict) -> bool:
    if "error" not in body:
        return False
    code = body["error"]["code"]
    return code in {408, 429} or 500 <= code < 600


def reserve_unknown_fee(event: dict, out: Path) -> float:
    path = ROOT / ".local/billing_reservations.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    reservation = {"request_key": digest({"payload_key": event["request_key"], "time": event["time"]}),
                   "payload_request_key": event["request_key"], "scope": SCOPE,
                   "reserved_usd": RESERVATION, "run": str(out), "attempt": event["attempt"],
                   "reason": "Request may have been billed; no charge receipt returned"}
    headers = event.get("response_headers", {})
    if "x-generation-id" in headers:
        reservation["response_id"] = headers["x-generation-id"]
    rows.append(reservation)
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return RESERVATION


async def run_requests(packets: list[dict], out: Path, *, max_new_usd: float, concurrency: int = 8,
                       key_source: str = "env", refresh: bool = False, cache_write: bool = True) -> dict:
    ledger = ROOT / ".local/billing.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    history = [json.loads(line) for line in ledger.open()] if ledger.exists() else []
    project_before = sum(row["cost_usd"] for row in history)
    scope_before = sum(row["cost_usd"] for row in history if row.get("scope") == SCOPE)
    new = reserved = uncertain = 0.0
    counts = Counter()
    stopping = False
    by_model: dict[str, list[dict]] = {}
    for packet in packets:
        by_model.setdefault(packet["request"]["model"], []).append(packet)
    work = {model: iter(rows) for model, rows in by_model.items()}
    capacity = asyncio.Semaphore(concurrency)
    request_locks: dict[str, asyncio.Lock] = {}
    model_start_locks: dict[str, asyncio.Lock] = {}
    model_next_start: dict[str, float] = {}
    clock = asyncio.get_running_loop().time
    cache = ROOT / "data/interim/action_choice_cache"
    cache.mkdir(parents=True, exist_ok=True)
    @functools.cache
    def authorization() -> dict[str, str]:
        return {"Authorization": "Bearer " + api_key(key_source)}
    async with httpx.AsyncClient(headers={"X-OpenRouter-Title": "virtue-choices/action-choice-eval"}, timeout=180) as client:
        with (out / "receipts.jsonl").open("w") as receipts:
            async def worker(model_work):
                nonlocal new, reserved, uncertain
                for packet in model_work:
                    if stopping:
                        return
                    payload = packet["request"]
                    key = digest(payload)
                    async with request_locks.setdefault(key, asyncio.Lock()):
                        path = cache / (key + ".json.gz")
                        cached = path.exists() and not refresh
                        if cached:
                            record = json.loads(gzip.decompress(path.read_bytes()))
                            assert record["request"] == payload and record["request_key"] == key
                            if transient_provider_error(record["response"]):
                                cached = False
                                counts["cached_infrastructure_errors_retried"] += 1
                        if cached:
                            counts["cache_hits"] += 1
                        else:
                            assert project_before + new + reserved + outstanding(ROOT) + RESERVATION <= PROJECT_CAP_USD, "Cumulative project cap reached"
                            assert scope_before + new + reserved + outstanding(ROOT, SCOPE) + RESERVATION <= EVALUATION_CAP_USD, "Evaluation scope cap reached"
                            assert new + uncertain + reserved + RESERVATION <= max_new_usd, "This evaluation run's cap reached"
                            reserved += RESERVATION
                            try:
                                for attempt in range(4):
                                    async with model_start_locks.setdefault(payload["model"], asyncio.Lock()):
                                        while (delay := model_next_start.get(payload["model"], 0) - clock()) > 0:
                                            await asyncio.sleep(delay)
                                        model_next_start[payload["model"]] = clock() + packet["metadata"].get("minimum_request_interval_seconds", 0)
                                    assert project_before + new + reserved + outstanding(ROOT) <= PROJECT_CAP_USD, "Project cap reached during retries"
                                    assert scope_before + new + reserved + outstanding(ROOT, SCOPE) <= EVALUATION_CAP_USD, "Evaluation cap reached during retries"
                                    assert new + uncertain + reserved <= max_new_usd, "This evaluation run's cap reached during retries"
                                    event = {"time": datetime.now(timezone.utc).isoformat(), "request_key": key,
                                             "metadata": packet["metadata"], "attempt": attempt, "request": payload}
                                    counts["http_attempts"] += 1
                                    chunks = []
                                    try:
                                        async with capacity:
                                            async with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions", json=payload, headers=authorization()) as response:
                                                event |= {"http_status": response.status_code, "response_headers": dict(response.headers)}
                                                async for chunk in response.aiter_bytes():
                                                    chunks.append(chunk)
                                    except httpx.TransportError as error:
                                        event |= {"error": repr(error), "partial_body_base64": base64.b64encode(b"".join(chunks)).decode("ascii")}
                                        if not isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
                                            uncertain += reserve_unknown_fee(event, out)
                                        with (out / "events.jsonl").open("a") as stream:
                                            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                                        if attempt < 3:
                                            await asyncio.sleep(2 ** attempt)
                                            continue
                                        record = {"request_key": key, "request": payload, "http_status": None,
                                                  "response_origin": "transport_failure", "response": {"transport_error": repr(error), "attempts": attempt + 1}}
                                        counts["new_requests"] += 1
                                        counts["exhausted_transport_errors"] += 1
                                        break
                                    raw_body = b"".join(chunks)
                                    try:
                                        event["response_text"] = raw_body.decode("utf-8")
                                    except UnicodeDecodeError as error:
                                        event |= {"error": repr(error), "response_body_base64": base64.b64encode(raw_body).decode("ascii")}
                                        uncertain += reserve_unknown_fee(event, out)
                                        with (out / "events.jsonl").open("a") as stream:
                                            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                                        raise
                                    with (out / "events.jsonl").open("a") as stream:
                                        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                                    if response.status_code == 200 or response.headers.get("content-type", "").startswith("application/json"):
                                        try:
                                            body = json.loads(event["response_text"])
                                        except json.JSONDecodeError:
                                            uncertain += reserve_unknown_fee(event, out)
                                            raise
                                    else:
                                        body = {"error": {"message": event["response_text"]}}
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
                                        uncertain += reserve_unknown_fee(event, out)
                                        raise ValueError("Successful chat response lacks an actual cost receipt")
                                    elif response.status_code in {408, 504} or ("error" in body and body["error"].get("code") in {408, 504}):
                                        uncertain += reserve_unknown_fee(event, out)
                                    provider_timeout = response.status_code == 200 and transient_provider_error(body)
                                    rate_limited = response.status_code == 429 or ("error" in body and body["error"].get("code") == 429)
                                    if rate_limited:
                                        model_next_start[payload["model"]] = max(model_next_start[payload["model"]], clock() + 30)
                                    if (response.status_code in {408, 429} or 500 <= response.status_code < 600 or provider_timeout) and attempt < 3:
                                        await asyncio.sleep(2 ** attempt)
                                        continue
                                    record = {"request_key": key, "request": payload, "http_status": response.status_code, "response": body}
                                    counts["new_requests"] += 1
                                    if response.status_code == 200 and not provider_timeout and cache_write:
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
            workers_per_model = max(1, concurrency // len(work))
            tasks = [asyncio.create_task(worker(model_work)) for model_work in work.values() for _ in range(workers_per_model)]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                stopping = True
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
    assert counts["completed"] == len(packets)
    return dict(counts) | {"new_cost_usd": new, "project_total_usd": project_before + new,
                           "evaluation_scope_usd": scope_before + new,
                           "evaluation_scope_native_estimates_usd": noncash_reviews(ROOT, SCOPE),
                           "evaluation_scope_conservative_total_usd": scope_before + new + outstanding(ROOT, SCOPE),
                           "unbilled_cost_reserved_usd": unknown_fees(ROOT),
                           "noncash_review_estimate_usd": noncash_reviews(ROOT),
                           "conservative_budget_total_usd": project_before + new + outstanding(ROOT), "scope": SCOPE}
