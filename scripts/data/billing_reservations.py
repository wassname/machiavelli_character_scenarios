"""Keep unknown fees and native usage estimates separate from charge receipts. -- PI/Sol and PI/OpenAI"""

import json
import math
from pathlib import Path

PROJECT_CAP_USD = 80
EVALUATION_CAP_USD = 10.25
EVALUATION_SCOPE = "action-choice-eval-v1"
NATIVE_USAGE_SCOPES = {EVALUATION_SCOPE, "virtue-labels-v2", "legacy-unassigned"}


def unknown_fees(root: Path, scope: str | None = None) -> float:
    path = root / ".local/billing_reservations.json"
    if not path.exists():
        return 0.0
    rows = json.loads(path.read_text())
    identities = [("response", row["response_id"]) if "response_id" in row else ("request", row["request_key"]) for row in rows]
    assert len(set(identities)) == len(rows)
    assert all(math.isfinite(row["reserved_usd"]) and row["reserved_usd"] >= 0 for row in rows)
    return sum(row["reserved_usd"] for row in rows if scope is None or row["scope"] == scope)


def noncash_reviews(root: Path, scope: str | None = None) -> float:
    path = root / ".local/review_usage_estimates.json"
    if not path.exists():
        return 0.0
    rows = json.loads(path.read_text())
    assert len({row["run_id"] for row in rows}) == len(rows)
    assert all(row["scope"] in NATIVE_USAGE_SCOPES for row in rows), "Unknown native usage scope"
    values = [row["reported_cost_usd"]["costUsd"] for row in rows]
    assert all(math.isfinite(value) and value >= 0 for value in values)
    return sum(row["reported_cost_usd"]["costUsd"] for row in rows
               if scope is None or row["scope"] == scope)


def outstanding(root: Path, scope: str | None = None) -> float:
    return unknown_fees(root, scope) + noncash_reviews(root, scope)
