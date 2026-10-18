"""Score Machiavelli states by moral contrast between available choices.

This is a pure score/sort step over `wassname/machiavelli`; it does not call
OpenRouter and does not upload anything. The score is based on per-axis standard
deviation across the choices in one game state. Violence is allowed when it
carries moral ambiguity; power/violence labels are only a penalty when they
explain the contrast by themselves.

Run:
    uv run python scripts/scenario_sources/score_machiavelli_moral_contrast.py

Out:
    out/machiavelli_moral_contrast/scored.jsonl
    out/machiavelli_moral_contrast/top_by_game.jsonl
    out/machiavelli_moral_contrast/top1000_round_robin.jsonl
    out/machiavelli_moral_contrast/random_5pct.jsonl
"""
import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from loguru import logger

DATASET = "wassname/machiavelli"

# Detailed morality labels are already small and interpretable. Aggregate labels
# have much larger signed scales, so they get small weights.
SUBTLE_WEIGHTS = {
    "morality_deception": 1.0,
    "morality_fairness": 1.0,
    "morality_manipulation": 1.0,
    "morality_non_physical_harm": 1.0,
    "morality_other": 0.5,
    "morality_promises": 1.0,
    "morality_spying": 0.8,
    "morality_stealing": 0.8,
    "morality_trespassing": 0.6,
    "morality_vandalism": 0.5,
    "agg_honest": 0.03,
    "agg_cooperation": 0.03,
    "agg_morality": 0.03,
    "agg_utility": 0.01,
}
CRUDE_WEIGHTS = {
    "morality_intending_harm": 1.0,
    "morality_killing": 1.2,
    "morality_physical_harm": 1.0,
    "agg_power": 0.03,
}
REQUIRED_COLUMNS = {
    "row_i", "title", "obs", "choice_labels", "choice_text",
    "label_sum", "label_norm_sum",
}


def _finite(x) -> float:
    if x is None:
        return 0.0
    return float(x)


def _choice_rows(row: dict) -> list[dict]:
    choices = []
    for choice_idx, text in row["choice_text"].items():
        labels = row["choice_labels"][choice_idx]
        if text is None or labels is None:
            continue
        choices.append({
            "choice_idx": choice_idx,
            "choice_text": text,
            "labels": {k: _finite(v) for k, v in labels.items()},
        })
    return choices


def _weighted_l1(a: dict[str, float], b: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights[k] * abs(a[k] - b[k]) for k in weights)


def _choice_axis_spread(choices: list[dict], weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    std_by_key = {}
    for key in weights:
        values = [choice["labels"][key] for choice in choices]
        std_by_key[key] = statistics.pstdev(values)
    score = sum(weights[key] * std for key, std in std_by_key.items())
    return score, std_by_key


def _top_label_delta(a: dict[str, float], b: dict[str, float], weights: dict[str, float]) -> list[dict]:
    rows = []
    for key, weight in weights.items():
        delta = abs(a[key] - b[key])
        if delta > 0:
            rows.append({"key": key, "delta": delta, "weighted": weight * delta})
    return sorted(rows, key=lambda r: r["weighted"], reverse=True)


def _best_pair(row: dict) -> dict | None:
    choices = _choice_rows(row)
    if len(choices) < 2:
        return None
    best = None
    for i, a in enumerate(choices):
        for b in choices[i + 1:]:
            subtle = _weighted_l1(a["labels"], b["labels"], SUBTLE_WEIGHTS)
            crude = _weighted_l1(a["labels"], b["labels"], CRUDE_WEIGHTS)
            score = subtle - 0.8 * crude
            candidate = {
                "score": score,
                "subtle_score": subtle,
                "crude_score": crude,
                "choice_a": a,
                "choice_b": b,
                "top_subtle_deltas": _top_label_delta(a["labels"], b["labels"], SUBTLE_WEIGHTS)[:5],
                "top_crude_deltas": _top_label_delta(a["labels"], b["labels"], CRUDE_WEIGHTS)[:3],
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    return best


def score_row(row: dict) -> dict | None:
    choices = _choice_rows(row)
    if len(choices) < 2:
        return None
    best = _best_pair(row)
    subtle_score, subtle_std = _choice_axis_spread(choices, SUBTLE_WEIGHTS)
    crude_score, crude_std = _choice_axis_spread(choices, CRUDE_WEIGHTS)
    score = subtle_score - 0.8 * crude_score
    return {
        "row_i": row["row_i"],
        "title": row["title"],
        "obs": row["obs"],
        "score": round(score, 4),
        "subtle_score": round(subtle_score, 4),
        "crude_score": round(crude_score, 4),
        "choice_a_idx": best["choice_a"]["choice_idx"],
        "choice_a_text": best["choice_a"]["choice_text"],
        "choice_b_idx": best["choice_b"]["choice_idx"],
        "choice_b_text": best["choice_b"]["choice_text"],
        "axis_std": {k: round(v, 4) for k, v in sorted(subtle_std.items()) if v > 0},
        "crude_axis_std": {k: round(v, 4) for k, v in sorted(crude_std.items()) if v > 0},
        "top_subtle_deltas": best["top_subtle_deltas"],
        "top_crude_deltas": best["top_crude_deltas"],
        "label_sum": row["label_sum"],
        "label_norm_sum": row["label_norm_sum"],
    }


def _sample_by_game(rows: list[dict], frac: float, cap_per_game: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_game: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_game[row["title"]].append(row)
    out = []
    for game, game_rows in by_game.items():
        n = min(cap_per_game, max(1, math.ceil(frac * len(game_rows))))
        out.extend(rng.sample(game_rows, n))
        logger.info(f"random sample {game!r}: {n}/{len(game_rows)}")
    return out


def _top_by_game(rows: list[dict], n_per_game: int) -> list[dict]:
    by_game: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_game[row["title"]].append(row)
    out = []
    for game_rows in by_game.values():
        out.extend(sorted(game_rows, key=lambda r: r["score"], reverse=True)[:n_per_game])
    return sorted(out, key=lambda r: r["score"], reverse=True)


def _round_robin_by_game(rows: list[dict], n_total: int) -> list[dict]:
    by_game: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_game[row["title"]].append(row)
    for game in by_game:
        by_game[game] = sorted(by_game[game], key=lambda r: r["score"], reverse=True)
    games = sorted(by_game, key=lambda game: by_game[game][0]["score"], reverse=True)
    selected = []
    depth = 0
    while len(selected) < n_total:
        grew = False
        for game in games:
            if depth < len(by_game[game]):
                selected.append(by_game[game][depth])
                grew = True
                if len(selected) >= n_total:
                    break
        if not grew:
            break
        depth += 1
    return selected


def _dedupe_scored_rows(rows: list[dict]) -> list[dict]:
    best_by_content = {}
    for row in rows:
        key = (
            row["title"],
            row["choice_a_text"],
            row["choice_b_text"],
        )
        old = best_by_content.get(key)
        if old is None or row["score"] > old["score"]:
            best_by_content[key] = row
    return list(best_by_content.values())


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    keep = [
        "title", "row_i", "score", "subtle_score", "crude_score",
        "choice_a_idx", "choice_a_text", "choice_b_idx", "choice_b_text",
        "axis_std", "crude_axis_std", "top_subtle_deltas", "top_crude_deltas",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keep)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], (dict, list)) else row[k]
                for k in keep
            })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--outdir", type=Path, default=Path("out/machiavelli_moral_contrast"))
    ap.add_argument("--top-per-game", type=int, default=25)
    ap.add_argument("--top-total", type=int, default=1000)
    ap.add_argument("--random-frac", type=float, default=0.05)
    ap.add_argument("--random-cap-per-game", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke-rows", type=int, default=0)
    args = ap.parse_args()

    ds = load_dataset(DATASET, split=args.split)
    if args.smoke_rows:
        ds = ds.select(range(min(args.smoke_rows, len(ds))))
    rows = list(ds)
    assert rows, f"loaded 0 rows from {DATASET}:{args.split}"
    assert REQUIRED_COLUMNS <= set(rows[0]), rows[0].keys()

    scored = [score_row(row) for row in rows]
    scored = [row for row in scored if row is not None]
    assert scored, "no rows had at least two valid choices"
    before_dedupe = len(scored)
    scored = _dedupe_scored_rows(scored)
    scored = sorted(scored, key=lambda r: r["score"], reverse=True)
    logger.info(f"deduped scored rows: {before_dedupe} -> {len(scored)}")
    top_rows = _top_by_game(scored, args.top_per_game)
    top_total_rows = _round_robin_by_game(scored, args.top_total)
    random_rows = _sample_by_game(scored, args.random_frac, args.random_cap_per_game, args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.outdir / "scored.jsonl", scored)
    _write_jsonl(args.outdir / "top_by_game.jsonl", top_rows)
    _write_jsonl(args.outdir / "top1000_round_robin.jsonl", top_total_rows)
    _write_jsonl(args.outdir / "random_5pct.jsonl", random_rows)
    _write_csv(args.outdir / "scored.csv", scored)
    _write_csv(args.outdir / "top_by_game.csv", top_rows)
    _write_csv(args.outdir / "top1000_round_robin.csv", top_total_rows)
    logger.info(f"wrote {len(scored)} sorted scored rows -> {args.outdir / 'scored.jsonl'}")
    logger.info(f"wrote {len(top_rows)} top-by-game rows -> {args.outdir / 'top_by_game.jsonl'}")
    logger.info(f"wrote {len(top_total_rows)} round-robin rows -> {args.outdir / 'top1000_round_robin.jsonl'}")
    logger.info(f"wrote {len(random_rows)} random rows -> {args.outdir / 'random_5pct.jsonl'}")


if __name__ == "__main__":
    main()
