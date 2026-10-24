"""Export full-model virtue profiles with the production grouped bootstrap. -- PI/OpenAI"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from score_action_choice import paired_summary

MODELS = {
    "deepseek/deepseek-v4.1-flash": "DeepSeek",
    "google/gemma-4-31b-it": "Gemma",
    "anthropic/claude-haiku-5.5": "Haiku",
}


def export(run: Path, virtues: Path, out: Path, replicates: int) -> dict:
    scenes = {r["source_id"]: r for r in map(json.loads, (run / "scenes.jsonl").open())}
    predictions = {}
    for r in map(json.loads, (run / "choices.jsonl").open()):
        if r["model"] in MODELS:
            key = (r["source_id"], r["model"], r["order"])
            assert key not in predictions, "Duplicate actor reply"
            assert r["status"] == "ok", "Incomplete actor response, not a score"
            predictions[key] = r
    assert set(predictions) == {(sid, m, order) for sid in scenes for m in MODELS for order in ("forward", "reverse")}
    raw = {(r["source_id"], r["choice_id"]): r["virtues"] for r in pq.read_table(virtues).to_pylist()
           if r["source_id"] in scenes}
    first = next(iter(raw.values()))
    direct = sorted(i for i in first if first[i]["derived_from"] is None)
    traditions = sorted({i.split("|")[0] for i in direct})
    assert len(direct) == 66 and len(traditions) == 9
    items = {t: [i for i in direct if i.startswith(t + "|")] for t in traditions}
    vectors = {m: {} for m in MODELS}
    width = len(traditions)
    for sid, scene in scenes.items():
        choice_ids = set(scene["choice_text"])
        assert {c for s, c in raw if s == sid} == choice_ids
        scores = {c: [sum(raw[sid, c][i]["signed"] for i in items[t]) / len(items[t])
                      for t in traditions] for c in choice_ids}
        baseline = [sum(v[i] for v in scores.values()) / len(scores) for i in range(width)]
        opportunity = [max(v[i] for v in scores.values()) - baseline[i] for i in range(width)]
        for m in MODELS:
            selected = [scores[predictions[sid, m, order]["original_choice_id"]] for order in ("forward", "reverse")]
            delta = [(a + b) / 2 - base for a, b, base in zip(*selected, baseline, strict=True)]
            vectors[m][sid] = delta + opportunity
    sources = sorted({r["source"] for r in scenes.values()})
    weights = {s: 1 / len(sources) for s in sources}
    reference = {(r["model_" + side], r["tradition"]): r["opportunity_capture_" + side]
                 for r in map(json.loads, (run / "paired_differences.jsonl").open())
                 if r["cohort"] == "equal_sources" and r["scenes"] == len(scenes)
                 for side in ("a", "b") if r["model_" + side] in MODELS}
    assert len(reference) == len(MODELS) * width
    rows = []
    ratios = tuple((i, width + i) for i in range(width))
    for model in MODELS:
        point, intervals = paired_summary(vectors[model], scenes, set(scenes), weights,
                                         replicates=replicates, ratios=ratios)
        for i, tradition in enumerate(traditions):
            ratio_index = 2 * width + i
            assert abs(point[ratio_index] - reference[model, tradition]) < 1e-12, "Does not match production paired score"
            rows.append({"model": model, "name": MODELS[model], "tradition": tradition,
                         "delta_from_uniform": point[i], "delta_ci95": intervals[i],
                         "opportunity_capture": point[ratio_index], "capture_ci95": intervals[ratio_index],
                         "available_above_uniform": point[width + i]})
    result = {"author": "PI/OpenAI", "scenes": len(scenes), "models": MODELS,
              "source_counts": dict(Counter(r["source"] for r in scenes.values())), "source_weights": weights,
              "independent_items": len(direct), "bootstrap_replicates": replicates,
              "orders": "Original order plus reversal, averaged per scene; no initial shuffle.",
              "protocol": "Legacy character-role instruction; original pool, before next-state eligibility filtering.",
              "uncertainty": "95% game/topic-grouped bootstrap within source. Ratio numerator and denominator jointly resampled. Excludes label uncertainty, source validity and role-prompt interpretation.",
              "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                               for name, path in {"scenes": run / "scenes.jsonl", "choices": run / "choices.jsonl", "virtues": virtues}.items()},
              "rows": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--virtues", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/virtue_profiles_3860.json"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    result = export(args.run, args.virtues, args.out, args.bootstrap)
    print(f'{result["scenes"]} paired scenes; {len(result["rows"])} model/tradition estimates with grouped intervals; {args.out}')
