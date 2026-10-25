"""Score every saved behaviour, virtue and D&D dimension without inference. -- PI/Astra"""

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from action_choice import behavior_eligible, ELIGIBILITY_PATH
from relative_virtue_profiles import MODELS

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_visible(scene: dict, row: dict) -> dict:
    assert row["text"] == scene["text"], f"Changed visible story: {scene['source_id']}"
    assert json.loads(row["choice_text"]) == scene["choice_text"], f"Changed choices: {scene['source_id']}"
    labels = json.loads(row["choice_labels_visible"])
    assert labels.keys() == scene["choice_text"].keys()
    for key, label in labels.items():
        assert label["source_id"] == scene["source_id"] and label["choice_id"] == key
        assert label["action"] == scene["choice_text"][key], f"Wrong labelled act: {scene['source_id']}/{key}"
    return labels


def complete_scene_ids(values: dict, ids: list[str], indices: list[int]) -> list[str]:
    return [sid for sid in ids if np.isfinite(np.stack(list(values[sid].values()))[:, indices]).all()]


def bootstrap(values: np.ndarray, groups: list[str], replicates: int) -> np.ndarray:
    clusters = sorted(set(groups))
    groups = np.asarray(groups)
    counts = np.array([(groups == key).sum() for key in clusters])
    totals = np.stack([values[groups == key].sum(0) for key in clusters])
    rng = random.Random(20261024)
    draws = np.array([rng.choices(range(len(clusters)), k=len(clusters)) for _ in range(replicates)])
    return totals[draws].sum(1) / counts[draws].sum(1)[:, None, None]


def analyze(run: Path, dataset: Path, out: Path, replicates: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    scene_rows = list(map(json.loads, (run / "scenes.jsonl").open()))
    scenes = {r["source_id"]: r for r in scene_rows}
    assert len(scenes) == len(scene_rows), "Duplicate source IDs"
    models = list(MODELS)
    records = [r for r in map(json.loads, (run / "choices.jsonl").open()) if r["model"] in models]
    replies = {(r["source_id"], r["model"], r["order"]): r for r in records}
    assert len(replies) == len(records) == len(scenes) * len(models) * 2
    assert all(r["status"] == "ok" for r in records)
    raw = {}
    for batch in pq.ParquetFile(run / "virtues.parquet").iter_batches(columns=["source_id", "choice_id", "virtues"]):
        for row in batch.to_pylist():
            if row["source_id"] in scenes:
                key = row["source_id"], row["choice_id"]
                assert key not in raw
                raw[key] = row["virtues"]
    first = next(iter(raw.values()))
    items = sorted(first)
    direct = [item for item in items if first[item]["derived_from"] is None]
    rubric = json.loads((ROOT / "rubrics/virtues_v2.json").read_text())
    kinds = {item["item_id"]: item["kind"] for item in rubric["items"]}
    kinds.update({pair["vice_id"]: "vices" for pair in rubric["contraries"]})
    assert len(items) == 80 and len(direct) == 66 and set(kinds) == set(items)
    dimensions = [{"id": item, "family": "virtues", "tradition": item.split("|")[0], "label": item.split("|")[-1],
                   "kind": kinds[item], "derived_from": first[item]["derived_from"],
                   "direction": "higher = more of the named trait; vice signs reversed from favorable signed labels"} for item in items]
    behaviour_names = json.loads((ROOT / "rubrics/actions.json").read_text())["dimensions"]
    assert len(behaviour_names) == 26
    dimensions += [{"id": key, "family": "behaviours", "label": key, "kind": "behaviour", "derived_from": None,
                    "direction": "higher = greater judged probability of this behaviour, not always better"} for key in behaviour_names]
    dimensions += [{"id": axis, "family": "dnd", "label": axis, "kind": "alignment", "derived_from": None,
                    "direction": "higher = more lawful" if axis == "order" else "higher = more good"} for axis in ("order", "morality")]
    visible = {}
    columns = ["source_id", "text", "choice_text", "choice_labels_visible"]
    for batch in pq.ParquetFile(dataset).iter_batches(batch_size=32, columns=columns):
        for row in batch.to_pylist():
            sid = row["source_id"]
            if sid in scenes:
                assert sid not in visible
                visible[sid] = checked_visible(scenes[sid], row)
    assert visible.keys() == scenes.keys()
    dimension_ids = [d["id"] for d in dimensions]
    values, action_rows = {}, []
    for sid, scene in scenes.items():
        assert {c for s, c in raw if s == sid} == scene["choice_text"].keys()
        values[sid] = {}
        for choice_id in scene["choice_text"]:
            labels, old = raw[sid, choice_id], visible[sid][choice_id]
            assert labels.keys() == first.keys() and old["visible_action_values"].keys() == set(behaviour_names)
            for item in items:
                parent = labels[item]["derived_from"]
                assert parent == first[item]["derived_from"], f"Changed derivation: {sid}/{choice_id}/{item}"
                if parent is not None:
                    assert labels[item]["signed"] == labels[parent]["signed"]
            v = [labels[item]["signed"] * (-1 if kinds[item] == "vices" else 1) for item in items]
            v += [old["visible_action_values"][key] for key in behaviour_names]
            axes = old["dnd_action_alignment"]
            assert axes.keys() == {"order", "morality"}
            v += [axes[axis]["score"] if axes[axis]["score"] is not None else np.nan for axis in ("order", "morality")]
            array = np.array(v)
            assert np.isfinite(array[:106]).all()
            assert np.abs(array[:80]).max() <= 1.00000001
            assert (array[80:106] >= 0).all() and (array[80:106] <= 1).all()
            assert all(np.isnan(x) or -5 <= x <= 5 for x in array[106:])
            values[sid][choice_id] = array
            action_rows.append({"source_id": sid, "choice_id": choice_id, "scores": v})
    pq.write_table(pa.Table.from_pylist(action_rows), out / "action_scores.parquet")
    reviews = json.loads(ELIGIBILITY_PATH.read_text())["cases"]
    result = {"author": "PI/Astra", "models": MODELS, "dimensions": dimensions, "replicates": replicates,
              "input_sha256": {"scenes": sha(run / "scenes.jsonl"), "choices": sha(run / "choices.jsonl"),
                               "virtues": sha(run / "virtues.parquet"), "visible_labels": sha(dataset), "eligibility": sha(ELIGIBILITY_PATH),
                               "virtue_rubric": sha(ROOT / "rubrics/virtues_v2.json"), "behaviour_rubric": sha(ROOT / "rubrics/actions.json")},
              "inputs": {"run": str(run), "visible_labels": str(dataset)}, "script_sha256": sha(Path(__file__)),
              "cohort": "Historical full shared three-model cohort; no source filtering. Exclusion review only marks candidates for future calls.",
              "scores": "Selected scores average both orders. Above_uniform subtracts the same scene's mean over all actions. Versus_DeepSeek subtracts its paired score; no rescaling.",
              "missingness": "Each family uses questions with all offered actions finite on every family dimension, shared by all models, independent of chosen actions. Missing D&D is never zero.",
              "uncertainty": f"{replicates} jointly paired game/topic-cluster resamples, pointwise mean SE and percentile95% intervals. Labels and replies fixed; no epistemic calibration or intrinsic-preference claim.",
              "display_polarity": "For all17 vice records, use minus favorable signed score (named-vice presence direction); 14 are derived duplicates. Covariance uses66 independent favorable-polarity labels.",
              "sources": {}, "rows": [], "candidates": []}
    saved = {}
    direct_ix = [items.index(item) for item in direct]
    favorable_signs = np.array([-1 if kinds[item] == "vices" else 1 for item in direct])
    for source in sorted({scene["source"] for scene in scenes.values()}):
        ids = sorted(sid for sid, scene in scenes.items() if scene["source"] == source)
        covariance = np.zeros((66, 66))
        for sid in ids:
            action_ids = list(scenes[sid]["choice_text"])
            matrix = np.stack([values[sid][c][direct_ix] * favorable_signs for c in action_ids])
            centered = matrix - matrix.mean(0)
            covariance += centered.T @ centered / len(matrix) / len(ids)
            if behavior_eligible(scenes[sid], reviews):
                quality = matrix.mean(1)
                for a, b in combinations(range(len(matrix)), 2):
                    gap = matrix[a] - matrix[b]
                    pos, neg = np.flatnonzero(gap >= .35), np.flatnonzero(gap <= -.35)
                    if len(pos) >= 2 and len(neg) >= 2 and abs(quality[a] - quality[b]) <= .12:
                        strength = min(gap[pos].mean(), -gap[neg].mean())
                        result["candidates"].append({"source_id": sid, "source": source, "choices": [action_ids[a], action_ids[b]],
                            "strength": float(strength), "overall_gap": float(quality[a] - quality[b]),
                            "distance_below_best_overall": float(quality.max() - min(quality[a], quality[b])),
                            "positive_items": [direct[i] for i in pos], "negative_items": [direct[i] for i in neg]})
        std = np.sqrt(np.diag(covariance))
        assert (std > 0).all()
        correlation = covariance / np.outer(std, std)
        eig = np.linalg.eigvalsh(covariance)[::-1]
        corr_eig = np.linalg.eigvalsh(correlation)[::-1]
        offdiag = correlation[np.triu_indices(66, 1)]
        positive_items = [k for k in direct if kinds[k] == "virtues"]
        info = {"historical_scenes": len(ids), "families": {}, "within_question_covariance": covariance.tolist(),
                "within_question_correlation": correlation.tolist(), "independent_items": direct,
                "pc_variance_fractions": (eig / eig.sum()).tolist(), "correlation_pc_fractions": (corr_eig / corr_eig.sum()).tolist(),
                "median_pair_correlation": float(np.median(offdiag)), "negative_correlation_pairs": int((offdiag < 0).sum())}
        result["sources"][source] = info
        for family in ("virtues", "behaviours", "dnd"):
            ix = [i for i, d in enumerate(dimensions) if d["family"] == family]
            complete = complete_scene_ids(values, ids, ix)
            assert complete
            groups = [scenes[sid]["eval_cluster"] for sid in complete]
            selected = np.stack([[np.mean([values[sid][replies[sid, model, order]["original_choice_id"]][ix]
                                                for order in ("forward", "reverse")], axis=0) for model in models] for sid in complete])
            uniform = np.stack([np.mean([v[ix] for v in values[sid].values()], axis=0) for sid in complete])
            raw_boot = bootstrap(selected, groups, replicates)
            above_boot = bootstrap(selected - uniform[:, None, :], groups, replicates)
            delta_boot = raw_boot - raw_boot[:, :1]
            assert np.max(np.abs(delta_boot - (above_boot - above_boot[:, :1]))) < 1e-12
            info["families"][family] = {"scenes": len(complete), "clusters": len(set(groups)),
                "excluded_missing_labels": sorted(set(ids) - set(complete))}
            if family == "virtues":
                favorable = (selected - uniform[:, None, :])[:, :, direct_ix] * favorable_signs
                info["favorable_overall_above_uniform"] = {MODELS[model]: float(favorable[:, m].mean()) for m, model in enumerate(models)}
                mean = selected.mean(0)
                ranks = Counter(">".join(MODELS[models[m]] for m in np.argsort(mean[:, items.index(key)])[::-1]) for key in positive_items)
                info["individual_virtue_rank_counts"] = dict(ranks)
            for mode, array, boot in (("raw", selected, raw_boot), ("above_uniform", selected - uniform[:, None, :], above_boot),
                                      ("versus_deepseek", selected - selected[:, :1], delta_boot)):
                point, se = array.mean(0), boot.std(0, ddof=1)
                ci = np.quantile(boot, [.025, .975], axis=0)
                for m, model in enumerate(models):
                    for j, k in enumerate(ix):
                        result["rows"].append({"source": source, "family": family, "mode": mode, "model": MODELS[model],
                            "item": dimension_ids[k], "mean": float(point[m, j]), "mean_se": float(se[m, j]),
                            "ci95": [float(ci[0, m, j]), float(ci[1, m, j])], "scenes": len(complete)})
                if family == "dnd":
                    info["families"][family][mode + "_bootstrap_covariance"] = {MODELS[models[m]]: np.cov(boot[:, m].T).tolist() for m in range(3)}
                saved[source + ":" + family + ":" + mode] = boot
        print(source, json.dumps({"first_PC": info["pc_variance_fractions"][0], "median_corr": info["median_pair_correlation"],
                                  "individual_virtue_ranks": info["individual_virtue_rank_counts"], "family_n": {f:v["scenes"] for f,v in info["families"].items()}}), flush=True)
    result["candidates"].sort(key=lambda c: (-c["strength"], c["source_id"], c["choices"]))
    result["checks"] = "PASS: exact source text, choice mapping and selected-action joins;80virtues/26behaviours/2axes;14derived parents agree; 66-item favorable means retained for reference checks; shared-baseline cancellation holds for every paired bootstrap draw."
    np.savez_compressed(out / "bootstrap_means.npz", **saved)
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(result["checks"])
    print(out / "analysis.json")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/20261024_action_preferences_resumed")
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs/20261024_release21_numeric/parquet/train.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/20261025_individual_dimensions")
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    analyze(args.run, args.dataset, args.out, args.replicates)
