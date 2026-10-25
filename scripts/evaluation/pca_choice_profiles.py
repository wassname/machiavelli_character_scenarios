"""Remove one shared action-label component from saved choice profiles. -- PI/OpenAI"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pyarrow.parquet as pq
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_component(values: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return values - (values @ direction)[..., None] * direction


def analyze(directory: Path, out: Path) -> dict:
    directory, out = directory.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.md"
    log.write_text(f"# PCA residual choice profiles\n\nInput: `{directory}`\n\n"
                   "SHOULD: action-wise projection and projection of mean profiles agree; residual PC1 dot products are zero. -- PI/OpenAI\n\n")
    analysis = json.loads((directory / "analysis.json").read_text())
    run = Path(analysis["inputs"]["run"])
    assert sha(run / "scenes.jsonl") == analysis["input_sha256"]["scenes"]
    assert sha(run / "choices.jsonl") == analysis["input_sha256"]["choices"]
    scenes = {r["source_id"]: r for r in map(json.loads, (run / "scenes.jsonl").open())}
    model_ids, names = list(analysis["models"]), list(analysis["models"].values())
    assert names[0] == "DeepSeek"
    choices = {(r["source_id"], r["model"], r["order"]): r["original_choice_id"]
               for r in map(json.loads, (run / "choices.jsonl").open()) if r["model"] in model_ids}
    assert len(choices) == len(scenes) * len(names) * 2
    dimensions = [d for d in analysis["dimensions"] if d["family"] == "virtues"]
    direct = [d for d in dimensions if d["derived_from"] is None]
    assert len(dimensions) == 80 and len(direct) == 66
    items = [d["id"] for d in direct]
    indices = [dimensions.index(d) for d in direct]
    signs = np.array([-1 if d["kind"] == "vices" else 1 for d in direct])
    action_rows = pq.read_table(directory / "action_scores.parquet").to_pylist()
    actions = {(r["source_id"], r["choice_id"]): np.asarray(r["scores"])[indices] * signs for r in action_rows}
    assert len(actions) == len(action_rows)
    assert set(actions) == {(sid, c) for sid, s in scenes.items() for c in s["choice_text"]}
    boot = np.load(directory / "bootstrap_means.npz")
    original = {(r["source"], r["model"], r["item"]): r["mean"] for r in analysis["rows"]
                if r["family"] == "virtues" and r["mode"] == "above_uniform"}
    result = {"author": "PI/OpenAI", "models": names, "dimensions": direct, "sources": {}, "rows": [],
              "method": "Per-source covariance PCA on 66 direct favorable-direction labels after subtracting each question's action mean. Equal question weights, then equal action weights. No variance scaling. Fit uses all offered actions and no actor choices.",
              "meaning": "Residual = score beyond its fitted PC1 contribution, not an absolute virtue score or capability correction. Zero is uniform-choice expectation. An all-positive direction forces every nonzero orthogonal model difference to have mixed signs; crossings alone are not evidence of distinct values.",
              "uncertainty": "Transform the saved paired game/topic-bootstrap means with the fixed source-specific projection. Pointwise intervals condition on labels, responses and fitted PC1; no loading-estimation or label uncertainty.",
              "cohort": analysis["cohort"], "replicates": analysis["replicates"],
              "input_sha256": {str(p): sha(p) for p in (directory / "analysis.json", directory / "action_scores.parquet", directory / "bootstrap_means.npz", run / "scenes.jsonl", run / "choices.jsonl")},
              "script_sha256": sha(Path(__file__))}
    arrays, summary = {}, []
    for source, info in analysis["sources"].items():
        assert info["independent_items"] == items
        ids = sorted(s for s in scenes if scenes[s]["source"] == source)
        covariance = np.zeros((66, 66))
        selected = []
        centered_actions = []
        weights = []
        for sid in ids:
            matrix = np.stack([actions[sid, c] for c in scenes[sid]["choice_text"]])
            centered = matrix - matrix.mean(0)
            assert np.isfinite(centered).all()
            assert np.max(np.abs(centered.mean(0))) < 1e-12
            covariance += centered.T @ centered / len(matrix) / len(ids)
            selected.append([np.mean([actions[sid, choices[sid, m, order]] for order in ("forward", "reverse")], axis=0) - matrix.mean(0) for m in model_ids])
            centered_actions.extend(centered)
            weights.extend([1 / len(ids) / len(matrix)] * len(matrix))
        assert np.allclose(covariance, info["within_question_covariance"], atol=1e-12, rtol=0)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, -1]
        direction *= 1 if direction.sum() >= 0 else -1
        fraction = float(eigenvalues[-1] / eigenvalues.sum())
        assert np.isclose(fraction, info["pc_variance_fractions"][0])
        selected = np.asarray(selected)
        point = selected.mean(0)
        expected = np.array([[original[source, m, i] for i in items] for m in names]) * signs
        assert np.allclose(point, expected, atol=1e-12, rtol=0)
        residual = remove_component(selected, direction)
        residual_point = residual.mean(0)
        assert np.allclose(residual_point, remove_component(point, direction), atol=1e-12, rtol=0)
        assert np.max(np.abs(residual @ direction)) < 1e-12
        assert np.max(np.abs(remove_component(3 * direction, direction))) < 1e-12
        assert np.allclose(remove_component(residual, direction), residual, atol=1e-12, rtol=0)
        centered_actions = np.asarray(centered_actions)
        action_residual = remove_component(centered_actions, direction)
        variance_ratio = np.sum(np.asarray(weights)[:, None] * action_residual ** 2) / np.trace(covariance)
        assert np.isclose(variance_ratio, 1 - fraction)
        samples = boot[source + ":virtues:above_uniform"][:, :, indices] * signs
        assert samples.shape == (analysis["replicates"], len(names), 66)
        groups = [scenes[sid]["eval_cluster"] for sid in ids]
        clusters = sorted(set(groups))
        totals = np.stack([selected[np.array(groups) == c].sum(0) for c in clusters])
        sizes = np.array([groups.count(c) for c in clusters])
        rng = random.Random(20261024)
        counts = np.array([np.bincount(rng.choices(range(len(clusters)), k=len(clusters)), minlength=len(clusters))
                           for _ in range(analysis["replicates"])])
        rebuilt_samples = np.einsum("rc,cmv->rmv", counts, totals) / (counts @ sizes)[:, None, None]
        bootstrap_error = float(np.max(np.abs(samples - rebuilt_samples)))
        assert bootstrap_error < 1e-12
        residual_samples = remove_component(samples, direction)
        assert np.max(np.abs(residual_samples @ direction)) < 1e-12
        arrays[source + ":original"] = samples
        arrays[source + ":residual"] = residual_samples
        arrays[source + ":pc1"] = direction
        virtue_ix = [i for i, d in enumerate(direct) if d["kind"] == "virtues"]
        ranks = {}
        for mode, values, draws in (("original", point, samples), ("residual", residual_point, residual_samples)):
            ranks[mode] = dict(Counter(">".join(names[m] for m in np.argsort(values[:, i])[::-1]) for i in virtue_ix))
            ci = np.quantile(draws, [.025, .975], axis=0)
            for m, name in enumerate(names):
                for j, item in enumerate(items):
                    result["rows"].append({"source": source, "mode": mode, "model": name, "item": item,
                                           "mean": float(values[m, j]), "ci95": [float(ci[0, m, j]), float(ci[1, m, j])]})
        result["sources"][source] = {"scenes": len(ids), "actions": len(centered_actions), "pc1_fraction": fraction,
            "loadings": dict(zip(items, direction.tolist(), strict=True)), "positive_loadings": int((direction > 0).sum()),
            "rank_counts_63_virtues": ranks, "pc1_model_coordinates": dict(zip(names, (point @ direction).tolist(), strict=True)),
            "max_bootstrap_reconstruction_error": bootstrap_error,
            "max_residual_dot_pc1": float(np.max(np.abs(residual @ direction))),
            "residual_variance_fraction": float(variance_ratio)}
        summary.append([source, f"{fraction:.1%}", len(ids), ranks["original"].get("Haiku>DeepSeek>Gemma", 0), ranks["residual"].get("Haiku>DeepSeek>Gemma", 0)])
    result["checks"] = "PASS: original source/choice hashes; exact action coverage; recomputed covariance; original means; projection/averaging commutation; residual orthogonality; idempotence; pure-PC1 control; variance decomposition; full paired cluster-bootstrap reconstruction and transformation."
    np.savez_compressed(out / "bootstrap_means.npz", **arrays)
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    table = tabulate(summary, headers=["Source", "Label variance removed", "Scenes", "H>D>G before /63", "H>D>G after /63"], tablefmt="pipe")
    text = "## Result\n\n" + table + "\n\nRank counts are descriptive. Crossings are not evidence of valid moral distinctions.\n\n" + result["checks"] + "\n\n" + result["uncertainty"] + "\n\n" + str(log) + "\n"
    with log.open("a") as f:
        f.write(text)
    print(text)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/20261025_individual_dimensions")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/20261026_pca_profiles")
    args = parser.parse_args()
    analyze(args.input, args.out)
