# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow>=24.0.0", "matplotlib>=3.9"]
# [tool.uv]
# exclude-newer = "8 days"
# ///
"""Plot each model's chosen-action score per tradition, and DeepSeek's score divided by Gemma's. Unpaid; reads a saved run. -- PI/Claude-opus-5-5"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS, GE = "deepseek/deepseek-v4.1-flash", "google/gemma-4-31b-it"
RED, BLUE = "#c0392b", "#2c6fbb"


def scene_deltas(run: Path, virtues: Path):
    """Per paired scene and tradition: (DeepSeek, Gemma) chosen-action score minus the uniform-choice score."""
    scenes = {r["source_id"]: r for r in map(json.loads, (run / "scenes.jsonl").open())}
    pred = {(r["source_id"], r["model"], r["order"]): r for r in map(json.loads, (run / "choices.jsonl").open())}
    raw = {(r["source_id"], r["choice_id"]): r["virtues"] for r in pq.read_table(virtues).to_pylist() if r["source_id"] in scenes}
    first = next(iter(raw.values()))
    direct = sorted(i for i in first if first[i]["derived_from"] is None)  # derived contraries duplicate their parent
    trads = sorted({i.split("|")[0] for i in direct})
    assert len(direct) == 66 and len(trads) == 9
    items = {t: [i for i in direct if i.startswith(t + "|")] for t in trads}
    paired = sorted(s for s in scenes if all(pred[s, m, o]["status"] == "ok" for m in (DS, GE) for o in ("forward", "reverse")))

    def mean_score(v, t):
        return sum(v[i]["signed"] for i in items[t]) / len(items[t])

    out = {}
    for s in paired:
        options = [raw[s, c] for c in scenes[s]["choice_text"]]
        row = []
        for t in trads:
            uniform = sum(mean_score(v, t) for v in options) / len(options)
            chosen = [sum(mean_score(raw[s, pred[s, m, o]["original_choice_id"]], t) for o in ("forward", "reverse")) / 2 for m in (DS, GE)]
            row.append((chosen[0] - uniform, chosen[1] - uniform))
        out[s] = row
    return trads, out, {s: scenes[s]["game_id"] for s in paired}


def summarise(trads, deltas, games, *, replicates: int, seed: int):
    by_game = defaultdict(list)
    for s, row in deltas.items():
        by_game[games[s]].append(row)
    keys = sorted(by_game)

    def stats(draw):
        rows = [r for g in draw for r in by_game[g]]
        ds = [sum(r[j][0] for r in rows) / len(rows) for j in range(len(trads))]
        ge = [sum(r[j][1] for r in rows) / len(rows) for j in range(len(trads))]
        return ds, ge, [a / b for a, b in zip(ds, ge)]

    point = stats(keys)
    rng = random.Random(seed)
    boots = [stats(rng.choices(keys, k=len(keys))) for _ in range(replicates)]  # resample games, as in score_action_choice.py

    def ci(k, j):
        xs = sorted(b[k][j] for b in boots)
        return xs[int(.025 * (replicates - 1))], xs[int(.975 * (replicates - 1))]

    return [{"tradition": t, "deepseek": point[0][j], "gemma": point[1][j], "ratio": point[2][j], "ratio_ci95": ci(2, j)} for j, t in enumerate(trads)]


def plot(rows, published: Path, path: Path):
    # left panel uses the published per-tradition scores and intervals; the ratio comes from this script
    pub = {(r["model"], r["tradition"]): r for r in map(json.loads, published.open())}
    for r in rows:
        for key, model in (("deepseek", DS), ("gemma", GE)):
            p = pub[model, r["tradition"]]
            assert abs(p["delta_from_uniform"] - r[key]) < 1e-12, "Recomputed score differs from per_tradition.jsonl"
            r[key + "_ci95"] = (p["ci95_low"], p["ci95_high"])
    rows = sorted(rows, key=lambda r: r["deepseek"])  # bottom to top, so DeepSeek's highest is on top
    y = range(len(rows))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8, 4), sharey=True, gridspec_kw={"width_ratios": [3, 2]}, constrained_layout=True)
    for off, key, colour, label in ((0.15, "deepseek", RED, "DeepSeek v4.1 Flash"), (-0.15, "gemma", BLUE, "Gemma 4 31B")):
        for i, r in enumerate(rows):
            lo, hi = r[key + "_ci95"]
            ax.plot([lo, hi], [i + off] * 2, color=colour, lw=1.2, alpha=.6)
        ax.plot([r[key] for r in rows], [i + off for i in y], "o", color=colour, ms=5)
        top = rows[-1]
        ax.annotate(label, (top[key + "_ci95"][1], len(rows) - 1 + off), xytext=(4, 0), textcoords="offset points", va="center", color=colour, fontsize=8)
    ax.set_xlim(0, 0.16)
    ax.set_xlabel("Chosen action's score minus random choice's")
    ax.set_yticks(list(y), [r["tradition"].capitalize() for r in rows])
    for i, r in enumerate(rows):
        lo, hi = r["ratio_ci95"]
        bx.plot([lo, hi], [i, i], color="0.4", lw=1.2)
        bx.plot(r["ratio"], i, "o", color="0.15", ms=5)
    bx.axvline(1, color="0.7", lw=.8)
    bx.annotate("equal", (1, len(rows) - 0.5), xytext=(3, 0), textcoords="offset points", color="0.5", fontsize=8, va="bottom")
    bx.set_xlabel("DeepSeek's score ÷ Gemma's score")
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(length=0)
        a.grid(axis="x", color="0.92", lw=.6)
        a.set_axisbelow(True)
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("outputs/20261023_action_choice_completed"))
    parser.add_argument("--virtues", type=Path, default=Path("outputs/20261023_virtues_v2_contrast_2500_resumed/virtues.parquet"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/action_choice_by_tradition.png"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    trads, deltas, games = scene_deltas(args.run, args.virtues)
    assert len(deltas) == 499
    rows = summarise(trads, deltas, games, replicates=args.bootstrap, seed=20261024)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({"scenes": len(deltas), "games": len(set(games.values())), "bootstrap": args.bootstrap, "rows": rows}, indent=1) + "\n")
    plot(rows, args.run / "per_tradition.jsonl", args.out)
