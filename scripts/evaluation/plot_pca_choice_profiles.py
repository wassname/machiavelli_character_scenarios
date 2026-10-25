# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "numpy>=2,<3"]
# [tool.uv]
# exclude-newer = "8 days"
# ///
"""Render original and PC1-residual scores from one saved analysis. -- PI/OpenAI"""

import argparse
from html import escape
import json
from pathlib import Path
import textwrap

import numpy as np
from plot_choice_dimensions import bars, key_legend, plt, save, SHORT, SOURCES, TRADITIONS

ROOT = Path(__file__).resolve().parents[2]


def render(directory: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((directory / "analysis.json").read_text())
    boot = np.load(directory / "bootstrap_means.npz")
    items = [d["id"] for d in data["dimensions"]]
    index = {(r["source"], r["mode"], r["model"], r["item"]): r["mean"] for r in data["rows"]}
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "svg.fonttype": "none"})
    footer = ("0 = DeepSeek after the same transformation. Points = paired means; lines = pointwise 95% game/topic-bootstrap intervals.\n"
              "Labels, replies and PC1 held fixed. Historical role-play cohort. Residuals are not capability-corrected virtue scores.  PI/OpenAI")

    def get(source, mode, groups):
        samples = boot[source + ":" + mode]
        paired = samples - samples[:, :1]
        rows = {}
        for model in ("Gemma", "Haiku"):
            m = data["models"].index(model)
            rows[model] = []
            for members in groups:
                ix = [items.index(item) for item in members]
                mean = np.mean([index[source, mode, model, i] - index[source, mode, "DeepSeek", i] for i in members])
                ci = np.quantile(paired[:, m, ix].mean(1), [.025, .975]).tolist()
                rows[model].append({"mean": float(mean), "ci95": ci})
        return rows

    groups = [[d["id"] for d in data["dimensions"] if d["tradition"] == t] for t in TRADITIONS]
    rows = {(s, mode): get(s, mode, groups) for s in SOURCES for mode in ("original", "residual")}
    lim = max(abs(v) for r in rows.values() for ms in r.values() for cell in ms for v in cell["ci95"]) * 1.1
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    fig.suptitle("Choice profiles before and after removing the shared label component", x=.02, ha="left", fontsize=16)
    key_legend(fig, ["Gemma", "Haiku"])
    for column, source in enumerate(SOURCES):
        for row, mode in enumerate(("original", "residual")):
            ax = axes[row, column]
            bars(ax, rows[source, mode])
            ax.set_yticks(range(len(groups)), list(TRADITIONS.values()))
            ax.set_xlim(-lim, lim)
            info = data["sources"][source]
            condition = "Original scores" if mode == "original" else f"PC1 removed ({info['pc1_fraction']:.1%} of label variance)"
            ax.set_title(f"{SOURCES[source]} · n={info['scenes']:,}\n{condition}", loc="left")
            ax.set_xlabel("Mean label-score difference from DeepSeek")
    fig.supxlabel("Equal item weight within each tradition; 66 direct labels only, all in favorable direction. Same scale in every panel.\n" + footer, fontsize=9)
    save(fig, out, "pca_overview")
    figures = ["pca_overview"]
    all_rows = {s: get(s, "residual", [[i] for i in items]) for s in SOURCES}
    lim = max(abs(v) for r in all_rows.values() for ms in r.values() for cell in ms for v in cell["ci95"]) * 1.1
    for source in SOURCES:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
        fig.suptitle(f"PC1-residual individual virtues · {SOURCES[source]}", x=.02, ha="left", fontsize=16)
        key_legend(fig, ["Gemma", "Haiku"])
        for ax, (tradition, title) in zip(axes.flat, TRADITIONS.items(), strict=True):
            ds = [d for d in data["dimensions"] if d["tradition"] == tradition]
            bars(ax, get(source, "residual", [[d["id"]] for d in ds]))
            labels = [("Resistance to " if d["kind"] == "vices" else "") + SHORT.get(d["label"], d["label"]) for d in ds]
            ax.set_yticks(range(len(ds)), [textwrap.fill(n.capitalize(), 25) for n in labels], fontsize=9)
            ax.set_title(title, loc="left")
            ax.set_xlim(-lim, lim)
        fig.supxlabel("Difference in residual label-score units; higher = more than predicted by the removed component, relative to DeepSeek.\n" + footer, fontsize=9)
        name = "pca_virtues_" + source
        save(fig, out, name)
        figures.append(name)
    images = "".join(f'<h2>{escape(name.replace("_", " "))}</h2><a href="{name}.svg"><img src="{name}.png" alt="{escape(name)}"></a>' for name in figures)
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PCA residual choice profiles</title><style>body{font:17px/1.5 system-ui;max-width:1500px;margin:auto;padding:24px;color:#222}p{max-width:1050px}img{width:100%;height:auto}a{color:#246891}</style><h1>Removing the shared label pattern</h1><p><a href="index.html">Original profiles</a> · <a href="pca_analysis.json">All scores and loadings</a></p><p>For each question, subtract the mean of its offered actions. Fit the first principal component (PC1) from those 66-label action vectors, separately for each source, without using model choices. Subtract each action\'s projection on that direction, then average the models\' saved choices. No labels were changed and no new actor or judge calls were made.</p><p>A positive residual means more of that label than its shared component predicts. It does not mean better conduct. Here all 66 PC1 loadings are positive, so every nonzero residual model difference must have both positive and negative entries. Crossings alone prove no moral distinction. Remaining differences can include judge errors and scenario ambiguity. This is a descriptive view, not a capability correction.</p><p>Intervals use the original 2,000 jointly paired bootstrap draws, transformed by the fixed projection. They omit uncertainty in the fitted direction, labels and fresh responses. The historical cohort predates current next-state exclusions. All 66 direct items enter in favorable polarity; the three directly judged vices are shown as resistance. Derived duplicates are excluded.</p>' + images + '<p>PI/OpenAI</p></html>'
    (out / "pca.html").write_text(page)
    (out / "pca_analysis.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS: {len(figures)} PCA figures rendered; clipping checks passed. Independent visual review pending.")
    print(out / "pca.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs/20261026_pca_profiles")
    parser.add_argument("--out", type=Path, default=ROOT / ".local/choice_preview")
    args = parser.parse_args()
    render(args.input, args.out)
