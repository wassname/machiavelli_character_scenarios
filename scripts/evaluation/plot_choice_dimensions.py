# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "numpy>=2,<3"]
# [tool.uv]
# exclude-newer = "8 days"
# ///
"""Render saved choice dimensions and their paired uncertainty. No inference. -- PI/Astra"""

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"machiavelli": "Game-state summaries", "daily_dilemmas": "Everyday dilemmas"}
COLORS = {"DeepSeek": "#9a542f", "Gemma": "#246891", "Haiku": "#39744e"}
MARKERS = {"DeepSeek": "x", "Gemma": "o", "Haiku": "^"}
TRADITIONS = {"buddhist": "☸ Buddhist", "christian": "✝ Christian", "confucian": "Confucian", "hindu": "Hindu",
              "islamic": "☪ Islamic", "jewish": "✡ Jewish", "rationalist": "Rationalist", "stoic": "Stoic", "taoist": "☯ Taoist"}
BEHAVIOURS = {
    "violence": "⚔ Violence (including defence)", "morality_killing": "Killing / attempted killing",
    "morality_physical_harm": "Physical harm", "morality_intending_harm": "Intending harm",
    "morality_non_physical_harm": "Non-physical harm", "morality_deception": "Deception",
    "morality_manipulation": "Manipulation", "morality_fairness": "Unfairness",
    "morality_promises": "Promise-breaking", "morality_spying": "Spying", "morality_stealing": "Stealing",
    "morality_trespassing": "Trespassing", "morality_vandalism": "Vandalism", "morality_other": "Other moral violation",
    "sycophancy": "Sycophancy", "loyalty": "Loyalty", "compassion": "Compassion", "ambition": "Ambition",
    "authority_obedience": "Obedience to authority", "independence": "Independent judgement", "risk_taking": "Risk-taking",
    "agg_honest": "Honesty", "agg_cooperation": "Cooperation", "agg_power": "Seeking power", "agg_utility": "Promoting welfare",
    "agg_morality": "Everyday moral approval"}
SHORT = {"Abstention from unrighteous appropriation": "No wrongful appropriation", "Control of the Sense-organs": "Control of sense-organs",
         "Sincerity (good faith)": "Sincerity / good faith", "Gentleness (compassion, 慈)": "Gentleness / compassion",
         "Shrinking from taking precedence of others": "Not seeking precedence", "liberality to kith and kin": "Generosity to kin",
         "Being bold without gentleness": "Bold without gentleness", "Being liberal without economy (lavishness)": "Lavishness"}


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0, pad=5)
    ax.grid(axis="x", color=".92", linewidth=.6)
    ax.set_axisbelow(True)
    ax.axvline(0, color=".65", linewidth=.8)


def key_legend(fig, names):
    fig.legend([Line2D([], [], color=COLORS[m], marker=MARKERS[m], linewidth=1) for m in names], names,
               loc="outside upper right", ncol=len(names), frameon=False)


def save(fig, out, name):
    fig.canvas.draw()
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    assert box.x0 > -.02 and box.y0 > -.02 and box.x1 < w + .02 and box.y1 < h + .02, (name, box)
    for suffix in ("png", "svg"):
        fig.savefig(out / f"{name}.{suffix}", dpi=150, facecolor="white")
    plt.close(fig)


def bars(ax, rows, factor=1):
    for offset, model in ((-.14, "Gemma"), (.14, "Haiku")):
        r = rows[model]
        y = np.arange(len(r)) + offset
        x = np.array([v["mean"] for v in r]) * factor
        ci = np.array([v["ci95"] for v in r]).T * factor
        ax.hlines(y, ci[0], ci[1], color=COLORS[model], linewidth=1.2)
        ax.plot(x, y, MARKERS[model], color=COLORS[model], markersize=4)
    ax.set_ylim(len(r) - .5, -.5)
    style(ax)


def render(path: Path, out: Path):
    data = json.loads(path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "svg.fonttype": "none"})
    index = {(r["source"], r["mode"], r["model"], r["item"]): r for r in data["rows"]}
    dimensions = data["dimensions"]
    assert {d["id"] for d in dimensions if d["family"] == "behaviours"} == BEHAVIOURS.keys()
    def get(source, items):
        return {m: [index[source, "versus_deepseek", m, item] for item in items] for m in ("Gemma", "Haiku")}
    def limit(family):
        return max(abs(v) for r in data["rows"] if r["family"] == family and r["mode"] == "versus_deepseek" for v in r["ci95"]) * 1.12
    vlim = limit("virtues")
    footer = "Points = paired mean differences; lines = pointwise 95% game/topic-bootstrap intervals. Labels and replies held fixed.\nHistorical role-play prompts; two orders averaged. No variance scaling or model-mean centering.  PI/Astra"
    figures = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 11), constrained_layout=True)
    fig.suptitle("Behaviours chosen, compared with DeepSeek", fontsize=16, x=.02, ha="left")
    key_legend(fig, ["Gemma", "Haiku"])
    for ax, source in zip(axes, SOURCES, strict=True):
        bars(ax, get(source, list(BEHAVIOURS)), factor=100)
        ax.set_yticks(range(26), list(BEHAVIOURS.values()))
        ax.set_xlim(-100 * limit("behaviours"), 100 * limit("behaviours"))
        ax.set_title(f"{SOURCES[source]} · n={data['sources'][source]['historical_scenes']:,}", loc="left")
        ax.set_xlabel("Less ← Difference in judge-probability points → More")
        for boundary in (13.5, 20.5):
            ax.axhline(boundary, color=".9", linewidth=.7)
    fig.supxlabel("Right means more of the named behaviour, not better. 0 = DeepSeek. One point = 0.01 in the judge's probability, not event frequency.\n" + footer, fontsize=9)
    save(fig, out, "behaviours")
    figures.append("behaviours")

    direct = [d for d in dimensions if d["family"] == "virtues" and d["derived_from"] is None]
    assert len(direct) == 66
    for source in SOURCES:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
        fig.suptitle(f"Individual virtues · {SOURCES[source]}", fontsize=16, x=.02, ha="left")
        key_legend(fig, ["Gemma", "Haiku"])
        for ax, (tradition, title) in zip(axes.flat, TRADITIONS.items(), strict=True):
            ds = [d for d in direct if d["tradition"] == tradition]
            bars(ax, get(source, [d["id"] for d in ds]))
            names = [SHORT.get(d["label"], d["label"]).capitalize() + (" †" if d["kind"] == "vices" else "") for d in ds]
            ax.set_yticks(range(len(ds)), [textwrap.fill(n, 24) for n in names], fontsize=9)
            ax.set_title(title, loc="left", fontsize=12)
            ax.set_xlim(-vlim, vlim)
            ax.set_xticks([-.05, 0, .05])
        fig.supxlabel("Chosen-action label score minus DeepSeek (same −1…+1 units). Right = more of the named trait; † directly judged vice, sign reversed.\n66 directly judged items; derived opposites are shown separately. " + footer, fontsize=9)
        name = f"virtues_{source}"
        save(fig, out, name)
        figures.append(name)

    derived = [d for d in dimensions if d["family"] == "virtues" and d["derived_from"] is not None]
    assert len(derived) == 14
    fig, axes = plt.subplots(1, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("Derived vices · the opposite signs of 14 parent virtues", fontsize=15, x=.02, ha="left")
    key_legend(fig, ["Gemma", "Haiku"])
    for ax, source in zip(axes, SOURCES, strict=True):
        bars(ax, get(source, [d["id"] for d in derived]))
        ax.set_yticks(range(14), [textwrap.fill(f"{d['tradition'].capitalize()}: {SHORT.get(d['label'], d['label'])}", 30) for d in derived], fontsize=9)
        ax.set_xlim(-vlim, vlim)
        ax.set_title(SOURCES[source], loc="left")
        ax.set_xlabel("Less ← Named vice score minus DeepSeek → More")
    fig.supxlabel("These are derived records, not new judgements; they add no independent evidence and are excluded from clusters and covariance.\n" + footer, fontsize=9)
    save(fig, out, "derived_vices")
    figures.append("derived_vices")

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), constrained_layout=True)
    fig.suptitle("D&D · changes from DeepSeek, on both axes", fontsize=16, x=.02, ha="left")
    for ax, source in zip(axes, SOURCES, strict=True):
        for model in COLORS:
            rows = [index[source, "versus_deepseek", model, axis] for axis in ("order", "morality")]
            x, y = [r["mean"] for r in rows]
            ax.hlines(y, *rows[0]["ci95"], color=COLORS[model], linewidth=1.2)
            ax.vlines(x, *rows[1]["ci95"], color=COLORS[model], linewidth=1.2)
            ax.plot(x, y, MARKERS[model], color=COLORS[model], markersize=6)
            shifts = {"DeepSeek": (8, -18), "Gemma": (-40, -22), "Haiku": (8, 8)}
            ax.annotate(model, (x, y), xytext=shifts[model], textcoords="offset points", color=COLORS[model], fontsize=10)
        style(ax)
        ax.axhline(0, color=".65", linewidth=.8)
        ax.set(xlim=(-.25, .25), ylim=(-.25, .25), aspect="equal")
        coverage = data["sources"][source]
        ax.set_title(f"{SOURCES[source]}\n{coverage['families']['dnd']['scenes']:,} / {coverage['historical_scenes']:,} complete questions", loc="left")
        ax.set_xlabel("← More chaotic     Δ order     More lawful →")
        ax.set_ylabel("← Lower goodness     Δ morality     Higher goodness →")
    fig.supxlabel("Shifts from DeepSeek, not absolute alignments. Native axes run −5…+5; this view is zoomed.\n"
                  "Only questions with both axes graded for EVERY offered action; missing labels are not zero.\n"
                  "Crosses = paired 95% intervals on each axis, not a joint region. Raw positions are in the table. Labels/replies fixed.  PI/Astra", fontsize=9)
    save(fig, out, "dnd")
    figures.append("dnd")

    group_path = ROOT / "rubrics/virtue_display_groups.json"
    groups = json.loads(group_path.read_text())["groups"]
    virtue_ids = [d["id"] for d in dimensions if d["family"] == "virtues"]
    positive_ids = {d["id"] for d in direct if d["kind"] == "virtues"}
    members = [item for items in groups.values() for item in items]
    assert len(members) == len(set(members)) and set(members) <= positive_ids
    boot = np.load(path.parent / "bootstrap_means.npz")
    fig = plt.figure(figsize=(12, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    fig.suptitle("Related virtues · exploratory groups, not traditions", fontsize=16, x=.02, ha="left")
    key_legend(fig, list(COLORS))
    cluster_rows = []
    angle = np.linspace(0, 2 * np.pi, len(groups), endpoint=False)
    for column, source in enumerate(SOURCES):
        radar = fig.add_subplot(grid[0, column], projection="polar")
        radar.set_theta_offset(np.pi / 2)
        radar.set_theta_direction(-1)
        radar.set_xticks(angle, list(groups), fontsize=10)
        radar.set_ylim(0, .35)
        radar.set_yticks([.1, .2, .3], [".1", ".2", ".3"], fontsize=8, color=".4")
        radar.set_rlabel_position(100)
        radar.grid(color=".85", linewidth=.6)
        radar.spines["polar"].set_visible(False)
        radar.set_title(SOURCES[source] + "\nScore above uniform choice", pad=24, fontsize=12)
        paired = {m: [] for m in ("Gemma", "Haiku")}
        for m, model in enumerate(COLORS):
            means = []
            for group, items in groups.items():
                ix = [virtue_ids.index(item) for item in items]
                raw = np.mean([index[source, "above_uniform", model, item]["mean"] for item in items])
                assert 0 <= raw <= .35
                means.append(raw)
                delta = np.mean([index[source, "versus_deepseek", model, item]["mean"] for item in items])
                samples = boot[source + ":virtues:versus_deepseek"][:, m, ix].mean(axis=1)
                ci = np.quantile(samples, [.025, .975]).tolist()
                row = {"source": source, "model": model, "group": group, "mean": float(delta), "above_uniform": float(raw), "ci95": ci}
                cluster_rows.append(row)
                if model != "DeepSeek":
                    paired[model].append(row)
            radar.plot(np.r_[angle, angle[0]], means + means[:1], marker=MARKERS[model], color=COLORS[model], linewidth=1.2, markersize=4)
        ax = fig.add_subplot(grid[1, column])
        bars(ax, paired)
        ax.set_yticks(range(len(groups)), list(groups))
        ax.set_xlim(-vlim, vlim)
        ax.set_xlabel("Same groups: label score minus DeepSeek")
    fig.supxlabel(f"{len(members)} of 63 directly judged virtues grouped by meaning; complete membership is listed below. Not a fitted or validated taxonomy.\n"
                  "Top: unscaled above-uniform scores, no filled areas. Bottom: paired 95% mean intervals.\n"
                  "Items receive equal weight within each group; labels/replies fixed. See all individual items before interpreting a group.  PI/Astra", fontsize=9)
    save(fig, out, "virtue_clusters")
    figures.append("virtue_clusters")
    (out / "cluster_scores.json").write_text(json.dumps({"definition_sha256": hashlib.sha256(group_path.read_bytes()).hexdigest(), "groups": groups, "rows": cluster_rows}, indent=2) + "\n")

    rows = []
    for source in SOURCES:
        for model in COLORS:
            r = [index[source, "raw", model, axis] for axis in ("order", "morality")]
            rows.append(f"<tr><td>{SOURCES[source]}</td><td>{model}</td><td>{r[0]['mean']:+.3f}</td><td>{r[1]['mean']:+.3f}</td><td>{r[0]['scenes']:,}</td></tr>")
    membership = "".join(f"<details><summary>{escape(name)} ({len(items)})</summary><ul>" + "".join(f"<li>{escape(item.replace('|', ' / '))}</li>" for item in items) + "</ul></details>" for name, items in groups.items())
    def image(name):
        return f'<a href="{name}.svg"><img src="{name}.png" alt="{name.replace("_", " ")}"></a>'
    source_counts = ", ".join(f"{data['sources'][s]['historical_scenes']:,} {SOURCES[s].lower()}" for s in SOURCES)
    ranks = " ".join(f"{SOURCES[s]}: " + "; ".join(f"{n} {escape(rank.replace('>', ' > '))}" for rank, n in data['sources'][s]['individual_virtue_rank_counts'].items()) + "." for s in SOURCES)
    factors = "; ".join(f"{SOURCES[s].lower()}: {data['sources'][s]['pc_variance_fractions'][0]:.1%}" for s in SOURCES)
    diagnostic_link = '<a href="diagnostics.html">Read the controlled checks and complete examples.</a>' if (out / "diagnostics.html").exists() else ''
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Choice profiles: behaviours, virtues, D&D</title>
<style>body{font:17px/1.5 system-ui,sans-serif;max-width:1500px;margin:auto;padding:24px;color:#222}h1,h2{font-weight:600}nav{display:flex;gap:24px}img{display:block;width:100%;height:auto}p{max-width:1050px}table{border-collapse:collapse}td,th{padding:7px 14px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}details{margin:12px 0}a{color:#246891}.note{color:#555;font-size:15px}</style>
<h1>Choice profiles</h1><nav><a href="#behaviours">Behaviours</a><a href="#dnd">D&D</a><a href="#virtues">Virtues</a><a href="#audit">Checks and limits</a></nav>
<p>These plots use the saved choices of DeepSeek, Gemma and Haiku: """ + source_counts + """, two option orders each. Rebuilding the plots needs no inference. They retain the original role-play prompts and historical pool.</p>
<p>Known judge misreadings and scenario ambiguity limit what these profiles measure. Treat them as summaries of saved labels, not validated measurements of distinct model virtues. """ + diagnostic_link + """</p>
<p>Zero in a comparison plot means DeepSeek. Lines show uncertainty in the paired mean, not the spread of individual choices. They omit label error and fresh-generation variability. Click a plot for its SVG.</p>
<h2 id="behaviours">Behaviours</h2><p>Read each row in its named direction. More violence, loyalty or ambition is not automatically better or worse. Values are mean judge probabilities, not counted events.</p>""" + image("behaviours") + """
<h2 id="dnd">D&D</h2><p>Horizontal movement compares order; vertical movement compares goodness. Each source has its own DeepSeek reference at zero. These are shifts in mean labels, not absolute character alignments. The native grades are ordinal: the plotted means assume equal-sized steps.</p>""" + image("dnd") + "<details open><summary>Raw D&D means (native −5…+5 axes; same complete cohort)</summary><table><tr><th>Source</th><th>Model</th><th>Order → lawful</th><th>Morality → good</th><th>Questions</th></tr>" + "".join(rows) + "</table></details>" + """
<h2 id="virtues">Virtues</h2><p>The spider view groups related virtues rather than averaging whole traditions. The groups are descriptive, not equivalent teachings. The paired plots below it give more precise comparisons.</p>""" + image("virtue_clusters") + membership + """
<h3>Every directly judged item</h3><p>Model-order counts across the 63 virtues (before tradition averaging): """ + ranks + """ These counts describe point estimates, not significant pairwise differences.</p>""" + image("virtues_machiavelli") + image("virtues_daily_dilemmas") + '<details><summary>14 derived vices (not independent judgements)</summary>' + image("derived_vices") + "</details>" + """
<h2 id="audit">Checks and limits</h2><p>Every score is joined by source ID and original choice ID; exact scenario and action text match the saved labels. Missing D&D grades remain missing: a question is used only if every offered action has both axes graded.</p>
<p>After subtracting each question's action mean, the largest covariance component explains the following fraction of variation across the 66 directly judged labels: """ + factors + """. This concerns the labels across offered actions, not bits of model identity. Shared moral content, judge bias and question design remain possible causes; this calculation does not decide among them.</p>
<p>The labeller uses one teaching with up to 40 separately scored choices in an HTTP packet. Its documented question isolation differs from an ordinary model generating 40 answers sequentially. Saved replies contain probabilities and input diagnostics, not judge chain of thought. Passing an input diagnostic does not validate a returned grade.</p>
<p class="note">""" + f"{data['replicates']:,}" + """ paired game/topic-cluster resamples; pointwise 95% intervals. No source pooling, model-dependent centering or variance scaling. The profile figures use historical replies; follow-up diagnostics are separate runs. Figures and analysis: PI/Astra.</p></html>"""
    (out / "index.html").write_text(page)
    metadata = {"author": "PI/Astra", "analysis_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "figures": {name: hashlib.sha256((out / f"{name}.png").read_bytes()).hexdigest() for name in figures}, "clipping_check": "passed", "independent_review": "pending"}
    (out / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"PASS: {len(figures)} figures; 26 behaviours, 66 direct and 14 derived items, 2 D&D axes; clipping checks passed.")
    print(out / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ROOT / "outputs/20261025_individual_dimensions/analysis.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/20261025_individual_dimensions")
    args = parser.parse_args()
    render(args.analysis, args.out)
