# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# [tool.uv]
# exclude-newer = "8 days"
# ///
"""Plot the saved full-cohort virtue estimates without inference. -- PI/OpenAI"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot(data: dict, out: Path) -> None:
    rows = data["rows"]
    traditions = sorted({r["tradition"] for r in rows})
    names = ["DeepSeek", "Gemma", "Haiku"]
    colours = ["#b64b35", "#246891", "#39744e"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.8), sharey=True, constrained_layout=True)
    fig.suptitle(f'Virtue profiles: {data["scenes"]:,} scenes, three models', fontsize=15)
    for i, (name, colour) in enumerate(zip(names, colours, strict=True)):
        indexed = {r["tradition"]: r for r in rows if r["name"] == name}
        assert indexed.keys() == set(traditions)
        y = [j + (i - 1) * .19 for j in range(len(traditions))]
        for ax, metric, interval in zip(axes, ("delta_from_uniform", "opportunity_capture"),
                                        ("delta_ci95", "capture_ci95"), strict=True):
            values = [indexed[t][metric] for t in traditions]
            lo = [indexed[t][interval][0] for t in traditions]
            hi = [indexed[t][interval][1] for t in traditions]
            ax.errorbar(values, y, xerr=[[v - l for v, l in zip(values, lo, strict=True)],
                                        [h - v for h, v in zip(hi, values, strict=True)]],
                        fmt="o", color=colour, label=name, markersize=5, linewidth=1.1, capsize=2)
    axes[0].set_yticks(range(len(traditions)), [t.capitalize() for t in traditions])
    axes[0].invert_yaxis()
    axes[0].set_title("Above uniform choice", loc="left", fontsize=11)
    axes[0].set_xlabel("Chosen-action score − mean available score")
    axes[1].set_title("Share of available improvement", loc="left", fontsize=11)
    axes[1].set_xlabel("0 = uniform choice; 1 = best available score")
    axes[0].set_xlim(left=0)
    axes[1].set_xlim(0, 1)
    axes[0].legend(loc="upper left", frameon=False)
    for ax in axes:
        ax.grid(axis="x", color="0.91", linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=6)
        ax.spines[["top", "right", "left"]].set_visible(False)
    counts = data["source_counts"]
    fig.text(.02, -.025,
             f'{counts["machiavelli"]:,} game + {counts["daily_dilemmas"]:,} everyday scenes; equal source weight; two orders averaged; 66 direct items.\n'
             "Lines: 95% game/topic-grouped bootstrap intervals; exclude label and source-validity uncertainty.\n"
             "Original role-play protocol and pool, before next-state filtering; not intrinsic moral identities.  — PI/OpenAI",
             fontsize=9, color="0.3", va="top")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_relative(data: dict, out: Path, level: str) -> None:
    plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Serif CJK JP"]
    rows = [r for r in data["rows"] if r["kind"] == level]
    keys = sorted({r["item_id"] for r in rows})
    height = max(5.8, len(keys) * .34 + 1.5)
    fig, ax = plt.subplots(figsize=(9, height), constrained_layout=True)
    fig.suptitle(f'Relative virtue profiles: {data["scenes"]:,} scenes', fontsize=15)
    ax.axvline(100, color="0.45", linewidth=1, linestyle="--", label="DeepSeek v4.1 Flash = 100%")
    for shift, name, colour in [(-.12, "Gemma", "#246891"), (.12, "Haiku", "#39744e")]:
        indexed = {r["item_id"]: r for r in rows if r["name"] == name}
        valid = [(i, indexed[k]) for i, k in enumerate(keys) if indexed[k]["relative_percent"] is not None]
        y = [i + shift for i, r in valid]
        ax.hlines(y, [r["relative_ci95"][0] for i, r in valid], [r["relative_ci95"][1] for i, r in valid], color=colour, linewidth=1.2)
        ax.plot([r["relative_percent"] for i, r in valid], y, "o", color=colour, label={"Gemma": "Gemma 4 31B", "Haiku": "Haiku 5.5"}[name], markersize=5)
        for i, key in enumerate(keys):
            if indexed[key]["relative_percent"] is None:
                ax.text(.99, i + shift, f"{name}: unstable baseline", transform=ax.get_yaxis_transform(), ha="right", fontsize=7, color=colour)
    vice_labels = {f"islamic|quran_16_90|{name}": f"Resistance to {name}" for name in ("all shameful deeds", "injustice", "rebellion")}
    labels = [k.capitalize() if level == "tradition" else f'{vice_labels[k] if k in vice_labels else k.split("|")[-1]} ({k.split("|")[0]})' for k in keys]
    ax.set_yticks(range(len(keys)), labels)
    ax.set_ylim(len(keys) - .5, -.5)
    if level == "virtue":
        ax.tick_params(axis="x", labeltop=True, top=True)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Above-uniform label score, relative to DeepSeek Flash")
    ax.grid(axis="x", color="0.91", linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.01), ncol=3, frameon=False, fontsize=9)
    fig.supxlabel(f'Same {data["scenes"]:,} scenes; equal source weight; both orders averaged. Lines: pointwise 95% game/topic bootstrap.\n'
                  "Actor/label uncertainty excluded; vice rows show resistance. Original protocol. — PI/OpenAI", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("docs/figures/virtue_profiles_3860.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/virtue_profiles_3860.png"))
    parser.add_argument("--relative", action="store_true", help="DeepSeek=100% using jointly resampled above-uniform ratios")
    parser.add_argument("--level", choices=["tradition", "virtue"], default="tradition")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    if args.relative:
        plot_relative(data, args.out, args.level)
    else:
        plot(data, args.out)
