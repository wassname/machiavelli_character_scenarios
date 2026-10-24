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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("docs/figures/virtue_profiles_3860.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/figures/virtue_profiles_3860.png"))
    args = parser.parse_args()
    plot(json.loads(args.data.read_text()), args.out)
