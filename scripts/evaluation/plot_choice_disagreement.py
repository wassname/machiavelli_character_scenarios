# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# [tool.uv]
# exclude-newer = "8 days"
# ///
"""Plot saved choice disagreement and the separate repeated-call pilot. -- PI/OpenAI"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {
    "deepseek/deepseek-v4.1-flash": "DeepSeek",
    "google/gemma-4-31b-it": "Gemma",
    "anthropic/claude-haiku-5.5": "Haiku",
}


def collect(coverage: Path, repeat_analysis: Path) -> dict:
    full = json.loads(coverage.read_text())
    pilot = json.loads(repeat_analysis.read_text())
    controls = {frozenset((r["a"][1], r["b"][1])): r for r in pilot["results"]
                if r["kind"] == "models" and r["cohort"] == "equal_sources"
                and r["a"][0] == r["b"][0] == "baseline"}
    rows = []
    for r in full:
        pair = frozenset((r["model_a"], r["model_b"]))
        if not pair <= NAMES.keys():
            continue
        assert r["complete_scenes"] == r["planned_scenes"] == 3860
        assert r["matched_views"] == 2 * r["complete_scenes"]
        c = controls[pair]
        assert c["scenes"] == 64
        assert abs(c["between"] - c["within"] - c["between_minus_within"]) < 1e-12
        assert c["ci95"][0] <= c["between_minus_within"] <= c["ci95"][1]
        rows.append({"pair": " / ".join(sorted(NAMES[m] for m in pair)),
                     "full_scenes": r["complete_scenes"],
                     "full_disagreement": r["different_actions_matched_views"] / r["matched_views"],
                     "pilot_scenes": c["scenes"], "pilot_between": c["between"],
                     "pilot_within": c["within"], "pilot_difference": c["between_minus_within"],
                     "pilot_ci95": c["ci95"]})
    assert len(rows) == 3
    return {"author": "PI/OpenAI", "protocol": "Legacy character-role prompt; original order plus reversal, averaged per scene.",
            "scope": "Historical pool, before forward-looking-choice eligibility review. Separate cohorts; no moral quality ranking.",
            "full_source_counts": {s: r["complete_scenes"] for s, r in full[0]["by_source"].items()},
            "full_weighting": "Each scene equally weighted: 65% game scenes, 35% everyday. Pilot sources each receive 50% weight.",
            "pilot_source_counts": {"machiavelli": 32, "daily_dilemmas": 32},
            "pilot_estimand": pilot["estimand"],
            "pilot_uncertainty": "2,000 game/topic-grouped bootstrap resamples within source; each scene retains its correlated repeats. Assumes repeated calls approximate independent draws under recorded service settings. No labels or classifier scores enter.",
            "pilot_limits": "64 fixed-seed, source-balanced scenes, not an estimate for all 3,860. Choice distributions do not identify intrinsic moral values; interpretation and temporal service changes remain possible confounds.",
            "input_sha256": {"pair_coverage": hashlib.sha256(coverage.read_bytes()).hexdigest(),
                             "repeat_analysis": hashlib.sha256(repeat_analysis.read_bytes()).hexdigest()},
            "rows": sorted(rows, key=lambda r: -r["full_disagreement"])}


def plot(data: dict, path: Path) -> None:
    rows = data["rows"]
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True, constrained_layout=True)
    fig.suptitle("Choice disagreement: full pool and repeated-call pilot", fontsize=14)
    for ax in axes:
        ax.set_xlim(0, 40)
        ax.set_xticks(range(0, 41, 10))
        ax.set_ylim(len(rows) - .5, -.9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(length=0, pad=7)
        ax.grid(axis="x", color="0.9", linewidth=.7)
        ax.set_axisbelow(True)
    left, right = axes
    left.set_title("Original pool: 3,860 scenes", loc="left", pad=15)
    right.set_title("Repeated-call pilot: 64 scenes", loc="left", pad=15)
    left.set_yticks(range(len(rows)), [r["pair"] for r in rows])
    for y, row in enumerate(rows):
        left.plot(100 * row["full_disagreement"], y, "o", color="#246891", markersize=7)
        left.annotate(f'{100 * row["full_disagreement"]:.1f}%',
                      (100 * row["full_disagreement"], y), xytext=(9, 0),
                      textcoords="offset points", va="center", color="#246891")
        value = 100 * row["pilot_difference"]
        lo, hi = [100 * v for v in row["pilot_ci95"]]
        right.errorbar(value, y, xerr=[[value - lo], [hi - value]],
                       fmt="o", color="#246891", markersize=7, linewidth=1.5, capsize=3)
    left.set_xlabel("Different choices (%)\nEqual scene weight: 65% game / 35% everyday")
    right.set_xlabel("Difference beyond repeat noise (percentage points)\n95% game/topic-grouped bootstrap intervals")
    fig.text(.02, -.035,
             "Pilot: between-model disagreement minus mean within-model repeat disagreement; 3 replies per exact prompt/order.\n"
             "Pilot sources each receive 50% weight. It is a separate cohort, not a correction for the full pool.\n"
             "Legacy role-play protocol and source pool; not a validated test of intrinsic moral values.  — PI/OpenAI",
             fontsize=9, color="0.3", va="top")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path)
    parser.add_argument("--repeat-analysis", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/choice_disagreement.png"))
    args = parser.parse_args()
    if args.coverage is None and args.repeat_analysis is None:
        data = json.loads(args.out.with_suffix(".json").read_text())
    else:
        assert args.coverage is not None and args.repeat_analysis is not None
        data = collect(args.coverage, args.repeat_analysis)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
    plot(data, args.out)
