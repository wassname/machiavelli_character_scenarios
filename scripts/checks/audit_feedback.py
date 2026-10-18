"""Read every diagnostic result and build reproducible category spot checks. PI/OpenAI."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from tabulate import tabulate

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from dataset_io import published_rows, write_jsonl


def audit(labels: Path, rows: list[dict], out: Path, seed: int, previous: Path | None = None) -> dict:
    parents = {row["source_id"]: row for row in rows}
    groups = defaultdict(list)
    all_values = []
    issue_values = defaultdict(list)
    total_mass = Counter()
    nonmax = []
    with labels.open() as stream:
        for line_number, line in enumerate(stream, 1):
            item = json.loads(line)
            feedback = item["judge_feedback"]
            word = feedback["choice"]
            probabilities = feedback["probabilities"]
            assert word in probabilities
            assert abs(sum(probabilities.values()) - 1) <= 0.05 + 1e-8
            if probabilities[word] < max(probabilities.values()):
                nonmax.append(line_number)
            total_mass.update(probabilities)
            for key, value in item["issues"].items():
                issue_values[key].append(value)
            parent = parents[item["source_id"]]
            record = {"source_id": item["source_id"], "choice_id": item["choice_id"], "action": item["action"], "word": word, "word_probability": probabilities[word], "none_probability": probabilities["none"], "issues": item["issues"], "request_key": item["request_key"], "source": f"{labels}:{line_number}", "prompt_changed": parent["text"] != parent["original_prompt"]["text"] if "original_prompt" in parent else False}
            all_values.append(record)
            groups[word].append(record)
    assert len({(r["source_id"], r["choice_id"]) for r in all_values}) == len(all_values)
    expected = {(row["source_id"], key) for row in rows for key, value in row["choice_text"].items() if value}
    assert {(r["source_id"], r["choice_id"]) for r in all_values} == expected
    rng = random.Random(seed)
    samples = []
    for word, members in sorted(groups.items()):
        ordered = sorted(members, key=lambda r: (r["source_id"], int(r["choice_id"])))
        selected = rng.sample(ordered, min(3, len(ordered)))
        selected += [max(ordered, key=lambda r: r["word_probability"])]
        for record in {r["source"]: r for r in selected}.values():
            row = parents[record["source_id"]]
            samples.append(record | {"scenario_text": row["text"], "original_prompt": row.get("original_prompt"), "input_repairs": row.get("input_repairs", {}), "input_issues": row.get("input_issues", [])})
    write_jsonl(out / "all_feedback.jsonl", all_values)
    write_jsonl(out / "category_samples.jsonl", samples)
    counts = Counter(r["word"] for r in all_values)
    category_stats = {word: {"actions": len(members), "scenarios": len({r["source_id"] for r in members}), "mean_selected_probability": sum(r["word_probability"] for r in members) / len(members), "changed_prompt_actions": sum(r["prompt_changed"] for r in members)} for word, members in groups.items()}
    summary = {"actions_read": len(all_values), "scenarios": len(rows), "selected_word_counts": dict(counts.most_common()), "non_none_actions": sum(v for k, v in counts.items() if k != "none"), "non_none_scenarios": len({r["source_id"] for r in all_values if r["word"] != "none"}), "all_probability_categories_read": len(total_mass), "mean_probability_by_category": {k: v / len(all_values) for k, v in total_mass.most_common()}, "selected_not_argmax_lines": nonmax, "category_stats": category_stats, "mean_issue_probabilities": {k: sum(v) / len(v) for k, v in issue_values.items()}, "sample_count": len(samples), "sample_seed": seed, "labels_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(), "caveat": "Every diagnostic value and probability was parsed. Text is a single selected word, not a free-text explanation. Category samples are targeted plus seeded random, not a defect prevalence estimate."}
    if previous is not None:
        old = {(r["source_id"], r["choice_id"]): r["judge_feedback"]["choice"] for r in (json.loads(line) for line in previous.open())}
        current = {(r["source_id"], r["choice_id"]): r["word"] for r in all_values}
        assert old.keys() <= current.keys(), "A previously available choice was lost"
        common = old.keys() & current.keys()
        summary["paired_feedback"] = {"previous_labels": str(previous), "common_actions": len(common), "new_actions": len(current.keys() - old.keys()), "flag_to_none": sum(old[key] != "none" and current[key] == "none" for key in common), "none_to_flag": sum(old[key] == "none" and current[key] != "none" for key in common), "flag_to_flag": sum(old[key] != "none" and current[key] != "none" for key in common), "new_actions_flagged": sum(current[key] != "none" for key in current.keys() - old.keys()), "changed_prompt_actions": sum(r["prompt_changed"] for r in all_values)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    table = tabulate([[word, count, f"{100 * count / len(all_values):.3f}", category_stats[word]["scenarios"]] for word, count in counts.most_common()], headers=["selected diagnostic", "choices", "% of choices", "scenarios"], tablefmt="pipe")
    lines = [f"# Diagnostic audit\n\nAll {len(all_values)} selected words and all {len(total_mass)} probability entries per record were read by the audit. These are model flags, not confirmed defects.\n\n{table}\n\nSamples: three seeded random records and the most confident selected word per category, deduplicated. Source: `{labels}`.\n"]
    for record in samples:
        lines.append(f"## {record['word']}: {record['source_id']} / {record['choice_id']}\n\nSource: `{record['source']}`. Request: `{record['request_key']}`.\n\n```text\n{record['scenario_text']}\n```\n\nChosen action:\n\n```text\n{record['action']}\n```\n\nIssues:\n\n```json\n{json.dumps(record['issues'], indent=2)}\n```\n")
    lines.append(f"\n{out / 'run.md'}\n")
    (out / "run.md").write_text("\n".join(lines))
    print(table)
    print(out / "run.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--previous-labels", type=Path, help="Compare flags on shared choice IDs; additions counted separately")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261018)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = published_rows() if args.inputs is None else [json.loads(line) for line in args.inputs.open()]
    audit(args.labels, rows, args.out, args.seed, args.previous_labels)


if __name__ == "__main__":
    main()
