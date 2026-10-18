"""Audit the published prompts against their pinned source rows. PI/OpenAI."""

import argparse
import json
import random
import re
import shlex
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from tabulate import tabulate

import summarise_machiavelli as generation
from dataset_io import PROVENANCE, ROOT, published_rows, sample_by_game, source_rows, write_jsonl


def check_row(row: dict, source: dict) -> dict:
    rebuilt = generation._scenario_text(source, row["decision_context"], row["final_question"])
    issues = []
    if rebuilt != row["text"]:
        issues.append("prompt_roundtrip_mismatch")
    if row["choice_labels"] != source["choice_labels"]:
        issues.append("source_labels_mismatch")
    if row["short_summary"] != generation._premise(source):
        issues.append("source_premise_mismatch")
    if row["player_character"] != generation._protagonist(source):
        issues.append("source_character_mismatch")
    if row["actions_text"] != generation._actions_text(source["choice_text"]):
        issues.append("source_actions_mismatch")
    try:
        generation._validate_scenario(source, row["text"], row["decision_context"])
    except ValueError as error:
        issues.append(f"structural_validation: {error}")
    variables = sorted(set(generation._VAR.findall("\n".join(text for text in source["choice_text"].values() if text))))
    literal_variables = []
    for name in variables:
        cleaned = generation._clean("${" + name + "}", variable_text="someone")
        if cleaned not in {"someone", *generation._VAR_SUFFIX_WORDS.values(), "run", "walk", "move", "flee", "leave"} and cleaned in row["actions_text"]:
            literal_variables.append({"variable": name, "rendered": cleaned})
    if literal_variables:
        issues.append("unresolved_choice_variable_candidate")
    history_text = "\n".join([source["obs"], *source["early_history"], *source["recent_history"]])
    return {
        "source_id": row["source_id"], "game_id": row["game_id"], "issues": issues,
        "choice_variables": variables, "literal_variables": literal_variables,
        "metadata_character_absent_from_source_window": source["player_character"].casefold() not in history_text.casefold(),
        "early_history_scenes": len(source["early_history"]), "recent_history_scenes": len(source["recent_history"]),
        "summary_words": len(row["decision_context"].split()),
        "input_tokens_chars_over_four": (len(generation.SYSTEM) + len(generation._context(source))) / 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Audit only this many published rows; omit for all rows")
    parser.add_argument("--per-game", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--out", type=Path, default=Path("outputs") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_audit"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    run_log = args.out / "run.md"
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{message}\n", colorize=False)
    logger.add(run_log, level="DEBUG", format="{message}\n", colorize=False)
    started = time.monotonic()
    config = vars(args) | {"out": str(args.out), "provenance": PROVENANCE}
    metadata = {"argv": sys.argv, "config": config, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("# Published dataset audit\n\nCommand:\n\n```text\n" + shlex.join(sys.argv) + "\n```\n\n## Resolved config\n\n```json\n" + json.dumps(config, indent=2) + "\n```")
    try:
        logger.info("## Audit -> load pinned published and source data")
        rows = published_rows()
        if args.limit is not None:
            assert args.limit > 0
            rows = rows[:args.limit]
        source = source_rows(rows)
        logger.info(f"Matched {len(source)}/{len(rows)} source IDs at the pinned revision.")
        checks = [check_row(row, source[row["source_id"]]) for row in rows]
        write_jsonl(args.out / "checks.jsonl", checks)
        sample = sample_by_game(rows, args.per_game, args.seed)
        write_jsonl(args.out / "sample.jsonl", [{"published": row, "source": source[row["source_id"]]} for row in sample])
        logger.info(f"## Audit -> review sample\n\nSeed {args.seed}; {args.per_game} row(s) per game, uniformly sampled within each game. {len(sample)} sampled rows across {len({r['game_id'] for r in sample})} games. This is game-balanced, not a row-weighted estimate of defect prevalence. Full inputs: `{args.out / 'sample.jsonl'}`.")
        logger.debug("## Audit -> first full sample\n\n```json\n" + json.dumps({"published": sample[0], "source": source[sample[0]["source_id"]]}, ensure_ascii=False, indent=2) + "\n```")
        counts = Counter(issue for check in checks for issue in check["issues"])
        tokens = sum(check["input_tokens_chars_over_four"] for check in checks)
        summary = {
            "rows": len(rows), "games": len({r["game_id"] for r in rows}), "matched_source_rows": len(source),
            "issue_counts": dict(counts), "duplicate_text_rows": len(rows) - len({r["text"] for r in rows}),
            "rows_with_choice_variables": sum(bool(r["choice_variables"]) for r in checks),
            "metadata_character_absent_from_source_window": sum(r["metadata_character_absent_from_source_window"] for r in checks),
            "early_history_scene_counts": dict(Counter(r["early_history_scenes"] for r in checks)),
            "recent_history_scene_counts": dict(Counter(r["recent_history_scenes"] for r in checks)),
            "input_million_tokens_chars_over_four": tokens / 1e6,
            "visible_output_million_tokens_chars_over_four": sum(len(r["decision_context"]) for r in rows) / 4e6,
            "sample_source_ids": [r["source_id"] for r in sample], "seed": args.seed,
            "caveat": "Mechanical audit only. Flags are review candidates, not semantic failures. No judging, relabelling or paid API calls.",
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        logger.info(f"## Audit -> finished\n\nElapsed: {time.monotonic() - started:.1f}s. CPU only; no GPU used.\n\nTODO validate: source-aligned formatting does not establish summary fidelity or decision sufficiency. Candidate variable and character flags need source reading.")
        table_rows = [[name, count, len(rows)] for name, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
        if not table_rows:
            table_rows = [["mechanical issue candidates", 0, len(rows)]]
        table = tabulate(table_rows, headers=["check", "flagged rows↓", "audited rows"], tablefmt="pipe")
        logger.info(f"## Result\n\nCompleted source-aligned audit: {len(rows)} prompts, {summary['games']} games. Semantic quality remains unmeasured.\n\n{table}\n\nFull checks: `{args.out / 'checks.jsonl'}`. Full game-balanced sample: `{args.out / 'sample.jsonl'}`. Counts: `{args.out / 'summary.json'}`.\n\n{run_log}")
    except Exception:
        logger.error("## Audit failed\n\n```text\n" + traceback.format_exc() + f"\n```\n\n{run_log}")
        raise


if __name__ == "__main__":
    main()
