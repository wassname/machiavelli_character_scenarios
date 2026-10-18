"""Restore displayed choices without changing summaries or reading outcomes. PI/OpenAI."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from dataset_io import PROVENANCE, published_rows, source_rows, write_jsonl

INPUT_REVISION = "rendered-choices-v3"
MARKUP = re.compile(r"\[/?(?:b|i|u|s|strong|em)\]", re.I)
OPTION = re.compile(r"(?ms)^(\d+):[ \t]*(.*?)(?=^\d+:[ \t]|\Z)")
UNDO_MENU = re.compile(r"\bundo\b.{0,35}\bdecision\b", re.I)
ORIGINAL_FIELDS = ("text", "actions_text", "choice_text", "decision_context", "schema_version")


def observation_body(text: str) -> str:
    return MARKUP.sub("", re.split(r"(?m)^0:[ \t]", text)[0]).strip()


def rendered_choices(observation: str) -> dict[str, str]:
    starts = list(re.finditer(r"(?m)^0:[ \t]", observation))
    assert starts, "No displayed choice block"
    items = OPTION.findall(observation[starts[-1].start():])
    assert [key for key, _ in items] == [str(i) for i in range(len(items))], "Noncontiguous displayed IDs"
    choices = {key: " ".join(MARKUP.sub("", text).split()) for key, text in items}
    assert choices and all(choices.values()), "Empty displayed choice"
    assert all("${" not in text and "@{" not in text for text in choices.values()), "Unresolved source choice code"
    return choices


def repair(row: dict, original: dict) -> dict:
    choices = rendered_choices(original["obs"])
    actions = "\n".join("- " + text for text in choices.values())
    before = "Actions to choose:\n" + row["actions_text"] + "\n\n"
    assert row["text"].count(before) == 1
    text = row["text"].replace(before, "Actions to choose:\n" + actions + "\n\n")
    recent = original["recent_history"]
    interface = choices["0"] == "Very well. I'll just have to try harder next time." and any(UNDO_MENU.search(turn) for turn in [original["obs"], *recent])
    context = row["decision_context"]
    if interface:
        previous = "\n\n".join(observation_body(turn) for turn in recent)
        context = "You are discussing undoing a past decision with the game's narrator. These exchanges have already happened.\n\nRecent dialogue:\n" + previous + "\n\nCurrent reply:\n" + observation_body(original["obs"])
        scene = "Current scene:\n" + row["decision_context"] + "\n\nActions to choose:"
        assert text.count(scene) == 1
        text = text.replace(scene, "Current scene:\n" + context + "\n\nActions to choose:")
    changes = {key: {"before": row["choice_text"].get(key), "after": value} for key, value in choices.items() if row["choice_text"].get(key) != value}
    assert not {key for key, value in row["choice_text"].items() if value} - choices.keys(), "Published choice missing from source observation"
    return row | {
        "text": text, "actions_text": actions, "choice_text": choices,
        "schema_version": 16, "input_revision": INPUT_REVISION, "decision_context": context,
        "original_prompt": {key: row[key] for key in ORIGINAL_FIELDS},
        "input_repairs": changes, "input_context_repaired": interface,
        "input_issues": (["game_interface"] if interface else []) + (["source_user_input_placeholder"] if any("user_input" in value for value in choices.values()) else []),
    }


def verify_inputs(rows: list[dict], originals: list[dict], sources: dict[str, dict]) -> None:
    assert [row["source_id"] for row in rows] == [row["source_id"] for row in originals]
    changed_fields = set(ORIGINAL_FIELDS)
    new_fields = {"input_revision", "original_prompt", "input_repairs", "input_context_repaired", "input_issues"}
    for row, original in zip(rows, originals):
        assert row["input_revision"] == INPUT_REVISION, "Verify with the matching input-repair version"
        assert set(row) == set(original) | new_fields
        assert all(row[key] == original[key] for key in original.keys() - changed_fields)
        assert row["original_prompt"] == {key: original[key] for key in changed_fields}
        assert row == repair(original, sources[row["source_id"]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    originals = published_rows()
    sources = source_rows(originals)
    rows = [repair(row, sources[row["source_id"]]) for row in originals]
    verify_inputs(rows, originals, sources)
    path = args.out / "inputs.jsonl"
    write_jsonl(path, rows)
    modified = [row for row in rows if row["text"] != row["original_prompt"]["text"]]
    added = [(row["source_id"], key) for row in rows for key in row["choice_text"] if not row["original_prompt"]["choice_text"].get(key)]
    summary = {"input_revision": INPUT_REVISION, "scenarios": len(rows), "actions": sum(len(row["choice_text"]) for row in rows), "changed_prompt_rows": len(modified), "changed_choice_texts": sum(len(row["input_repairs"]) for row in rows), "restored_choices": len(added), "restored_ids_beyond_nine": sum(int(key) >= 10 for _, key in added), "source_placeholder_rows": sum("source_user_input_placeholder" in row["input_issues"] for row in rows), "interface_context_repaired_rows": sum(row["input_context_repaired"] for row in rows), "original_annotations_unchanged": True, "all_other_summaries_unchanged": True, "only_current_observation_choices_used": True, "api_calls": 0, "inputs_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "provenance": PROVENANCE}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
