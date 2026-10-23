"""Check exact paired actions, source preservation and hidden-annotation blinding. -- PI/Sol"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from virtue_questions import ROOT, choices, load_rubric, request


def check(inputs: Path) -> None:
    rows = [json.loads(line) for line in inputs.open()]
    assert len(rows) == 1360 and len({row["source_id"] for row in rows}) == len(rows)
    rubric = load_rubric()
    for row in rows:
        assert row["source"] == "daily_dilemmas" and row["source_license"] == "cc-by-4.0" and row["player_character"] == "You"
        assert set(row["choice_text"]) == {"0", "1"}
        assert {annotation["action_type"] for annotation in row["source_annotations"].values()} == {"to_do", "not_to_do"}
        for key, annotation in row["source_annotations"].items():
            assert row["choice_text"][key] == annotation["action"]
            assert row["decision_context"] == annotation["dilemma_situation"]
        assert row["actions_text"] == "\n".join("- " + row["choice_text"][key] for key in ("0", "1"))
    original = request(rubric, rubric["items"][0], choices(rows[:1]), full_feedback=True)
    poisoned = rows[0] | {"source_annotations": {"negative_consequence": "HIDDEN_FUTURE_SENTINEL", "values_aggregated": "HIDDEN_VALUES_SENTINEL"}}
    assert request(rubric, rubric["items"][0], choices([poisoned]), full_feedback=True) == original
    assert "HIDDEN_" not in json.dumps(original)
    print("PASS: all1360scenes retain exact paired source actions and annotations, sourceCCBY4, agentYou; hidden consequences/values cannot affect judge requests. No API calls. -- PI/Sol")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    args = parser.parse_args()
    check(args.inputs)
