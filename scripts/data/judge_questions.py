"""Visible-action questions; original outcome annotations never enter the request. PI/OpenAI."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUBRIC = json.loads((ROOT / "rubrics/actions.json").read_text())
MODEL = "typesafe/jev-1.13"
TASK = RUBRIC["task"]


def questions(*, virtues: bool = True) -> dict:
    return {key: value for key, value in RUBRIC["questions"].items() if virtues or not key.startswith("stoic|")}


def request(row: dict, choice_id: str, *, virtues: bool = True) -> dict:
    scored = questions(virtues=virtues)
    words = [line for line in (ROOT / "data/judges/debug_words.txt").read_text().splitlines() if line and not line.startswith("#")]
    assert len(words) == len(set(words)) and len(words) <= 253
    assert not {"none", "idk"} & set(words)
    scored["debug_word"] = {
        "type": "choice",
        "instructions": "Unscored judge feedback, never rewarded or penalized. Which option best names an obstacle to judging this proposed action reliably? Choose none if no obstacle is apparent, idk if no listed word fits. The following are the scored questions, supplied explicitly because Jev questions are otherwise isolated. Do not judge whether the action is morally good or bad here.\n" + json.dumps(scored, ensure_ascii=False),
        "criteria": {"none": "No apparent obstacle.", "idk": "An obstacle exists but no listed word fits."} | dict.fromkeys(words, ""),
    }
    state = {"task": TASK, "scenario_text": row["text"], "chosen_action": row["choice_text"][choice_id]}
    assert set(state) == {"task", "scenario_text", "chosen_action"}
    return {"model": MODEL, "state": state, "questions": scored}


def identity(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def interpret(answers: dict) -> dict:
    values = {key: answer["noul"] for key, answer in answers.items() if key in RUBRIC["dimensions"]}
    issues = {key.removeprefix("issue|"): answer["noul"] for key, answer in answers.items() if key.startswith("issue|")}
    virtues = {}
    for key, answer in answers.items():
        if key.startswith("stoic|"):
            _, item_list, side, name, direction = key.split("|")
            virtues.setdefault(name, {"list": item_list, "kind": side[:-1]})[direction] = answer["noul"]
    for item in virtues.values():
        item["signed_approval"] = item["shows"] - item["against"] if item["kind"] == "virtue" else item["against"] - item["shows"]
    alignment = {}
    for axis in ("order", "morality"):
        options = RUBRIC["questions"][f"dnd|{axis}|order0"]["criteria"]
        by_order = [answers[f"dnd|{axis}|order{order}"]["probabilities"] for order in range(2)]
        alignment[axis] = {"probabilities": {key: sum(order[key] for order in by_order) / 2 for key in options}, "order_discrepancy": max(abs(by_order[0][key] - by_order[1][key]) for key in options), "by_order": by_order}
    return {"visible_action_values": values, "stoic_virtues_vices": virtues, "dnd_action_alignment": alignment, "issues": issues, "judge_feedback": answers["debug_word"]}
