"""Frozen tiny-virtues five-level rubric applied to proposed actions. -- PI/Sol"""

import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUBRIC_PATH = ROOT / "rubrics/virtues_v2.json"
PROPOSED_SCOPE = (
    "Judge only the selected PROPOSED choice, before it has happened. The agent is the named player character. "
    "Use the exact visible story to interpret what choosing this action would display under the teaching. "
    "Do not score earlier acts, other characters, unchosen alternatives, hidden outcomes or a generated answer. "
    "Do not assume the proposed act succeeds or invent its consequences."
)


def load_rubric(path: Path = RUBRIC_PATH) -> dict:
    rubric = json.loads(path.read_text())
    assert len(rubric["items"]) == 66 and len(rubric["contraries"]) == 14
    direct = {item["item_id"] for item in rubric["items"]}
    derived = {pair["vice_id"] for pair in rubric["contraries"]}
    assert len(direct) == 66 and len(derived) == 14 and not direct & derived
    assert all(pair["virtue_id"] in direct for pair in rubric["contraries"])
    assert rubric["model"] == "typesafe/jev-1.14"
    return rubric


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def question_id(source_id: str, choice_id: str) -> str:
    return "s" + hashlib.sha256(json.dumps([source_id, choice_id]).encode()).hexdigest()[:16] + "_level"


def choices(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        for key, action in row["choice_text"].items():
            assert isinstance(action, str) and action.strip()
            result.append({"source_id": row["source_id"], "choice_id": key, "action": action,
                           "agent": row["player_character"], "story": row["text"]})
    assert len({(row["source_id"], row["choice_id"]) for row in result}) == len(result)
    return result


def request(rubric: dict, item: dict, rows: list[dict], *, full_feedback: bool) -> dict:
    assert 1 <= len(rows) <= 40
    template = rubric["template"]
    form = template["forms"][item["kind"]]["forward"]
    tradition, _, name = item["item_id"].split("|")
    question = {"tradition": tradition, "list": item["list_name"], "item": name, **form["question"],
                **{key: item[key] for key in ("applicability", "source_context") if key in item}}
    state = {"layout_scope": template["layout_scope"],
             "judging_task": template["judging_task"] + " " + PROPOSED_SCOPE,
             "source_led_teaching": item["teaching"]}
    questions = {}
    review = {}
    for row in rows:
        assert row["agent"].strip() and row["story"].strip() and row["action"].strip()
        key = question_id(row["source_id"], row["choice_id"])
        assert key not in questions
        questions[key] = {"type": "score", "instructions": {
            "question": copy.deepcopy(question), "agent": row["agent"], "story": row["story"],
            "selected_act_quote": row["action"], "target_scope": PROPOSED_SCOPE,
        }, "criteria": copy.deepcopy(form["criteria"])}
        review[key] = copy.deepcopy(questions[key]) if full_feedback else {
            "id": key, "agent": row["agent"], "selected_act_quote": row["action"]}
    feedback = rubric["feedback"]
    words = feedback["words"]
    assert len(words) <= 252 and len(words) == len(set(words)) and not {"none", "idk", "review_this"} & set(words)
    questions["debug_word"] = {
        "type": "choice", "instructions": {"task": feedback["hybrid_task"], "questions_to_review": review},
        "criteria": {
            "none": "Nothing apparent needs attention; this is not a guarantee the setup is correct.",
            "idk": "Cannot determine whether this setup is adequate from the supplied information.",
            "review_this": "Something warrants checking, but no other option names it adequately.",
        } | dict.fromkeys(words, ""),
    }
    return {"model": rubric["model"], "state": state, "questions": questions,
            "provider": {"only": ["typesafe"], "allow_fallbacks": False}}


def parse_answers(body: dict, payload: dict, item: dict, rows: list[dict]) -> list[dict]:
    assert set(body["answers"]) == set(payload["questions"]), "Missing or extra answers"
    results = []
    for row in rows:
        key = question_id(row["source_id"], row["choice_id"])
        answer = body["answers"][key]
        assert answer["type"] == "score"
        raw = answer["probabilities"]
        assert set(raw) == {str(i) for i in range(5)}
        vector = [raw[str(i)] for i in range(5)]
        assert all(type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1 for p in vector)
        mass = sum(vector)
        assert math.isclose(mass, 1, abs_tol=.02 + 1e-8), "Five-level probability mass outside rounding tolerance"
        normalized = [p / mass for p in vector]
        criteria = payload["questions"][key]["criteria"]
        assert answer["legend"] == {str(i): text for i, text in enumerate(criteria)}
        assert type(answer["score"]) in (float, int) and math.isfinite(answer["score"]) and 0 <= answer["score"] <= 4
        favorable = normalized[4] + .5 * normalized[3]
        adverse = normalized[0] + .5 * normalized[1]
        shows, against = (favorable, adverse) if item["kind"] == "virtues" else (adverse, favorable)
        signed = sum(i * p for i, p in enumerate(normalized)) / 2 - 1
        results.append({"source_id": row["source_id"], "choice_id": row["choice_id"], "item_id": item["item_id"],
                        "shows": shows, "against": against, "signed": signed, "vice_presence": -signed,
                        "raw_probabilities": raw, "normalized_probabilities": normalized, "raw_probability_mass": mass,
                        "provider_score": answer["score"], "criteria": criteria,
                        "signed_definition": "sum(i * normalized_probability[i]) / 2 - 1",
                        "evidence_semantics": "Weighted evidence, not calibrated truth or an independent yes/no probability."})
    feedback = body["answers"]["debug_word"]
    assert feedback["type"] == "choice" and feedback["choice"] in payload["questions"]["debug_word"]["criteria"]
    probabilities = feedback["probabilities"]
    assert set(probabilities) == set(payload["questions"]["debug_word"]["criteria"])
    assert all(type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
    assert math.isclose(sum(probabilities.values()), 1, abs_tol=.02 + 1e-8)
    return results


def derive_contrary(parent: dict, vice_id: str) -> dict:
    return {"source_id": parent["source_id"], "choice_id": parent["choice_id"], "item_id": vice_id,
            "shows": parent["against"], "against": parent["shows"], "signed": parent["signed"],
            "vice_presence": -parent["signed"], "derived_from": parent["item_id"],
            "raw_probabilities": parent["raw_probabilities"],
            "evidence_semantics": "Source-named contrary; not an independent judgement. Raw probabilities belong to derived_from."}
