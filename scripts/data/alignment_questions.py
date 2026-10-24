"""Grade both D&D axes numerically; the API Score primitive permits only ten levels. -- PI/OpenAI"""

import hashlib
import json
import math
from pathlib import Path

from virtue_questions import ROOT, PROPOSED_SCOPE, question_id

RUBRIC_PATH = ROOT / "rubrics/dnd_alignment.json"


def load_rubric(path: Path = RUBRIC_PATH) -> dict:
    rubric = json.loads(path.read_text())
    assert {item["id"] for item in rubric["items"]} == {"order", "morality"}
    assert all([x["score"] for x in item["levels"]] == list(range(5, -6, -1)) for item in rubric["items"])
    return rubric


def request(rubric: dict, item: dict, rows: list[dict], *, full_feedback: bool) -> dict:
    assert 1 <= len(rows) <= 40
    state = {"task": rubric["task"], "rules": rubric["rules"], "axis_rubric": item, "target_scope": PROPOSED_SCOPE,
             "grading": "Select the numerical grade defined by the axis rubric. Numbers are grades, not arbitrary category IDs. Unclear is missing evidence, never neutral."}
    questions, review = {}, {}
    for row in rows:
        key = question_id(row["source_id"], row["choice_id"])
        levels = ["grade_" + str(i) for i in range(-5, 6)] + ["unclear"]
        if int(hashlib.sha256((key + item["id"]).encode()).hexdigest(), 16) % 2:
            levels.reverse()
        assert key not in questions
        questions[key] = {"type": "choice", "instructions": {"question": item["question"], "agent": row["agent"],
            "story": row["story"], "selected_act_quote": row["action"]},
            "criteria": dict.fromkeys(levels, "")}
        if full_feedback and int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 == 0:
            review[key] = {"agent": row["agent"], "selected_act_quote": row["action"], "story": row["story"]}
    questions["debug_word"] = {"type": "choice", "instructions": {"task":
        "Unscored feedback: inspect only the sampled actions copied here; other scoring questions are deliberately invisible. If the sample is empty, choose none. Name an interpretation obstacle, not moral approval. None is not validation.",
        "questions_to_review": review}, "criteria": {"none": "No apparent obstacle.", "missing": "Missing context or evidence.",
        "ambiguous": "Ambiguous wording or reference.", "scope": "Wrong actor, retrospective act or background choice.", "rubric": "Unclear or conflicting rubric.", "other": "Another obstacle."}}
    return {"model": rubric["model"], "state": state, "questions": questions,
            "provider": {"only": ["typesafe"], "allow_fallbacks": False}}


def numerical_score(probabilities: dict) -> float | None:
    known = sum(probabilities[str(i)] for i in range(-5, 6))
    if probabilities["unclear"] >= max(probabilities[str(i)] for i in range(-5, 6)):
        return None
    assert known > 0
    return sum(i * probabilities[str(i)] for i in range(-5, 6)) / known


def parse_answers(body: dict, payload: dict, item: dict, rows: list[dict]) -> list[dict]:
    assert body["answers"].keys() == payload["questions"].keys()
    result = []
    for row in rows:
        key = question_id(row["source_id"], row["choice_id"])
        answer = body["answers"][key]
        assert answer["type"] == "choice" and answer["choice"] in payload["questions"][key]["criteria"]
        native = answer["probabilities"]
        assert native.keys() == payload["questions"][key]["criteria"].keys()
        assert all(type(p) in (float, int) and math.isfinite(p) and 0 <= p <= 1 for p in native.values())
        raw = {grade.removeprefix("grade_"): value for grade, value in native.items()}
        mass = sum(raw.values())
        assert math.isclose(mass, 1, abs_tol=len(raw) * .005 + 1e-8) and mass > 0
        normalized = {key: p / mass for key, p in raw.items()}
        result.append({"source_id": row["source_id"], "choice_id": row["choice_id"], "item_id": item["item_id"],
            "axis": item["id"], "score": numerical_score(normalized), "raw_probabilities": raw,
            "normalized_probabilities": normalized, "unclear_probability": normalized["unclear"],
            "criterion_order": list(payload["questions"][key]["criteria"]), "native_probabilities": native,
            "full_context_diagnostic": key in payload["questions"]["debug_word"]["instructions"]["questions_to_review"],
            "raw_probability_mass": mass, "selected_grade": answer["choice"].removeprefix("grade_")})
    feedback = body["answers"]["debug_word"]
    assert feedback["type"] == "choice" and feedback["choice"] in payload["questions"]["debug_word"]["criteria"]
    assert feedback["probabilities"].keys() == payload["questions"]["debug_word"]["criteria"].keys()
    assert all(math.isfinite(p) and 0 <= p <= 1 for p in feedback["probabilities"].values())
    assert math.isclose(sum(feedback["probabilities"].values()), 1, abs_tol=.03 + 1e-8)
    return result
