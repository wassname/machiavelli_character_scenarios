"""Unpaid regression checks of real requests and five-level score parsing. -- PI/Sol"""

import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from virtue_questions import ROOT, choices, derive_contrary, identity, load_rubric, parse_answers, question_id, request


def rejected(body: dict, payload: dict, item: dict, batch: list[dict]) -> None:
    try:
        parse_answers(body, payload, item, batch)
    except (AssertionError, KeyError):
        return
    raise AssertionError("Malformed judgement accepted")


def main() -> None:
    rubric = load_rubric()
    with (ROOT / "outputs/20261018_rendered_inputs_v3/inputs.jsonl").open() as stream:
        source = json.loads(next(stream))
    batch = choices([source])
    item = rubric["items"][0]
    payload = request(rubric, item, batch, full_feedback=True)
    assert payload == request(rubric, item, batch, full_feedback=True)
    assert identity(payload) != identity(request(rubric, item, batch, full_feedback=False))
    changed_labels = source | {"choice_labels": {"hidden_outcome": "SENTINEL_NOT_VISIBLE"}, "generated_answer": "SENTINEL_NOT_VISIBLE"}
    assert request(rubric, item, choices([changed_labels]), full_feedback=True) == payload
    alone = request(rubric, item, batch[:1], full_feedback=True)
    assert alone["questions"][question_id(batch[0]["source_id"], batch[0]["choice_id"])] == payload["questions"][question_id(batch[0]["source_id"], batch[0]["choice_id"])]
    for key, question in payload["questions"].items():
        if key != "debug_word":
            assert question["instructions"]["story"] == source["text"]
            assert question["instructions"]["agent"] == source["player_character"]
            assert "PROPOSED choice" in question["instructions"]["target_scope"]
    assert payload["questions"]["debug_word"]["instructions"]["questions_to_review"] == {
        key: question for key, question in payload["questions"].items() if key != "debug_word"}
    short = request(rubric, item, batch, full_feedback=False)
    assert all(set(summary) == {"id", "agent", "selected_act_quote"} for summary in
               short["questions"]["debug_word"]["instructions"]["questions_to_review"].values())
    body = {"answers": {}}
    for key, question in payload["questions"].items():
        if key == "debug_word":
            body["answers"][key] = {"type": "choice", "choice": "none", "probabilities": {
                word: float(word == "none") for word in question["criteria"]}}
        else:
            body["answers"][key] = {"type": "score", "score": 2.5,
                "probabilities": {str(i): p for i, p in enumerate([.1, .1, .2, .3, .3])},
                "legend": {str(i): text for i, text in enumerate(question["criteria"])}}
    scores = parse_answers(body, payload, item, batch)
    assert all(abs(score["signed"] - .3) < 1e-12 for score in scores)
    contrary = derive_contrary(scores[0], "test-contrary")
    assert contrary["signed"] == scores[0]["signed"] and contrary["shows"] == scores[0]["against"]
    assert contrary["against"] == scores[0]["shows"] and contrary["derived_from"] == item["item_id"]
    key = question_id(batch[0]["source_id"], batch[0]["choice_id"])
    for damage in ("missing_score", "missing_feedback", "nan", "wrong_mass", "wrong_legend"):
        broken = copy.deepcopy(body)
        if damage == "missing_score":
            del broken["answers"][key]
        elif damage == "missing_feedback":
            del broken["answers"]["debug_word"]
        elif damage == "nan":
            broken["answers"][key]["probabilities"]["0"] = float("nan")
        elif damage == "wrong_mass":
            broken["answers"][key]["probabilities"]["0"] = 1
        elif damage == "wrong_legend":
            broken["answers"][key]["legend"]["0"] = "wrong"
        rejected(broken, payload, item, batch)
    vice = next(item for item in rubric["items"] if item["kind"] == "vices")
    vice_request = request(rubric, vice, batch, full_feedback=False)
    assert "vice was not in play" in vice_request["questions"][key]["instructions"]["question"]["evidence_rule"]
    assert len(batch) == len(source["choice_text"])
    print("PASS: real prompt and agent preserved; deterministic requests; hybrid diagnostic copies; signed projection and contrary mapping; malformed replies rejected; vice-temptation cue retained. No API calls. -- PI/Sol")


if __name__ == "__main__":
    main()
