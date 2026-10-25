"""Unpaid checks for exact action joins, shared cohorts and paired plots. -- PI/Astra"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest import TestCase

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from score_choice_dimensions import bootstrap, checked_visible, complete_scene_ids


def check_functions():
    scene = {"source_id": "scene", "text": "Visible story", "choice_text": {"0": "Help", "1": "Leave"}}
    labels = {key: {"source_id": "scene", "choice_id": key, "action": text} for key, text in scene["choice_text"].items()}
    row = {"text": scene["text"], "choice_text": json.dumps(scene["choice_text"]), "choice_labels_visible": json.dumps(labels)}
    assert checked_visible(scene, row) == labels
    for field, wrong in (("action", "Leave"), ("source_id", "different"), ("choice_id", "1")):
        changed = copy.deepcopy(labels)
        changed["0"][field] = wrong
        with TestCase().assertRaises(AssertionError):
            checked_visible(scene, row | {"choice_labels_visible": json.dumps(changed)})
    with TestCase().assertRaises(AssertionError):
        checked_visible(scene, row | {"choice_text": json.dumps({"0": "Leave", "1": "Help"})})
    with TestCase().assertRaises(AssertionError):
        checked_visible(scene, row | {"text": "A different story"})
    values = {"complete": {"a": [1, 0], "b": [2, 0]}, "missing": {"a": [1, 0], "b": [2, np.nan]}}
    assert complete_scene_ids(values, list(values), [0]) == ["complete", "missing"]
    assert complete_scene_ids(values, list(values), [0, 1]) == ["complete"]
    first = np.array([0., 2., 10.])
    values = np.stack([first, first + 7], axis=1)[:, :, None]
    draws = bootstrap(values, ["game1", "game1", "game2"], 2000)
    assert set(draws[:, 0, 0]) == {1., 4., 10.}
    assert np.array_equal(draws[:, 1, 0] - draws[:, 0, 0], np.full(2000, 7.))
    assert np.array_equal(draws, bootstrap(values, ["game1", "game1", "game2"], 2000))
    return ["Changed story, source ID, choice ID and swapped action joins rejected", "Missing unchosen action excludes its question; real zero remains a grade", "Unequal game sizes retained; paired bootstrap preserves constant differences exactly"]


def check_artifact(path: Path, reference: Path | None):
    d = json.loads(path.read_text())
    index = {(r["source"], r["mode"], r["model"], r["item"]): r for r in d["rows"]}
    assert len(index) == len(d["rows"]) == len(d["sources"]) * 3 * 3 * len(d["dimensions"])
    assert len(d["dimensions"]) == 108
    for row in d["rows"]:
        assert np.isfinite([row["mean"], row["mean_se"], *row["ci95"]]).all()
        assert row["scenes"] == d["sources"][row["source"]]["families"][row["family"]]["scenes"]
        if row["mode"] == "versus_deepseek":
            raw = index[row["source"], "raw", row["model"], row["item"]]
            base = index[row["source"], "raw", "DeepSeek", row["item"]]
            assert abs(row["mean"] - (raw["mean"] - base["mean"])) < 1e-12
            if row["model"] == "DeepSeek":
                assert [row["mean"], row["mean_se"], *row["ci95"]] == [0., 0., 0., 0.]
    for dim in d["dimensions"]:
        if dim["derived_from"] is None:
            continue
        for source in d["sources"]:
            for mode in ("raw", "above_uniform", "versus_deepseek"):
                for model in d["models"].values():
                    child = index[source, mode, model, dim["id"]]
                    parent = index[source, mode, model, dim["derived_from"]]
                    assert np.allclose([child["mean"], *child["ci95"]], [-parent["mean"], -parent["ci95"][1], -parent["ci95"][0]], atol=1e-12, rtol=0)
    if reference is not None:
        old = json.loads(reference.read_text())
        for source, info in d["sources"].items():
            for model, mean in info["favorable_overall_above_uniform"].items():
                assert abs(mean - old["sources"][source]["rows"][model + ":overall"]["mean"]) < 1e-12
    return {"analysis_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows_checked": len(d["rows"]),
            "paired_point_means": "match raw differences to 1e-12", "derived_vice_means_and_intervals": "exact sign-reversed parent values",
            "prior_source_means": "match to 1e-12" if reference is not None else "not requested"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {"status": "PASS", "author": "PI/Astra", "regressions": check_functions()}
    if args.analysis is not None:
        result["artifact"] = check_artifact(args.analysis, args.reference)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
