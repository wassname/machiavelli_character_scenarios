"""Export a complete choice-level virtue overlay, without modifying earlier labels. -- PI/Sol"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from virtue_questions import ROOT, load_rubric


def export(run: Path) -> dict:
    rubric = load_rubric()
    direct = {item["item_id"] for item in rubric["items"]}
    parents = {pair["vice_id"]: pair["virtue_id"] for pair in rubric["contraries"]}
    expected = direct | parents.keys()
    selection = {row["source_id"]: row for row in map(json.loads, (run / "selection.jsonl").open())}
    config = json.loads((run / "metadata.json").read_text())["config"]
    sources = {row["source_id"]: row for row in map(json.loads, Path(config["inputs"]).open()) if row["source_id"] in selection}
    assert sources.keys() == selection.keys()
    choices = {}
    for source_id, row in sources.items():
        assert len(row["choice_text"]) == selection[source_id]["choices"]
        for choice_id in row["choice_text"]:
            choices[source_id, choice_id] = {}
    for label in map(json.loads, (run / "labels.jsonl").open()):
        key = label["source_id"], label["choice_id"]
        item = label["item_id"]
        assert key in choices and item in expected and item not in choices[key], "Extra or duplicate label"
        probabilities = [float(label["raw_probabilities"][str(i)]) for i in range(5)]
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities)
        mass = sum(probabilities)
        assert math.isclose(mass, 1, abs_tol=.02 + 1e-8)
        signed = sum(i * p / mass for i, p in enumerate(probabilities)) / 2 - 1
        assert math.isclose(label["signed"], signed, abs_tol=1e-12)
        parent = label["derived_from"] if item in parents else None
        assert parent == parents.get(item)
        choices[key][item] = {"signed": label["signed"], "raw_probabilities": probabilities, "derived_from": parent}
    rows = []
    for (source_id, choice_id), virtues in choices.items():
        assert virtues.keys() == expected, f"Incomplete 80-item coverage: {source_id}, {choice_id}"
        for contrary, parent in parents.items():
            assert virtues[contrary]["signed"] == virtues[parent]["signed"]
            assert virtues[contrary]["raw_probabilities"] == virtues[parent]["raw_probabilities"]
        rows.append({"source_id": source_id, "choice_id": choice_id,
                     "selection_priority": selection[source_id]["selection_priority"], "virtues": virtues})
    path = run / "virtues.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
    exported = pq.read_table(path).to_pylist()
    assert exported == rows, "Parquet round-trip changed values"
    summary = {"status": "PASS", "scenes": len(selection), "choices": len(rows),
               "direct_items_per_choice": len(direct), "derived_items_per_choice": len(parents),
               "cells": len(rows) * len(expected), "all_selected_scene_choices_retained": True,
               "existing_labels_modified": False, "parquet": str(path),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "raw_probability_semantics": "Direct records retain native five-level probabilities. Derived contraries copy the parent's signed weight and raw vector and identify it in derived_from; they are not independent judgements.",
               "author": "PI/Sol"}
    (run / "export_verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.run), indent=2))
