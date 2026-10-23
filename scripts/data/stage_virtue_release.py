"""Stage and load-check a single local HF scene table; never upload. -- PI/Sol"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from virtue_questions import ROOT, identity, load_rubric, visible_identity


def stage(base: Path, run: Path, native_verification: Path, out: Path) -> dict:
    verified = json.loads((run / "export_verification.json").read_text())
    native = json.loads(native_verification.read_text())
    assert verified["status"] == native["status"] == "PASS"
    overlay_path = run / "virtues.parquet"
    assert native["parquet_sha256"] == verified["sha256"] == hashlib.sha256(overlay_path.read_bytes()).hexdigest(), "Native proof belongs to another overlay"
    original = pq.read_table(base)
    assert original.num_rows == 10492
    assert original["source"].unique().to_pylist() == ["machiavelli"]
    source_ids = original["source_id"].to_pylist()
    assert len(set(source_ids)) == len(source_ids)
    choices_by_source = dict(zip(source_ids, map(json.loads, original["choice_text"].to_pylist()), strict=True))
    priorities, overlay = {}, {}
    rubric = load_rubric()
    assert native["frozen_rubric_sha256"] == identity(rubric), "Native proof belongs to another frozen rubric"
    direct = {item["item_id"] for item in rubric["items"]}
    parents = {pair["vice_id"]: pair["virtue_id"] for pair in rubric["contraries"]}
    expected = direct | parents.keys()
    for row in pq.read_table(overlay_path).to_pylist():
        source_id, choice_id = row["source_id"], row["choice_id"]
        assert source_id in choices_by_source and choice_id in choices_by_source[source_id]
        assert row["virtues"].keys() == expected
        assert choice_id not in overlay.setdefault(source_id, {})
        for item, parent in parents.items():
            assert row["virtues"][item]["derived_from"] == parent
            assert row["virtues"][item]["signed"] == row["virtues"][parent]["signed"]
            assert row["virtues"][item]["raw_probabilities"] == row["virtues"][parent]["raw_probabilities"]
        overlay[source_id][choice_id] = row["virtues"]
        if source_id in priorities:
            assert priorities[source_id] == row["selection_priority"]
        priorities[source_id] = row["selection_priority"]
    for source_id, labels in overlay.items():
        assert labels.keys() == choices_by_source[source_id].keys(), "Partial scene choice coverage"
    assert len(overlay) == verified["scenes"] == native["scenes"] == 2500
    assert sum(map(len, overlay.values())) == verified["choices"] == native["choices"] == 8723
    visible = []
    for row in original.select(["source_id", "player_character", "text", "choice_text"]).to_pylist():
        if row["source_id"] in overlay:
            visible.append(row | {"choice_text": json.loads(row["choice_text"])})
    assert visible_identity(visible) == native["visible_inputs_sha256"], "Base prompts differ from the judged inputs"
    assert not {"has_virtue_labels", "choice_virtue_labels", "virtue_selection_priority"} & set(original.column_names)
    merged = original.append_column("has_virtue_labels", pa.array([key in overlay for key in source_ids]))
    merged = merged.append_column("choice_virtue_labels", pa.array([
        json.dumps(overlay[key], ensure_ascii=False, separators=(",", ":")) if key in overlay else "{}" for key in source_ids]))
    merged = merged.append_column("virtue_selection_priority", pa.array([priorities.get(key) for key in source_ids], type=pa.float64()))
    assert merged.select(original.column_names).equals(original), "Earlier source or label fields changed"
    out.mkdir(parents=True, exist_ok=False)
    parquet = out / "parquet/train.parquet"
    parquet.parent.mkdir()
    pq.write_table(merged, parquet, compression="zstd")
    reloaded = pq.read_table(parquet)
    assert reloaded.equals(merged), "Parquet round-trip changed fields"
    shutil.copyfile(ROOT / "CITATION.cff", out / "CITATION.cff")
    card = "---\n" + (ROOT / "dataset_metadata.yaml").read_text().strip() + "\n---\n\n" + (ROOT / "README.md").read_text()
    (out / "README.md").write_text(card)
    examples = out / "examples"
    examples.mkdir()
    shutil.copyfile(ROOT / "examples/judge_request.json", examples / "visible_actions_v4_jev_1_13_request.json")
    from datasets import load_dataset
    from huggingface_hub import DatasetCard
    metadata = DatasetCard.load(out / "README.md").data.to_dict()
    assert len(metadata["configs"]) == 1 and metadata["configs"][0]["config_name"] == "default"
    assert metadata["configs"][0]["data_files"] == [{"split": "train", "path": "parquet/train.parquet"}]
    dataset = load_dataset("parquet", data_files=str(parquet), split="train")
    assert len(dataset) == original.num_rows
    assert dataset.data.column("source_id").to_pylist() == source_ids
    assert dataset.data.column("source").unique().to_pylist() == ["machiavelli"]
    assert pc.sum(dataset.data.column("has_virtue_labels")).as_py() == len(overlay)
    for index, source_id in enumerate(source_ids):
        actual = json.loads(dataset.data.column("choice_virtue_labels")[index].as_py())
        assert actual == overlay.get(source_id, {}), source_id
    result = {"status": "PASS", "author": "PI/Sol", "scenes": len(source_ids),
              "virtue_labelled_scenes": len(overlay), "virtue_labelled_choices": 8723, "virtue_cells": 697840,
              "source": "machiavelli", "existing_fields_and_labels_unchanged": True,
              "all_source_ids_and_choices_preserved": True, "all_selected_scene_choices_labelled": True,
              "unlabelled_choice_virtue_labels": {}, "hf_local_parquet_load": "PASS", "card_config_parse": "PASS",
              "rubric_version": rubric["version"], "tiny_virtues_commit": rubric["source"]["commit"],
              "frozen_rubric_sha256": identity(rubric), "full_teachings_or_native_traces_in_bundle": False,
              "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
              "validation_scope": "Source and artifact integrity; semantic accuracy and probability calibration are unmeasured."}
    checks = out / "checks"
    checks.mkdir()
    (checks / "virtues_v2.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "staging_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "outputs/20261018_release_labels/parquet/train.parquet")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--native-verification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.base, args.run, args.native_verification, args.out), indent=2))
