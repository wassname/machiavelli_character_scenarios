"""Stage and load-check one local HF scene table; never upload. -- PI/Sol"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from virtue_questions import ROOT, identity, load_rubric, visible_identity


def checked_overlay(run: Path, proof: Path, rubric: dict) -> tuple[dict, dict, dict]:
    exported = json.loads((run / "export_verification.json").read_text())
    native = json.loads(proof.read_text())
    assert exported["status"] == native["status"] == "PASS"
    path = run / "virtues.parquet"
    assert native["parquet_sha256"] == exported["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), "Native proof belongs to another overlay"
    assert native["frozen_rubric_sha256"] == identity(rubric), "Native proof belongs to another frozen rubric"
    direct = {item["item_id"] for item in rubric["items"]}
    parents = {pair["vice_id"]: pair["virtue_id"] for pair in rubric["contraries"]}
    expected = direct | parents.keys()
    labels, priorities = {}, {}
    for row in pq.read_table(path).to_pylist():
        source_id, choice_id = row["source_id"], row["choice_id"]
        assert choice_id not in labels.setdefault(source_id, {})
        assert row["virtues"].keys() == expected
        for item, parent in parents.items():
            assert row["virtues"][item]["derived_from"] == parent
            assert row["virtues"][item]["signed"] == row["virtues"][parent]["signed"]
            assert row["virtues"][item]["raw_probabilities"] == row["virtues"][parent]["raw_probabilities"]
        labels[source_id][choice_id] = row["virtues"]
        if source_id in priorities:
            assert priorities[source_id] == row["selection_priority"]
        priorities[source_id] = row["selection_priority"]
    assert len(labels) == exported["scenes"] == native["scenes"]
    assert sum(map(len, labels.values())) == exported["choices"] == native["choices"]
    return labels, priorities, native


def check_visible(rows: list[dict], labels: dict, native: dict) -> None:
    sources = {row["source_id"]: row for row in rows}
    assert labels.keys() <= sources.keys()
    for source_id, choices in labels.items():
        assert choices.keys() == sources[source_id]["choice_text"].keys(), "Partial or changed scene choice coverage"
    assert visible_identity([sources[key] for key in labels]) == native["visible_inputs_sha256"], "Base prompts differ from the judged inputs"


def checked_visible_labels(run: Path, inputs: Path, rows: list[dict]) -> tuple[dict, dict]:
    proof = json.loads((run / "verification.json").read_text())
    parquet = run / "parquet/train.parquet"
    assert proof["status"] == "PASS" and proof["every_saved_score_matches_raw_response"]
    assert proof["input_fields_preserved_in_exports"] and proof["original_annotations_unchanged"]
    assert proof["inputs_sha256"] == hashlib.sha256(inputs.read_bytes()).hexdigest()
    assert proof["parquet_sha256"] == hashlib.sha256(parquet.read_bytes()).hexdigest()
    assert proof["scenarios"] == 1360 and proof["complete_actions"] == 2720
    assert proof["rubric_version"] == "visible-action-values-v4"
    assert proof["served_models"] == ["typesafe/jev-1.13-20260917"]
    table = pq.read_table(parquet, columns=["source_id", "choice_labels_visible"])
    labels = {row["source_id"]: json.loads(row["choice_labels_visible"]) for row in table.to_pylist()}
    assert labels.keys() == {row["source_id"] for row in rows}
    dimensions = set(json.loads((ROOT / "rubrics/actions.json").read_text())["dimensions"])
    for row in rows:
        assert labels[row["source_id"]].keys() == row["choice_text"].keys()
        for label in labels[row["source_id"]].values():
            assert set(label["visible_action_values"]) == dimensions
            assert len(label["stoic_virtues_vices"]) == 8
            assert set(label["dnd_action_alignment"]) == {"order", "morality"}
    return labels, proof


def stage(base: Path, run: Path, native_verification: Path, out: Path, *, dd_inputs: Path | None = None, dd_run: Path | None = None,
          dd_visible_run: Path | None = None) -> dict:
    assert (dd_inputs is None) == (dd_run is None), "Supply both DailyDilemmas inputs and its complete labelled run"
    assert dd_visible_run is None or dd_inputs is not None
    rubric = load_rubric()
    original = pq.read_table(base)
    assert original.num_rows == 10492 and original["source"].unique().to_pylist() == ["machiavelli"]
    source_ids = original["source_id"].to_pylist()
    assert len(set(source_ids)) == len(source_ids)
    labels, priorities, native = checked_overlay(run, native_verification, rubric)
    assert len(labels) == 2500 and sum(map(len, labels.values())) == 8723
    visible = [row | {"choice_text": json.loads(row["choice_text"])} for row in original.select(["source_id", "player_character", "text", "choice_text"]).to_pylist()]
    check_visible(visible, labels, native)
    new_fields = {"has_virtue_labels", "choice_virtue_labels", "virtue_selection_priority"}
    assert not new_fields & set(original.column_names)
    merged = original.append_column("has_virtue_labels", pa.array([key in labels for key in source_ids]))
    merged = merged.append_column("choice_virtue_labels", pa.array([json.dumps(labels[key], ensure_ascii=False, separators=(",", ":")) if key in labels else "{}" for key in source_ids]))
    merged = merged.append_column("virtue_selection_priority", pa.array([priorities.get(key) for key in source_ids], type=pa.float64()))
    assert merged.select(original.column_names).equals(original), "Earlier source or label fields changed"
    source_licenses = {"machiavelli": "mit"}
    dd_rows = []
    dd_visible = {}
    dd_visible_proof = None
    if dd_inputs is not None:
        dd_rows = [json.loads(line) for line in dd_inputs.open()]
        assert len(dd_rows) == 1360 and len({row["source_id"] for row in dd_rows}) == 1360
        assert all(row["source"] == "daily_dilemmas" and row["source_license"] == "cc-by-4.0" for row in dd_rows)
        assert all("choice_labels_visible" not in row for row in dd_rows), "Use original prepared inputs plus a verified visible-label run"
        if dd_visible_run is not None:
            dd_visible, dd_visible_proof = checked_visible_labels(dd_visible_run, dd_inputs, dd_rows)
        assert not set(source_ids) & {row["source_id"] for row in dd_rows}
        dd_labels, dd_priorities, dd_native = checked_overlay(dd_run, dd_run / "native_verification.json", rubric)
        assert len(dd_labels) == 1360 and sum(map(len, dd_labels.values())) == 2720, "Do not stage a partial DailyDilemmas run as complete"
        check_visible(dd_rows, dd_labels, dd_native)
        provenance = ["source_repository", "source_revision", "source_license", "source_attribution", "source_url", "source_annotations", "topic_group"]
        refs = json.loads((ROOT / "data/sources.json").read_text())
        original_provenance = {"source_repository": refs["source_dataset"]["repository"],
            "source_revision": refs["source_dataset"]["revision"], "source_license": "mit",
            "source_attribution": "MACHIAVELLI benchmark", "source_url": "https://huggingface.co/datasets/wassname/machiavelli",
            "source_annotations": None, "topic_group": None}
        assert not set(provenance) & set(merged.column_names)
        for field in provenance:
            merged = merged.append_column(field, pa.array([original_provenance[field]] * len(source_ids), type=pa.string()))
        extra = []
        for row in dd_rows:
            converted = {}
            for field in merged.column_names:
                value = row[field] if field in row else None
                converted[field] = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else value
            if dd_visible_run is not None:
                converted["choice_labels_visible"] = json.dumps(dd_visible[row["source_id"]], ensure_ascii=False, separators=(",", ":"))
            converted.update(world="DailyDilemmas", input_revision="daily-dilemmas-paired-actions-v1",
                has_virtue_labels=True, choice_virtue_labels=json.dumps(dd_labels[row["source_id"]], ensure_ascii=False, separators=(",", ":")),
                virtue_selection_priority=dd_priorities[row["source_id"]])
            extra.append(converted)
        merged = pa.concat_tables([merged, pa.Table.from_pylist(extra, schema=merged.schema)])
        source_ids += [row["source_id"] for row in dd_rows]
        labels |= dd_labels
        source_licenses["daily_dilemmas"] = "cc-by-4.0"
    assert len(set(source_ids)) == len(source_ids)
    assert merged.select(original.column_names).slice(0, original.num_rows).equals(original)
    review_path = ROOT / "data/scenario_reviews.json"
    review_metadata = json.loads(review_path.read_text())
    reviews = review_metadata["cases"]
    assert review_metadata["reviewer"] and review_metadata["scope"] and review_metadata["source_reference"]
    assert set(reviews) <= set(source_ids), "Review metadata names a missing scenario"
    texts = dict(zip(source_ids, merged["text"].to_pylist(), strict=True))
    for source_id, review in reviews.items():
        assert isinstance(review["flags"], list) and review["flags"] and review["visible_evidence"]
        assert hashlib.sha256(texts[source_id].encode()).hexdigest() == review["visible_prompt_sha256"], "Review flags belong to a changed prompt"
        assert all(quote in texts[source_id] for quote in review["visible_evidence"]), "Review visible quote differs from the staged prompt"
    merged = merged.append_column("scenario_review_flags", pa.array([json.dumps(reviews[key]["flags"], ensure_ascii=False) if key in reviews else "[]" for key in source_ids]))
    merged = merged.append_column("scenario_review_status", pa.array(["reviewed_with_flags" if key in reviews else "unreviewed" for key in source_ids]))
    card = "---\n" + (ROOT / "dataset_metadata.yaml").read_text().strip() + "\n---\n\n" + (ROOT / "README.md").read_text()
    from huggingface_hub import DatasetCard
    metadata = DatasetCard(card).data.to_dict()
    assert len(metadata["configs"]) == 1 and metadata["configs"][0]["config_name"] == "default"
    assert metadata["configs"][0]["data_files"] == [{"split": "train", "path": "parquet/train.parquet"}]
    if dd_rows:
        licenses = metadata["license"] if isinstance(metadata["license"], list) else [metadata["license"]]
        assert {"mit", "cc-by-4.0"} <= set(licenses), "Combined card must preserve per-source MIT/CC-BY-4.0; main owns the prose/metadata"
    out.mkdir(parents=True, exist_ok=False)
    parquet = out / "parquet/train.parquet"
    parquet.parent.mkdir()
    pq.write_table(merged, parquet, compression="zstd")
    assert pq.read_table(parquet).equals(merged), "Parquet round-trip changed fields"
    shutil.copyfile(ROOT / "CITATION.cff", out / "CITATION.cff")
    (out / "README.md").write_text(card)
    for relative in re.findall(r"!\[[^\]]*\]\((docs/figures/[^)]+)\)", card):
        assert (ROOT / relative).resolve().is_relative_to(ROOT / "docs/figures")
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    examples = out / "examples"
    examples.mkdir()
    shutil.copyfile(ROOT / "examples/judge_request.json", examples / "visible_actions_v4_jev_1_13_request.json")
    from datasets import load_dataset
    dataset = load_dataset("parquet", data_files=str(parquet), split="train")
    assert len(dataset) == merged.num_rows and dataset.data.column("source_id").to_pylist() == source_ids
    source_counts = Counter(merged["source"].to_pylist())
    assert Counter(dataset.data.column("source").to_pylist()) == source_counts
    assert pc.sum(dataset.data.column("has_virtue_labels")).as_py() == len(labels)
    for index, source_id in enumerate(source_ids):
        assert json.loads(dataset.data.column("choice_virtue_labels")[index].as_py()) == labels.get(source_id, {})
        expected_flags = reviews[source_id]["flags"] if source_id in reviews else []
        assert json.loads(dataset.data.column("scenario_review_flags")[index].as_py()) == expected_flags
        assert dataset.data.column("scenario_review_status")[index].as_py() == ("reviewed_with_flags" if source_id in reviews else "unreviewed")
    if dd_rows:
        for index, row in enumerate(dd_rows, original.num_rows):
            assert json.loads(dataset.data.column("source_annotations")[index].as_py()) == row["source_annotations"]
            if dd_visible_run is not None:
                assert json.loads(dataset.data.column("choice_labels_visible")[index].as_py()) == dd_visible[row["source_id"]]
    result = {"status": "PASS", "author": "PI/Sol", "scenes": len(source_ids), "source_counts": dict(source_counts),
              "virtue_labelled_scenes": len(labels), "virtue_labelled_choices": sum(map(len, labels.values())),
              "virtue_cells": sum(map(len, labels.values())) * 80, "source_licenses": source_licenses,
              "existing_fields_and_labels_unchanged": True, "all_source_ids_and_choices_preserved": True,
              "all_selected_scene_choices_labelled": True, "unlabelled_choice_virtue_labels": {},
              "daily_dilemmas_annotations_preserved": bool(dd_rows), "daily_dilemmas_visible_labels": len(dd_visible), "hf_local_parquet_load": "PASS", "card_config_parse": "PASS",
              "known_review_flagged_scenes": len(reviews), "unreviewed_scenes": len(source_ids) - len(reviews),
              "review_status_semantics": "An empty flag list with unreviewed means no recorded review, not validation.",
              "rubric_version": rubric["version"], "tiny_virtues_commit": rubric["source"]["commit"],
              "frozen_rubric_sha256": identity(rubric), "full_teachings_or_native_traces_in_bundle": False,
              "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(), "publishing_performed": False,
              "validation_scope": "Source and artifact integrity; semantic accuracy and probability calibration are unmeasured. Editorial sign-off is separate."}
    checks = out / "checks"
    checks.mkdir()
    (checks / "virtues_v2.json").write_text(json.dumps(result, indent=2) + "\n")
    shutil.copyfile(ROOT / "data/sources.json", checks / "sources.json")
    shutil.copyfile(review_path, checks / "scenario_reviews.json")
    if dd_visible_proof is not None:
        public_proof = {key: dd_visible_proof[key] for key in ("status", "scenarios", "complete_actions", "labels_sha256", "parquet_sha256", "inputs_sha256", "rubric_version", "served_models", "feedback_counts", "every_saved_score_matches_raw_response", "parquet_source_fields_preserved")}
        (checks / "daily_dilemmas_visible_labels.json").write_text(json.dumps(public_proof, indent=2) + "\n")
    (out / "staging_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "outputs/20261018_release_labels/parquet/train.parquet")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--native-verification", type=Path, required=True)
    parser.add_argument("--dd-inputs", type=Path)
    parser.add_argument("--dd-run", type=Path)
    parser.add_argument("--dd-visible-run", type=Path, help="Verified DailyDilemmas behaviour, Stoic and D&D label run")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.base, args.run, args.native_verification, args.out, dd_inputs=args.dd_inputs, dd_run=args.dd_run, dd_visible_run=args.dd_visible_run), indent=2))
