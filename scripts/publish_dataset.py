"""Bundle checked data and publish one atomic Hugging Face update. PI/OpenAI."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))

from dataset_io import SOURCES, ROOT
from prepare_inputs import INPUT_REVISION

REPO = "wassname/virtue-choices"
CODE = "https://github.com/wassname/virtue-choices"


def dataset_card() -> str:
    return "---\n" + (ROOT / "dataset_metadata.yaml").read_text().strip() + "\n---\n\n" + (ROOT / "README.md").read_text()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--negative-checks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--expected-parent", default=SOURCES["original_prompts"]["revision"])
    args = parser.parse_args()
    verified = json.loads((args.run / "verification.json").read_text())
    prepared = json.loads((args.prepared / "summary.json").read_text())
    feedback = json.loads((args.feedback / "summary.json").read_text())
    negatives = json.loads(args.negative_checks.read_text())
    metadata = json.loads((args.run / "metadata.json").read_text())
    assert verified["status"] == "PASS" and not verified["raw_http_or_transport_errors"]
    for key in ("input_fields_preserved_in_exports", "original_annotations_unchanged", "original_prompts_retained", "only_visible_prompt_and_choice_sent", "every_saved_score_matches_raw_response", "complete_configured_questions_checked", "parquet_source_fields_preserved"):
        assert verified[key], key
    assert verified["input_revision"] == prepared["input_revision"] == INPUT_REVISION
    assert prepared["inputs_sha256"] == metadata["config"]["inputs_sha256"]
    assert verified["scenarios"] == prepared["scenarios"] == feedback["scenarios"] == 10492
    assert verified["complete_actions"] == prepared["actions"] == feedback["actions_read"]
    assert verified["feedback_counts"] == feedback["selected_word_counts"]
    assert feedback["labels_sha256"] == hashlib.sha256((args.run / "labels.jsonl").read_bytes()).hexdigest()
    assert negatives["all_real_inputs_pass"]
    assert all(negatives[key] == "rejected" for key in ("removed_displayed_option", "altered_upstream_annotations", "missing_behaviour_question", "missing_stoic_questions"))
    assert negatives["in_world_undo_dialogue"] == negatives["stale_undo_history"] == "not_rewritten"
    assert negatives["interface_repair_rows"] == prepared["interface_context_repaired_rows"]
    assert verified["project_total_usd_at_verification"] <= 40
    args.out.mkdir(parents=True, exist_ok=False)
    files = {
        "CITATION.cff": ROOT / "CITATION.cff",
        "parquet/train.parquet": args.run / "parquet/train.parquet",
        "checks/verification.json": args.run / "verification.json",
        "checks/input_repairs.json": args.prepared / "summary.json",
        "checks/feedback_summary.json": args.feedback / "summary.json",
        "checks/category_samples.jsonl": args.feedback / "category_samples.jsonl",
        "checks/negative_checks.json": args.negative_checks,
    }
    hashes = {}
    for relative, source in files.items():
        target = args.out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    card = args.out / "README.md"
    card.write_text(dataset_card())
    hashes["README.md"] = hashlib.sha256(card.read_bytes()).hexdigest()
    release = {"version": "1.0.0", "dataset": REPO, "code": CODE, "input_revision": INPUT_REVISION, "rubric_version": metadata["config"]["rubric_version"], "model": metadata["config"]["model"], "scenarios": verified["scenarios"], "choices": verified["complete_actions"], "changed_prompt_rows": prepared["changed_prompt_rows"], "restored_choices": prepared["restored_choices"], "input_sha256": prepared["inputs_sha256"], "files_sha256": hashes, "code_files_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((ROOT / "scripts").rglob("*.py"))}, "validation": "Artifact integrity and source/receipt checks, not semantic accuracy or calibration.", "project_inference_usd": verified["project_total_usd_at_verification"], "local_generation_commit": metadata["commit"]}
    (args.out / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    print(json.dumps(release, indent=2))
    if args.publish:
        api = HfApi()
        assert api.dataset_info(REPO).sha == args.expected_parent, "HF changed since preparation; inspect before publishing"
        operations = [CommitOperationAdd(path_in_repo=str(path.relative_to(args.out)), path_or_fileobj=str(path)) for path in sorted(args.out.rglob("*")) if path.is_file()]
        info = api.create_commit(repo_id=REPO, repo_type="dataset", operations=operations, parent_commit=args.expected_parent, commit_message="Restore displayed choices and add visible-action labels, flags and citation")
        result = {"url": f"https://huggingface.co/datasets/{REPO}", "revision": info.oid, "commit_url": info.commit_url}
        (ROOT / ".local/publication.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
