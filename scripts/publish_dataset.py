"""Verify the curated release directory and atomically publish it to Hugging Face. -- PI/OpenAI"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/data"))
from alignment_questions import numerical_score
REPO = "wassname/virtue-choices"
CORE_FILES = {
    "README.md", "CITATION.cff", "parquet/train.parquet", "staging_verification.json",
    "checks/virtues_v2.json", "checks/sources.json", "checks/scenario_reviews.json",
    "examples/visible_actions_v4_jev_1_13_request.json", "checks/daily_dilemmas_visible_labels.json",
    "checks/dnd_alignment.json", "rubrics/dnd_alignment.json",
}
EARLIER_CHECKS = {
    "checks/budget.json", "checks/category_samples.jsonl", "checks/feedback_summary.json",
    "checks/input_repairs.json", "checks/negative_checks.json", "checks/verification.json",
}


def verify(staged: Path) -> dict:
    card = (staged / "README.md").read_text()
    expected_card = "---\n" + (ROOT / "dataset_metadata.yaml").read_text().strip() + "\n---\n\n" + (ROOT / "README.md").read_text()
    assert card == expected_card, "Staged card differs from the approved README and metadata"
    assert (staged / "CITATION.cff").read_bytes() == (ROOT / "CITATION.cff").read_bytes()
    images = set(re.findall(r"!\[[^\]]*\]\((docs/figures/[^)]+)\)", card))
    expected = CORE_FILES | images
    actual = {str(p.relative_to(staged)) for p in staged.rglob("*") if p.is_file()}
    assert actual - {"release.json"} == expected, f"Release file allowlist differs: {actual ^ expected}"
    proof = json.loads((staged / "staging_verification.json").read_text())
    assert proof["status"] == "PASS" and proof["all_preexisting_non_dnd_labels_preserved"]
    assert proof["dnd_alignment_relabelled_choices"] == 39896
    dnd_proof = json.loads((staged / "checks/dnd_alignment.json").read_text())
    assert dnd_proof["status"] == "PASS" and dnd_proof["every_label_reconstructed_from_native_response"]
    assert dnd_proof["choices"] == 39896 and dnd_proof["direct_cells"] == 79792
    assert dnd_proof["served_model"] == "typesafe/jev-1.13-20260917"
    assert (staged / "rubrics/dnd_alignment.json").read_bytes() == (ROOT / "rubrics/dnd_alignment.json").read_bytes()
    assert hashlib.sha256((staged / "rubrics/dnd_alignment.json").read_bytes()).hexdigest() == dnd_proof["rubric_sha256"]
    assert proof["all_source_ids_and_choices_preserved"] and proof["all_selected_scene_choices_labelled"]
    assert proof["daily_dilemmas_annotations_preserved"] and not proof["full_teachings_or_native_traces_in_bundle"]
    assert proof["daily_dilemmas_visible_labels"] == 1360, "Include the complete shared DailyDilemmas label run"
    visible_proof = json.loads((staged / "checks/daily_dilemmas_visible_labels.json").read_text())
    assert visible_proof["status"] == "PASS" and visible_proof["complete_actions"] == 2720
    assert visible_proof["served_models"] == ["typesafe/jev-1.13-20260917"]
    parquet = staged / "parquet/train.parquet"
    assert hashlib.sha256(parquet.read_bytes()).hexdigest() == proof["parquet_sha256"]
    rows = pq.read_table(parquet, columns=["source", "has_virtue_labels", "choice_text", "choice_labels_visible"]).to_pylist()
    assert len(rows) == proof["scenes"] == 11852
    assert Counter(r["source"] for r in rows) == proof["source_counts"] == {"machiavelli": 10492, "daily_dilemmas": 1360}
    assert sum(r["has_virtue_labels"] for r in rows) == proof["virtue_labelled_scenes"] == 3860
    choices = sum(len(json.loads(r["choice_text"])) for r in rows)
    assert choices == 39896
    assert sum(len(json.loads(r["choice_text"])) for r in rows if r["has_virtue_labels"]) == proof["virtue_labelled_choices"] == 11443
    for row in rows:
        labels = json.loads(row["choice_labels_visible"])
        assert labels.keys() == json.loads(row["choice_text"]).keys()
        for label in labels.values():
            assert label["dnd_action_alignment"].keys() == {"order", "morality"}
            for axis in label["dnd_action_alignment"].values():
                assert axis["rubric_version"] == dnd_proof["rubric_version"] and axis["served_model"] == dnd_proof["served_model"]
                raw, normalized = axis["raw_probabilities"], axis["normalized_probabilities"]
                assert raw.keys() == normalized.keys() == {str(i) for i in range(-5, 6)} | {"unclear"}
                assert all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in list(raw.values()) + list(normalized.values()))
                mass = sum(raw.values())
                assert mass > 0 and math.isclose(mass, 1, abs_tol=len(raw) * .005 + 1e-8)
                assert all(math.isclose(normalized[key], p / mass, abs_tol=1e-12) for key, p in raw.items())
                assert axis["unclear_probability"] == normalized["unclear"]
                expected_score = numerical_score(normalized)
                score = axis["score"]
                assert (score is None and expected_score is None) or (
                    type(score) in (int, float) and expected_score is not None and math.isfinite(score)
                    and -5 <= score <= 5 and math.isclose(score, expected_score, abs_tol=1e-12))
    for relative in expected:
        path = staged / relative
        assert not path.is_symlink()
        if path.suffix in {".md", ".json", ".cff"}:
            text = path.read_text()
            assert not any(marker in text for marker in ("/workspace/", "/home/code/", "sk-or-", "Bearer ")), relative
    files = {name: hashlib.sha256((staged / name).read_bytes()).hexdigest() for name in sorted(expected)}
    release = {"version": "2.1.0", "dataset": REPO, "code": "https://github.com/" + REPO,
               "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "scenarios": len(rows), "choices": choices, "virtue_labelled_scenes": 3860,
               "virtue_labelled_choices": proof["virtue_labelled_choices"], "visible_labelled_scenes": len(rows), "visible_labelled_choices": choices, "source_counts": proof["source_counts"],
               "rubric_version": proof["rubric_version"], "dnd_rubric_version": dnd_proof["rubric_version"], "files_sha256": files,
               "validation": proof["validation_scope"]}
    (staged / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    return release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--expected-parent")
    args = parser.parse_args()
    release = verify(args.staged)
    print(json.dumps(release, indent=2))
    if not args.publish:
        return
    assert args.expected_parent, "Publishing needs an explicitly inspected HF parent revision"
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "README.md", "CITATION.cff", "dataset_metadata.yaml",
                    "data/scenario_reviews.json", "data/sources.json", "rubrics/dnd_alignment.json", "scripts/data/alignment_questions.py", "scripts/publish_dataset.py",
                    "scripts/data/stage_virtue_release.py", "docs/figures"], cwd=ROOT, check=True)
    api = HfApi()
    assert api.dataset_info(REPO).sha == args.expected_parent, "HF changed; inspect before publishing"
    files = {str(p.relative_to(args.staged)): p for p in args.staged.rglob("*") if p.is_file()}
    remote = set(api.list_repo_files(REPO, repo_type="dataset"))
    assert remote <= files.keys() | EARLIER_CHECKS | {".gitattributes"}, "Unexpected remote files; inspect before replacing"
    operations = [CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in sorted(files.items())]
    operations += [CommitOperationDelete(path_in_repo=name) for name in sorted(remote - files.keys() - {".gitattributes"})]
    result = api.create_commit(repo_id=REPO, repo_type="dataset", parent_commit=args.expected_parent,
                               operations=operations, commit_message="Publish Virtue Choices: game and everyday action choices with nine-tradition labels")
    receipt = {"url": "https://huggingface.co/datasets/" + REPO, "revision": result.oid,
               "commit_url": result.commit_url, "scenarios": release["scenarios"],
               "choices": release["choices"], "virtue_labelled_scenes": release["virtue_labelled_scenes"],
               "virtue_labelled_choices": release["virtue_labelled_choices"]}
    (ROOT / ".local/publication.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
