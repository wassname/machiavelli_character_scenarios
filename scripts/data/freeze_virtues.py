"""Freeze source-led tiny-virtues teachings for proposed-choice judging. -- PI/Sol"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def freeze(source: Path, out: Path) -> None:
    sys.path.insert(0, str(source / "src"))
    from tiny_virtues import item_major as major
    from tiny_virtues.judge_feedback import DEBUG_TASK, HYBRID_TASK, FEEDBACK_WORDS

    items = major.load_items(root=source, scope="major")
    pairs_path = source / "data/provenance/rubrics/named_contraries_20261022.json"
    pairs = json.loads(pairs_path.read_text())
    for binding in pairs["source_bindings"]:
        assert hashlib.sha256((source / binding["path"]).read_bytes()).hexdigest() == binding["sha256"]
    derived_ids = {pair["vice_id"] for pair in pairs["pairs"]}
    direct = [item for item in items if item["item_id"] not in derived_ids]
    assert len(items) == 80 and len(direct) == 66 and len(derived_ids) == 14
    assert {pair["virtue_id"] for pair in pairs["pairs"]} <= {item["item_id"] for item in direct}
    assert derived_ids <= {item["item_id"] for item in items}
    template = json.loads(major.FIVE_TEMPLATE.read_text())
    frozen = {
        "version": "tiny-virtues-rubric-v2-machiavelli-proposed-choice-v1",
        "author": "PI/Sol",
        "source": {
            "repository": "https://github.com/wassname/tiny-virtues",
            "commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
            "format": major.FIVE_LEVEL,
            "template_sha256": hashlib.sha256(major.FIVE_TEMPLATE.read_bytes()).hexdigest(),
            "contraries_sha256": hashlib.sha256(pairs_path.read_bytes()).hexdigest(),
        },
        "model": major.MODEL,
        "served_model": major.SERVED_MODEL,
        "template": {key: template[key] for key in ("layout_scope", "judging_task", "forms")},
        "pricing_observation": template["pricing_observation"],
        "feedback": {"task": DEBUG_TASK, "hybrid_task": HYBRID_TASK, "words": FEEDBACK_WORDS},
        "derivation": pairs["derivation"],
        "contraries": pairs["pairs"],
        "items": [{
            **{key: item[key] for key in ("item_id", "kind", "list_name", "status", "tested_real_examples", "rubric_path", "rubric_sha256")},
            **{key: item[key] for key in ("applicability", "source_context") if key in item},
            "source_binding": item["binding"],
            "teaching": major.render_teaching(item["guide"], item["sources"], format=major.FIVE_LEVEL),
        } for item in direct],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"path": str(out), "direct": len(direct), "derived": len(derived_ids), "source": frozen["source"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "rubrics/virtues_v2.json")
    args = parser.parse_args()
    freeze(args.source.resolve(), args.out)
