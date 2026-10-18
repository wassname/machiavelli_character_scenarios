"""Snapshot attributed tiny-virtues rubrics without importing its credentials. PI/OpenAI."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = "wassname/tiny-virtues-dev"
TRADITIONS = ("stoic", "confucian", "taoist", "buddhist", "christian", "jewish", "islamic", "hindu", "rationalist")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="f1e8a66ae9fcc89baf2202f1547c0a696fdc7757")
    parser.add_argument("--out", type=Path, default=Path("data/raw/virtue_rubrics"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    entries = []
    for school in TRADITIONS:
        source = f"docs/virtues/lists/{school}.json"
        raw = subprocess.check_output(["gh", "api", f"repos/{REPO}/contents/{source}?ref={args.revision}", "--header", "Accept: application/vnd.github.raw"])
        rubric = json.loads(raw)
        assert rubric["school"] == school
        (args.out / f"{school}.json").write_bytes(raw)
        entries.append({"school": school, "path": source, "sha256": hashlib.sha256(raw).hexdigest(), "lists": len(rubric["lists"]), "items": sum(len(items) for item_list in rubric["lists"] for side, items in item_list.items() if side in {"virtues", "vices"})})
    manifest = {"repository": REPO, "revision": args.revision, "private_reference": True, "files": entries, "author": "PI/OpenAI"}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
