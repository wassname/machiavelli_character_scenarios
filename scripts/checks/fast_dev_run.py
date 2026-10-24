"""Run the real choice-evaluation pipeline on two scenes, then exercise its cache. -- PI/OpenAI"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/processed/parquet/train.parquet")
    parser.add_argument("--key-source", choices=["env", "pi"], default="env")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_smoke"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    log_path = args.out / "run.md"
    env = os.environ | {"BEARTYPE": "1"}
    results = []
    with log_path.open("w") as log:
        log.write("# Real action-choice smoke run\n\n-- PI/OpenAI\n\nSHOULD: eight fresh valid replies, then eight exact cache hits; both runs score the same two scenes. Tiny scores do not establish model quality.\n")
        for phase in ("fresh", "cached"):
            run = args.out / phase
            command = [sys.executable, str(ROOT / "scripts/evaluation/action_choice.py"),
                       "--inputs", str(args.inputs), "--count", "2", "--allow-paid", "--max-new-usd", ".05",
                       "--concurrency", "2", "--key-source", args.key_source, "--out", str(run)]
            if phase == "fresh":
                command.append("--refresh")
            log.write("\n## " + phase + "\n\n`" + shlex.join(command) + "`\n\n")
            log.flush()
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            command = [sys.executable, str(ROOT / "scripts/evaluation/score_action_choice.py"), str(run),
                       "--virtues", str(run / "virtues.parquet"), "--bootstrap", "100"]
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            usage = json.loads((run / "summary.json").read_text())
            scores = json.loads((run / "score_summary.json").read_text())
            manifest = json.loads((run / "manifest.json").read_text())
            assert manifest["runtime_type_checking"] and scores["paired_complete_scenes"] == 2
            assert usage["status_counts"] == {"ok": 8}
            if phase == "fresh":
                assert usage["new_requests"] == 8
            else:
                assert usage["cache_hits"] == 8 and usage["new_cost_usd"] == 0
            results.append([phase, 2, usage["new_requests"] if phase == "fresh" else 0, usage["cache_hits"] if phase == "cached" else 0, usage["new_cost_usd"], "PASS"])
        fresh = [json.loads(line) for line in (args.out / "fresh/choices.jsonl").open()]
        cached = [json.loads(line) for line in (args.out / "cached/choices.jsonl").open()]
        assert {r["request_key"]: (r["original_choice_id"], r["content"]) for r in fresh} == {r["request_key"]: (r["original_choice_id"], r["content"]) for r in cached}
        table = tabulate(results, headers=["phase", "scenes", "new requests", "cache hits", "US$", "result"], tablefmt="pipe", floatfmt=".6f")
        tail = "\n## Result\n\n" + table + "\n\n" + str(log_path) + "\n"
        log.write(tail)
    print(tail)


if __name__ == "__main__":
    main()
