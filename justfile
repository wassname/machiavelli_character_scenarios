set shell := ["bash", "-cu"]

# PI/OpenAI: Show the available commands.
default:
    @just --list

# PI/OpenAI: Check the real source and request path without API calls.
offline-check:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p outputs
    out="$(mktemp -d outputs/offline_XXXXXX)"
    HF_HUB_OFFLINE=1 uv run --locked python scripts/data/prepare_inputs.py --limit 2 --out "$out/inputs"
    HF_HUB_OFFLINE=1 uv run --locked python scripts/data/label_actions.py --inputs "$out/inputs/inputs.jsonl" --limit 2 --dry-run --out "$out/requests"

# PI/OpenAI: Real API, cache and export on two scenarios.
fast-dev-run *args:
    BEARTYPE=1 uv run --locked python scripts/data/label_actions.py --limit 2 --concurrency 2 {{args}}

# PI/OpenAI: Restore displayed choices without model calls.
prepare out:
    uv run --locked python scripts/data/prepare_inputs.py --out "{{out}}"

# PI/OpenAI: Label actions from the visible prompt.
label *args:
    uv run --locked python scripts/data/label_actions.py {{args}}

# PI/OpenAI: Verify source fields, configured questions, responses and exports.
verify run:
    uv run --locked python scripts/checks/verify_labels.py "{{run}}"

# PI/OpenAI: Inspect every diagnostic value and seeded category examples.
feedback labels out *args:
    uv run --locked python scripts/checks/audit_feedback.py "{{labels}}" --out "{{out}}" {{args}}

# PI/OpenAI: Generate scene summaries from source histories.
summarise *args:
    uv run --locked python scripts/generation/summarise_machiavelli.py {{args}}
