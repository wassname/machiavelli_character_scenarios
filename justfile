set shell := ["bash", "-cu"]

# PI/OpenAI: Discover reproducible local commands.
default:
    @just --list

# PI/OpenAI: Unpaid structural/source audit on two real rows; not the API smoke test.
offline-check:
    HF_HUB_OFFLINE=1 uv run --locked python scripts/data/check_dataset.py --limit 2
    HF_HUB_OFFLINE=1 uv run --locked python scripts/data/label_actions.py --limit 2 --dry-run

# PI/OpenAI: Real model, cache and export on two scenarios.
fast-dev-run *args:
    BEARTYPE=1 uv run --locked python scripts/data/label_actions.py --limit 2 --concurrency 2 {{args}}

# PI/OpenAI: Audit every prompt against its pinned source row.
audit *args:
    uv run --locked python scripts/data/check_dataset.py {{args}}

# PI/OpenAI: Snapshot private, attributed rubrics without importing credentials.
rubrics:
    uv run --locked python scripts/data/import_virtue_rubrics.py

# PI/OpenAI: Only visible prompts enter the judge; all previous pilot charges count toward $40.
label *args:
    uv run --locked python scripts/data/label_actions.py --budget-usd 40 {{args}}

# PI/OpenAI: Summarise source histories locally; publishing is separate.
summarise *args:
    uv run --locked python scripts/data/summarise_machiavelli.py {{args}}

# PI/OpenAI: Restore displayed actions and narrator dialogue without inference.
prepare out:
    uv run --locked python scripts/data/prepare_inputs.py --out "{{out}}"

# PI/OpenAI: Read all diagnostic values and create seeded category samples.
feedback labels out *args:
    uv run --locked python scripts/data/audit_feedback.py "{{labels}}" --out "{{out}}" {{args}}

# PI/OpenAI: Unpaid integrity check against the saved real API requests and pinned source.
verify run:
    uv run --locked python scripts/data/verify_labels.py "{{run}}"

results:
    @cat results.md
