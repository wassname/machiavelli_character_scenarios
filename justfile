set shell := ["bash", "-cu"]

# PI/OpenAI: Show the available commands.
default:
    @just --list

# PI/OpenAI: Download the published dataset for evaluation.
data:
    uv run --locked hf download wassname/virtue-choices parquet/train.parquet --repo-type dataset --local-dir data/processed

# PI/OpenAI: Check the real source and request path without API calls.
offline-check:
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p outputs
    out="$(mktemp -d outputs/offline_XXXXXX)"
    HF_HUB_OFFLINE=1 uv run --locked python scripts/data/prepare_inputs.py --limit 2 --out "$out/inputs"
    HF_HUB_OFFLINE=1 uv run --locked python scripts/data/label_actions.py --inputs "$out/inputs/inputs.jsonl" --limit 2 --dry-run --out "$out/requests"

# PI/OpenAI: Real API, scoring, runtime types and cache on two scenarios (at most $0.05).
fast-dev-run *args: data
    BEARTYPE=1 uv run --locked python scripts/checks/fast_dev_run.py {{args}}

# PI/Astra: Recompute behaviour, individual-virtue and D&D figures from saved choices; no inference.
choice-profiles run="outputs/20261024_action_preferences_resumed" dataset="outputs/20261024_release21_numeric/parquet/train.parquet" out="outputs/20261025_individual_dimensions":
    uv run --locked python scripts/evaluation/score_choice_dimensions.py --run "{{run}}" --dataset "{{dataset}}" --out "{{out}}"
    uv run --locked python scripts/checks/check_choice_dimensions.py --analysis "{{out}}/analysis.json" --out "{{out}}/verification.json"
    uv run scripts/evaluation/plot_choice_dimensions.py --analysis "{{out}}/analysis.json" --out "{{out}}"

# PI/OpenAI: Remove the shared label component from saved profiles; no inference.
choice-pca input="outputs/20261025_individual_dimensions" out="outputs/20261026_pca_profiles" preview=".local/choice_preview":
    OPENBLAS_NUM_THREADS=1 uv run --locked python scripts/evaluation/pca_choice_profiles.py --input "{{input}}" --out "{{out}}"
    uv run --script scripts/evaluation/plot_pca_choice_profiles.py --input "{{out}}" --out "{{preview}}"

# PI/OpenAI: Rebuild and verify the frozen four-model pilot from native receipts; no inference.
choice-pilot run="outputs/20261026_differentiated_pilot_v2" preview=".local/choice_preview/pilot":
    OPENBLAS_NUM_THREADS=1 uv run --locked python scripts/evaluation/score_differentiated_pilot.py --run "{{run}}"
    uv run --locked python scripts/checks/check_differentiated_pilot.py --run "{{run}}"
    uv run --script scripts/evaluation/plot_differentiated_pilot.py --run "{{run}}" --out "{{preview}}"

# PI/OpenAI: Normalize labels within each question and plot saved historical/pilot replies; no inference.
choice-question-normalization:
    OPENBLAS_NUM_THREADS=1 uv run --locked --with matplotlib python scripts/evaluation/question_normalized_profiles.py

# PI/OpenAI: Build the random-set page (http://localhost:8081/random1000/) from a run's question tables; no inference.
page-random run="outputs/20261027_random1000_v10" out=".local/choice_preview/random1000":
    OPENBLAS_NUM_THREADS=1 uv run --locked --with matplotlib python scripts/evaluation/plot_probability_eval.py --tables "{{run}}/question_tables.jsonl" --out "{{out}}"

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

# PI/Sol: Freeze local tiny-virtues teachings without inference.
freeze-virtues source:
    uv run --locked python scripts/data/freeze_virtues.py "{{source}}"

# PI/Sol: Default is unpaid request construction and cost estimation.
virtues *args:
    uv run --locked python scripts/data/label_virtues.py {{args}}

# PI/Sol: Check local virtue requests and parsing without inference.
virtues-check:
    uv run --locked python scripts/checks/check_virtues.py

# PI/Sol: Reconstruct native requests/replies, every parquet score and its ledger charge.
virtues-verify run:
    uv run --locked python scripts/checks/verify_virtues.py "{{run}}"

# PI/Sol: Export a complete 80-item choice-level overlay without inference.
virtues-export run:
    uv run --locked python scripts/data/export_virtues.py "{{run}}"

# PI/Sol: Stage and locally load-check the full scene table; never upload.
stage-virtues run out *args:
    HF_HUB_OFFLINE=1 uv run --locked python scripts/data/stage_virtue_release.py --run "{{run}}" --native-verification "{{run}}/native_verification.json" --out "{{out}}" {{args}}

# PI/Sol: Choose actions with models; default is an unpaid request/cost preflight.
action-eval *args:
    uv run --locked python scripts/evaluation/action_choice.py {{args}}

# PI/Sol: Check prompt blinding, order mapping, parsing and the uniform null.
action-eval-check:
    uv run --locked python scripts/checks/check_action_choice.py

# PI/Sol: Score saved choices against the independently labelled action table.
action-eval-score run virtues:
    uv run --locked python scripts/evaluation/score_action_choice.py "{{run}}" --virtues "{{virtues}}"

# PI/Sol: Pair original DailyDilemmas actions, retaining hidden fields only as annotations.
prepare-dailydilemmas out:
    uv run --locked python scripts/data/prepare_dailydilemmas.py --out "{{out}}"

# PI/Sol: Check exact DD source actions, license and hidden-field blinding without inference.
dailydilemmas-check inputs:
    uv run --locked python scripts/checks/check_dailydilemmas.py "{{inputs}}"

# PI/OpenAI: GitHub Pages site (React): copy the D&D data, build, and screenshot the chart for the README.
site-build run="outputs/20261027_random1000_v12":
    cp "{{run}}/calibration_heldout/dnd_web.json" site/src/data/dnd_web.json
    cd site && pnpm install --frozen-lockfile && pnpm exec tsc --noEmit && pnpm exec vite build

site-shot:
    #!/usr/bin/env bash
    set -euo pipefail
    (cd site && exec node_modules/.bin/vite preview --host 127.0.0.1 --port 8083 --strictPort > /dev/null) & server=$!
    trap 'kill $server' EXIT
    sleep 3
    google-chrome --headless=new --no-sandbox --disable-gpu --hide-scrollbars --window-size=1300,1000 --virtual-time-budget=4000 --screenshot=docs/figures/dnd_alignment.png "http://127.0.0.1:8083/?shot"
