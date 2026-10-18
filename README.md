# MACHIAVELLI character scenarios

Short game scenes for studying the choices a language model makes. The dataset has 10,492 decision points from 92 games, with a prompt and a set of possible actions for each.

```python
import json
from datasets import load_dataset

row = load_dataset("wassname/machiavelli_character_scenarios", split="train")[0]
print(row["text"])  # Send the complete prompt, including its game instructions.
labels = json.loads(row["choice_labels_visible"])
```

[Dataset and card](https://huggingface.co/datasets/wassname/machiavelli_character_scenarios) · [Label guide](docs/labels.md) · [Quality checks](docs/quality.md)

## Which labels should I use?

`choice_labels_visible` describes each proposed action using only the prompt the model sees. It includes honesty, cooperation, harm, other behaviours, Stoic virtues and vices, and the two Dungeons & Dragons alignment dimensions. `typesafe/jev-1.13` (JEV) supplied these estimates via OpenRouter. They have not been validated against human judgements.

`choice_labels` keeps the upstream MACHIAVELLI annotations. Those can depend on what happens after the action. They answer a different question and use different scales.

Choices now come from the game's displayed action list. `original_prompt` preserves the earlier export. The source still has `user_input` placeholders in 160 scenes' actions; other context gaps can remain. Use the context flags to find cases to inspect. A diagnostic word of `none` does not certify a scene.

## Reproduce the work

Use Python, [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```sh
uv sync --locked
just prepare outputs/repaired_inputs
just fast-dev-run --inputs outputs/repaired_inputs/inputs.jsonl --no-virtues
just label --inputs outputs/repaired_inputs/inputs.jsonl --no-virtues --out outputs/labels
just verify outputs/labels
```

Labelling requires this project's `OPENROUTER_API_KEY`. Requests are cached; all new inference shares a US$40 project limit. The examples omit Stoic questions with the explicit `--no-virtues` option. Reproducing the published Stoic questions also requires the pinned reference file from `tiny-virtues-dev`, which is not redistributed here. See [provenance.json](provenance.json).

## Sources and citation

Scenes and original annotations come from [MACHIAVELLI](https://github.com/aypan17/machiavelli), via [wassname/machiavelli](https://huggingface.co/datasets/wassname/machiavelli). The summaries use DeepSeek V4 Flash. Generation scripts were moved from [persona-steering-template-library](https://github.com/wassname/persona-steering-template-library); their original hashes are recorded in the provenance file.

Use [CITATION.cff](CITATION.cff) to cite this dataset, and also cite the original MACHIAVELLI paper listed there. Credit for the games belongs to their authors.

<!-- PI/OpenAI: README prepared with annoy-less at wassname's request. -->
