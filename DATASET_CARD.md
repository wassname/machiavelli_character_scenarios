---
license: mit
language:
- en
task_categories:
- text-generation
- text-classification
pretty_name: MACHIAVELLI character scenarios
tags:
- moral-dilemmas
- roleplay
- ai-safety
- synthetic
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: parquet/train.parquet
---

# MACHIAVELLI character scenarios

This dataset turns game histories into short scenes where a language model must choose an action. It contains 10,492 decision points from 92 games, with 37,176 available choices. Use it to study moral decisions or how a model's character changes its choices.

[Code and checks](https://github.com/wassname/machiavelli_character_scenarios) · [Label guide](https://github.com/wassname/machiavelli_character_scenarios/blob/main/docs/labels.md) · [Quality report](https://github.com/wassname/machiavelli_character_scenarios/blob/main/docs/quality.md)

## Use a prompt

```python
import json
from datasets import load_dataset

row = load_dataset("wassname/machiavelli_character_scenarios", split="train")[0]
print(row["text"])
choices = json.loads(row["choice_text"])
labels = json.loads(row["choice_labels_visible"])
```

Send the complete `text` field. It explains that this is a game, identifies the player character, describes the scene, lists the actions and asks for a choice with reasoning. Rebuilding it from selected fields can leave out those instructions.

Nested fields are stored as JSON strings. Choice IDs are strings such as `"0"`. IDs match the game's displayed choices; some scenes have more than ten.

## Two sets of labels

`choice_labels_visible` contains new estimates from `typesafe/jev-1.13`. The judge sees the prompt and one proposed choice. It assesses the intended act and reasonable inferences about it, before the action happens. It receives no hidden branch outcomes, rewards, original labels or model-generated answer.

These labels cover the existing 18 dimensions, including honesty, cooperation, harm and deception; eight further behaviours, including sycophancy and loyalty; four Stoic virtues and four vices; and both Dungeons & Dragons alignment dimensions. Two inherited names need care: `morality_fairness` means unfairness, and `morality_promises` means promise-breaking. Separate context flags let the judge say that information is missing or the action is unclear. An unscored diagnostic word flags other possible obstacles.

`choice_labels` preserves the upstream annotations. These can describe action outcomes rather than what a player can infer beforehand. Their numbers are not comparable to the new probabilities. Newly restored choices beyond ID 9 have no upstream annotation.

The new labels are model estimates, not human-validated moral answers. D&D labels describe an action's suggested tendency, not a person's stable alignment. Both option orders and their differences are saved. Neutral means a supported middle stance; unclear means there is too little evidence to identify a stance.

## Fields to know

| Field | What it contains |
| --- | --- |
| `text` | Complete prompt to give a model. |
| `choice_text` | Available actions, keyed by choice ID. |
| `choice_labels_visible` | New action estimates, context flags and diagnostic feedback. |
| `choice_labels` | Unchanged upstream annotations. |
| `decision_context` | Shortened game history used in the prompt. |
| `original_prompt` | Prompt fields before the rendering repair. |
| `input_repairs`, `input_context_repaired`, `input_issues` | Recorded repairs and remaining source issues. |
| `source_id`, `game_id`, `row_i` | Links back to the source decision. |

The remaining fields include the game premise, character name and the original selection scores. Those scores selected scenes with variation in upstream moral annotations; they are not scores for the newly judged actions.

## Context and quality

DeepSeek V4 Flash shortened the available game histories. A character name or short premise is not a full biography. Some prompts omit information needed to distinguish motives or relationships. Setup choices and exchanges with the game's narrator are retained and flagged rather than treated as ordinary moral decisions.

The rendering repair restores displayed choices instead of deleting unresolved game-code fragments. It also restores lost options and keeps the earlier prompt fields for comparison. Narrator exchanges about undoing a decision use their visible dialogue instead of an unrelated story summary. The quality report records the counts and category spot checks.

Checks verify coverage, source matching, saved API replies and the exported fields. They do not establish moral accuracy or calibrated probabilities. A diagnostic word of `none` does not certify a prompt; other words can describe valid choice content rather than a defect.

## Sources and citation

The source is [wassname/machiavelli](https://huggingface.co/datasets/wassname/machiavelli), which packages the [MACHIAVELLI benchmark](https://github.com/aypan17/machiavelli). Credit for the original games belongs to their authors. The generation scripts were imported from [persona-steering-template-library](https://github.com/wassname/persona-steering-template-library). The current code and pinned source revisions are in the [canonical repo](https://github.com/wassname/machiavelli_character_scenarios/blob/main/provenance.json).

The Stoic questions use a pinned reference from `tiny-virtues-dev`; its source extracts are not redistributed here. Reproducing those exact questions requires access to that reference; the other label questions are public. The D&D questions use operational readings of the [official alignment descriptions](https://www.dndbeyond.com/sources/dnd/free-rules/creating-a-character#Alignment).

Use [CITATION.cff](https://github.com/wassname/machiavelli_character_scenarios/blob/main/CITATION.cff) to cite this dataset. Also cite Pan et al. (2023), *Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the Machiavelli Benchmark*, listed there.

<!-- PI/OpenAI: dataset card prepared with annoy-less at wassname's request. -->
