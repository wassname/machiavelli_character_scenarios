# MACHIAVELLI character scenarios

10,492 short scenes from 92 text-based games, with 37,176 possible actions. Each prompt gives the character, scene and choices. Use them to study how language models choose actions and explain their reasoning.

[Dataset](https://huggingface.co/datasets/wassname/machiavelli_character_scenarios) · [Code](https://github.com/wassname/machiavelli_character_scenarios)

```python
import json
from datasets import load_dataset

row = load_dataset("wassname/machiavelli_character_scenarios", split="train")[0]
print(row["text"])
choices = json.loads(row["choice_text"])
labels = json.loads(row["choice_labels_visible"])
```

## How it was made

DeepSeek V4 Flash shortened the available game histories from [MACHIAVELLI](https://huggingface.co/datasets/wassname/machiavelli). Actions come from the game's displayed choices.

`typesafe/jev-1.13` labels each proposed action using the complete visible prompt and that choice. It sees no hidden action outcomes, original annotations or model-generated answer. The [full rubric](https://github.com/wassname/machiavelli_character_scenarios/blob/main/rubrics/actions.json) is in JSON. A [complete saved judge request](https://github.com/wassname/machiavelli_character_scenarios/blob/main/examples/judge_request.json) shows the first scene and its first action exactly as sent, with every question.

## Labels

`choice_labels_visible` contains:

- The original 18 MACHIAVELLI columns, relabelled by JEV.
- Eight additions: sycophancy, violence, loyalty, compassion, ambition, authority obedience, independence and risk-taking.
- Four Stoic virtues and four vices, using their individual teaching excerpts.
- Dungeons & Dragons action alignment: law/chaos and good/evil.
- Separate context and diagnostic flags.

These are model estimates, not ground-truth moral answers.

`choice_labels` keeps the original annotations, which can depend on game outcomes. They use different scales from the JEV estimates. See the [label fields](https://github.com/wassname/machiavelli_character_scenarios/blob/main/docs/labels.md).

## Citation

Use [CITATION.cff](https://github.com/wassname/machiavelli_character_scenarios/blob/main/CITATION.cff), and also cite the original [MACHIAVELLI paper](https://arxiv.org/abs/2304.03279). Credit for the games belongs to their authors.

<!-- PI/OpenAI -->
