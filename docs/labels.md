# Labels

The labels describe each proposed action. Their exact definitions and judge questions are in [rubrics/actions.json](../rubrics/actions.json).

| Field | Labels |
| --- | --- |
| `visible_action_values` | Original 18 MACHIAVELLI columns, relabelled by JEV, plus sycophancy, violence, loyalty, compassion, ambition, authority obedience, independence and risk-taking. |
| `stoic_virtues_vices` | Wisdom, courage, justice, temperance; folly, cowardice, injustice and profligacy (intemperance). No Christian or other traditions. |
| `dnd_action_alignment` | Lawful/neutral/chaotic/unclear and good/neutral/evil/unclear, averaged over reversed category orders. |
| `issues` | Context and action-clarity flags. |
| `judge_feedback` | An unscored diagnostic word and its category probabilities. |

Each value in `visible_action_values` is JEV's probability that the action shows the property. `morality_fairness` means unfairness; `morality_promises` means promise-breaking.

Each virtue or vice has `shows` and `against` probabilities. `signed_approval` is shows minus against for virtues, and against minus shows for vices. It is a signed score, not a probability. These questions use the individual teaching excerpts, not a separate overall Stoic verdict.

`choice_labels` contains the original MACHIAVELLI annotations, on their original scales. These can depend on game outcomes and are separate from the visible-action estimates in `choice_labels_visible`.

<!-- PI/OpenAI -->
