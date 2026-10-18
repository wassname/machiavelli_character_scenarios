# Reading the labels

Each entry in `choice_labels_visible` describes one proposed action. The shared input contains only the complete visible prompt and that choice. The judge estimates properties of the intended act; it does not predict the game's hidden result or assess a generated answer.

The numbers come from `typesafe/jev-1.13` through OpenRouter's decisions API. Narrow yes/no questions return probabilities between zero and one. A low value means little support for that property in the supplied text. It does not prove its opposite. These probabilities have not been calibrated against human judgements on this dataset.

## Behaviours

`visible_action_values` contains the 18 dimension names inherited from the source, with newly asked questions, plus eight further behaviours. The definitions are in [rubrics/actions.json](../rubrics/actions.json).

- The aggregate names cover honesty, cooperation, morality, power and utility.
- The other inherited names cover deception, unfairness, intended harm, killing, manipulation, non-physical harm, other harm, physical harm, broken promises, spying, stealing, trespassing and vandalism.
- The further behaviours are sycophancy, violence, loyalty, compassion, ambition, obedience to authority, independence and risk-taking.

Two inherited names need care: `morality_fairness` asks about **unfairness**, and `morality_promises` asks about **breaking promises**. These keep the original naming direction. Use the definitions, rather than assuming that every high number means something desirable.

`choice_labels` retains the old source annotations separately. It uses different scales and may depend on outcomes unavailable to the player. It also lacks entries for restored IDs above nine. Do not replace a missing upstream annotation with zero.

## Virtues and vices

`stoic_virtues_vices` covers wisdom, courage, justice and temperance, plus folly, cowardice, injustice and profligacy (intemperance). Each has separate `shows` and `against` probabilities. For a vice, `against` asks whether the act actively resists that vice; merely not showing it is insufficient.

`signed_approval` is `shows - against` for a virtue and `against - shows` for a vice. It ranges from minus one to one and is not a probability. The source wording is pinned by hash in [provenance.json](../provenance.json); the private reference extracts are not included in this repo.

## Dungeons & Dragons alignment

`dnd_action_alignment` contains two axes: `order` (lawful, neutral, chaotic, unclear) and `morality` (good, neutral, evil, unclear). Neutral means a supported middle stance; unclear means there is too little evidence to choose a stance.

Each axis is asked twice with the category order reversed. `probabilities` averages those two vectors; `by_order` retains them. `order_discrepancy` is the largest absolute change for any category. Reversing the options checks one source of judge sensitivity; it does not establish correctness or complete order invariance.

## Context and diagnostic feedback

`issues` contains seven separate probabilities: character/setup choice, unclear action, missing relationship, missing motive, conflicting facts, dependence on unseen outcomes, and insufficient context. No quality threshold has been validated for these. Use them to select cases for inspection.

`judge_feedback` contains one selected diagnostic word and the full category probabilities. JEV cannot write a free-text explanation. The word is kept out of every moral score. A flag such as consent can describe the action's content; none is not proof of adequate context.

The [quality report](quality.md) records the complete category counts and the repair-or-keep decisions.

<!-- PI/OpenAI: label guide. -->
