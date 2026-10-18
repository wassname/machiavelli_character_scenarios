# Quality checks and repairs

The diagnostic field helps find cases to inspect. It is one selected word, with category probabilities, rather than a free-text explanation. Every saved diagnostic value and its probability entries were read by the audit. A seeded example of each selected category was read in full and compared with the source where an input defect was suspected.

## What changed?

The repair uses actions already displayed in the current source observation. It does not read hidden action outcomes or ask a model to invent missing text.

- Displayed choices differ in 1,750 scenes. Some differences are formatting or names/titles; others restore lost content. The exporter had deleted conditional game text, mistaken variable names for pronouns, stripped ordinary words such as "Intimidation", or stopped after ten options.
- The repair restores 41 missing choices, including nine with IDs above nine. There are now 37,176 choices across the same 10,492 scenes.
- Another 23 scenes contain exchanges with the game's narrator about undoing a decision. Their context now quotes the visible recent dialogue and current reply, rather than a summary of an earlier zombie attack.
- All 1,773 changed prompts are rejudged in full. Unchanged prompts reuse exact-request cached replies. The earlier prompt fields remain in `original_prompt`; the upstream annotations remain in `choice_labels`.

For example, one old action was `Dare Hjalta to see Hallvard Name's .` The source already displayed `Dare Hjalta to see Hallvard's impending death as an opportunity for her and for us.` Re-summarising the story would not have repaired that deleted action text.

The remaining 160 scenes with `user_input` in their displayed actions have a `source_user_input_placeholder` flag. Those placeholders already occur in the source. They are preserved instead of guessing a name. Other missing relationship or motive information can remain; the context flags support further inspection.

Residual flags need reading too. Some `typo` flags concern intentional pronouns such as `ym` or `zhir`; these are kept. Some `truncated` flags concern deliberate ellipses or interrupted speech; these are also kept. One displayed source action is still `Bribe them with Hippolyta's .` The missing object is unknown, so the release retains the action and its warning rather than inventing it.

## Read every category, then repair or keep

Before repairs, 1,709 of 37,135 choices had a word other than `none`, across 1,342 scenes. In the release, 1,385 of 37,176 choices have such a word, across 1,109 scenes. These are model flag counts, not confirmed-defect rates. The extra 41 choices change the denominator.

Every selected word and all 218 probability entries per record were parsed. A full seeded example of every observed category was read in each pass, including `none`. Samples and final counts are in the dataset's `checks/category_samples.jsonl` and `checks/feedback_summary.json`.

On the 37,135 shared choices, 427 flags changed to `none`, 103 changed from `none` to a flag, and 1,282 remained flagged. None of the 41 restored choices received a non-none word. These changes measure the judge's feedback after input repair, not human-validated accuracy.

| Selected word | Final choices | Before repairs | Repair or keep decision |
| --- | ---: | ---: | --- |
| none | 35,791 | 35,426 | Keep. This word does not validate an input; inspect the separate context flags. |
| ambiguous | 971 | 1,122 | Keep valid ambiguity. Restore unresolved action names through the general renderer. |
| underspecified | 253 | 300 | Restore literal options such as Choice2 from displayed actions. Keep genuinely open-ended choices. |
| placeholder | 98 | 44 | Restore raw variable names. Keep and flag placeholders already in the source; these explain the increase after rendering. |
| typo | 16 | 95 | Restore deleted words and name/pronoun corruption. Keep source spelling and intentional pronouns. |
| truncated | 13 | 83 | Restore exporter deletions. Keep source ellipses; retain a warning where the source itself is incomplete. |
| hedging | 11 | 11 | Keep tentative plans. Tentativeness can be intended. |
| consent | 5 | 6 | Keep. Consent is choice content, not automatically bad data; relationship context can still be missing. |
| overconfident | 5 | 5 | Keep. A character can adopt an unjustifiably confident stance. |
| missing | 3 | 7 | Restore displayed names. Keep uncertainty where a needed relationship is still unspecified. |
| fabricated | 2 | 3 | Keep fictional lore and missing-name warnings. This word does not prove the exporter invented a fact. |
| persona | 2 | 2 | Keep character-creation choices and their setup flag. Do not infer morality from appearance. |
| roleplay | 2 | 2 | Keep background choices and their setup flag. |
| generic | 1 | 1 | Keep the spoken slogan. A virtue word alone does not establish virtuous conduct. |
| context | 1 | 2 | Repair narrator framing. One remaining complaint about game hints lacks the earlier event needed to check its truth; keep the warning. |
| repetition | 1 | 3 | Keep. The repeated phrase also appears in the source and does not change the proposed act. |
| hypothetical | 1 | 0 | Keep a character's conditional thought about a rival's death. It does not establish that death happened or that the character will cause it. |
| empty | 0 | 11 | Restore actions reduced to a full stop. |
| mismatch | 0 | 9 | Restore displayed names/pronouns; retain separate context flags for unresolved referents. |
| garbled | 0 | 2 | Restore displayed text instead of template fragments. |
| wording | 0 | 1 | Restore displayed wording instead of a literal variable name. |

A small blind pilot review by `deepseek/deepseek-v4.1-flash` also supplied unscored notes. Every note was read; these pilot notes are not per-choice fields in the release. Some requested useful context; others demanded actual success, a hidden outcome or an assistant's rationale. Those demands were rejected because the task concerns the proposed action before it happens.

## What the checks establish

The checks compare every choice with the pinned source observation, verify every saved label against its API reply or exact-request cache, and check every exported field. Negative checks remove a displayed option, alter an original annotation, or omit configured judge questions; the checks reject each. Ordinary story dialogue mentioning undo and a stale undo discussion do not trigger the narrator repair. Evidence: `checks/negative_checks.json` in the dataset.

This establishes data integrity and records the repairs. It does not establish moral accuracy, probability calibration, or complete scene understanding. The full audit summaries and release counts accompany the dataset under `checks/`.

<!-- PI/OpenAI: quality report. -->
