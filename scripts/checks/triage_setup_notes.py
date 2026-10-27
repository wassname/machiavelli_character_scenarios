"""Count what models flag in their setup notes for a run, by kind of problem, with hash-chosen examples, so data bugs are triaged systematically. -- PI/OpenAI"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[2]

# Kind of complaint -> pattern. Overlapping kinds are allowed; "other" = a non-empty note matching none.
KINDS = {
    'name placeholder / template text': r'user_input|placeholder|\[chosen name|template|variable artifact',
    'garbled, cut-off or typo text': r'truncat|cut off|garbled|typo|incomplete sentence|malformed',
    'option text ambiguous': r'ambiguous|unclear (what|whether|who)|could mean|interpret',
    'missing middle option / either-or': r'binary|dichotom|middle|compromise|false choice|only two|omit',
    'missing facts / underspecified': r'unknown|unspecified|not specified|does not specify|missing (detail|info|context)|underspecified',
    'options choose past or backstory': r'already (happened|occurred|done)|backstory|past event|in the past',
    'role / frame (character vs self)': r'past life|inherited|previous (occupant|personality)|in character|the character\'s|role-?play',
    'duplicate or overlapping options': r'overlap|duplicate|same (action|option)|identical',
}


def main(runs: list[Path], out: Path) -> None:
    notes = []
    for run in runs:
        for stage in ('smoke', 'rest'):
            f = run / stage / 'choices.jsonl'
            if f.exists():
                notes += [(r['model'].split('/')[-1], r['source'], r['source_id'], r['setup_note'] or '') for r in map(json.loads, f.open()) if r['status'] == 'ok']
    counts, examples = defaultdict(Counter), defaultdict(list)
    for m, src, sid, note in sorted(notes, key=lambda x: hashlib.sha256((x[2] + x[0]).encode()).hexdigest()):
        hit = [k for k, p in KINDS.items() if re.search(p, note, re.I)]
        for k in hit or (['other'] if note.strip() else ['empty']):
            counts[k][src] += 1
            counts[k][m] += 1
            if len(examples[k]) < 4:
                examples[k].append(f'{m} · {sid}: {note[:240]}')
    models = sorted({n[0] for n in notes})
    total = Counter(n[1] for n in notes)
    rows = [[k, *(f"{counts[k][s] / total[s]:.0%}" for s in ('daily_dilemmas', 'machiavelli')), *(counts[k][m] for m in models)] for k in [*KINDS, 'other', 'empty']]
    report = (f'# What models flag in setup notes\n\nRuns: {", ".join(str(r.relative_to(ROOT)) for r in runs)}; {len(notes):,} replies. '
              'A note can match several kinds; patterns in `scripts/checks/triage_setup_notes.py`.\n\n'
              + tabulate(rows, headers=['Kind', 'Everyday (% of replies)', 'Games (% of replies)', *models], tablefmt='github')
              + '\n\n## Hash-chosen examples\n\n' + '\n\n'.join(f'### {k}\n\n' + '\n'.join(f'- {e}' for e in v) for k, v in examples.items()) + '\n\n-- PI/OpenAI\n')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'setup_note_triage.md').write_text(report)
    print(report[:3000])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    main([r.resolve() for r in a.runs], a.out)
