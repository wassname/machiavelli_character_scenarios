"""Role-play check for the page: per model, how much its stated choice changes when only the character's trait flips, against how much it changes when the same scene is asked again.
Both use the total-variation distance between means of 2 answers on each side (draws 0-1 and 2-3 of 4), so the two numbers are comparable. -- PI/OpenAI"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MIND = {'chess': 'Chess ambition', 'everest': 'Everest dream', 'gamble': 'Love of gambling', 'mission': 'Spy mission'}  # traits of the mind: should not carry over


def main(runs: list[Path], out: Path) -> None:
    D = {}
    for run in runs:
        for stage in ('smoke', 'rest'):
            f = run / stage / 'choices.jsonl'
            if f.exists():
                for r in map(json.loads, f.open()):
                    if r['status'] == 'ok':
                        D.setdefault((r['source_id'], r['model']), {})[r['draw']] = r['original_probabilities']
    mean = lambda d, idx: {o: np.mean([d[i][o] for i in idx]) / 100 for o in d[0]}
    tv = lambda a, b: 0.5 * sum(abs(a[o] - b[o]) for o in a)
    halves = ([0, 1], [2, 3])
    rows, example = {}, {}
    for m in sorted({m for _, m in D}):
        flip, noise = {}, []
        for sid, mm in D:
            k = sid.split('_')[-2]
            if mm != m or not sid.endswith('_A') or k not in MIND:
                continue
            A, B = D[sid, m], D[sid[:-2] + '_B', m]
            assert len(A) == len(B) == 4, (sid, m)
            flip[MIND[k]] = float(np.mean([tv(mean(A, h), mean(B, h)) for h in halves]))  # A vs B, same draws
            noise += [tv(mean(A, halves[0]), mean(A, halves[1])), tv(mean(B, halves[0]), mean(B, halves[1]))]  # same scene asked again
        rows[m] = {'mind': {k: round(v, 4) for k, v in flip.items()}, 'mind_mean': round(float(np.mean(list(flip.values()))), 4), 'same_scene_again': round(float(np.mean(noise)), 4)}
        a, b = D['probe4_everest_A', m], D['probe4_everest_B', m]
        example[m] = {'dream': round(float(np.mean([a[i]['0'] for i in a])) / 100, 3), 'afraid': round(float(np.mean([b[i]['0'] for i in b])) / 100, 3)}
    out.write_text(json.dumps({'author': 'PI/OpenAI', 'runs': [str(r.relative_to(ROOT)) for r in runs],
                               'metric': 'total-variation distance between means of 2 answers (draws 0-1 vs 2-3); flip = A vs B scene, same_scene_again = same scene, first 2 vs last 2 answers',
                               'models': rows, 'everest_take_slot': example}, indent=1) + '\n')
    print(out, {m.split('/')[-1]: (r['mind_mean'], r['same_scene_again']) for m, r in rows.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    main([r.resolve() for r in a.runs], a.out)
