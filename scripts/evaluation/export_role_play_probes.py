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
    rng = np.random.default_rng(20261028)
    for m in sorted({m for _, m in D}):
        flip, noise, pairs = {}, [], []
        for sid, mm in D:
            k = sid.split('_')[-2]
            if mm != m or not sid.endswith('_A') or k not in MIND:
                continue
            A, B = D[sid, m], D[sid[:-2] + '_B', m]
            assert len(A) == len(B) == 4, (sid, m)
            flip[MIND[k]] = float(np.mean([tv(mean(A, h), mean(B, h)) for h in halves]))  # A vs B, same draws
            noise += [tv(mean(A, halves[0]), mean(A, halves[1])), tv(mean(B, halves[0]), mean(B, halves[1]))]  # same scene asked again
            pairs.append((A, B))
        gap = lambda ps: (np.mean([np.mean([tv(mean(A, h), mean(B, h)) for h in halves]) for A, B in ps])
                          - np.mean([tv(mean(X, halves[0]), mean(X, halves[1])) for A, B in ps for X in (A, B)]))
        def shuffle(A, B):  # null of no role-play: the 8 answers of a scene pair are exchangeable between versions A and B
            pool = [A[i] for i in A] + [B[i] for i in B]
            k = rng.permutation(8)
            return dict(enumerate(pool[i] for i in k[:4])), dict(enumerate(pool[i] for i in k[4:]))
        null = np.array([gap([shuffle(A, B) for A, B in pairs]) for _ in range(2000)])
        rows[m] = {'mind': {k: round(v, 4) for k, v in flip.items()}, 'mind_mean': round(float(np.mean(list(flip.values()))), 4), 'same_scene_again': round(float(np.mean(noise)), 4),
                   'role_play': round(float(gap(pairs)), 4), 'null_95': round(float(np.percentile(null, 95)), 4), 'p_value': round(float((1 + (null >= gap(pairs)).sum()) / (1 + len(null))), 4)}
        a, b = D['probe4_everest_A', m], D['probe4_everest_B', m]
        example[m] = {'dream': round(float(np.mean([a[i]['0'] for i in a])) / 100, 3), 'afraid': round(float(np.mean([b[i]['0'] for i in b])) / 100, 3)}
    out.write_text(json.dumps({'author': 'PI/OpenAI', 'runs': [str(r.relative_to(ROOT)) for r in runs],
                               'metric': 'total-variation distance between means of 2 answers (draws 0-1 vs 2-3); flip = A vs B scene, same_scene_again = same scene, first 2 vs last 2 answers; role_play = flip - same_scene_again; null_95 / p_value: 2,000 shuffles of the 8 answers of each scene pair between versions A and B (no role-play), scenes fixed',
                               'models': rows, 'everest_take_slot': example}, indent=1) + '\n')
    for m, r in rows.items():
        print(f"{m.split('/')[-1]:28s} role-play {r['role_play']:+.3f}  null 95% {r['null_95']:+.3f}  p {r['p_value']:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    main([r.resolve() for r in a.runs], a.out)
