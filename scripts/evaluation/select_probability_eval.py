"""Choose the tiny eval from DeepSeek's stated probabilities only: uncertain dev set, random controls, confident landmarks. -- PI/OpenAI"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def entropy(p: dict) -> float:
    # Normalized entropy of the mean stated distribution: 0 = certain, 1 = uniform.
    return -sum(v * math.log(v) for v in p.values() if v > 0) / math.log(len(p))


def select(references: list[Path], out: Path, dev_fraction: float, cap: int, controls: int, landmarks: int) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    scenes = {s['source_id']: s for s in map(json.loads, (references[0] / 'scenes.jsonl').open())}
    rows = [json.loads(l) for ref in references for path in sorted(ref.glob('*/choices.jsonl')) for l in path.open()]
    # One reply per question/draw/reasoning level; a later valid redraw replaces a failed attempt.
    best = {}
    for r in rows:
        k = (r['source_id'], r['draw'], r.get('level'))
        if r['status'] == 'ok' or k not in best:
            best[k] = r
    failed = [k for k, r in best.items() if r['status'] != 'ok']
    assert not failed, failed
    by = defaultdict(list)
    for r in best.values():
        by[r['source_id']].append(r)
    assert set(by) == set(scenes) and all(len(v) >= 2 for v in by.values()), Counter(len(v) for v in by.values())
    info = {}
    for sid, rs in by.items():
        mean = {o: sum(r['original_probabilities'][o] for r in rs) / (100 * len(rs)) for o in scenes[sid]['choice_text']}
        info[sid] = {'entropy': entropy(mean), 'mean_probabilities': mean,
                     'draws': len(rs), 'draw_choices': [r['original_choice_id'] for r in rs], 'draws_disagree': len({r['original_choice_id'] for r in rs}) > 1}
    picked = {}
    for source in ('daily_dilemmas', 'machiavelli'):
        ids = sorted((s for s in scenes if scenes[s]['source'] == source), key=lambda s: (-info[s]['entropy'], h('dev:' + s)))
        n_dev = round(dev_fraction * len(ids))
        clusters = defaultdict(list)
        for s in ids:
            clusters[scenes[s]['eval_cluster']].append(s)
        dev = [v[0] for v in clusters.values()]  # most uncertain question of every game/topic
        per = Counter(scenes[s]['eval_cluster'] for s in dev)
        for s in ids:
            if len(dev) >= n_dev:
                break
            c = scenes[s]['eval_cluster']
            if s not in dev and per[c] < cap:
                dev.append(s)
                per[c] += 1
        rest = [s for s in ids if s not in set(dev)]
        # Controls ignore DeepSeek's answers entirely: hash order only.
        ctrl = sorted(rest, key=lambda s: h('control:' + s))[:controls]
        left = [s for s in rest if s not in set(ctrl)]
        # Landmarks: the dev set is mostly mid-to-high uncertainty, so add the extremes it lacks for calibration:
        # half the most confident and half the least confident of the remainder, at most 2 per game/topic.
        landmark_ids, landmark_per = [], Counter()
        ranked = sorted(left, key=lambda s: (info[s]['entropy'], h('landmark:' + s)))
        for end in (ranked, ranked[::-1]):
            taken = 0
            for s in end:
                c = scenes[s]['eval_cluster']
                if taken < landmarks // 2 and s not in landmark_ids and landmark_per[c] < 2:
                    landmark_ids.append(s)
                    landmark_per[c] += 1
                    taken += 1
        for role, group in (('dev', dev), ('control', ctrl), ("landmark", landmark_ids)):
            for s in group:
                picked[s] = role
    counts = Counter((scenes[s]['source'], r) for s, r in picked.items())
    result = {'author': 'PI/OpenAI', 'references': [str(r) for r in references], 'counts': {f'{a}/{b}': n for (a, b), n in sorted(counts.items())},
              'rule': f'Per source: dev = most uncertain question of every game/topic, then by DeepSeek normalized entropy (mean of all valid rotated draws: 2 everyday at medium effort, 4 games = 2 medium + 2 at reasoning 10k) up to {dev_fraction:.0%} of the source, at most {cap} per game/topic. '
                      f'Controls = {controls} hash-random from the rest (not using DeepSeek answers). Landmarks = {landmarks}: half the most and half the least confident of the remainder by DeepSeek entropy, at most 2 per game/topic, added so calibration covers the extremes the dev set lacks.',
              'questions': [{'source_id': s, 'role': r, 'source': scenes[s]['source'], 'eval_cluster': scenes[s]['eval_cluster']} | info[s] for s, r in picked.items()]}
    (out / 'selection.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'scenes.jsonl').write_text(''.join(json.dumps(scenes[s] | {'eval_role': r}, ensure_ascii=False) + '\n' for s, r in picked.items()))
    entropies = {role: sorted(info[s]['entropy'] for s, r in picked.items() if r == role) for role in ('dev', 'control', 'landmark')}
    print(json.dumps(result['counts'], indent=2))
    print({role: (round(v[0], 2), round(v[len(v) // 2], 2), round(v[-1], 2)) for role, v in entropies.items()}, '(min, median, max entropy)')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references', type=Path, nargs='+', default=[ROOT / 'outputs/20261026_probability_reference', ROOT / 'outputs/20261026_probability_reference_redraw', ROOT / 'outputs/20261026_probability_reference_games_10k'])
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/20261026_probability_eval_selection')
    parser.add_argument('--dev-fraction', type=float, default=.10)
    parser.add_argument('--cap', type=int, default=8)
    parser.add_argument('--controls', type=int, default=50)
    parser.add_argument('--landmarks', type=int, default=20)
    a = parser.parse_args()
    select(a.references, a.out, a.dev_fraction, a.cap, a.controls, a.landmarks)
