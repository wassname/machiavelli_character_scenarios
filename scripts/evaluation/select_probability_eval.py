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


def select(reference: Path, out: Path, dev_fraction: float, cap: int, controls: int, landmarks: int) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    scenes = {s['source_id']: s for s in map(json.loads, (reference / 'scenes.jsonl').open())}
    rows = [json.loads(l) for stage in ('smoke', 'rest') for l in (reference / stage / 'choices.jsonl').open()]
    assert all(r['status'] == 'ok' for r in rows), Counter(r['status'] for r in rows)
    by = defaultdict(list)
    for r in rows:
        by[r['source_id']].append(r)
    assert set(by) == set(scenes) and all(len(v) == 2 for v in by.values())
    info = {}
    for sid, rs in by.items():
        mean = {o: sum(r['original_probabilities'][o] for r in rs) / 200 for o in scenes[sid]['choice_text']}
        info[sid] = {'entropy': entropy(mean), 'mean_probabilities': mean,
                     'draw_choices': [r['original_choice_id'] for r in rs], 'draws_disagree': rs[0]['original_choice_id'] != rs[1]['original_choice_id']}
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
        landmark_ids, landmark_per = [], Counter()
        for s in sorted(left, key=lambda s: (info[s]['entropy'], h('landmark:' + s))):
            c = scenes[s]['eval_cluster']
            if len(landmark_ids) < landmarks and landmark_per[c] < 2:
                landmark_ids.append(s)
                landmark_per[c] += 1
        for role, group in (('dev', dev), ('control', ctrl), ("landmark", landmark_ids)):
            for s in group:
                picked[s] = role
    counts = Counter((scenes[s]['source'], r) for s, r in picked.items())
    result = {'author': 'PI/OpenAI', 'reference': str(reference), 'counts': {f'{a}/{b}': n for (a, b), n in sorted(counts.items())},
              'rule': f'Per source: dev = most uncertain question of every game/topic, then by DeepSeek normalized entropy (mean of 2 rotated draws) up to {dev_fraction:.0%} of the source, at most {cap} per game/topic. '
                      f'Controls = {controls} hash-random from the rest (not using DeepSeek answers). Landmarks = {landmarks} lowest-entropy from the remainder, at most 2 per game/topic, used only to calibrate each model\'s scale.',
              'questions': [{'source_id': s, 'role': r, 'source': scenes[s]['source'], 'eval_cluster': scenes[s]['eval_cluster']} | info[s] for s, r in picked.items()]}
    (out / 'selection.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'scenes.jsonl').write_text(''.join(json.dumps(scenes[s] | {'eval_role': r}, ensure_ascii=False) + '\n' for s, r in picked.items()))
    entropies = {role: sorted(info[s]['entropy'] for s, r in picked.items() if r == role) for role in ('dev', 'control', 'landmark')}
    print(json.dumps(result['counts'], indent=2))
    print({role: (round(v[0], 2), round(v[len(v) // 2], 2), round(v[-1], 2)) for role, v in entropies.items()}, '(min, median, max entropy)')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=ROOT / 'outputs/20261026_probability_reference')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/20261026_probability_eval_selection')
    parser.add_argument('--dev-fraction', type=float, default=.10)
    parser.add_argument('--cap', type=int, default=8)
    parser.add_argument('--controls', type=int, default=50)
    parser.add_argument('--landmarks', type=int, default=50)
    a = parser.parse_args()
    select(a.reference, a.out, a.dev_fraction, a.cap, a.controls, a.landmarks)
