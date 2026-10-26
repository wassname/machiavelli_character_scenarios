"""Select a frozen differentiated cohort and random controls from historical choices. -- PI/OpenAI"""

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tabulate import tabulate

from action_choice import behavior_eligible, behavior_question_key, ELIGIBILITY_PATH
from pca_choice_profiles import remove_component
from relative_virtue_profiles import MODELS

ROOT = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(run: Path, dimensions: Path, pca: Path, out: Path, seed: int = 20261026) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    scenes = {r['source_id']: r for r in map(json.loads, (run / 'scenes.jsonl').open())}
    replies = {(r['source_id'], r['model'], r['order']): r for r in map(json.loads, (run / 'choices.jsonl').open()) if r['model'] in MODELS}
    assert len(replies) == len(scenes) * len(MODELS) * 2
    assert all(r['status'] == 'ok' for r in replies.values())
    description = json.loads((dimensions / 'analysis.json').read_text())
    assert description['input_sha256']['scenes'] == sha(run / 'scenes.jsonl')
    assert description['input_sha256']['choices'] == sha(run / 'choices.jsonl')
    component = json.loads((pca / 'analysis.json').read_text())
    for path, expected in component['input_sha256'].items():
        assert sha(Path(path)) == expected, f'Changed PCA input: {path}'
    assert component['input_sha256'][str((dimensions / 'analysis.json').resolve())] == sha(dimensions / 'analysis.json')
    ds = description['dimensions']
    direct = [i for i, d in enumerate(ds) if d['family'] == 'virtues' and d['derived_from'] is None]
    items = [ds[i]['id'] for i in direct]
    signs = np.array([-1 if ds[i]['kind'] == 'vices' else 1 for i in direct])
    assert len(direct) == 66
    actions = {(r['source_id'], r['choice_id']): np.asarray(r['scores'])[direct] * signs
               for r in pq.read_table(dimensions / 'action_scores.parquet').to_pylist()}
    directions = {s: np.array([v['loadings'][i] for i in items]) for s, v in component['sources'].items()}
    assert all(np.isclose(u @ u, 1) for u in directions.values())
    reviews = json.loads(ELIGIBILITY_PATH.read_text())['cases']
    rows = []
    for sid, scene in scenes.items():
        if not behavior_eligible(scene, reviews):
            continue
        chosen = {name: [replies[sid, model, o]['original_choice_id'] for o in ('forward', 'reverse')] for model, name in MODELS.items()}
        stable = {m: c[0] for m, c in chosen.items() if c[0] == c[1]}
        gaps = [actions[sid, a] - actions[sid, b] for a, b in combinations(stable.values(), 2)]
        all_stable = len(stable) == 3
        residual_gap = float(np.sqrt(np.mean(remove_component(np.stack(gaps), directions[scene['source']]) ** 2))) if all_stable else None
        raw_gap = float(np.sqrt(np.mean(np.stack(gaps) ** 2))) if all_stable else None
        rows.append({'source_id': sid, 'source': scene['source'], 'eval_cluster': scene['eval_cluster'],
                     'residual_choice_gap': residual_gap, 'raw_choice_gap': raw_gap,
                     'stable_models': len(stable), 'stable_pairs_different': sum(a != b for a, b in combinations(stable.values(), 2)),
                     'native_tradition_tradeoff': scene['tradition_tradeoff_spread'], 'historical_choices': chosen,
                     'reviewed': sid in reviews, 'visible_question_sha256': behavior_question_key(scene),
                     'random_priority': hashlib.sha256(f'{seed}:{sid}'.encode()).hexdigest()})
    selected = []
    for source in sorted(directions):
        pool = [r for r in rows if r['source'] == source]
        controls = sorted(pool, key=lambda r: r['random_priority'])[:10]
        control_ids = {r['source_id'] for r in controls}
        ranked = sorted((r for r in pool if r['stable_models'] == 3 and r['stable_pairs_different'] > 0 and r['source_id'] not in control_ids),
                        key=lambda r: (-r['residual_choice_gap'], r['source_id']))
        clusters, targets = Counter(), []
        for rank, row in enumerate(ranked, 1):
            if clusters[row['eval_cluster']] < 3 and len(targets) < 40:
                targets.append(row | {'selection': 'differentiated', 'candidate_rank': rank})
                clusters[row['eval_cluster']] += 1
        assert len(targets) == 40, f'Not enough source-diverse stable differences: {source}'
        selected.extend(targets + [r | {'selection': 'random_control', 'candidate_rank': None} for r in controls])
    assert len(selected) == len({r['source_id'] for r in selected}) == 100
    assert Counter(r['selection'] for r in selected) == {'differentiated': 80, 'random_control': 20}
    manifest = {'author': 'PI/OpenAI', 'seed': seed, 'questions': selected,
                'definition': 'RMS across three historical actor-pair differences and 66 direct favorable labels after fixed source-PC1 removal. All three actors must agree with themselves across original/reversed order. Not proof of virtue disagreement.',
                'selection': 'Per source: 10 hash-random eligible controls first, then 40 largest residual choice gaps, at most three per game/topic. Controls are not conditioned on disagreement or rank. Parent source review precedes calls.',
                'models_used_for_selection': MODELS, 'model_selection_bias': 'Existing three actors and labels choose targets; four new model families test transfer. No population estimate from this selected mixture.',
                'pool_scenes': len(rows), 'stable_different_candidates': sum(r['stable_models'] == 3 and r['stable_pairs_different'] > 0 for r in rows),
                'selected_reviewed': sum(r['reviewed'] for r in selected),
                'input_sha256': {str(p.resolve()): sha(p) for p in (run / 'scenes.jsonl', run / 'choices.jsonl', dimensions / 'analysis.json', dimensions / 'action_scores.parquet', pca / 'analysis.json', ELIGIBILITY_PATH)},
                'script_sha256': sha(Path(__file__))}
    (out / 'selection.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'candidates.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / 'scenes.jsonl').write_text(''.join(json.dumps(scenes[r['source_id']], ensure_ascii=False) + '\n' for r in selected))
    headers = ['Question', 'Residual gap↑', 'Raw gap', 'Stable models /3', 'Different stable pairs /3', 'Native tradeoff', 'Selection']
    cells = [[f"{'D' if r['source'] == 'daily_dilemmas' else 'G'}{n % 50 + 1:02d}", 'unstable' if r['residual_choice_gap'] is None else f"{r['residual_choice_gap']:.3f}",
              'unstable' if r['raw_choice_gap'] is None else f"{r['raw_choice_gap']:.3f}", r['stable_models'],
              r['stable_pairs_different'], f"{r['native_tradition_tradeoff']:.3f}", r['selection']] for n, r in enumerate(selected)]
    table = tabulate(cells, headers=headers, tablefmt='pipe', disable_numparse=True, colalign=('left', 'right', 'right', 'right', 'right', 'right', 'left'))
    text = '# Pilot question selection\n\n' + manifest['definition'] + '\n\n' + manifest['selection'] + '\n\n' + table + '\n\n'
    text += 'D = everyday, G = game; question numbers follow the ten complete review files. Native tradeoff concerns any Pareto-nondominated options, not necessarily the historical choices; zero does not exclude within-tradition differences. No new actor replies used. -- PI/OpenAI\n'
    (out / 'selection.md').write_text(text)
    for start in range(0, 100, 10):
        text = '# Complete source questions and choices\n\nThe actor wrapper is the same own-judgement-v4 instruction for all models. These are complete decision contexts and literal options; all upstream text is preserved in scenes.jsonl. -- PI/OpenAI\n\n'
        for n, row in enumerate(selected[start:start + 10], start + 1):
            s = scenes[row['source_id']]
            assert s['decision_context'] in s['text'] and all(c in s['text'] for c in s['choice_text'].values())
            text += f"## {n}. {row['source_id']}\n\nControlled character: {s['player_character']}\n\n{s['decision_context']}\n\n"
            text += '\n'.join(f"- {k}: {v}" for k, v in s['choice_text'].items()) + '\n\n'
        (out / f'review_{start // 10 + 1:02d}.md').write_text(text)
    print(json.dumps({k: manifest[k] for k in ('pool_scenes', 'stable_different_candidates', 'selected_reviewed')}, indent=2))
    print(out / 'selection.md')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/20261024_action_preferences_resumed')
    parser.add_argument('--dimensions', type=Path, default=ROOT / 'outputs/20261025_individual_dimensions')
    parser.add_argument('--pca', type=Path, default=ROOT / 'outputs/20261026_pca_profiles')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/20261026_differentiated_pilot_selection')
    args = parser.parse_args()
    select(args.run, args.dimensions, args.pca, args.out)
