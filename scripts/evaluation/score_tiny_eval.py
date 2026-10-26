"""Score stated option probabilities: per-question nats vs the model average, draw noise, position bias. -- PI/OpenAI"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[2]
NAMES = {'meta/muse-spark-1.3-contributor': 'Muse', 'qwen/qwen3.8-27b': 'Qwen', 'google/gemini-3.8-flash': 'Gemini', 'openai/gpt-5.6-luna': 'Luna'}


def logp(p: dict, k: int) -> dict:
    # Add-one smoothing on the 0..100 scale, so a stated 0 is finite.
    return {o: np.log((v + 1) / (100 + k)) for o, v in p.items()}


def analyze(run: Path) -> dict:
    rows = [json.loads(l) for stage in ('smoke', 'trial') for l in (run / stage / 'choices.jsonl').open()]
    scenes = {s['source_id']: s for s in map(json.loads, (run / 'scenes.jsonl').open())}
    status = defaultdict(int)
    for r in rows:
        status[r['status']] += 1
    ok = [r for r in rows if r['status'] == 'ok']
    by = defaultdict(list)
    for r in ok:
        by[r['source_id'], NAMES[r['model']]].append(r)
    models = list(NAMES.values())
    questions, noise_ratio = [], []
    for sid, s in scenes.items():
        opts = sorted(s['choice_text'], key=int)
        k = len(opts)
        if not all(len(by[sid, m]) == 4 for m in models):
            continue
        # draws x options log-probabilities per model
        L = {m: np.array([[logp(r['original_probabilities'], k)[o] for o in opts] for r in by[sid, m]]) for m in models}
        mean_L = np.stack([L[m].mean(0) for m in models])          # models x options
        nats = mean_L - mean_L.mean(0, keepdims=True)              # vs per-question model average
        within = np.mean([L[m].var(0, ddof=1).mean() for m in models])
        between = mean_L.var(0, ddof=1).mean()
        noise_ratio.append(between / (within / 4 + 1e-12))
        P = {m: np.mean([[r['original_probabilities'][o] for o in opts] for r in by[sid, m]], 0) for m in models}
        questions.append({'source_id': sid, 'source': s['source'], 'options': {o: s['choice_text'][o] for o in opts},
                          'mean_stated_percent': {m: dict(zip(opts, np.round(P[m], 1).tolist())) for m in models},
                          'nats_vs_model_average': {m: dict(zip(opts, np.round(nats[i], 2).tolist())) for i, m in enumerate(models)},
                          'draws': {m: [r['original_probabilities'] for r in by[sid, m]] for m in models},
                          'choices': {m: [r['original_choice_id'] for r in by[sid, m]] for m in models},
                          'within_model_var_logp': float(within), 'between_model_var_mean_logp': float(between)})
    stated = np.array([v for r in ok for v in r['original_probabilities'].values()])
    pos = defaultdict(list)
    for r in ok:
        k = len(r['displayed_to_original'])
        for d, o in r['displayed_to_original'].items():
            pos[k, int(d)].append(r['probabilities'][d])
    result = {'author': 'PI/OpenAI', 'status_counts': dict(status), 'replies': len(rows),
              'choice_is_top': sum(r['choice_is_top'] for r in ok), 'ok': len(ok),
              'stated_extremes': {'exactly_0_or_100': int(((stated == 0) | (stated == 100)).sum()), 'values': int(stated.size)},
              'position_mean_percent': {f'{k} options, position {i}': round(float(np.mean(v)), 1) for (k, i), v in sorted(pos.items())},
              'between_over_within_mean_noise': {'median': float(np.median(noise_ratio)), 'per_question': [round(float(x), 1) for x in noise_ratio],
                  'meaning': 'Between-model variance of 4-draw mean log-probabilities / variance expected from draw noise alone (within-model variance / 4). Under an independent identical-draws null the numerator and denominator have similar expectations, but the expected ratio is not exactly 1 (ratio of noisy estimates; rotated option orders also break independence). Descriptive only.'},
              'cost_usd': sum(r['usage']['cost'] for r in rows if 'usage' in r),
              'questions': questions}
    (run / 'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Tiny eval trial: stated option probabilities\n', f"Replies {result['replies']}; status {result['status_counts']}; highest stated option = chosen option in {result['choice_is_top']}/{result['ok']}.",
             f"Stated values exactly 0 or 100: {result['stated_extremes']['exactly_0_or_100']}/{result['stated_extremes']['values']}. Cost ${result['cost_usd']:.4f}.",
             f"Between-model / draw-noise variance ratio, median over questions: {result['between_over_within_mean_noise']['median']:.1f} (descriptive; not a test against 1).\n"]
    for q in questions:
        lines.append(f"## {q['source_id']}\n")
        table = [[o, t[:70]] + [f"{q['mean_stated_percent'][m][o]:.0f}% ({q['nats_vs_model_average'][m][o]:+.2f})" for m in models] for o, t in q['options'].items()]
        lines.append(tabulate(table, headers=['id', 'option'] + [m + ' mean% (nats)' for m in models], tablefmt='pipe') + '\n')
        lines.append(f"Draw-to-draw variance of log p {q['within_model_var_logp']:.3f}; between-model variance of mean log p {q['between_model_var_mean_logp']:.3f}.\n")
    lines.append('mean% = arithmetic mean of the 4 stated percentages. Nats = mean over the 4 draws of log stated probability (add-one smoothed), minus the average of that over the 4 models, per option. Stated percentages are self-reports, not logprobs.\n\n-- PI/OpenAI')
    (run / 'results.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:4]))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/20261026_tiny_eval_trial10')
    analyze(parser.parse_args().run)
