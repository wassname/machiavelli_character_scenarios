"""Does the reasoning budget change stated option probabilities beyond draw-to-draw noise? -- PI/OpenAI"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[2]
LEVELS = ['0', '10k', '20k', '40k', 'max']
NAMES = {'deepseek/deepseek-v4.1-flash': 'DeepSeek', 'meta/muse-spark-1.3-contributor': 'Muse', 'qwen/qwen3.8-27b': 'Qwen',
         'google/gemini-3.8-flash': 'Gemini', 'openai/gpt-5.6-luna': 'Luna'}


def logp(p: dict, opts: list) -> np.ndarray:
    k = len(opts)
    return np.log((np.array([p[o] for o in opts]) + 1) / (100 + k))  # add-one smoothing on the 0..100 scale


def analyze(run: Path) -> None:
    rows = [json.loads(l) for l in (run / 'rest/choices.jsonl').open()]
    scenes = {s['source_id']: s for s in map(json.loads, (run / 'scenes.jsonl').open())}
    by = defaultdict(list)
    for r in rows:
        by[NAMES[r['model']], r['level'], r['source_id']].append(r)
    table, detail = [], {}
    for model in NAMES.values():
        noise = []
        for level in LEVELS:
            cell = [r for (m, l, _), rs in by.items() if m == model and l == level for r in rs]
            ok = [r for r in cell if r['status'] == 'ok']
            first = [((r.get('first_reply_usage') or r['usage']).get('completion_tokens_details') or {}).get('reasoning_tokens') or 0 for r in ok]
            deltas, sd = [], []
            for sid, s in scenes.items():
                opts = sorted(s['choice_text'], key=int)
                a = [r for r in by[model, level, sid] if r['status'] == 'ok']
                b = [r for r in by[model, 'max', sid] if r['status'] == 'ok']
                if len(a) == 2:
                    la = np.stack([logp(r['original_probabilities'], opts) for r in a])
                    sd.append(np.abs(la[0] - la[1]).mean() / np.sqrt(2))  # noise floor for a difference of two 2-draw means
                    if len(b) == 2:
                        lb = np.stack([logp(r['original_probabilities'], opts) for r in b])
                        deltas.append(np.abs(la.mean(0) - lb.mean(0)).mean())
            noise += sd
            detail[model, level] = {'ok': len(ok), 'replies': len(cell), 'followups': sum(r.get('followup_used', False) for r in ok),
                                    'reasoning_tokens_median': float(np.median(first)) if first else None,
                                    'reasoning_tokens_max': int(max(first)) if first else None,
                                    'median_abs_change_vs_max_nats': float(np.median(deltas)) if deltas else None,
                                    'draw_noise_nats': float(np.median(sd)) if sd else None,
                                    'cost_usd': sum((r.get('usage') or {}).get('cost', 0) for r in cell)}
            d = detail[model, level]
            table.append([model, level, f"{d['ok']}/{d['replies']}", d['followups'], d['reasoning_tokens_median'], d['reasoning_tokens_max'],
                          '' if d['median_abs_change_vs_max_nats'] is None else f"{d['median_abs_change_vs_max_nats']:.2f}",
                          '' if d['draw_noise_nats'] is None else f"{d['draw_noise_nats']:.2f}", f"{d['cost_usd']:.3f}"])
    headers = ['model', 'budget', 'valid', 'follow-ups', 'reasoning tokens (median)', 'reasoning tokens (max)',
               'median change vs max, nats', 'draw noise, nats', 'cost $']
    text = ('# Reasoning-budget test\n\n10 questions (5 everyday, 5 game) x 2 draws (rotated option order) x 5 budgets. '
            'Change = per question, mean over options of |mean log p at this budget - mean log p at max|, mean over 2 draws, add-one smoothed; median over questions. '
            'Draw noise = the same statistic expected from draw-to-draw variation alone at this budget (|draw0 - draw1| / sqrt 2), so change near the noise means the budget makes no visible difference. '
            'Reasoning tokens are from the first reply (before any follow-up).\n\n'
            + tabulate(table, headers=headers, tablefmt='pipe') + '\n\n-- PI/OpenAI\n')
    (run / 'results.md').write_text(text)
    (run / 'analysis.json').write_text(json.dumps({f'{m}|{l}': v for (m, l), v in detail.items()}, indent=2) + '\n')
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/20261026_reasoning_test')
    analyze(parser.parse_args().run)
