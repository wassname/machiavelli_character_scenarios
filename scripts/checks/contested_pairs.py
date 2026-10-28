"""Do models look alike because obvious pairs (truth vs murder) dominate the label-gap weighting?
Recompute overall goodness (PC1) vs DeepSeek using only contested pairs: both options get at least CUT % on average over all six models.
Report, per source: how many questions have a dropped 'nobody picks this' option, and the between-model spread on all pairs vs contested pairs,
next to DeepSeek against itself (answer 1 vs answer 2). Calibrated log-probs (split-half maps). -- PI/OpenAI"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from plot_probability_eval import ROOT, SOURCES, calibrated_logp, load_calibration, short, virtue_basis  # noqa: E402

CUT = 10.0
tables = ROOT / 'outputs/20261027_random1000_v12/question_tables.jsonl'
lines, maps, identity, fold = load_calibration(tables)
groups = json.loads((ROOT / 'rubrics/virtue_display_groups.json').read_text())['groups']
basis = virtue_basis(lines, groups)
ds = 'deepseek/deepseek-v4.1-flash'
models = [m for m in lines[1]['models'] if m != ds]

for src, sname in SOURCES.items():
    v1 = basis['src'][src]['V'][0]
    acc = {k: {m: [0., 0.] for m in models + ['self']} for k in ('all', 'contested')}
    n_q = n_junk = n_contested_q = 0
    for q in lines[1:]:
        if q['source'] != src or set(models + [ds]) - set(q['models']) or any(len(q['models'][m]['draws']) < 2 for m in q['models']):
            continue
        o = list(q['options'])
        X = np.array([q['option_scores'][x] for x in o], float)[:, basis['direct']] * basis['fav']
        if not np.isfinite(X).all():
            continue
        g = (X - X.mean(0)) @ v1
        pooled = np.mean([[q['models'][m]['mean_stated_percent'][x] for x in o] for m in q['models']], 0)
        keep = pooled >= CUT
        n_q += 1
        n_junk += (~keep).any()
        iu = np.triu_indices(len(o), 1)
        both = keep[iu[0]] & keep[iu[1]]
        n_contested_q += both.any()
        dg = g[iu[0]] - g[iu[1]]
        ref = calibrated_logp(q, ds, maps, fold)
        d0, d1 = [np.log(np.array([dr['probabilities'][x] for x in o], float) + 1) for dr in q['models'][ds]['draws'][:2]]
        for m in models + ['self']:
            delta = (d0 - d1) if m == 'self' else calibrated_logp(q, m, maps, fold) - ref
            dd = delta[iu[0]] - delta[iu[1]]
            for k, mask in (('all', np.ones_like(both)), ('contested', both)):
                acc[k][m][0] += (np.abs(dg) * np.sign(dg) * dd * mask).sum()
                acc[k][m][1] += (np.abs(dg) * mask).sum()
    print(f'\n## {sname}: {n_q} questions; {n_junk} have an option nobody picks (pooled < {CUT:.0f}%); {n_contested_q} have a contested pair')
    print(f"{'':22s}{'all pairs':>12s}{'contested only':>16s}")
    for m in models + ['self']:
        name = 'DeepSeek vs itself' if m == 'self' else short(m)
        a, c = (acc[k][m][0] / acc[k][m][1] for k in ('all', 'contested'))
        print(f'{name:22s}{a:+12.3f}{c:+16.3f}')
    for k in ('all', 'contested'):
        vals = [acc[k][m][0] / acc[k][m][1] for m in models]
        print(f'spread between models ({k}): sd {np.std(vals):.3f}, range {max(vals) - min(vals):.3f}')
