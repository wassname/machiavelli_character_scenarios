"""Is the shared shape of the absolute spider (all models high on wisdom and inquiry) an artefact?
Run synthetic choosers through the same goodness-removed, pair-weighted formula and compare with DeepSeek:
  goodness-only: log p = beta * PC1 score (cares only about overall goodness, as decisive as DeepSeek on PC1)
If goodness-only reproduces DeepSeek's shape, the shape comes from the questions and the method, not from the models. -- PI/OpenAI"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from plot_probability_eval import ROOT, SOURCES, calibrated_logp, virtue_basis  # noqa: E402

tables = ROOT / 'outputs/20261027_random1000_v12/question_tables.jsonl'
lines = [json.loads(x) for x in tables.open()]
groups = json.loads((ROOT / 'rubrics/virtue_display_groups.json').read_text())['groups']
basis = virtue_basis(lines, groups)
gidx = [[basis['ids'].index(i) for i in items] for items in groups.values()]
ds = 'deepseek/deepseek-v4.1-flash'
everyone = list(lines[1]['models'])

for src, sname in SOURCES.items():
    v1 = basis['src'][src]['V'][0]
    num = {k: np.zeros(len(groups) + 1) for k in ('deepseek', 'goodness')}
    den = np.zeros(len(groups) + 1)
    for q in lines[1:]:
        if q['source'] != src or set(everyone) - set(q['models']) or any(len(q['models'][m]['draws']) < 2 for m in q['models']):
            continue
        X = np.array([q['option_scores'][x] for x in q['options']], float)[:, basis['direct']] * basis['fav']
        if not np.isfinite(X).all():
            continue
        pc1 = (X - X.mean(0)) @ v1
        R = X - np.outer(X @ v1, v1)
        L = np.column_stack([np.stack([R[:, ix].mean(1) for ix in gidx], 1), pc1])  # options x (groups + PC1)
        iu = np.triu_indices(len(X), 1)
        dL = L[iu[0]] - L[iu[1]]
        w = np.abs(dL) * np.sign(dL)
        for k, lp in (('deepseek', calibrated_logp(q, ds, {}, {})), ('goodness', pc1)):
            num[k] += (w * (lp[iu[0]] - lp[iu[1]])[:, None]).sum(0)
        den += np.abs(dL).sum(0)
    lvl = {k: v / den for k, v in num.items()}
    lvl['goodness'] = lvl['goodness'] * lvl['deepseek'][-1] / lvl['goodness'][-1]  # as decisive as DeepSeek on PC1, so only the shape differs
    print(f'\n{sname}: nats vs an even split (last column = PC1, matched)')
    print(f"{'':16s}" + ''.join(f'{g[:11]:>12s}' for g in list(groups) + ['PC1']))
    for k, v in lvl.items():
        print(f'{k:16s}' + ''.join(f'{x:+12.3f}' for x in v))
    a, b = lvl['deepseek'][:-1], lvl['goodness'][:-1]
    print(f'correlation of DeepSeek shape with goodness-only shape: {np.corrcoef(a, b)[0, 1]:+.2f}')
