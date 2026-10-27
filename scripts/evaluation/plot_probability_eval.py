"""Results page for the stated-probability tiny eval: per source, each model's preference per label in nats vs DeepSeek. -- PI/OpenAI"""

import argparse
from collections import defaultdict
import hashlib
from html import escape
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_choice_dimensions import bootstrap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {'daily_dilemmas': 'Everyday dilemmas', 'machiavelli': 'Game summaries'}
UNIT = {'expected': 'Label-score difference vs DeepSeek (probability-weighted)', 'raw': 'Nats per unit of label score, model minus DeepSeek',
        'calibrated': 'Nats per unit of label score, model minus DeepSeek (calibrated)'}
COMBOS = ((0, 0), (0, 1), (1, 1))  # draw pairs used to resample 2 draws within a question
COLORS = {'Muse': '#7b3294', 'Qwen': '#c65a00', 'Gemini': '#1b7837', 'Luna': '#007a92', 'DeepSeek self': '#999999'}


def short(model: str) -> str:
    for k in COLORS:
        if k.split()[0].lower() in model.lower():
            return 'DeepSeek self' if 'deepseek' in model.lower() else k
    return model


def dot_panel(ax, rows: list[dict], labels: list[str], title: str, xlabel: str) -> None:
    models = sorted({short(r['model']) for r in rows}, key=list(COLORS).index)
    # Rare labels have tiny within-question variance and huge intervals; clip the axis and mark clipped ends with arrows.
    bounds = np.abs([x for r in rows if r['beta_nats_per_label_unit'] is not None for x in r['ci95']])
    lim = 1.15 * float(np.quantile(bounds, .9)) if len(bounds) else 1
    for j, m in enumerate(models):
        pts = {r['dimension']['id']: r for r in rows if short(r['model']) == m}
        y = np.arange(len(labels)) + (j - (len(models) - 1) / 2) * .16
        for yi, lid in zip(y, labels):
            r = pts.get(lid)
            if r is None or r['beta_nats_per_label_unit'] is None:
                continue
            lo, hi = r['ci95']
            ax.hlines(yi, max(lo, -lim), min(hi, lim), color=COLORS[m], lw=1, alpha=.6)
            for end, marker in ((lo, '<'), (hi, '>')):
                if abs(end) > lim:
                    ax.plot(np.sign(end) * lim, yi, marker, ms=4, color=COLORS[m], alpha=.6)
            ax.plot(np.clip(r['beta_nats_per_label_unit'], -lim, lim), yi, 'o', ms=3.5, color=COLORS[m], label=m if yi == y[0] else None)
    ax.axvline(0, color='#555555', lw=.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(len(labels) - .5, -.8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', alpha=.15)
    ax.tick_params(axis='y', length=0)


def render(scores: Path, scenes_path: Path, out: Path, mode: str, role: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(scores.read_text())
    rows = [r for r in data['rows'] if r['mode'] == mode and r['role'] == role]
    assert rows, (mode, role)
    plt.rcParams.update({'font.size': 9, 'font.family': ['DejaVu Sans', 'Noto Serif CJK SC'], 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.spines.left': False, 'savefig.facecolor': 'white'})
    made = []
    xlabel = UNIT[mode]
    for family, height in (('virtues', 22), ('behaviours', 10), ('dnd', 2.6)):
        dims = []
        for r in rows:
            d = r['dimension']
            if d['family'] == family and d.get('derived_from') is None and d['id'] not in [x['id'] for x in dims]:
                dims.append(d)
        if not dims:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(14, height), sharey=True, constrained_layout=True)
        for ax, (src, name) in zip(axs, SOURCES.items(), strict=True):
            sub = [r for r in rows if r['source'] == src and r['dimension']['family'] == family]
            n = max((r['informative_questions'] for r in sub), default=0)
            dot_panel(ax, sub, [d['id'] for d in dims], f'{name} \u00b7 {n} {role} questions (fewer for labels missing on some menus)', xlabel)
        lab = [f"{d['label']} [{d.get('tradition', '')}]" + (' \u2022 vice' if d['kind'] == 'vices' else '') if family == 'virtues'
               else d['label'].removeprefix('agg_').replace('_', ' ') for d in dims]
        axs[0].set_yticks(range(len(dims)), lab)
        axs[0].legend(loc='upper left', bbox_to_anchor=(0, 1.0), ncol=len({short(r['model']) for r in rows}), frameon=True, framealpha=.95, fontsize=8, markerscale=1.5)
        fig.suptitle(f'{family.capitalize()}: which options does each model put more probability on than DeepSeek? ({mode}, {role} questions)', fontsize=12)
        fig.supxlabel('Positive = the model gives more probability than DeepSeek to options scoring higher on this label (vices: more of the named vice). '
                      + ('Lines: 95% two-level bootstrap (games/topics, then draws; includes draw noise). ' if mode == 'expected' else 'Lines: 95% game/topic cluster bootstrap over questions (draw noise not included). ') + 'Arrows: the interval (or the point, drawn at the edge) runs past the axis. '
                      '"DeepSeek self" = first half of DeepSeek\'s draws minus second half, an empirical noise check.', fontsize=9)
        path = out / f'{family}_{role}_{mode}.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        made.append(path)
    return made


def heatmap(scores: Path, out: Path, mode: str, role: str) -> tuple[Path, list[list]]:
    # Compact view: one cell per (virtue, model); colour = nats per label unit; dot = 95% interval excludes 0.
    rows = [r for r in json.loads(scores.read_text())['rows'] if r['mode'] == mode and r['role'] == role
            and r['dimension']['family'] == 'virtues' and r['dimension'].get('derived_from') is None]
    dims = list({r['dimension']['id']: r['dimension'] for r in rows}.values())
    models = [m for m in COLORS if any(short(r['model']) == m for r in rows)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 17), sharey=True, constrained_layout=True)
    counts = []
    clip = float(np.quantile(np.abs([r['beta_nats_per_label_unit'] for r in rows if r['beta_nats_per_label_unit'] is not None]), .95))
    for ax, (src, name) in zip(axs, SOURCES.items(), strict=True):
        grid = np.full((len(dims), len(models)), np.nan)
        sig = np.zeros_like(grid, dtype=bool)
        for r in rows:
            if r['source'] == src and r['beta_nats_per_label_unit'] is not None:
                i, j = [d['id'] for d in dims].index(r['dimension']['id']), models.index(short(r['model']))
                grid[i, j] = r['beta_nats_per_label_unit']
                lo, hi = r['ci95']
                sig[i, j] = lo > 0 or hi < 0
        im = ax.imshow(np.clip(grid, -clip, clip), cmap='RdBu', vmin=-clip, vmax=clip, aspect='auto')
        yy, xx = np.nonzero(sig)
        ax.plot(xx, yy, 'k.', ms=3)
        ax.set_xticks(range(len(models)), models, rotation=45, ha='right')
        ax.set_title(f"{name.split()[0]} ({max(r['informative_questions'] for r in rows if r['source'] == src)} q)", fontsize=10)
        for j, m in enumerate(models):
            pos, neg = int(((grid[:, j] > 0) & sig[:, j]).sum()), int(((grid[:, j] < 0) & sig[:, j]).sum())
            counts.append([name, m, pos, neg, int(np.isfinite(grid[:, j]).sum())])
    axs[0].set_yticks(range(len(dims)), [f"{d['label']} [{d.get('tradition', '')}]" + (' \u2022 vice' if d['kind'] == 'vices' else '') for d in dims], fontsize=7)
    fig.colorbar(im, ax=axs, shrink=.3, label=f'{UNIT[mode]} (clipped at \u00b1{clip:.2g})')
    fig.suptitle(f'66 virtues, {role} questions ({mode}): blue = model puts more probability than DeepSeek on options scoring higher on this label\n(for rows marked vice: more of the vice)', fontsize=10)
    fig.supxlabel('Dot = 95% interval excludes 0 (' + ('two-level bootstrap, includes draw noise' if mode == 'expected' else 'game/topic bootstrap, draw noise not included') + '). "DeepSeek self" = first half of DeepSeek draws vs second half, an empirical noise check.', fontsize=8)
    path = out / f'heatmap_virtues_{role}_{mode}.png'
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path, counts


def top_differences(scores: Path, out: Path, mode: str, role: str, k: int = 6) -> Path:
    # Readable summary: per model and source, the k virtues most above and most below DeepSeek (by estimate / bootstrap SE).
    rows = [r for r in json.loads(scores.read_text())['rows'] if r['mode'] == mode and r['role'] == role
            and r['dimension']['family'] == 'virtues' and r['dimension'].get('derived_from') is None and r['beta_nats_per_label_unit'] is not None]
    models = [m for m in COLORS if m != 'DeepSeek self' and any(short(r['model']) == m for r in rows)]
    fig, axs = plt.subplots(len(models), 2, figsize=(13, 3.1 * len(models)), constrained_layout=True)
    for i, m in enumerate(models):
        for j, (src, name) in enumerate(SOURCES.items()):
            ax = axs[i, j]
            sub = sorted((r for r in rows if short(r['model']) == m and r['source'] == src), key=lambda r: r['beta_nats_per_label_unit'] / max(r['bootstrap_se'], 1e-12))
            pick = sub[:k] + sub[-k:]
            y = np.arange(len(pick))
            vals = np.array([r['beta_nats_per_label_unit'] for r in pick])
            lo, hi = np.array([r['ci95'][0] for r in pick]), np.array([r['ci95'][1] for r in pick])
            sig = (lo > 0) | (hi < 0)
            ax.barh(y, vals, color=[COLORS[m] if s else '#cccccc' for s in sig], height=.6)
            ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt='none', ecolor='#444444', lw=.8, capsize=2)
            ax.axvline(0, color='#555555', lw=.8)
            ax.set_yticks(y, [f"{r['dimension']['label']} [{r['dimension'].get('tradition', '')}]" + (' \u2022 vice' if r['dimension']['kind'] == 'vices' else '') for r in pick], fontsize=8)
            ax.axhline(k - .5, color='#bbbbbb', lw=.6, ls='--')
            ax.set_title(f'{m} \u00b7 {name}', fontsize=10, loc='left', color=COLORS[m])
            ax.tick_params(axis='x', labelsize=8)
    fig.suptitle(f'Where each model differs most from DeepSeek ({role} questions, {mode}): bottom {k} and top {k} of 66 virtues, ranked by estimate / its standard error\n'
                 f'Bar = {UNIT[mode]}; grey bar = 95% interval includes 0. Compare with the DeepSeek-self check on the page.', fontsize=11)
    path = out / f'top_differences_{role}_{mode}.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def dnd_2d(scores: Path, out: Path, mode: str, role: str) -> Path:
    # Two separate axes, not averaged: x = lawfulness ("order"), y = goodness ("morality"); crosses are per-axis 95% intervals.
    rows = [r for r in json.loads(scores.read_text())['rows'] if r['mode'] == mode and r['role'] == role and r['dimension']['family'] == 'dnd']
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    for ax, (src, name) in zip(axs, SOURCES.items(), strict=True):
        n = 0
        for m in COLORS:
            pts = {r['dimension']['id']: r for r in rows if r['source'] == src and short(r['model']) == m}
            if 'order' not in pts or 'morality' not in pts:
                continue
            x, y = pts['order'], pts['morality']
            n = max(n, x['informative_questions'])
            ax.errorbar(x['beta_nats_per_label_unit'], y['beta_nats_per_label_unit'],
                        xerr=[[x['beta_nats_per_label_unit'] - x['ci95'][0]], [x['ci95'][1] - x['beta_nats_per_label_unit']]],
                        yerr=[[y['beta_nats_per_label_unit'] - y['ci95'][0]], [y['ci95'][1] - y['beta_nats_per_label_unit']]],
                        fmt='o', color=COLORS[m], ms=7, lw=1, capsize=3, label=m)
            ax.annotate(m, (x['beta_nats_per_label_unit'], y['beta_nats_per_label_unit']), xytext=(6, 6), textcoords='offset points', color=COLORS[m], fontsize=9)
        ax.axhline(0, color='#888888', lw=.8)
        ax.axvline(0, color='#888888', lw=.8)
        ax.set_title(f'{name} ({n} {role} questions with both axes graded)', fontsize=10)
        ax.set_xlabel('\u2190 more chaotic   lawfulness vs DeepSeek   more lawful \u2192')
        ax.set_ylabel('\u2190 more evil   goodness vs DeepSeek   more good \u2192')
    fig.suptitle(f'D&D alignment of probability-weighted options relative to DeepSeek (origin) \u00b7 {role} questions, {mode}', fontsize=12)
    fig.supxlabel('Each point: how much more probability the model puts than DeepSeek on options graded more lawful (x) or more good (y). Crosses: separate 95% intervals per axis.', fontsize=9)
    path = out / f'dnd2d_{role}_{mode}.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def spider(scores: Path, out: Path, mode: str, role: str, shape_only: bool = False) -> Path:
    # Exploratory virtue groups (rubrics/virtue_display_groups.json); spoke = mean slope of the group's virtues; dashed ring = DeepSeek (0).
    groups = json.loads((ROOT / 'rubrics/virtue_display_groups.json').read_text())['groups']
    rows = [r for r in json.loads(scores.read_text())['rows'] if r['mode'] == mode and r['role'] == role and r['dimension']['family'] == 'virtues']
    angle = np.linspace(0, 2 * np.pi, len(groups), endpoint=False)
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.5), subplot_kw={'projection': 'polar'}, constrained_layout=True)
    for ax, (src, name) in zip(axs, SOURCES.items(), strict=True):
        vals = {}
        for m in COLORS:
            by = {r['dimension']['id']: r['beta_nats_per_label_unit'] for r in rows if r['source'] == src and short(r['model']) == m and r['beta_nats_per_label_unit'] is not None}
            vals[m] = [float(np.mean([by[i] for i in items if i in by])) for items in groups.values()]
            if shape_only:  # remove the model's overall level so only its profile shape remains
                vals[m] = list(np.array(vals[m]) - np.mean(vals[m]))
        lim = max(abs(v) for vs in vals.values() for v in vs) * 1.1
        for m, v in vals.items():
            ax.plot(np.r_[angle, angle[0]], np.r_[v, v[:1]], color=COLORS[m], lw=1.6 if m != 'DeepSeek self' else 1, ls='-' if m != 'DeepSeek self' else ':', marker='o', ms=3, label=m)
        ax.plot(np.linspace(0, 2 * np.pi, 200), np.zeros(200), color='#333333', lw=.8, ls='--')
        ax.set_ylim(-lim, lim)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angle, list(groups), fontsize=9)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_title(name, pad=20)
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False)
    fig.suptitle(f'Virtue groups relative to DeepSeek (dashed ring = DeepSeek) \u00b7 {role} questions, {mode}' + (' \u00b7 shape only: each model\'s average over the 9 groups removed' if shape_only else ''), fontsize=12)
    fig.supxlabel(f'Spoke = mean over the group\'s virtues of: {UNIT[mode]}' + ('; each model\'s own average over the 9 groups is subtracted, so outside the dashed ring = the model favours that group more than its own average (relative to DeepSeek). '
                  if shape_only else '; outside the dashed ring = model favours that group more than DeepSeek. ')
                  + 'Groups are exploratory, not traditions; no intervals here (see the per-virtue plots).', fontsize=9)
    path = out / f'spider_{role}_{mode}{"_shape" if shape_only else ""}.png'
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path


def examples(question_tables: Path, k: int = 6) -> str:
    qs = [q for q in map(json.loads, question_tables.open()) if '_meta' not in q and q['role'] == 'dev']
    qs.sort(key=lambda q: hashlib.sha256(('example:' + q['source_id']).encode()).hexdigest())
    blocks = []
    for q in [q for src in SOURCES for q in [x for x in qs if x['source'] == src][:k // 2]]:
        models = q['models']
        names = {m: short(m) for m in models}
        table = [[o, escape(t[:90])] + [f"{models[m]['mean_stated_percent'][o]:.0f}%" + ('' if 'deepseek' in m else f" ({models[m]['calibrated_nats'][o]:+.2f})")
                                        for m in models] for o, t in q['options'].items()]
        heads = ['id', 'option'] + [('DeepSeek' if 'deepseek' in m else names[m]) + ('' if 'deepseek' in m else ' (nats)') for m in models]
        replies = ''.join(f"<details><summary>{escape('DeepSeek' if 'deepseek' in m else names[m])}: one native reply (choice and probabilities use the displayed IDs; displayed-to-original: {escape(json.dumps(v['native_example'].get('displayed_to_original')))})</summary><pre>{escape(v['native_example']['reply'])}</pre></details>"
                          for m, v in models.items() if v.get('native_example'))
        blocks.append(f"<h4>{escape(q['source_id'])}</h4><pre>{escape(q['text'][-900:])}</pre>" + tabulate(table, headers=heads, tablefmt='html') + replies)
    return ('<h2>Example questions</h2><p>Hash-chosen dev questions, not typical or best. Percent = mean of the 4 stated draws; '
            'calibrated nats vs DeepSeek in brackets.</p>' + ''.join(blocks))


def expected_rows(question_tables: Path, out: Path, replicates: int = 300) -> Path:
    # No division by label variance: per question, (model's mean stated distribution - DeepSeek's) @ option label scores.
    # DeepSeek self = first half of its draws minus second half. Equal question weight; two-level bootstrap (clusters, then draws).
    lines = list(map(json.loads, question_tables.open()))
    dims = lines[0]['_meta']['dimensions']
    qs = lines[1:]
    ds = 'deepseek/deepseek-v4.1-flash'
    models = [m for m in qs[0]['models'] if m != ds] + ['DeepSeek self']
    rows = []
    for src in SOURCES:
        for role in sorted({q['role'] for q in qs}):
            for family in ('virtues', 'behaviours', 'dnd'):
                ix = [i for i, d in enumerate(dims) if d['family'] == family]
                vals, groups, per_draw = [], [], []
                for q in qs:
                    if q['source'] != src or q['role'] != role:
                        continue
                    opts = list(q['options'])
                    S = np.array([q['option_scores'][o] for o in opts], float)[:, ix]
                    if not np.isfinite(S).all():
                        continue
                    # per draw: stated distribution @ label scores, for every model including DeepSeek
                    pd = {m: np.stack([np.array([x['probabilities'][o] / 100 for o in opts]) @ S for x in sorted(q['models'][m]['draws'], key=lambda x: x['draw'])]) for m in q['models']}
                    half = len(pd[ds]) // 2  # DeepSeek self: first half of its draws minus second half
                    v = [pd[m].mean(0) - pd[ds].mean(0) for m in models[:-1]] + [pd[ds][:half].mean(0) - pd[ds][half:].mean(0)]
                    vals.append(np.stack(v))
                    per_draw.append(pd)
                    groups.append(q['eval_cluster'])
                if not vals:
                    continue
                v = np.stack(vals)
                # Two-level bootstrap: resample games/topics, then each model's draws within each question, so draw noise is in the interval.
                rng = np.random.default_rng(20261027)
                clusters = sorted(set(groups))
                members = {c: [i for i, g in enumerate(groups) if g == c] for c in clusters}
                b = np.empty((replicates, len(models), len(ix)))
                for r in range(replicates):
                    pick = [i for c in rng.choice(clusters, len(clusters)) for i in members[c]]
                    acc = np.zeros((len(models), len(ix)))
                    for i in pick:
                        pd = per_draw[i]
                        mean = {m: pd[m][rng.integers(0, len(pd[m]), len(pd[m]))].mean(0) for m in pd}
                        dsd = pd[ds]
                        h = len(dsd) // 2
                        self_diff = dsd[rng.integers(0, h, h)].mean(0) - dsd[h + rng.integers(0, len(dsd) - h, len(dsd) - h)].mean(0)
                        acc += np.stack([mean[m] - mean[ds] for m in models[:-1]] + [self_diff])
                    b[r] = acc / len(pick)
                lo, hi = np.quantile(b, [.025, .975], axis=0)
                for mi, m in enumerate(models):
                    for j, k in enumerate(ix):
                        rows.append({'model': 'deepseek self' if m == 'DeepSeek self' else m, 'mode': 'expected', 'role': role, 'source': src, 'dimension': dims[k],
                                     'beta_nats_per_label_unit': float(v[:, mi, j].mean()), 'ci95': [float(lo[mi, j]), float(hi[mi, j])],
                                     'bootstrap_se': float(b[:, mi, j].std(ddof=1)), 'informative_questions': len(vals), 'clusters': len(set(groups))})
    path = out / 'expected_label_scores.json'
    path.write_text(json.dumps({'author': 'PI/OpenAI', 'method': 'Per question: (model mean stated distribution - DeepSeek mean stated distribution) @ option label scores; '
                                'no division by label variance. Key beta_nats_per_label_unit holds this difference (name kept for plotting code reuse).', 'rows': rows}) + '\n')
    return path


def page(scoring: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((scoring / 'summary.json').read_text())
    noise = json.loads((scoring / 'draw_noise.json').read_text())['summaries']
    figs, counts = {}, {}
    expected = expected_rows(scoring / 'question_tables.jsonl', out)
    for mode, file, roles in (('expected', expected, ('dev', 'control')), ('raw', scoring / 'scored_identity.json', ('dev', 'control')),
                              ('calibrated', scoring / 'scored_quantile.json', ('dev',))):
        for role in roles:
            figs[mode, role] = render(file, None, out, mode, role)
            h, c = heatmap(file, out, mode, role)
            figs[mode, role].insert(0, h)
            counts[mode, role] = c
    def count_table(key):
        return tabulate([[src, m, f'{pos}/{n}', f'{neg}/{n}'] for src, m, pos, neg, n in counts[key]],
                        headers=['Source', 'Model', 'Virtues: interval above 0', 'Virtues: interval below 0'], tablefmt='html')
    noise_rows = [[SOURCES[r['source']], r['role'], r['questions'], f"{r['pooled_between_over_draw_noise']:.1f}", f"{r['ci95'][0]:.1f} to {r['ci95'][1]:.1f}"]
                  for r in noise if r['model_group'] == 'comparisons_only' and r['role'] in ('dev', 'control')]
    noise_table = tabulate(noise_rows, headers=['Source', 'Questions', 'n', 'Between-model / draw-noise variance', '95% interval'], tablefmt='html')
    gate = summary['quantile_clean_dev_gate']
    img = lambda p: f'<img src="{p.name}" alt="{escape(p.stem)}">'
    blocks = ['<h1>Which virtues does each model favour, compared with DeepSeek?</h1>',
              '<p>Five models stated a percentage for every option of 505 moral-choice questions, 4 draws each with rotated option order and reasoning (10k-token budget). '
              'Main number, per question: each option\'s label score weighted by the model\'s average stated percentage, minus the same for DeepSeek; then averaged over questions '
              '(virtue scores run -1 to +1, behaviours 0 to 1, D&D -5 to +5). Positive = the model puts more probability than DeepSeek on options with more of that label '
              '(for vices: more of the vice). Nothing is divided by how much the options differ, so rare labels do not blow up. Everyday dilemmas and game summaries are kept separate. '
              'Questions were chosen with DeepSeek only: its most uncertain 10% (dev) and random controls.</p>',
              '<p><b>Read with care.</b> These numbers still mix a model\'s values with how decisive it is: a model that spreads its percentages more evenly than DeepSeek moves '
              'probability away from whatever DeepSeek favours. Calibrating each model\'s confidence scale failed checks (it biased DeepSeek against itself on control questions), '
              'so no calibration is applied. The nats-slope views lower down have the same problem plus large intervals for rare labels.</p>',
              '<h2>How many virtues differ from DeepSeek? (dev questions)</h2>', count_table(('expected', 'dev')),
              '<p>"DeepSeek self" compares DeepSeek\'s draws 1-2 with draws 3-4. Draw-to-draw noise is not in the intervals, and the 66 virtue labels move together, so one shared '
              'fluctuation can push several past zero; use its count as the noise level.</p>',
              '<h2>Did choosing uncertain questions help?</h2>', noise_table,
              '<p>Variance of the four models\' mean log-probabilities divided by their draw-to-draw variance, pooled over questions. Dev is not higher than control, so choosing '
              'DeepSeek-uncertain questions did not separate the models better than random questions.</p>',
              '<h2>Where each model differs most (dev, raw)</h2>', img(top_differences(expected, out, 'expected', 'dev')),
              '<h2>Virtue groups (spider, dev)</h2>', img(spider(expected, out, 'expected', 'dev')),
              '<p>The 66 virtue labels share one strong good/bad direction, so a model that is overall more virtuous than DeepSeek sits outside the ring on every spoke. '
              'The second spider removes each model\'s overall level, leaving only which groups it favours relative to its own average.</p>',
              img(spider(expected, out, 'expected', 'dev', shape_only=True)),
              '<h2>D&D lawfulness and goodness (dev, raw)</h2>', img(dnd_2d(expected, out, 'expected', 'dev')),
              '<h2>All 66 virtues (dev)</h2>', img(figs['expected', 'dev'][0]),
              '<details><summary>Virtues with intervals (dev)</summary>' + img(figs['expected', 'dev'][1]) + '</details>',
              '<details open><summary>Behaviours, including violence (dev)</summary>' + img(figs['expected', 'dev'][2]) + '</details>',
              '<h2>Random control questions</h2>', count_table(('expected', 'control')), ''.join(f'<details><summary>{escape(x.stem)}</summary>{img(x)}</details>' for x in figs['expected', 'control']),
              '<h2>Nats views (raw slopes; mix values with decisiveness)</h2>', count_table(('raw', 'dev')),
              ''.join(f'<details><summary>{escape(x.stem)}</summary>{img(x)}</details>' for role in ('dev', 'control') for x in figs['raw', role]),
              '<h2>Calibrated view (dev only, secondary)</h2>', count_table(('calibrated', 'dev')),
              ''.join(f'<details><summary>{escape(x.stem)}</summary>{img(x)}</details>' for x in figs['calibrated', 'dev']),
              examples(scoring / 'question_tables.jsonl'),
              '<h2>Limits</h2><p>Stated percentages are self-reports, not logprobs. Labels are automatic virtue grades of each option, not ground truth; label error is not in the intervals. '
              'Intervals are per-label 95% game/topic cluster bootstraps over questions, with draws fixed and no correction for 66 comparisons. The per-virtue slope is pooled across questions, so questions whose options '
              'differ more on that virtue count more. The 66 virtue labels are strongly correlated, so many same-sign results can reflect one shared good/bad direction. '
              'Luna reasons far less than the others under a token budget (about 80 tokens).</p>',
              f'<p>Data: <code>{escape(str(scoring))}</code></p><p>-- PI/OpenAI</p>']
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Stated-probability tiny eval</title><style>body{font:16px/1.5 system-ui;max-width:1400px;margin:30px auto;padding:0 24px;color:#222}'
            'img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px;margin:10px 0}td,th{padding:5px 10px;border-bottom:1px solid #ddd;text-align:left}'
            'details{border-top:1px solid #ddd;padding:10px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;background:#fafafa;padding:10px}p{max-width:1050px}</style><body>'
            + '\n'.join(blocks) + '</body></html>')
    (out / 'index.html').write_text(html)
    return out / 'index.html'


def calibration_before_after(tables: Path, out: Path) -> Path:
    # ECDF of pairwise |log-odds| per model, raw vs split-half calibrated (test-fold maps from calibration_heldout); DeepSeek in black.
    H = tables.parent / 'calibration_heldout'
    maps = json.loads((H / 'maps.json').read_text())
    fold = {q['source_id']: 'AB'.index(q['fold']) for q in json.loads((H / 'folds.json').read_text())['questions']}
    qs = list(map(json.loads, tables.open()))[1:]
    ds = 'deepseek/deepseek-v4.1-flash'
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    for row, (src, name) in enumerate(SOURCES.items()):
        mags = defaultdict(lambda: ([], [], []))  # raw, calibrated, weight (1/pairs, equal weight per question as in the fit)
        for q in qs:
            if q['source'] != src:
                continue
            o = list(q['options'])
            for m, d in q['models'].items():
                if len(d['draws']) < 2:  # calibration uses complete 2-draw questions only
                    continue
                lp = np.mean([np.log(np.array([dr['probabilities'][x] for x in o], float) + 1) for dr in d['draws']], 0) - np.log(100 + len(o))
                od = lp[:, None] - lp[None, :]
                iu = np.triu_indices(len(o), 1)
                raw = np.abs(od[iu])
                if m == ds:
                    cal = raw
                else:
                    mp = maps[src][m][str(fold[q['source_id']])]
                    v = np.interp(np.abs(od), mp['x'], mp['y'])
                    big = np.abs(od) > mp['x'][-1]
                    v[big] = mp['y'][-1] + (np.abs(od)[big] - mp['x'][-1]) * mp['end_slope']
                    z = (np.sign(od) * v).mean(1)
                    cal = np.abs((z[:, None] - z[None, :])[iu])
                mags[m][0].extend(raw)
                mags[m][1].extend(cal)
                mags[m][2].extend([1 / len(raw)] * len(raw))
        for col, title in enumerate(('uncalibrated', 'calibrated (map from other half)')):
            ax = axs[row, col]
            for m, vals in mags.items():
                ix = np.argsort(vals[col])
                x, w = np.array(vals[col])[ix], np.array(vals[2])[ix]
                c = 'black' if m == ds else COLORS[short(m)]
                ax.plot(x, np.cumsum(w) / w.sum(), color=c, lw=2.2 if m == ds else 1.4, label='DeepSeek (reference)' if m == ds else short(m))
            ax.set_title(f'{name}: {title}', loc='left', fontsize=10)
            ax.set_xlim(0, 5)
            ax.grid(alpha=.3)
        axs[row, 0].set_ylabel('share of option pairs')
    for ax in axs[1]:
        ax.set_xlabel('|log-odds| between two options of a question (nats; 0 = 50/50, 4.6 = 99/1)')
    axs[0, 0].legend(fontsize=9, loc='lower right')
    fig.suptitle('How decisive each model is, before and after calibration (curve left of black = less decisive than DeepSeek)', fontsize=11)
    path = out / 'calibration_before_after.png'
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def calibrated_logp(q: dict, model: str, maps: dict, fold: dict, pick: tuple = (0, 1)) -> np.ndarray:
    # Add-one-smoothed geometric pool over the picked draws, then the split-half quantile map (identity for DeepSeek); same steps as calibration_heldout/run.py.
    o = list(q['options'])
    draws = q['models'][model]['draws']
    lp = np.mean([np.log(np.array([draws[i]['probabilities'][x] for x in o], float) + 1) for i in pick], 0) - np.log(100 + len(o))
    if 'deepseek' not in model:
        od = lp[:, None] - lp[None, :]
        mp = maps[q['source']][model][str(fold[q['source_id']])]
        v = np.interp(np.abs(od), mp['x'], mp['y'])
        big = np.abs(od) > mp['x'][-1]
        v[big] = mp['y'][-1] + (np.abs(od)[big] - mp['x'][-1]) * mp['end_slope']
        lp = (np.sign(od) * v).mean(1)
    return lp - np.logaddexp.reduce(lp)


def nats_core(lines: list[dict], src: str, label_fn, maps: dict, fold: dict) -> tuple:
    # Log-odds shift toward the option with more of a label, model (calibrated) minus DeepSeek on the SAME question, averaged over option pairs weighted by label gap:
    #   nats = sum_q sum_{i<j} |x_i - x_j| * sign(x_i - x_j) * (delta_i - delta_j) / sum_q sum_{i<j} |x_i - x_j|,  delta = log p_model - log p_DeepSeek.
    # For a 2-option question this is exactly how many nats more the model's odds favour the higher-label option. label_fn(q) -> options x dims (NaN = missing).
    ds = 'deepseek/deepseek-v4.1-flash'
    models = [m for m in lines[1]['models'] if m != ds]
    num, num_self, den, groups = [], [], [], []
    for q in lines[1:]:
        if q['source'] != src or any(len(q['models'][m]['draws']) < 2 for m in q['models']):
            continue
        o = list(q['options'])
        X = label_fn(q)
        iu = np.triu_indices(len(o), 1)
        dX = X[iu[0]] - X[iu[1]]  # pairs x dims
        ok = np.isfinite(dX)
        w = np.where(ok, np.abs(dX), 0.)
        sgn = np.where(ok, np.sign(dX), 0.)
        pair = lambda dl: (w * sgn * (dl[iu[0]] - dl[iu[1]])[:, None]).sum(0)
        # draw resamples of 2 draws: (0,0), (0,1), (1,1) with probability 1/4, 1/2, 1/4; index [model, model combo, DeepSeek combo]
        refs = [calibrated_logp(q, ds, maps, fold, c) for c in COMBOS]
        cal = [[calibrated_logp(q, m, maps, fold, c) for c in COMBOS] for m in models]
        d0, d1 = [np.log(np.array([dr['probabilities'][x] for x in o], float) + 1) for dr in q['models'][ds]['draws'][:2]]
        num.append(np.array([[[pair(cal[j][a] - refs[b]) for b in range(3)] for a in range(3)] for j in range(len(models))]))
        num_self.append(pair(d0 - d1))  # DeepSeek draw 1 minus draw 2, smoothed, no calibration
        den.append(w.sum(0))
        groups.append(q['eval_cluster'])
    num, num_self, den = np.array(num), np.array(num_self), np.array(den)  # q x models x 3 x 3 x dims, q x dims, q x dims
    nq, nm = len(den), len(models)
    clusters = sorted(set(groups))
    cl = np.array([clusters.index(g) for g in groups])
    members = [np.flatnonzero(cl == c) for c in range(len(clusters))]
    rng = np.random.default_rng(20261027)
    boot = np.empty((1000, nm + 1, den.shape[1]))
    for r in range(1000):
        qi = np.concatenate([members[c] for c in rng.integers(0, len(clusters), len(clusters))])  # games/topics resampled; each copy gets fresh draws below
        a = rng.choice(3, size=(len(qi), nm), p=[.25, .5, .25])  # each model's 2 draws resampled with replacement, per question occurrence
        b = rng.choice(3, size=len(qi), p=[.25, .5, .25])  # DeepSeek's draws, shared by all models in that occurrence
        n_models = num[qi[:, None], np.arange(nm)[None, :], a, b[:, None]].sum(0)
        n_self = (rng.choice([-1., 1.], len(qi))[:, None] * num_self[qi]).sum(0)  # DeepSeek self: null band from random draw-label swaps
        with np.errstate(invalid='ignore', divide='ignore'):
            boot[r] = np.vstack([n_models, n_self]) / den[qi].sum(0)
    with np.errstate(invalid='ignore', divide='ignore'):
        point = np.vstack([num[:, :, 1, 1].sum(0), num_self.sum(0)[None]]) / den.sum(0)  # both draws pooled = combo (0, 1)
    lo, hi = np.nanquantile(boot, .025, 0), np.nanquantile(boot, .975, 0)
    return point, lo, hi, nq, (den > 0).sum(0)


def load_calibration(tables: Path) -> tuple[list[dict], dict, dict, dict]:
    # Question tables plus split-half maps and folds; identity maps give the uncalibrated view through the same code path.
    H = tables.parent / 'calibration_heldout'
    maps = json.loads((H / 'maps.json').read_text())
    fold = {q['source_id']: 'AB'.index(q['fold']) for q in json.loads((H / 'folds.json').read_text())['questions']}
    identity = {s: {m: {k: {'x': [0., 10.], 'y': [0., 10.], 'end_slope': 1.} for k in ('0', '1')} for m in v} for s, v in maps.items()}
    return list(map(json.loads, tables.open())), maps, identity, fold


def fmt(point: float, lo: float, hi: float, null: bool = False) -> str:
    return f"{point:+.3f} ({'swap null ' if null else ''}{lo:+.3f} to {hi:+.3f})"


def chosen_option_level(lines: list[dict], src: str, value_fn) -> tuple:
    # Virtue level of each draw's top option (ties averaged), model minus DeepSeek; bootstrap games/topics then draws. value_fn(q) -> per-option value.
    ds = 'deepseek/deepseek-v4.1-flash'
    models = [m for m in lines[1]['models'] if m != ds]
    top, groups = [], []
    for q in lines[1:]:
        if q['source'] != src or any(len(q['models'][m]['draws']) < 2 for m in q['models']):
            continue
        v = dict(zip(q['options'], value_fn(q)))
        if not np.isfinite(list(v.values())).all():
            continue
        t = lambda d: np.mean([v[o] for o, p in d['probabilities'].items() if p == max(d['probabilities'].values())])
        top.append([[t(d) for d in q['models'][m]['draws'][:2]] for m in models + [ds]])
        groups.append(q['eval_cluster'])
    top = np.array(top)  # q x (models + DeepSeek) x 2 draws
    clusters = sorted(set(groups))
    members = [np.flatnonzero(np.array(groups) == c) for c in clusters]
    rng = np.random.default_rng(20261027)
    boot = []
    for _ in range(1000):
        qi = np.concatenate([members[c] for c in rng.integers(0, len(clusters), len(clusters))])
        pick = rng.integers(0, 2, size=(len(qi), top.shape[1], 2))
        m = np.take_along_axis(top[qi], pick, 2).mean(2)
        boot.append((m[:, :-1] - m[:, -1:]).mean(0))
    point = (top.mean(2)[:, :-1] - top.mean(2)[:, -1:]).mean(0)
    return point, np.quantile(boot, .025, 0), np.quantile(boot, .975, 0)


def nats_overall(tables: Path) -> str:
    # One number per model: nats toward the option with the higher mean virtue score (66 direct virtues, vices flipped), uncalibrated and calibrated, plus top-option check.
    lines, maps, identity, fold = load_calibration(tables)
    dims = lines[0]['_meta']['dimensions']
    direct = [i for i, d in enumerate(dims) if d['family'] == 'virtues' and d['derived_from'] is None]
    fav = np.array([-1. if dims[i]['kind'] == 'vices' else 1. for i in direct])
    mean_virtue = lambda q: (np.array([q['option_scores'][x] for x in q['options']], float)[:, direct] * fav).mean(1)
    ds = 'deepseek/deepseek-v4.1-flash'
    models = [m for m in lines[1]['models'] if m != ds]
    rows = []
    for src, name in SOURCES.items():
        raw = nats_core(lines, src, lambda q: mean_virtue(q)[:, None], identity, fold)
        cal = nats_core(lines, src, lambda q: mean_virtue(q)[:, None], maps, fold)
        top = chosen_option_level(lines, src, mean_virtue)
        for j, m in enumerate(models + ['DeepSeek self']):
            self_ = m == 'DeepSeek self'
            rows.append([name, short(m), fmt(raw[0][j, 0], raw[1][j, 0], raw[2][j, 0], self_), '' if self_ else fmt(cal[0][j, 0], cal[1][j, 0], cal[2][j, 0]),
                         '' if self_ else fmt(top[0][j], top[1][j], top[2][j]), cal[3]])
    return tabulate(rows, headers=['Source', 'Model', 'Uncalibrated nats vs DeepSeek (95%)', 'Calibrated nats vs DeepSeek (95%)',
                                   'Chosen option only: virtue level vs DeepSeek (95%)', 'Questions'], tablefmt='html')


def nats_per_label(tables: Path, out: Path) -> tuple[list[Path], str]:
    # Per-label nats (see nats_core), uncalibrated and calibrated side by side. Vices flipped (favourable sign); behaviours and D&D not.
    lines, maps, identity, fold = load_calibration(tables)
    dims = lines[0]['_meta']['dimensions']
    keep = [i for i, d in enumerate(dims) if d['derived_from'] is None]
    sign = np.array([-1. if dims[i]['kind'] == 'vices' else 1. for i in keep])
    ds = 'deepseek/deepseek-v4.1-flash'
    models = [m for m in lines[1]['models'] if m != ds]
    label_fn = lambda q: np.array([q['option_scores'][x] for x in q['options']], float)[:, keep] * sign
    res = {v: {src: nats_core(lines, src, label_fn, mp, fold) for src in SOURCES} for v, mp in (('uncalibrated', identity), ('calibrated', maps))}
    cols = [('uncalibrated', j) for j in range(len(models))] + [('calibrated', j) for j in range(len(models) + 1)]  # DeepSeek self once, at the end
    names = [short(models[j]) if j < len(models) else 'DS self' for v, j in cols]
    figs, dnd_rows = [], []
    for fam, title, clip, height in (('virtues', 'virtue (vices flipped: blue = more virtuous, less of a vice)', .4, 17), ('behaviours', 'behaviour (blue = more of the behaviour, not always better)', .4, 8), ('dnd', 'D&D axis (order = lawful, morality = good)', .4, 2.4)):
        ix = [k for k, i in enumerate(keep) if dims[i]['family'] == fam]
        lab = [re.sub(r'[^\x00-\u024f]', '', dims[keep[k]]['label'] + (' (vice)' if dims[keep[k]]['kind'] == 'vices' else '') + (f" [{dims[keep[k]]['tradition']}]" if dims[keep[k]].get('tradition') else '')).replace(', )', ')') for k in ix]
        order = np.argsort([np.nanmean([res['calibrated'][s][0][:len(models), k] for s in SOURCES]) for k in ix])
        fig, axs = plt.subplots(1, 2, figsize=(14, height + 1.6), constrained_layout=True)
        for ax, (src, sname) in zip(axs, SOURCES.items()):
            M = np.array([[res[v][src][0][j, ix[k]] for v, j in cols] for k in order])
            sig = np.array([[(res[v][src][0][j, ix[k]] < res[v][src][1][j, ix[k]]) | (res[v][src][0][j, ix[k]] > res[v][src][2][j, ix[k]]) if j == len(models)
                             else (res[v][src][1][j, ix[k]] > 0) | (res[v][src][2][j, ix[k]] < 0) for v, j in cols] for k in order])
            im = ax.imshow(np.clip(M, -clip, clip), cmap='RdBu', vmin=-clip, vmax=clip, aspect='auto')
            yy, xx = np.nonzero(sig)
            ax.plot(xx, yy, 'k.', ms=3)
            ax.axvline(len(models) - .5, color='k', lw=1.5)
            ax.set_xticks(range(len(cols)), names, fontsize=8)
            ax.tick_params(axis='x', top=True, labeltop=True)
            for g, text in ((len(models) / 2 - .5, 'uncalibrated'), (len(models) + (len(models) + 1) / 2 - .5, 'calibrated')):
                ax.annotate(text, (g, 1), xycoords=ax.get_xaxis_transform(), xytext=(0, 24), textcoords='offset points', ha='center', fontsize=10, weight='bold')
            inf = res['calibrated'][src][4]
            ax.set_yticks(range(len(ix)), [f'{lab[k]} (n={inf[ix[k]]})' for k in order], fontsize=7)
            ax.set_title(f"{sname} ({res['calibrated'][src][3]} questions; n = questions where the label differs)", loc='left', fontsize=10, pad=48)
        fig.colorbar(im, ax=axs, shrink=.6 if fam != 'dnd' else 1, label=f'nats vs DeepSeek (clipped at \u00b1{clip})')
        fig.suptitle(f'Log-odds shift vs DeepSeek toward the option with more of each {title}; dot = 95% interval excludes 0 (DS self: outside swap band)', fontsize=10)
        path = out / f'nats_{fam}_random.png'
        fig.savefig(path, dpi=110)
        plt.close(fig)
        figs.append(path)
        if fam == 'dnd':
            for s in SOURCES:
                for k in ix:
                    for j, m in enumerate(models + ['DeepSeek self']):
                        r_raw, r_cal = res['uncalibrated'][s], res['calibrated'][s]
                        dnd_rows.append([SOURCES[s], short(m), dims[keep[k]]['label'], r_cal[4][k], fmt(r_raw[0][j, k], r_raw[1][j, k], r_raw[2][j, k], j == len(models)),
                                         '' if j == len(models) else fmt(r_cal[0][j, k], r_cal[1][j, k], r_cal[2][j, k])])
    return figs, tabulate(dnd_rows, headers=['Source', 'Model', 'D&D axis', 'Questions where it differs', 'Uncalibrated nats (95%)', 'Calibrated nats (95%)'], tablefmt='html')


def virtue_axes(tables: Path, out: Path) -> tuple[Path, str]:
    # Principal axes of the 66 direct virtue labels (favourable sign, centred within question), per source; models scored in calibrated nats along each axis.
    # Axis 1 = shared "overall goodness" direction; axes 2-3 = the largest trade-offs left after it.
    lines, maps, _, fold = load_calibration(tables)
    dims = lines[0]['_meta']['dimensions']
    direct = [i for i, d in enumerate(dims) if d['family'] == 'virtues' and d['derived_from'] is None]
    fav = np.array([-1. if dims[i]['kind'] == 'vices' else 1. for i in direct])
    name = [re.sub(r'[^\x00-\u024f]', '', dims[i]['label']).replace(' (compassion, )', '') for i in direct]
    ds = 'deepseek/deepseek-v4.1-flash'
    models = [m for m in lines[1]['models'] if m != ds]
    cols = models + ['DeepSeek self']
    scores = lambda q: np.array([q['option_scores'][x] for x in q['options']], float)[:, direct] * fav
    fig, axs = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    rows = []
    for c, (src, sname) in enumerate(SOURCES.items()):
        Z = np.vstack([x - x.mean(0) for x in (scores(q) for q in lines[1:] if q['source'] == src) if np.isfinite(x).all()])
        _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
        share = sv ** 2 / (sv ** 2).sum()
        V = Vt[:3] * np.sign(Vt[:3].sum(1, keepdims=True))  # sign: the pole with more virtue mass is positive
        point, lo, hi, n, _ = nats_core(lines, src, lambda q: scores(q) @ V.T, maps, fold)
        for k in range(3):
            ax = axs[k, c]
            neg, pos = np.argsort(V[k])[:5], np.argsort(V[k])[::-1][:5]
            for j, m in enumerate(cols):
                col = COLORS[short(m)]
                if m == 'DeepSeek self':  # swap-null band, not a confidence interval
                    ax.fill_betweenx([j - .3, j + .3], lo[j, k], hi[j, k], color='#dddddd')
                else:
                    ax.hlines(j, lo[j, k], hi[j, k], color=col, lw=2)
                ax.plot(point[j, k], j, 'o', color=col, ms=6)
            ax.axvline(0, color='#555555', lw=.8)
            ax.set_yticks(range(len(cols)), [short(m) for m in cols], fontsize=8)
            ax.invert_yaxis()
            title = ('overall goodness (all virtues load positively; largest: ' + ', '.join(name[i] for i in pos[:3]) + ')') if k == 0 else \
                    (', '.join(name[i] for i in neg[:3]) + '  \u2190 \u2192  ' + ', '.join(name[i] for i in pos[:3]))
            ax.set_title(f'{sname}, axis {k + 1} ({share[k]:.0%} of label variation)\n{title}', loc='left', fontsize=8)
            rows.append([sname, k + 1, f'{share[k]:.0%}', '' if k == 0 else ', '.join(name[i] for i in neg), ', '.join(name[i] for i in pos)]
                        + [fmt(point[j, k], lo[j, k], hi[j, k], m == 'DeepSeek self') for j, m in enumerate(cols)])
        axs[2, c].set_xlabel('calibrated nats vs DeepSeek along the axis (positive = toward the right-hand pole)')
    fig.suptitle('Virtue axes in calibrated nats (95% intervals: games/topics and draws; grey band = DeepSeek self swap null)', fontsize=10)
    path = out / 'virtue_axes_random_calibrated.png'
    fig.savefig(path, dpi=110)
    plt.close(fig)
    table = tabulate(rows, headers=['Source', 'Axis', 'Share of label variation', 'Negative pole (top 5)', 'Positive pole (top 5)'] + [short(m) for m in cols], tablefmt='html')
    return path, table


def page_random(tables: Path, out: Path, title: str) -> Path:
    # Page for a hash-random question set: everything in nats vs DeepSeek, uncalibrated and split-half calibrated.
    out.mkdir(parents=True, exist_ok=True)
    img = lambda p: f'<img src="{p.name}" alt="{escape(p.stem)}">'
    overall = nats_overall(tables)
    calib_png = calibration_before_after(tables, out)
    axes_png, axes_table = virtue_axes(tables, out)
    nats_pngs, nats_dnd = nats_per_label(tables, out)
    qs = list(map(json.loads, tables.open()))[1:]
    qs.sort(key=lambda q: hashlib.sha256(('example:' + q['source_id']).encode()).hexdigest())
    ex = []
    for q in [q for src in SOURCES for q in [x for x in qs if x['source'] == src][:3]]:
        models = q['models']
        table = [[o, escape(t[:90])] + [f"{models[m]['mean_stated_percent'][o]:.0f}%" for m in models] for o, t in q['options'].items()]
        replies = ''.join(f"<details><summary>{escape(short(m) if 'deepseek' not in m else 'DeepSeek')}: draw 1 reasoning and setup note</summary><pre>{escape(v['draws'][0]['reasoning'])}\n\nSetup note: {escape(v['draws'][0]['setup_note'])}</pre></details>" for m, v in models.items())
        ex.append(f"<h4>{escape(q['source_id'])}</h4><pre>{escape(q['text'][-900:])}</pre>" + tabulate(table, headers=['id', 'option'] + [short(m) if 'deepseek' not in m else 'DeepSeek' for m in models], tablefmt='html') + replies)
    takeaways = tables.parent / 'takeaways.html'  # hand-written summary of this run, kept next to its data
    blocks = [f'<h1>{escape(title)}</h1>',
              '<p>Five models stated a percentage for every option of 1,000 hash-random, eligibility-read questions (500 everyday dilemmas, 500 game summaries), 2 draws each with rotated option order, '
              'reasoning budget 10k tokens, first-person "you are now in this person\'s place" (isekai) prompt. One Gemini draw (content-policy block on daily_dilemmas_7099) is dropped; analyses that need both draws leave that question out for all models.</p>',
              '<p><b>Unit: nats vs DeepSeek.</b> For each question, each option\'s log-probability for the model minus DeepSeek\'s on the same question. For a label, these differences are combined over option pairs, '
              'weighted by how much the two options differ on the label: the result is how many nats more the model\'s odds favour the option with more of that label. '
              'In a 2-option question +0.1 nats is about 1.1 times DeepSeek\'s odds. <b>Calibrated</b> = each model\'s spread of percentages is first mapped onto DeepSeek\'s (one map per model and source, '
              'fitted on the other half of the questions), so a model that spreads its percentages evenly is not read as preferring worse options. Uncalibrated uses the same code with no map.</p>',
              takeaways.read_text() if takeaways.exists() else '',
              '<h2>1. Overall: does each model lean toward more virtuous options than DeepSeek?</h2>', overall,
              '<p>Label = mean of the 66 directly judged virtue scores of each option (vices flipped). The chosen-option column uses only each draw\'s top option (scale -1 to +1, not nats), '
              'so it does not depend on how a model spreads its percentages. "DeepSeek self" = DeepSeek draw 1 minus draw 2 (uncalibrated); its band comes from random draw-label swaps. '
              'Intervals resample games/topics and then each model\'s two draws (calibration maps held fixed).</p>',
              '<h2>2. Calibration: decisiveness before and after</h2>',
              '<p>Each curve shows, for one model, how far apart its stated percentages for two options of the same question are (absolute log-odds; each question weighted equally). '
              'Left of DeepSeek (black) = spreads its percentages more evenly. After calibration the curves should lie on black; option order within a question never changes. '
              'Percentages are add-one smoothed and pooled as a geometric mean over the 2 draws. Overlap on the held-out half shows the decisiveness scale transfers between halves, not that preferences are correct.</p>',
              img(calib_png),
              '<h2>3. Virtue axes</h2>',
              '<p>The 66 virtue labels are nearly one scale on these options: every pair of virtues is positively correlated (mean 0.6) and the first axis holds about 70% of the label variation, '
              'so a model that prefers the "good" option tends to win on most virtues at once, and most virtue groups show the same model order. The axes separate that shared direction from the two largest trade-offs left after it. '
              'Axes come from the label scores only (no model data), separately per source.</p>',
              img(axes_png), '<details><summary>Axis poles and numbers</summary>' + axes_table + '</details>',
              '<h2>4. Every label: uncalibrated vs calibrated</h2>',
              '<p>Same metric per label. Dividing by the absolute label gap (not its variance) avoids blow-ups from tiny gaps, but rare labels such as killing rest on few questions (n in the row labels). '
              'Because the virtue labels share one direction (section 3), most virtue rows repeat the overall result.</p>',
              ''.join(img(x) for x in nats_pngs), nats_dnd,
              '<h2>Known limits</h2><ul>'
              '<li>DeepSeek favours the option shown second: on everyday questions where draw 1 showed the less virtuous option first, draw 1 was more virtuous (+0.017 virtue level) and vice versa (-0.006). '
              'Models and DeepSeek see the same two orders, so this should largely cancel in model comparisons (not tested order by order), but it adds noise to the "DeepSeek self" check. '
              'On everyday questions DeepSeek self still sits just outside its swap band on overall goodness (+0.13 nats); read everyday differences of about that size with caution.</li>'
              '<li>Halves for calibration share games/topics: no test on unseen games. Maps are held fixed in the intervals.</li>'
              '<li>Muse and Luna return only reasoning summaries, so their motives cannot be checked from full reasoning.</li></ul>',
              '<h2>Example questions (hash-chosen, not typical or best)</h2>', ''.join(ex),
              f'<p>Data: <code>{escape(str(tables))}</code></p><p>-- PI/OpenAI</p>']
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>' + escape(title) + '</title><style>body{font:16px/1.5 system-ui;max-width:1400px;margin:30px auto;padding:0 24px;color:#222}'
            'img{max-width:100%;height:auto}table{border-collapse:collapse;font-size:14px;margin:10px 0}td,th{padding:5px 10px;border-bottom:1px solid #ddd;text-align:left}'
            'details{border-top:1px solid #ddd;padding:10px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;background:#fafafa;padding:10px}p,ul{max-width:1050px}</style><body>'
            + '\n'.join(blocks) + '</body></html>')
    (out / 'index.html').write_text(html)
    return out / 'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scoring', type=Path, default=ROOT / 'outputs/20261026_probability_eval_scored/scoring')
    parser.add_argument('--out', type=Path, default=ROOT / '.local/choice_preview/probability_eval')
    parser.add_argument('--tables', type=Path, help='question_tables.jsonl of a random run; builds the random-set page instead')
    parser.add_argument('--title', default='Which virtues does each model favour? 1,000 random questions')
    a = parser.parse_args()
    print(page_random(a.tables, a.out, a.title) if a.tables else page(a.scoring, a.out))
