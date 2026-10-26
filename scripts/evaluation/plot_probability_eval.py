"""Results page for the stated-probability tiny eval: per source, each model's preference per label in nats vs DeepSeek. -- PI/OpenAI"""

import argparse
from collections import defaultdict
import hashlib
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {'daily_dilemmas': 'Everyday dilemmas', 'machiavelli': 'Game summaries'}
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
    xlabel = 'Nats per unit of label score, model minus DeepSeek'
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
            dot_panel(ax, sub, [d['id'] for d in dims], f'{name} \u00b7 {role} questions (up to {n})', xlabel)
        lab = [f"{d['label']} [{d.get('tradition', '')}]" + (' \u2022 vice' if d['kind'] == 'vices' else '') if family == 'virtues'
               else d['label'].removeprefix('agg_').replace('_', ' ') for d in dims]
        axs[0].set_yticks(range(len(dims)), lab)
        handles, names = axs[0].get_legend_handles_labels()
        fig.suptitle(f'{family.capitalize()}: which options does each model put more probability on than DeepSeek? ({mode})', fontsize=12)
        fig.legend(handles, names, loc='outside lower center', ncol=len(names), frameon=False, markerscale=2)
        fig.supxlabel('Positive = the model gives more probability than DeepSeek to options scoring higher on this label (vices: more of the named vice). '
                      'Lines: 95% game/topic cluster bootstrap; arrows = interval runs past the axis. "DeepSeek self" = DeepSeek draws 0-1 vs 2-3, the noise floor.', fontsize=9)
        path = out / f'{family}_{role}_{mode}.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        made.append(path)
    return made


def examples(question_tables: Path, scenes_path: Path, k: int = 6) -> str:
    scenes = {s['source_id']: s for s in map(json.loads, scenes_path.open())}
    by = defaultdict(dict)
    for line in question_tables.open():
        q = json.loads(line)
        if '_meta' in q or q.get('map_kind') not in (None, 'identity'):
            continue
        by[q['source_id']][short(q['model'])] = q
    ids = sorted((s for s in by if s in scenes and scenes[s].get('eval_role') == 'dev'), key=lambda s: hashlib.sha256(('example:' + s).encode()).hexdigest())
    blocks = []
    for sid in [s for src in SOURCES for s in [x for x in ids if scenes[x]['source'] == src][:k // 2]]:
        s, qs = scenes[sid], by[sid]
        any_q = next(iter(qs.values()))
        opts = any_q['option_ids']
        table = [[o, escape(s['choice_text'][o][:90]), f"{100 * np.exp(any_q['baseline_logp'][i]):.0f}%"]
                 + [f"{100 * np.exp(qs[m]['raw_logp'][i]):.0f}% ({qs[m]['raw_nats'][i]:+.2f})" if m in qs else '' for m in COLORS if m != 'DeepSeek self']
                 for i, o in enumerate(opts)]
        blocks.append(f'<h4>{escape(sid)}</h4><p>{escape(s.get("decision_context", "")[-600:])}</p>'
                      + tabulate(table, headers=['id', 'option', 'DeepSeek'] + [m + ' (nats)' for m in COLORS if m != 'DeepSeek self'], tablefmt='html'))
    return ('<h2>Example questions</h2><p>Hash-chosen dev questions, not typical or best. Percent = geometric mean of the 4 stated draws (add-one smoothed, renormalised); nats vs DeepSeek in brackets.</p>'
            + ''.join(blocks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True, help='scorer JSON with rows (e.g. scored_identity.json)')
    parser.add_argument('--question-tables', type=Path, required=True)
    parser.add_argument('--scenes', type=Path, default=ROOT / 'outputs/20261026_probability_eval_scored/scenes.jsonl')
    parser.add_argument('--out', type=Path, default=ROOT / '.local/choice_preview/probability_eval')
    parser.add_argument('--mode', default='raw')
    a = parser.parse_args()
    paths = [p for role in ('dev', 'control') for p in render(a.scores, a.scenes, a.out, a.mode, role)]
    print('\n'.join(map(str, paths)))
    (a.out / 'examples.html').write_text(examples(a.question_tables, a.scenes))
