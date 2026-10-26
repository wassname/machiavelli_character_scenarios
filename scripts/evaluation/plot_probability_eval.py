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
            dot_panel(ax, sub, [d['id'] for d in dims], f'{name} \u00b7 {n} {role} questions (fewer for labels missing on some menus)', xlabel)
        lab = [f"{d['label']} [{d.get('tradition', '')}]" + (' \u2022 vice' if d['kind'] == 'vices' else '') if family == 'virtues'
               else d['label'].removeprefix('agg_').replace('_', ' ') for d in dims]
        axs[0].set_yticks(range(len(dims)), lab)
        axs[0].legend(loc='upper left', bbox_to_anchor=(0, 1.0), ncol=len({short(r['model']) for r in rows}), frameon=True, framealpha=.95, fontsize=8, markerscale=1.5)
        fig.suptitle(f'{family.capitalize()}: which options does each model put more probability on than DeepSeek? ({mode}, {role} questions)', fontsize=12)
        fig.supxlabel('Positive = the model gives more probability than DeepSeek to options scoring higher on this label (vices: more of the named vice). '
                      'Lines: 95% game/topic cluster bootstrap over questions (draw noise not included). Arrows: the interval (or the point, drawn at the edge) runs past the axis. '
                      '"DeepSeek self" = DeepSeek draws 1-2 vs 3-4, an empirical noise check.', fontsize=9)
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
    for ax, (src, name) in zip(axs, SOURCES.items(), strict=True):
        grid = np.full((len(dims), len(models)), np.nan)
        sig = np.zeros_like(grid, dtype=bool)
        for r in rows:
            if r['source'] == src and r['beta_nats_per_label_unit'] is not None:
                i, j = [d['id'] for d in dims].index(r['dimension']['id']), models.index(short(r['model']))
                grid[i, j] = r['beta_nats_per_label_unit']
                lo, hi = r['ci95']
                sig[i, j] = lo > 0 or hi < 0
        im = ax.imshow(np.clip(grid, -.6, .6), cmap='RdBu', vmin=-.6, vmax=.6, aspect='auto')
        yy, xx = np.nonzero(sig)
        ax.plot(xx, yy, 'k.', ms=3)
        ax.set_xticks(range(len(models)), models, rotation=45, ha='right')
        ax.set_title(f"{name.split()[0]} ({max(r['informative_questions'] for r in rows if r['source'] == src)} q)", fontsize=10)
        for j, m in enumerate(models):
            pos, neg = int(((grid[:, j] > 0) & sig[:, j]).sum()), int(((grid[:, j] < 0) & sig[:, j]).sum())
            counts.append([name, m, pos, neg, int(np.isfinite(grid[:, j]).sum())])
    axs[0].set_yticks(range(len(dims)), [f"{d['label']} [{d.get('tradition', '')}]" + (' \u2022 vice' if d['kind'] == 'vices' else '') for d in dims], fontsize=7)
    fig.colorbar(im, ax=axs, shrink=.3, label='Nats per label unit vs DeepSeek (clipped at \u00b10.6)')
    fig.suptitle(f'66 virtues, {role} questions ({mode}): blue = model puts more probability than DeepSeek on options scoring higher on this label\n(for rows marked vice: more of the vice)', fontsize=10)
    fig.supxlabel('Dot = 95% game/topic bootstrap interval excludes 0 (draw noise not included). "DeepSeek self" = DeepSeek draws 1-2 vs 3-4, an empirical noise check.', fontsize=8)
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
            sub = sorted((r for r in rows if short(r['model']) == m and r['source'] == src), key=lambda r: r['beta_nats_per_label_unit'] / r['bootstrap_se'])
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
    fig.suptitle(f'Where each model differs most from DeepSeek ({role} questions, {mode}): bottom {k} and top {k} of 66 virtues\n'
                 'Bar = nats per unit of virtue score, model minus DeepSeek; grey bar = 95% interval includes 0. DeepSeek vs itself gives about 5 of 66 below 0 by noise alone.', fontsize=11)
    path = out / f'top_differences_{role}_{mode}.png'
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


def page(scoring: Path, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((scoring / 'summary.json').read_text())
    noise = json.loads((scoring / 'draw_noise.json').read_text())['summaries']
    figs, counts = {}, {}
    for mode, file, roles in (('raw', 'scored_identity.json', ('dev', 'control')), ('calibrated', 'scored_quantile.json', ('dev',))):
        for role in roles:
            figs[mode, role] = render(scoring / file, None, out, mode, role)
            h, c = heatmap(scoring / file, out, mode, role)
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
              'For each question and option, nats = log(model %) minus log(DeepSeek %), so the question\'s base rate cancels. Per virtue, the number shown is how many nats more '
              '(or fewer) a model puts on options that score one unit higher on that virtue, compared with DeepSeek. Everyday dilemmas and game summaries are kept separate. '
              'Questions were chosen with DeepSeek only: its most uncertain 10% (dev) and random controls.</p>',
              '<p><b>Read with care.</b> These raw numbers mix two things: a model\'s values and how decisive it is. A model that spreads its percentages more evenly than DeepSeek '
              'gets negative nats on whatever DeepSeek favours. On the confident control questions this dominates (Qwen, the most even spreader, is below DeepSeek on most virtues). '
              'Calibrating each model\'s confidence scale to DeepSeek\'s (quantile matching) was meant to remove this; it passed its check on dev questions but produced large '
              'artefacts outside its fitted range (DeepSeek compared with itself on control questions: 51 of 66 virtues below zero), so it is shown only for dev, as a secondary view.</p>',
              '<h2>How many virtues differ from DeepSeek? (dev questions, raw)</h2>', count_table(('raw', 'dev')),
              '<p>"DeepSeek self" compares DeepSeek\'s draws 1-2 with draws 3-4. It still shows about 5 virtues below zero: draw-to-draw noise is not in the intervals, and the 66 '
              'virtue labels move together, so one shared fluctuation can push several past zero. Treat about 5 per model as the noise level.</p>',
              '<h2>Did choosing uncertain questions help?</h2>', noise_table,
              '<p>Variance of the four models\' mean log-probabilities divided by their draw-to-draw variance, pooled over questions. Dev is not higher than control, so choosing '
              'DeepSeek-uncertain questions did not separate the models better than random questions.</p>',
              '<h2>Where each model differs most (dev, raw)</h2>', img(top_differences(scoring / 'scored_identity.json', out, 'raw', 'dev')),
              '<h2>All 66 virtues (dev, raw)</h2>', img(figs['raw', 'dev'][0]),
              '<details open><summary>Virtues with intervals (dev, raw)</summary>' + img(figs['raw', 'dev'][1]) + '</details>',
              '<details open><summary>Behaviours, including violence (dev, raw)</summary>' + img(figs['raw', 'dev'][2]) + '</details>',
              '<details open><summary>D&D lawfulness ("order") and goodness ("morality") (dev, raw)</summary>' + img(figs['raw', 'dev'][3]) + '</details>',
              '<h2>Random control questions (raw)</h2>', count_table(('raw', 'control')), ''.join(img(x) for x in figs['raw', 'control']),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scoring', type=Path, default=ROOT / 'outputs/20261026_probability_eval_scored/scoring')
    parser.add_argument('--out', type=Path, default=ROOT / '.local/choice_preview/probability_eval')
    a = parser.parse_args()
    print(page(a.scoring, a.out))
