# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy>=2", "tabulate>=0.9"]
# ///
"""Local pilot page: held-out disagreement and unaveraged source-separated profiles. -- PI/OpenAI"""

import argparse
from html import escape as esc
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from tabulate import tabulate

COLORS = {'Muse': '#7b3294', 'Qwen': '#333333', 'Gemini': '#c65a00', 'Luna': '#007a92'}
NAMES = {'daily_dilemmas': 'Everyday dilemmas', 'machiavelli': 'Game-state summaries'}


def render(run: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((run / 'analysis.json').read_text())
    verification = json.loads((run / 'verification.json').read_text())
    figures = []
    plt.rcParams.update({'font.size': 11, 'font.family': ['DejaVu Sans', 'Noto Serif CJK SC'], 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.labelcolor': '#222222', 'savefig.facecolor': 'white'})
    models = [n for n in data['models'].values() if n != 'Qwen']
    rows = {(r['source'], r['selection'], r['mode'], r['model'], r['item']): r for r in data['rows']}
    def row(source, mode, model, item):
        return rows[source, 'differentiated', mode, model, item]
    def dots(ax, source, mode, dims):
        for j, model in enumerate(models):
            points = [row(source, mode, model, d['id']) for d in dims]
            y = np.arange(len(points)) + (j - 1) * .20
            ax.hlines(y, [p['ci95'][0] for p in points], [p['ci95'][1] for p in points], color=COLORS[model], alpha=.65, lw=1)
            ax.scatter([p['mean'] for p in points], y, s=17, color=COLORS[model], label=model, zorder=3)
        ax.axvline(0, color='#777777', lw=.8)
        ax.set_ylim(len(dims) - .4, -.8)
        ax.set_yticks(range(len(dims)))
        ax.tick_params(axis='y', length=0, pad=10)
        ax.grid(axis='x', alpha=.13)
        ax.set_xlabel('Difference from Qwen')
    def save(fig, name):
        fig.savefig(out / name, dpi=130, bbox_inches='tight')
        fig.savefig((out / name).with_suffix('.svg'), bbox_inches='tight')
        figures.append({'path': str((out / name).resolve()), 'sha256': hashlib.sha256((out / name).read_bytes()).hexdigest(),
            'axes': [{'title': ax.get_title(), 'xlabel': ax.get_xlabel(), 'ylabel': ax.get_ylabel(), 'xlim': list(ax.get_xlim()), 'ylim': list(ax.get_ylim())} for ax in fig.axes]})
        plt.close(fig)
    direct = [d for d in data['dimensions'] if d['family'] == 'virtues' and d['derived_from'] is None]
    limit = max(abs(x) for r in data['rows'] if r['selection'] == 'differentiated' and r['family'] == 'virtues'
                and r['mode'] in ('versus_qwen', 'residual_vs_qwen') and any(d['id'] == r['item'] for d in direct) for x in r['ci95']) * 1.08
    for source, title in NAMES.items():
        fig, axs = plt.subplots(1, 2, figsize=(16, 24), sharey=True, constrained_layout=True)
        for ax, mode, subtitle in zip(axs, ['versus_qwen', 'residual_vs_qwen'], ['Original labels', 'Fixed common component removed'], strict=True):
            dots(ax, source, mode, direct)
            ax.set_xlim(-limit, limit)
            ax.set_title(subtitle, pad=15)
        axs[0].set_yticklabels([f"{d['label']} [{d['tradition']}]" + (' • vice' if d['kind'] == 'vices' else '') for d in direct], fontsize=10)
        for i in range(1, len(direct)):
            if direct[i]['tradition'] != direct[i - 1]['tradition']:
                for ax in axs:
                    ax.axhline(i - .5, color='#dddddd', lw=.8)
        axs[0].legend(loc='lower left', ncol=3, bbox_to_anchor=(0, 1.015), frameon=False)
        fig.suptitle(f'{title}: 40 selected questions, 66 directly judged traits\nQwen = 0; points = paired mean differences; lines = pointwise 95% cluster-bootstrap intervals', fontsize=14)
        fig.supxlabel('Same horizontal scale in both sources and both views. Higher = more of the named trait, including vices.\nResiduals remove a shared label direction, not capability; 14 derived labels are excluded from these figures.', fontsize=11)
        save(fig, f'virtues_{source}.png')
    behaviours = [d for d in data['dimensions'] if d['family'] == 'behaviours']
    fig, axs = plt.subplots(1, 2, figsize=(14, 11), sharey=True, constrained_layout=True)
    lim = max(abs(x) for r in data['rows'] if r['selection'] == 'differentiated' and r['family'] == 'behaviours' and r['mode'] == 'versus_qwen' for x in r['ci95']) * 1.08
    for ax, (source, title) in zip(axs, NAMES.items(), strict=True):
        dots(ax, source, 'versus_qwen', behaviours)
        ax.set_xlim(-lim, lim)
        n = data['sources'][source]['strata']['differentiated']['families']['behaviours']['scenes']
        ax.set_title(f'{title} · {n} selected questions')
    axs[0].set_yticklabels([d['label'].removeprefix('agg_').replace('_', ' ') for d in behaviours])
    axs[0].legend(loc='lower left', bbox_to_anchor=(0, 1.015), ncol=3, frameon=False)
    fig.suptitle('Every behaviour dimension — no combined score\nMean differences from Qwen; pointwise 95% cluster-bootstrap intervals', fontsize=14)
    fig.supxlabel('Higher = greater judged probability of the named behaviour; this is not always better.', fontsize=11)
    save(fig, 'behaviours.png')
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    lim = max(abs(x) for r in data['rows'] if r['selection'] == 'differentiated' and r['family'] == 'dnd' and r['mode'] == 'versus_qwen' for x in r['ci95']) * 1.15
    offsets = {'Muse': (10, 18), 'Gemini': (10, -24), 'Luna': (-35, 28)}
    for ax, (source, title) in zip(axs, NAMES.items(), strict=True):
        for model in models:
            x, y = row(source, 'versus_qwen', model, 'order'), row(source, 'versus_qwen', model, 'morality')
            ax.hlines(y['mean'], *x['ci95'], color=COLORS[model], lw=1)
            ax.vlines(x['mean'], *y['ci95'], color=COLORS[model], lw=1)
            ax.scatter(x['mean'], y['mean'], color=COLORS[model], s=40)
            offset = {'Muse': (20, -25), 'Gemini': (20, 20), 'Luna': (-45, 0)}[model] if source == 'daily_dilemmas' else offsets[model]
            ax.annotate(model, (x['mean'], y['mean']), xytext=offset, textcoords='offset points', color=COLORS[model], arrowprops={'arrowstyle': '-', 'color': COLORS[model], 'lw': .6}, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 1})
        ax.scatter(0, 0, color=COLORS['Qwen'], marker='+', s=70)
        ax.annotate('Qwen', (0, 0), xytext=(-40, -18), textcoords='offset points')
        ax.axhline(0, color='#aaaaaa', lw=.7)
        ax.axvline(0, color='#aaaaaa', lw=.7)
        selected = data['sources'][source]['strata']['differentiated']
        n = selected['families']['dnd']['scenes']
        panel_title = f'{title}\n{n}/{selected["scenes"]} selected; {selected["scenes"] - n} excluded for missing labels'
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel='Lawfulness difference from Qwen →', ylabel='Goodness difference from Qwen →', title=panel_title)
        ax.set_aspect('equal')
    fig.suptitle('D&D alignment — two distinct dimensions\nSelected questions; paired pointwise 95% mean intervals', fontsize=14)
    save(fig, 'dnd.png')
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=True, constrained_layout=True)
    summary, labels = [], []
    for source, title in NAMES.items():
        for stratum, label, color in [('differentiated', 'selected', '#007a92'), ('random_control', 'random control', '#666666')]:
            info = data['sources'][source]['strata'][stratum]
            d = info['disagreement']
            y = len(labels)
            axs[0].hlines(y, *d['excess_ci95'], color=color, lw=2)
            axs[0].scatter(d['excess'], y, color=color, s=40)
            stable = info['stable_difference']
            axs[1].hlines(y, *stable['case_fraction_ci95'], color=color, lw=2)
            axs[1].scatter(stable['case_fraction'], y, color=color, s=40)
            labels.append(f'{title}: {label}\n{info["scenes"]} questions / {info["clusters"]} games or topics')
            summary.append([f'{title} / {label}', f'{d["excess"]:.1%}', ('degenerate' if d['excess_ci95'][0] == d['excess_ci95'][1] else f'{d["excess_ci95"][0]:.1%} to {d["excess_ci95"][1]:.1%}'), f'{stable["cases"]}/{info["scenes"]}', f'{d["between"]:.1%}', f'{d["within"]:.1%}'])
    for ax in axs:
        ax.axvline(0, color='#aaaaaa', lw=.8)
        ax.set_yticks(range(4), labels)
        ax.set_ylim(3.6, -.6)
        ax.tick_params(axis='y', length=0)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    axs[0].set_xlabel('Between models minus exact-repeat variation')
    axs[0].set_title('Excess disagreement', pad=16)
    axs[1].set_xlabel('Fraction of questions with a stable difference')
    axs[1].set_title('Two or more models each stable 6/6,\nbut choosing different actions', pad=16)
    axs[1].set_xlim(-.02, 1.02)
    fig.suptitle('Do historically selected cases distinguish the four new models?\n95% game/topic-cluster bootstrap intervals', fontsize=14)
    fig.supxlabel('The everyday controls all agree: their bootstrap interval is degenerate. This does not establish zero disagreement outside those 10 cases.', fontsize=10)
    save(fig, 'heldout_disagreement.png')
    scenes = {s['source_id']: s for s in map(json.loads, (run / 'scenes.jsonl').open())}
    reviews = json.loads((run / 'inference_source/data/evaluation_eligibility.json').read_text())['cases']
    replies = list(map(json.loads, (run / 'choices.jsonl').open()))
    model_names = data['models']
    question_table, case_html = [], []
    for n, q in enumerate(data['questions']):
        sid = q['source_id']
        label = f"{'D' if q['source'] == 'daily_dilemmas' else 'G'}{n % 50 + 1:02d}"
        gap = 'unstable' if q['residual_choice_gap'] is None else f'{q["residual_choice_gap"]:.3f}'
        title = scenes[sid]['decision_context'].replace('\n', ' ').removeprefix('- ')[:105] + '…'
        cells = [f'<a href="#case-{n}">{label}</a>', f'{q["excess_disagreement"]:.1%}', str(q['new_stable_pair_disagreements']), gap,
                 f'{q["within_model_disagreement"]:.1%}', f'{q["order_disagreement"]:.1%}', esc(q['selection']), esc(title)]
        question_table.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
        cases = sorted((r for r in replies if r['source_id'] == sid), key=lambda r: (list(model_names).index(r['model']), r['order'], r['repetition']))
        choice_table = tabulate([[name, ', '.join(v['forward']), ', '.join(v['reverse'])] for name, v in q['new_choices'].items()], headers=['Model', 'Forward original IDs', 'Reverse original IDs'], tablefmt='html')
        result_table = tabulate([[model_names[r['model']], r['order'], r['repetition'], r['original_choice_id'], r['reasoning'], r['setup_note']] for r in cases],
                                headers=['Model', 'Order', 'Repeat', 'Original choice', 'Explanation', 'Setup note'], tablefmt='html')
        case_html.append(f'<details id="case-{n}"><summary>{label} — {esc(title)}</summary><p><code>{esc(sid)}</code></p><pre>{esc(scenes[sid]["text"])}</pre><p>Parent source review: {esc(reviews[sid]["reason"])}</p>{choice_table}<details><summary>All 24 explanations and setup notes</summary>{result_table}</details></details>')
    headers = ['Question', 'New excess disagreement↑', 'Stable differing pairs /6', 'Old residual gap', 'Exact-repeat disagreement', 'Order disagreement', 'Selection', 'Visible opening']
    question_html = '<table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(question_table) + '</tbody></table>'
    blocks = [f'<h1>Four-model differentiated pilot</h1><p>Muse Spark 1.3 Contributor · Qwen 3.8 27B · Gemini 3.8 Flash · GPT-5.6 Luna</p>',
        '<p><a href="../index.html">Earlier full-cohort profiles</a> · <a href="#cohort">Frozen question table</a> · <a href="#cases">Complete scenarios and replies</a> · <a href="analysis.json">All numerical results</a></p>',
        '<p>100 personally reviewed questions: 40 selected + 10 random controls per source. Each model answered each question in two display orders, three fresh requests per order. Selection used only three historical actors; none of these four new actors selected the cases.</p>',
        '<p><strong>Selection is not validation.</strong> Some source ambiguities and known grading concerns remain visible in the case reviews. “Native tradeoff” in the selection artifact is scene-wide, not evidence that the old actors chose conflicting values.</p>',
        '<h2>Did selection transfer?</h2>', tabulate(summary, headers=['Source / sample', 'Excess disagreement↑', '95% interval', 'Stable differences↑', 'Between models↑', 'Exact repeats↓'], tablefmt='html'),
        '<img src="heldout_disagreement.png" alt="New-model disagreement beyond exact-repeat variation, selected versus random cases">',
        '<p>The 10 everyday controls all give zero disagreement, so resampling them cannot estimate a nonzero uncertainty range. Their degenerate bootstrap interval is not evidence of zero disagreement in the wider population.</p>',
        '<p>A stable difference means at least two models each chose the same original action in all six replies, and those actions differ. It does not require all four models to be stable. The per-question pair count uses all six possible model pairs as its denominator; an unstable pair does not count as demonstrated stable disagreement.</p>',
        f'<p>{esc(data["disagreement"])}</p><p>{esc(data["uncertainty"])}</p>',
        '<h2>Selected-case profiles</h2><p>The figures below use only the 40 selected questions per source; random-control numerical profiles remain in analysis.json. Qwen is the reference for these four-model comparisons. Subtracting a reference does not remove common ranking. The right-hand virtue panels additionally remove the fixed historical PC1; this is not a correction for capability.</p>',
        f'<p>{esc(data["residual_direction"])}</p>']
    for source, title in NAMES.items():
        blocks.append(f'<details><summary>{title}: all 66 directly judged virtues and vices, original versus PC1 residual</summary><img src="virtues_{source}.png" alt="{title} individual-trait paired comparisons"></details>')
    derived = [d for d in data['dimensions'] if d['family'] == 'virtues' and d['derived_from'] is not None]
    labels_by_id = {d['id']: d['label'] for d in data['dimensions']}
    derived_rows = []
    for source, title in NAMES.items():
        for d in derived:
            values = [row(source, 'versus_qwen', model, d['id']) for model in models]
            derived_rows.append([title, d['label'], labels_by_id[d['derived_from']]] + [f'{r["mean"]:+.3f} [{r["ci95"][0]:+.3f}, {r["ci95"][1]:+.3f}]' for r in values])
    blocks.append('<details><summary>14 derived vice labels, not independent measurements</summary><p>Original-label mean differences from Qwen with pointwise 95% intervals, on the same 40 selected questions per source. Each is the inverse of its named source virtue; these add no independent evidence.</p>' + tabulate(derived_rows, headers=['Source', 'Derived label', 'Inverse of', *models], tablefmt='html') + '</details>')
    dnd_counts = '; '.join(f'{title}: {data["sources"][source]["strata"]["differentiated"]["families"]["dnd"]["scenes"]}/40 selected questions' for source, title in NAMES.items())
    blocks += ['<details open><summary>Every behaviour dimension, including violence</summary><img src="behaviours.png" alt="All named behaviours by source"></details>',
               f'<details open><summary>D&D lawfulness and goodness, not averaged together</summary><p>{dnd_counts}. A question is excluded here if either axis is ungraded for any offered action. Missing grades are not neutral scores. Crossbars are separate axis intervals, not a joint confidence region.</p><img src="dnd.png" alt="Two-dimensional alignment comparisons by source"></details>',
               '<h2 id="cohort">Frozen cohort and differentiation columns</h2><p>Order remains the pre-run selection order. Old residual gap is RMS chosen-trait separation among historical actors stable under reversal. New excess disagreement uses the new replies. These are different measures. D = everyday; G = game. “Unstable” means the old-actor score was undefined, not zero.</p>', question_html,
               '<h2 id="cases">Complete visible questions and all 24 replies per question</h2>', *case_html,
               '<h2>Provenance and limits</h2>', f'<p>{esc(data["method"])}</p><p>{esc(data["checks"])}</p>',
               f'<p>All actor charges, including superseded smoke diagnostics: ${verification["all_pilot_attempts_cost_usd"]:.4f}. Active analysis replies: ${verification["used_reply_cost_usd"]:.4f}. New unknown-fee reservations: ${verification["pilot_unknown_fee_reservations_usd"]:.4f}. Review estimates are accounted separately against the additional $10 envelope.</p>',
               f'<p>Full frozen requests, native responses, usage, errors and code snapshots: <code>{esc(str(run.resolve()))}</code>.</p>',
               '<p>Model differences can reflect comprehension, prompt compliance, label errors, preferences or provider settings. Neither this selection nor PC1 isolates a pure character trait. No publication or dataset-wide new-model run.</p><p>— PI/OpenAI</p>']
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Four-model differentiated pilot</title><style>body{font:16px/1.5 system-ui,sans-serif;max-width:1450px;margin:35px auto;padding:0 25px;color:#222}h1{font-size:2rem}img{display:block;max-width:100%;height:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px;margin:15px 0}td,th{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}td:nth-child(2){font-weight:600}pre{white-space:pre-wrap;font:15px/1.5 system-ui;background:#fafafa;padding:14px}details{border-top:1px solid #ddd;padding:15px 0}summary{cursor:pointer;font-weight:600}a{color:#006b87}code{overflow-wrap:anywhere}p{max-width:1100px}</style><body>' + '\n'.join(blocks) + '</body></html>'
    (out / 'index.html').write_text(html)
    metadata = {'author': 'PI/OpenAI', 'analysis': str((run / 'analysis.json').resolve()),
        'analysis_sha256': hashlib.sha256((run / 'analysis.json').read_bytes()).hexdigest(), 'figures': figures,
        'profile_cohort': 'Virtues/behaviours:40 differentiated questions/source; D&D uses complete-label subsets specified below. Random controls are separate. No relabelling.',
        'family_cohorts': {source: {family: info['scenes'] for family, info in data['sources'][source]['strata']['differentiated']['families'].items()} for source in NAMES},
        'reference': 'Qwen=0 in all profile plots. Virtue figures preserve original named vice direction. PC1 residuals are not capability correction.',
        'intervals': 'Pointwise paired2000-resample game/topic cluster bootstrap. Labels, replies and PC1 fixed. The all-zero10-case everyday-control interval is degenerate, not a population bound.'}
    (out / 'figure_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for name in ('analysis.json', 'selection.json', 'results.md', 'verification.json'):
        shutil.copyfile(run / name, out / name)
    print(out.resolve() / 'index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/20261026_differentiated_pilot_v2'))
    parser.add_argument('--out', type=Path, default=Path('.local/choice_preview/pilot'))
    args = parser.parse_args()
    render(args.run, args.out)
