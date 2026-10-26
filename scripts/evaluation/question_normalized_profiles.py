"""Within-question scaling and ordinal comparisons of saved choices; no inference. -- PI/OpenAI"""

import argparse
from html import escape
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from tabulate import tabulate

from score_choice_dimensions import bootstrap, sha

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {'daily_dilemmas': 'Everyday dilemmas', 'machiavelli': 'Game-state summaries'}
METHODS = {'raw': 'Raw labels', 'centered': 'Question mean removed', 'question_sd': 'Question-centered / option SD',
           'question_range': 'Question-centered / option range', 'question_rank': 'Within-menu fractional rank', 'pairwise': 'Question-wise win advantage'}
COLORS = {'Muse': '#7b3294', 'Qwen': '#333333', 'Gemini': '#c65a00', 'Luna': '#007a92',
          'DeepSeek': '#333333', 'Gemma': '#007a92', 'Haiku': '#7b3294'}


def transforms(actions: np.ndarray) -> dict[str, np.ndarray]:
    assert len(actions) >= 2
    spread = np.ptp(actions, axis=0)
    centered = actions - actions.mean(axis=0)
    centered[:, spread == 0] = 0
    sd = actions.std(axis=0)
    sd[spread == 0] = 0
    scaled = lambda denominator: np.divide(centered, denominator, out=np.zeros_like(centered), where=denominator != 0)
    # PI/OpenAI: Average sign against every menu action equals centered fractional midrank, ties zero.
    rank = np.sign(actions[:, None, :] - actions[None, :, :]).sum(axis=1) / (2 * (len(actions) - 1))
    return {'raw': actions, 'centered': centered, 'question_sd': scaled(sd),
            'question_range': scaled(spread), 'question_rank': rank}


def win_advantage(replies: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.sign(replies[:, :, None, :] - reference[:, None, :, :]).mean(axis=(0, 1, 2)) / 2


def self_check() -> list[str]:
    a = np.array([[1., 2., 4.], [1., 2.00001, 3.], [1., 2., 3.]])
    t, u = transforms(a), transforms(a * 3 + 7)
    for mode in ('question_sd', 'question_range', 'question_rank'):
        assert np.allclose(t[mode], u[mode], atol=1e-8), mode
        assert np.all(t[mode][:, 0] == 0)
        assert np.allclose(t[mode].mean(axis=0), 0)
    binary = transforms(np.array([[0., 20.], [0.00001, 40.]]))
    assert np.allclose(binary['question_sd'], [[-1, -1], [1, 1]])
    assert np.allclose(binary['question_rank'], [[-.5, -.5], [.5, .5]])
    assert np.allclose(t['question_rank'][:, 2], [.5, -.25, -.25])
    sample = np.array([[[0., 1.], [2., 1.], [1., 1.]], [[1., 0.], [1., 2.], [1., 1.]]])
    assert np.allclose(win_advantage(sample, sample), 0)
    assert np.allclose(win_advantage(sample + 3, sample), .5)
    return ['Affine invariance of within-question SD/range/ranks', 'Exact ties and constant menus',
            'Binary-menu SD +/-1 and rank +/-0.5', 'Self-comparison win probability exactly one half']


def analyze(cohort: str, out: Path, replicates: int) -> dict:
    dimensions_dir = ROOT / 'outputs/20261025_individual_dimensions'
    definition = json.loads((dimensions_dir / 'analysis.json').read_text())
    dims = definition['dimensions']
    if cohort == 'pilot':
        run = ROOT / 'outputs/20261026_differentiated_pilot_v2'
        original = json.loads((run / 'analysis.json').read_text())
        model_names = original['models']
        strata = {r['source_id']: r['selection'] for r in original['questions']}
        reference, repeats = 'Qwen', 3
    else:
        run = ROOT / 'outputs/20261024_action_preferences_resumed'
        model_names = definition['models']
        reference, repeats = 'DeepSeek', 1
    scenes = {s['source_id']: s for s in map(json.loads, (run / 'scenes.jsonl').open())}
    if cohort == 'historical':
        strata = {sid: 'historical_full' for sid in scenes}
    records = [r for r in map(json.loads, (run / 'choices.jsonl').open()) if r['model'] in model_names]
    replies = {(r['source_id'], r['model'], r['order'], r.get('repetition', 0)): r for r in records}
    assert len(replies) == len(records) == len(scenes) * len(model_names) * 2 * repeats
    assert all(r['status'] == 'ok' for r in records)
    actions = {(r['source_id'], r['choice_id']): np.array(r['scores']) for r in pq.read_table(dimensions_dir / 'action_scores.parquet').to_pylist() if r['source_id'] in scenes}
    names = list(model_names.values())
    ref = names.index(reference)
    pairs = list(combinations(range(len(names)), 2))
    values, pair_values, flat_counts, tiny_counts, range_values, agreement = {}, {}, {}, {}, {}, {}
    for sid, scene in scenes.items():
        ids = list(scene['choice_text'])
        menu = np.stack([actions[sid, c] for c in ids])
        all_selected = np.stack([[[menu[ids.index(replies[sid, m, order, repeat]['original_choice_id'])]
                                   for repeat in range(repeats)] for order in ('forward', 'reverse')] for m in model_names])
        transformed = transforms(menu)
        values[sid] = {}
        pair_values[sid] = {}
        for method, matrix in transformed.items():
            chosen = np.stack([np.mean([matrix[ids.index(replies[sid, m, order, repeat]['original_choice_id'])]
                                       for order in ('forward', 'reverse') for repeat in range(repeats)], axis=0) for m in model_names])
            values[sid][method] = chosen
            pair_values[sid][method] = np.stack([chosen[a] - chosen[b] for a, b in pairs])
        values[sid]['pairwise'] = np.stack([win_advantage(all_selected[m], all_selected[ref]) for m in range(len(names))])
        pair_values[sid]['pairwise'] = np.stack([win_advantage(all_selected[a], all_selected[b]) for a, b in pairs])
        finite = np.isfinite(menu).all(axis=0)
        span = np.ptp(menu, axis=0)
        flat_counts[sid] = finite & (span == 0)
        full_scale = np.array([2. if d['family'] == 'virtues' else 10. if d['family'] == 'dnd' else 1. for d in dims])
        tiny_counts[sid] = finite & (span > 0) & (span < .01 * full_scale)
        range_values[sid] = span / full_scale
        assert np.allclose(pair_values[sid]['raw'][:, finite], pair_values[sid]['centered'][:, finite], atol=1e-12)
        if repeats == 3:
            # PI/OpenAI: Rank concordance across exact repeats, matching the display order; ties contribute zero, not agreement.
            signs = np.stack([np.sign(all_selected[a] - all_selected[b]) for a, b in pairs])
            agreement[sid] = np.stack([signs[:, :, a] * signs[:, :, b] for a, b in combinations(range(repeats), 2)]).mean(axis=(0, 1, 2))
    checks = self_check() + ['All native choice joins unique/complete', 'Raw and centered paired differences identical on every question']
    result = {'author': 'PI/OpenAI', 'cohort': cohort, 'models': names, 'reference': reference, 'dimensions': dims,
              'replicates': replicates, 'rows': [], 'summary': [], 'denominators': [], 'checks': checks,
              'inputs': {str(p.resolve()): sha(p) for p in [run / 'scenes.jsonl', run / 'choices.jsonl', dimensions_dir / 'action_scores.parquet', dimensions_dir / 'analysis.json']},
              'script_sha256': sha(Path(__file__)),
              'method': 'Transform all offered actions separately for each question and label, then average saved replies within question, then equally weight questions. Pairwise wins average all cross-model reply pairs within each display order, then orders. All-zero menu spans contribute zero, not exclusion. No division by a model score; no PC1 removal.',
              'uncertainty': 'Paired game/topic cluster bootstrap of questions, pointwise 95% mean intervals. Labels and saved replies fixed. Repeats are not independent questions. No new calls, graders or labels. Tiny label differences may become full ordinal wins; original margins remain in range diagnostics.',
              'scope': 'Pilot: selected/control sources separate, 3 replies per order, 4 new models.' if cohort == 'pilot' else 'Historical 3860-question full cohort, 3 older models, 1 reply per order, old role-play prompt and unfiltered historical eligibility. Not comparable as a matched seven-model run.'}
    for source in SOURCES:
        for stratum in sorted(set(strata.values())):
            source_ids = sorted(sid for sid in scenes if scenes[sid]['source'] == source and strata[sid] == stratum)
            for family in ('virtues', 'behaviours', 'dnd'):
                ix = [i for i, d in enumerate(dims) if d['family'] == family]
                ids = [sid for sid in source_ids if np.isfinite(values[sid]['raw'][:, ix]).all() and np.isfinite(range_values[sid][ix]).all()]
                assert ids
                groups = [scenes[sid]['eval_cluster'] for sid in ids]
                flat = np.stack([flat_counts[sid][ix] for sid in ids])
                tiny = np.stack([tiny_counts[sid][ix] for sid in ids])
                result['denominators'].append({'source': source, 'stratum': stratum, 'family': family, 'questions': len(ids), 'total_questions': len(source_ids),
                    'clusters': len(set(groups)), 'excluded_missing': sorted(set(source_ids) - set(ids)),
                    'constant_question_labels': int(flat.sum()), 'tiny_nonzero_question_labels': int(tiny.sum()), 'question_labels': int(flat.size),
                    'median_menu_range_fraction_full_scale': float(np.median(np.stack([range_values[sid][ix] for sid in ids]))),
                    'constant_counts_by_item': dict(zip([dims[i]['id'] for i in ix], flat.sum(axis=0).tolist(), strict=True))})
                for method in METHODS:
                    v = np.stack([values[sid][method][:, ix] for sid in ids])
                    pv = np.stack([pair_values[sid][method][:, ix] for sid in ids])
                    combined = np.concatenate([v, pv], axis=1)
                    draws = bootstrap(combined, groups, replicates)
                    pt = combined.mean(axis=0)
                    d = draws[:, :len(names)] if method == 'pairwise' else draws[:, :len(names)] - draws[:, ref:ref + 1]
                    means = pt[:len(names)] if method == 'pairwise' else pt[:len(names)] - pt[ref:ref + 1]
                    ci = np.quantile(d, [.025, .975], axis=0)
                    for m, name in enumerate(names):
                        for j, k in enumerate(ix):
                            result['rows'].append({'source': source, 'stratum': stratum, 'family': family, 'method': method, 'model': name,
                                'item': dims[k]['id'], 'mean': float(means[m, j]), 'ci95': ci[:, m, j].tolist(),
                                'mean_se': float(d[:, m, j].std(ddof=1)), 'absolute_mean': float(pt[m, j]),
                                'question_sd': float(v[:, m, j].std(ddof=1)), 'questions': len(ids)})
                    if family == 'virtues':
                        direct = [j for j, k in enumerate(ix) if dims[k]['derived_from'] is None]
                        pdraws = draws[:, len(names):, direct]
                        delta = pt[len(names):, direct]
                        pair_se = pdraws.std(axis=0, ddof=1)
                        precision = np.divide(np.abs(delta), pair_se, out=np.zeros_like(delta), where=pair_se != 0)
                        lo, hi = np.quantile(pdraws, [.025, .975], axis=0)
                        pooled = np.stack([np.sqrt((v[:, a, direct].var(axis=0, ddof=1) + v[:, b, direct].var(axis=0, ddof=1)) / 2) for a, b in pairs])
                        separation = np.divide(np.abs(delta), pooled, out=np.zeros_like(delta), where=pooled != 0)
                        result['summary'].append({'source': source, 'stratum': stratum, 'method': method, 'questions': len(ids),
                            'median_absolute_paired_mean_over_se': float(np.median(precision)),
                            'median_gap_over_question_sd': None if method == 'pairwise' else float(np.median(separation)),
                            'contrasts_excluding_zero': int(((lo > 0) | (hi < 0)).sum()), 'contrasts': int(delta.size),
                            'median_model_question_sd': float(np.median(v[:, :, direct].std(axis=0, ddof=1))),
                            'repeat_rank_concordance': None if repeats != 3 else float(np.mean(np.stack([agreement[sid][ix][direct] for sid in ids])))})
                        if method == 'raw':
                            raw_pair_draws = draws[:, len(names):]
                        if method == 'centered':
                            assert np.allclose(raw_pair_draws, draws[:, len(names):], atol=1e-12)
    result['checks'].append('Raw/centered paired bootstrap draws identical, not only aggregate means')
    previous = original if cohort == 'pilot' else definition
    old_mode = 'versus_qwen' if cohort == 'pilot' else 'versus_deepseek'
    old = {(r['source'], r.get('selection', 'historical_full'), r['family'], r['model'], r['item']): r for r in previous['rows'] if r['mode'] == old_mode}
    for r in result['rows']:
        if r['method'] == 'raw':
            earlier = old[r['source'], r['stratum'], r['family'], r['model'], r['item']]
            assert np.isclose(r['mean'], earlier['mean'], atol=1e-12)
            if replicates == previous['replicates']:
                assert np.allclose(r['ci95'], earlier['ci95'], atol=1e-12)
    result['checks'].append('Original paired means recovered for every label/source/stratum; equal-replicate intervals also recover')
    (out / f'{cohort}_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def render(data: dict, out: Path) -> list[dict]:
    cohort, reference = data['cohort'], data['reference']
    stratum = 'differentiated' if cohort == 'pilot' else 'historical_full'
    dims = data['dimensions']
    names = [m for m in data['models'] if m != reference]
    lookup = {(r['source'], r['family'], r['method'], r['model'], r['item']): r for r in data['rows'] if r['stratum'] == stratum}
    counts = {(d['source'], d['family']): d for d in data['denominators'] if d['stratum'] == stratum}
    metadata = []
    plt.rcParams.update({'font.size': 10, 'font.family': ['DejaVu Sans', 'Noto Serif CJK SC'], 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False, 'savefig.facecolor': 'white'})
    views = [('centered', 'Current paired label difference'), ('question_sd', 'Normalize each question first'), ('pairwise', 'Head-to-head wins within each question')]
    def get(source, family, method, model, item):
        return lookup[source, family, method, model, item]
    def save(fig, filename):
        p = out / f'{cohort}_{filename}.png'
        fig.savefig(p, dpi=130, bbox_inches='tight')
        fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')
        plt.close(fig)
        metadata.append({'path': str(p.resolve()), 'sha256': sha(p)})
    def panel(ax, source, family, method, ds):
        for m, model in enumerate(names):
            rows = [get(source, family, method, model, d['id']) for d in ds]
            y = np.arange(len(ds)) + (m - (len(names) - 1) / 2) * .20
            ax.hlines(y, [r['ci95'][0] for r in rows], [r['ci95'][1] for r in rows], color=COLORS[model], alpha=.65, lw=1)
            ax.scatter([r['mean'] for r in rows], y, s=13, color=COLORS[model], label=model, zorder=3)
        ax.axvline(0, color='#888888', lw=.7)
        ax.set_ylim(len(ds) - .4, -.8)
        ax.tick_params(axis='y', length=0)
        ax.set_yticks(range(len(ds)))
        ax.grid(axis='x', alpha=.12)
        extent = max(abs(x) for r in data['rows'] if r['stratum'] == stratum and r['family'] == family and r['method'] == method and (family != 'virtues' or any(d['id'] == r['item'] for d in ds)) for x in r['ci95'])
        ax.set_xlim(-extent * 1.06, extent * 1.06)
        if method == 'pairwise':
            ax.set_xlabel(f'Win probability minus 0.5 vs {reference}')
        else:
            ax.set_xlabel(f'Difference from {reference}' + (' (menu SD units)' if method == 'question_sd' else ' (label units)'))
    direct = [d for d in dims if d['family'] == 'virtues' and d['derived_from'] is None]
    for source, title in SOURCES.items():
        fig, axs = plt.subplots(1, 3, figsize=(21, 24), sharey=True, constrained_layout=True)
        for ax, (method, subtitle) in zip(axs, views, strict=True):
            panel(ax, source, 'virtues', method, direct)
            ax.set_title(subtitle, pad=18)
            for i in range(1, len(direct)):
                if direct[i]['tradition'] != direct[i - 1]['tradition']:
                    ax.axhline(i - .5, color='#dddddd', lw=.7)
        axs[0].set_yticklabels([d['label'] + ' [' + d['tradition'] + ']' + (' • vice' if d['kind'] == 'vices' else '') for d in direct], fontsize=9)
        axs[0].legend(loc='lower left', bbox_to_anchor=(0, 1.025), ncol=len(names), frameon=False)
        n = counts[source, 'virtues']['questions']
        label = 'selected pilot' if cohort == 'pilot' else 'historical full cohort'
        fig.suptitle(f'{title} · {n} questions · {label}\n66 directly judged traits; dots = means; lines = paired pointwise 95% game/topic bootstrap intervals', fontsize=14)
        fig.supxlabel('Same scale across sources within each method, not across methods. Higher = more of the named trait, including vices.\nMiddle: subtract menu mean and divide by option SD BEFORE averaging. Right: ties get half credit; +0.1 means 60% tie-adjusted win probability. No PC1 removal.', fontsize=11)
        save(fig, f'virtues_{source}')
    ds = [d for d in dims if d['family'] == 'behaviours']
    fig, axs = plt.subplots(2, 3, figsize=(21, 16), sharey=True, constrained_layout=True)
    for y, (source, title) in enumerate(SOURCES.items()):
        for x, (method, subtitle) in enumerate(views):
            panel(axs[y, x], source, 'behaviours', method, ds)
            axs[y, x].set_title(f'{title} · {counts[source, "behaviours"]["questions"]} questions\n{subtitle}')
        axs[y, 0].set_yticklabels([d['label'].removeprefix('agg_').replace('_', ' ') for d in ds])
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper right', ncol=len(names), frameon=False)
    fig.suptitle(f'All 26 behaviours · {cohort} ' + ('selected questions' if cohort == 'pilot' else 'full cohort') + '\nPointwise 95% intervals; no combined goodness score', fontsize=14)
    fig.supxlabel('Higher means more of the named behaviour, not always better. Flat menus contribute zero information, not missingness.', fontsize=11)
    save(fig, 'behaviours')
    fig, axs = plt.subplots(2, 3, figsize=(19, 12), constrained_layout=True)
    for y, (source, title) in enumerate(SOURCES.items()):
        for x, (method, subtitle) in enumerate(views):
            ax = axs[y, x]
            ax.axhline(0, color='#aaaaaa', lw=.7)
            ax.axvline(0, color='#aaaaaa', lw=.7)
            ax.scatter(0, 0, color=COLORS[reference], marker='+', s=70)
            for m, model in enumerate(names):
                a, b = get(source, 'dnd', method, model, 'order'), get(source, 'dnd', method, model, 'morality')
                ax.hlines(b['mean'], *a['ci95'], color=COLORS[model], lw=1)
                ax.vlines(a['mean'], *b['ci95'], color=COLORS[model], lw=1)
                ax.scatter(a['mean'], b['mean'], color=COLORS[model], label=model, s=35)
            extent = max(abs(z) for r in data['rows'] if r['stratum'] == stratum and r['family'] == 'dnd' and r['method'] == method for z in r['ci95']) * 1.15
            ax.set(xlim=(-extent, extent), ylim=(-extent, extent))
            units = 'win probability minus 0.5' if method == 'pairwise' else 'menu SD units' if method == 'question_sd' else 'label units'
            n = counts[source, 'dnd']
            ax.set_title(f'{title} · {n["questions"]}/{n["total_questions"]} complete-label questions\n{subtitle}', fontsize=11)
            ax.set_xlabel(f'Lawfulness vs {reference} ({units})', fontsize=9)
            ax.set_ylabel(f'Goodness vs {reference} ({units})', fontsize=9)
            ax.set_aspect('equal')
    axs[0, 0].legend(loc='lower left', bbox_to_anchor=(0, 1.07), ncol=len(names), frameon=False)
    fig.suptitle(f'D&D lawfulness and goodness · {cohort}\nCrossbars are separate paired 95% axis intervals, not joint regions; reference = origin', fontsize=14)
    save(fig, 'dnd')
    return metadata


def main(out: Path, preview: Path, cohorts: list[str], replicates: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    blocks = ['<h1>Normalize each question before aggregation</h1>', '<p><a href="../index.html">Original profiles</a> · <a href="../pilot/index.html">Four-model pilot</a></p>',
              '<p>Questions offer actions with different label baselines and ranges. Here every offered action is centered and scaled within its question, before averaging a model’s replies or combining questions. Sources and historical/new-model protocols stay separate.</p>',
              '<p>For each question and trait: normalized action score = (action score − mean score of offered actions) / SD of offered action scores. This happens before averaging the model’s chosen actions.</p>',
              '<p>Left: current paired differences, with the common question mean canceled. Middle: divide each question/label by its offered-action SD. Right: compare each native chosen action with the reference model on that same question, giving ties half credit. For example, +0.1 means a 60% tie-adjusted win probability, not 10% more virtuous. Flat menus contribute zero advantage.</p>',
              '<p>Centering alone removes question base rates from individual-model spread, but cancels in paired model differences. Scaling changes question weights and can change rankings. Very small nonzero label gaps become full rank wins: this view tests sensitivity to score magnitude, not label correctness or intrinsic values.</p>']
    reports, figures, all_summary = [], [], []
    for cohort in cohorts:
        data = analyze(cohort, out, replicates)
        figures += render(data, preview)
        selected = 'differentiated' if cohort == 'pilot' else 'historical_full'
        summary = [r for r in data['summary'] if r['stratum'] == selected]
        all_summary.extend([r | {'cohort': cohort} for r in summary])
        table = [[SOURCES[r['source']], METHODS[r['method']], f'{r["median_absolute_paired_mean_over_se"]:.2f}',
                  'n/a' if r['median_gap_over_question_sd'] is None else f'{r["median_gap_over_question_sd"]:.3f}',
                  f'{r["contrasts_excluding_zero"]}/{r["contrasts"]}', f'{r["median_model_question_sd"]:.3f}'] for r in summary]
        headers = ['Source', 'Per-question method', 'Median paired abs(gap) / SE', 'Median abs(gap) / question SD', '95% contrasts excluding 0', 'Median single-model question SD']
        blocks += [f'<h2>{escape(data["scope"])}</h2>', '<p>Precision compares paired means with their own uncertainty, not raw axis length. Contrasts use all model pairs × 66 directly judged traits. Counts are descriptive, with no multiple-test correction. Raw and centered paired contrasts are exactly identical. Single-model SDs have different units across scaling methods.</p>', tabulate(table, headers=headers, tablefmt='html')]
        for source in SOURCES:
            blocks.append(f'<details open><summary>{SOURCES[source]}: all 66 direct virtues</summary><img src="{cohort}_virtues_{source}.png"></details>')
        blocks += [f'<details open><summary>All behaviours including violence</summary><img src="{cohort}_behaviours.png"></details>', f'<details open><summary>D&D in two dimensions</summary><img src="{cohort}_dnd.png"></details>']
        denominators = [[SOURCES[r['source']], r['stratum'], r['family'], f'{r["questions"]}/{r["total_questions"]}', f'{r["constant_question_labels"]}/{r["question_labels"]}', f'{r["tiny_nonzero_question_labels"]}/{r["question_labels"]}'] for r in data['denominators']]
        blocks.append('<details><summary>Missing, constant and tiny label ranges; all strata</summary><p>Exactly constant question/labels carry zero information and remain in denominators. Tiny = positive menu range smaller than 1% of the full grade scale (a descriptive flag, not a cutoff). Virtue denominator counts include all 80 labels (14 derived); the plots and separation summaries use only 66 direct traits. Missing D&D excludes a question from both axes if any offered action is ungraded.</p>' + tabulate(denominators, headers=['Source', 'Stratum', 'Family', 'Complete questions', 'Constant question/labels', 'Tiny positive ranges'], tablefmt='html') + '</details>')
        blocks.append(f'<p><a href="{cohort}_analysis.json">All numeric results, including range/rank sensitivity and controls</a></p>')
        reports.append(f'## {cohort}\n\n' + tabulate(table, headers=headers, tablefmt='pipe') + '\n\n' + '\n'.join(data['checks']))
        (preview / f'{cohort}_analysis.json').write_bytes((out / f'{cohort}_analysis.json').read_bytes())
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    method_ids = list(METHODS)
    short = ['Raw', 'Centered', 'Question SD', 'Question range', 'Question rank', 'Pairwise wins']
    styles = {'pilot': ('#007a92', 'o'), 'historical': ('#7b3294', 's')}
    for cohort in cohorts:
        for source in SOURCES:
            rows = [next(r for r in all_summary if r['cohort'] == cohort and r['source'] == source and r['method'] == method) for method in method_ids]
            color, marker = styles[cohort]
            style = '-' if source == 'machiavelli' else '--'
            label = cohort + ': ' + SOURCES[source]
            axs[0].plot(range(5), [r['median_gap_over_question_sd'] for r in rows[:5]], color=color, marker=marker, ls=style, label=label)
            axs[1].plot(range(6), [r['median_absolute_paired_mean_over_se'] for r in rows], color=color, marker=marker, ls=style)
    axs[0].set_title('Separation relative to question spread')
    axs[0].set_ylabel('Median |model mean gap| / pooled question SD')
    axs[1].set_title('Separation relative to paired mean uncertainty')
    axs[1].set_ylabel('Median |paired mean gap| / bootstrap SE')
    for ax in axs:
        ax.set_xticks(range(6), short, rotation=25, ha='right')
        ax.grid(axis='y', alpha=.15)
        ax.set_ylim(bottom=0)
    axs[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Does per-question normalization reveal more model separation?\nSame saved replies, all model pairs × 66 direct traits; sources and cohorts separate', fontsize=14)
    fig.supxlabel('Left includes between-question baseline spread; right uses paired comparisons, where that baseline already cancels. Different y-axis quantities.', fontsize=10)
    p = preview / 'normalization_comparison.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)
    figures.append({'path': str(p.resolve()), 'sha256': sha(p)})
    blocks.insert(4, '<h2>What changed?</h2><img src="normalization_comparison.png"><p>These summaries compare all model pairs on the same questions. They are descriptive across traits, not a new accuracy measure or a claim of intrinsic values. On the right, raw and centered are the same by construction. Larger plotted label units alone do not count as improved separation.</p>')
    blocks += ['<h2>Limits and reproducibility</h2><p>Intervals resample game/topic clusters with labels and observed replies fixed. Three exact repeats per order already exist for the four-model pilot; no fresh actor/judge calls. Fixed settings make these repeatability checks, not broad sampling robustness. Historical old prompts and known excluded menus are retained only as historical evidence. No semantic relabeling or capability correction.</p>', '<p>Fractional rank is also computed: rank among all offered actions, including ties, then centered within question. Head-to-head wins are a direct ordinal comparison; fitting Elo would add a transitive one-dimensional strength assumption and would not create additional evidence. Kendall-style repeat rank concordance is saved for the pilot, with ties zero rather than silently counted as agreement.</p>', '<p>— PI/OpenAI</p>']
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Per-question normalized choice profiles</title><style>body{font:16px/1.5 system-ui;max-width:1700px;margin:30px auto;padding:0 24px;color:#222}img{width:100%;height:auto}table{border-collapse:collapse;font-size:14px}td,th{padding:7px 12px;border-bottom:1px solid #ddd;text-align:left}details{padding:18px 0;border-top:1px solid #ddd}summary{cursor:pointer}p{max-width:1100px}</style><body>' + '\n'.join(blocks) + '</body></html>'
    (preview / 'index.html').write_text(page)
    (preview / 'figure_metadata.json').write_text(json.dumps({'author': 'PI/OpenAI', 'figures': figures, 'analysis_paths': [str((out / f'{c}_analysis.json').resolve()) for c in cohorts]}, indent=2) + '\n')
    report = '# Within-question normalization\n\nNo new model calls. All transformations precede averaging.\n\n' + '\n\n'.join(reports) + '\n\n-- PI/OpenAI\n'
    (out / 'run.md').write_text(report)
    print(report)
    print(preview.resolve() / 'index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/20261026_question_normalization')
    parser.add_argument('--preview', type=Path, default=ROOT / '.local/choice_preview/question_normalization')
    parser.add_argument('--cohorts', nargs='+', choices=['pilot', 'historical'], default=['pilot', 'historical'])
    parser.add_argument('--replicates', type=int, default=2000)
    args = parser.parse_args()
    main(args.out, args.preview, args.cohorts, args.replicates)
