"""Classic D&D alignment chart, calibrated against an even split. -- PI/OpenAI

uv run --offline --with matplotlib --with scipy scripts/evaluation/plot_dnd_classic.py
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import PathPatch, Polygon, Rectangle
from matplotlib.path import Path as PlotPath
from matplotlib.ticker import FuncFormatter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate_probability_eval import REFERENCE, quantile_knots
from plot_probability_eval import COLORS, SOURCES, calibrated_logp, load_calibration, logo, nats_core, short

ROOT = Path(__file__).resolve().parents[2]
COMBOS = ((0, 0), (0, 1), (1, 1))
SEED = 20261027
METHOD = 'pair-gap-weighted calibrated log-odds; common page cohort; frozen observed DeepSeek anchor'
MODES = ('cluster_refit', 'full_refit', 'fixed_maps', 'clusters_only', 'draws_only', 'cluster_map_only', 'refit_only')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pack(questions, model, label_fn, folds):
    n = len(questions)
    k = np.array([len(q['options']) for q in questions])
    width = int(k.max())
    dimensions = label_fn(questions[0]).shape[1]
    coeff = np.zeros((n, width, dimensions))
    den = np.zeros((n, dimensions))
    logs = {m: np.zeros((n, 3, width)) for m in (model, REFERENCE)}
    for z, q in enumerate(questions):
        options = list(q['options'])
        X = label_fn(q)
        ii, jj = np.triu_indices(len(options), 1)
        dx = X[ii] - X[jj]
        dx = np.where(np.isfinite(dx), dx, 0.)
        np.add.at(coeff[z], ii, dx)
        np.add.at(coeff[z], jj, -dx)
        den[z] = np.abs(dx).sum(0)
        for m in logs:
            draws = sorted(q['models'][m]['draws'], key=lambda d: d['draw'])
            assert [d['draw'] for d in draws] == [0, 1]
            counts = np.array([[d['probabilities'][o] for o in options] for d in draws])
            assert np.all(counts.sum(1) == 100) and np.all(counts == counts.astype(int))
            for c, (a, b) in enumerate(COMBOS):
                logs[m][z, c, :k[z]] = np.log((counts[a] + 1) * (counts[b] + 1)) / 2 - np.log(100 + k[z])
    assert np.allclose(coeff.sum(1), 0, atol=1e-14)
    groups = np.array([q['eval_cluster'] for q in questions])
    members = [np.flatnonzero(groups == g) for g in sorted(set(groups))]
    return {'k': k, 'coeff': coeff, 'den': den, 'logs': logs, 'members': members,
            'fold': np.array([folds[q['source_id']] for q in questions])}


def transform(values, k, folds, maps):
    out = np.zeros_like(values)
    for f in (0, 1):
        mp = maps[str(f)]
        for size in np.unique(k[folds == f]):
            ix = np.flatnonzero((folds == f) & (k == size))
            v = values[ix, :size]
            odds = v[:, :, None] - v[:, None, :]
            mag = np.abs(odds)
            mapped = np.interp(mag, mp['x'], mp['y'])
            outside = mag > mp['x'][-1]
            mapped[outside] = mp['y'][-1] + (mag[outside] - mp['x'][-1]) * mp['end_slope']
            out[ix, :size] = (np.sign(odds) * mapped).mean(2)
    return out


def refit(lm, lb, k, folds):
    valid = np.arange(lm.shape[1])[None, :] < k[:, None]
    q, i, j = np.nonzero(valid[:, :, None] & valid[:, None, :] & np.triu(np.ones((lm.shape[1], lm.shape[1]), bool), 1)[None])
    x, y = abs(lm[q, i] - lm[q, j]), abs(lb[q, i] - lb[q, j])
    weights = 1 / (k[q] * (k[q] - 1) / 2)
    maps = {}
    for f in (0, 1):
        train = folds[q] != f
        assert train.any() and np.any(folds == f)
        maps[str(f)] = quantile_knots(x[train], y[train], weights[train])
    return maps


def estimate(coeff, den, delta):
    return np.einsum('nkd,nk->d', coeff, delta) / den.sum(0)


def marginal_maps(lm, lb, k, folds, weights):
    valid = np.arange(lm.shape[2])[None, :] < k[:, None]
    q, i, j = np.nonzero(valid[:, :, None] & valid[:, None, :] & np.triu(np.ones((lm.shape[2], lm.shape[2]), bool), 1)[None])
    x = abs(lm[q, :, i] - lm[q, :, j]).T
    y = abs(lb[q, :, i] - lb[q, :, j]).T
    pair_weights = np.array(weights)[:, None] / (k[q] * (k[q] - 1) / 2)[None, :]
    maps = {}
    for f in (0, 1):
        train = folds[q] != f
        w = pair_weights[:, train].ravel()
        use = w > 0
        maps[str(f)] = quantile_knots(x[:, train].ravel()[use], y[:, train].ravel()[use], w[use])
    return maps


def target_sensitivity(data, model):
    lm, lb = data['logs'][model], data['logs'][REFERENCE]
    k, f, c, d = (data[x] for x in ('k', 'fold', 'coeff', 'den'))
    separate = [estimate(c, d, transform(lm[:, a], k, f, refit(lm[:, a], lb[:, a], k, f)) - lb[:, a]) for a in (0, 2)]
    points = {'single_draw_separately_fitted': np.mean(separate, axis=0).tolist(), 'single_draw_separate_points': np.array(separate).tolist()}
    for name, weights in (('single_draw_marginal', [.5, 0., .5]), ('bootstrap_mixture', [.25, .5, .25])):
        maps = marginal_maps(lm, lb, k, f, weights)
        delta = sum(w * (transform(lm[:, a], k, f, maps) - lb[:, a]) for a, w in enumerate(weights))
        points[name] = estimate(c, d, delta).tolist()
    return points


def bootstrap_model(data, model, maps, replicates):
    rng = np.random.default_rng(SEED)
    lm, lb = data['logs'][model], data['logs'][REFERENCE]
    k, f, c, d = (data[x] for x in ('k', 'fold', 'coeff', 'den'))
    fixed = np.stack([transform(lm[:, j], k, f, maps) for j in range(3)], 1)
    samples = {mode: np.empty((replicates, c.shape[2])) for mode in MODES}
    for r in range(replicates):
        ix = np.concatenate([data['members'][j] for j in rng.integers(0, len(data['members']), len(data['members']))])
        a, b = rng.choice(3, (2, len(ix)), p=[.25, .5, .25])
        am, br = lm[ix, a], lb[ix, b]
        fitted = refit(am, br, k[ix], f[ix])
        cal = transform(am, k[ix], f[ix], fitted)
        samples['full_refit'][r] = estimate(c[ix], d[ix], cal - br)
        samples['fixed_maps'][r] = estimate(c[ix], d[ix], fixed[ix, a] - br)
        samples['clusters_only'][r] = estimate(c[ix], d[ix], fixed[ix, 1] - lb[ix, 1])
        cluster_fitted = refit(lm[ix, 1], lb[ix, 1], k[ix], f[ix])
        samples['cluster_refit'][r] = estimate(c[ix], d[ix], transform(lm[ix, 1], k[ix], f[ix], cluster_fitted) - lb[ix, 1])
        samples['cluster_map_only'][r] = estimate(c, d, transform(lm[:, 1], k, f, cluster_fitted) - lb[:, 1])
        aa, bb = rng.choice(3, (2, len(k)), p=[.25, .5, .25])
        allq = np.arange(len(k))
        samples['draws_only'][r] = estimate(c, d, fixed[allq, aa] - lb[allq, bb])
        samples['refit_only'][r] = estimate(c, d, transform(lm[:, 1], k, f, fitted) - lb[:, 1])
    return samples


def compute(tables, replicates):
    start = time.monotonic()
    lines, maps, _, folds = load_calibration(tables)
    dimensions = lines[0]['_meta']['dimensions']
    indices = [next(i for i, d in enumerate(dimensions) if d['id'] == key) for key in ('order', 'morality')]
    models = [m for m in lines[1]['models'] if m != REFERENCE]
    common = [q for q in lines[1:] if all(m in q['models'] and len(q['models'][m]['draws']) == 2 for m in models + [REFERENCE])]
    output = {'author': 'PI/OpenAI', 'method': METHOD, 'bootstrap_replicates': replicates, 'seed': SEED,
              'axes': ['lawfulness', 'goodness'], 'units': 'nats, positive = lawful / good; display x = minus lawfulness',
              'display_interval': 'cluster_refit',
              'warnings': {'full_refit': 'Diagnostic only: empirical resampling of just two draws systematically changes the calibration magnitude distribution. Percentiles are not validated confidence limits.', 'target_sensitivity': 'Signs can depend on whether calibration targets pooled or single-draw magnitudes; see pc1_refit.json and slop/20261028_dnd_refit_drift.json.'},
              'uncertainty': 'Displayed marginal bootstrap intervals resample clusters and refit opposite-fold maps, conditional on the two observed draws. Fixed-map cluster/draw intervals and biased nested-refit quantiles are separately recorded. Evaluation uses the common page cohort. Fitting retains each original map cohort (500/source except Haiku499games); omitted evaluation cases have zero scoring weight but remain eligible for map fitting. Offsets conditional on the observed absolute DeepSeek anchor, frozen labels and two draws. The flagged nested diagnostic resamples games/topics then model and DeepSeek draws and refits opposite-fold maps. Component intervals are separate diagnostics, not additive. Refit-only holds evaluation questions/draws fixed while refitting maps on resampled questions/draws.',
              'geometry': {'extent_nats': 3., 'neutral_octagon_circumradius_nats': .44, 'status': 'Illustrative boundaries, not empirically validated D&D cutoffs'}, 'sources': {}}
    for src in SOURCES:
        qs = [q for q in common if q['source'] == src]
        assert len(qs) == (500 if src == 'daily_dilemmas' else 499)
        result = {'questions': len(qs), 'question_ids': [q['source_id'] for q in qs],
                  'excluded_ids': [q['source_id'] for q in lines[1:] if q['source'] == src and q not in qs], 'models': {}}
        label_fn = lambda q: np.array([[q['option_scores'][o][i] for i in indices] for o in q['options']], float)
        point_page, _, _, n_page, info_page = nats_core(lines, src, label_fn, maps, folds, resample='clusters')
        assert n_page == len(qs)
        for j, m in enumerate(models):
            fit_qs = [q for q in lines[1:] if q['source'] == src and m in q['models'] and len(q['models'][m]['draws']) == 2]
            data = pack(fit_qs, m, label_fn, folds)
            evaluation_ids = {q['source_id'] for q in qs}
            evaluate = np.array([q['source_id'] in evaluation_ids for q in fit_qs])
            data['coeff'][~evaluate] = 0
            data['den'][~evaluate] = 0
            c, d, lm, lb = data['coeff'], data['den'], data['logs'][m], data['logs'][REFERENCE]
            cal = transform(lm[:, 1], data['k'], data['fold'], maps[src][m])
            point = estimate(c, d, cal - lb[:, 1])
            anchor = estimate(c, d, lb[:, 1])
            absolute = estimate(c, d, cal)
            assert np.max(abs(point - point_page[j])) < 1e-12
            assert np.max(abs(absolute - (anchor + point))) < 1e-12
            assert np.array_equal((d > 0).sum(0), info_page)
            assert np.max(abs(estimate(c, d, np.zeros_like(cal)))) == 0
            recovered = refit(lm[:, 1], lb[:, 1], data['k'], data['fold'])
            for f in ('0', '1'):
                for key in ('x', 'y', 'end_slope'):
                    assert recovered[f][key] == maps[src][m][f][key]
            samples = bootstrap_model(data, m, maps[src][m], replicates)
            stats = {mode: {'ci95': np.quantile(v, [.025, .975], axis=0).T.tolist(),
                            'width95': np.ptp(np.quantile(v, [.025, .975], axis=0), axis=0).tolist(),
                            'variance': np.var(v, axis=0, ddof=1).tolist(), 'mean': v.mean(0).tolist(),
                            'mean_minus_point': (v.mean(0) - point).tolist()} for mode, v in samples.items()}
            result['models'][short(m)] = {'model': m, 'absolute': absolute.tolist(), 'uncalibrated_absolute': estimate(c, d, lm[:, 1]).tolist(),
                                          'offset': point.tolist(), 'calibration_questions': len(fit_qs), 'target_sensitivity': target_sensitivity(data, m), 'uncertainty': stats}
            if j == 0:
                result['models']['DeepSeek'] = {'model': REFERENCE, 'absolute': anchor.tolist(), 'offset': [0., 0.]}
                result['informative_questions'] = info_page.tolist()
                result['pair_label_gap_weight'] = d.sum(0).tolist()
            print(f'{src} {short(m)}: absolute={absolute.round(4)}, offset={point.round(4)}', flush=True)
        widths = {name: row['uncertainty']['cluster_refit']['width95'] for name, row in result['models'].items() if name != 'DeepSeek'}
        result['representative_model'] = sorted(widths, key=lambda name: widths[name][0])[len(widths) // 2]
        result['width95_range'] = np.array([np.min(list(widths.values()), axis=0), np.max(list(widths.values()), axis=0)]).T.tolist()
        result['verification'] = {'matches_page_nats_core': True, 'same_page_cohort': True, 'absolute_equals_anchor_plus_offset': True,
                                  'all_observed_fold_maps_reproduced': True, 'uniform_chooser_zero': True, 'display_lawful_left_good_up': True}
        output['sources'][src] = result
    output['runtime_seconds'] = time.monotonic() - start
    return output


def background(ax, variant):
    x, y = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-1, 1, 400))
    paper = np.array(to_rgb('#faf7ed'))
    green, red = np.array(to_rgb('#51866d')), np.array(to_rgb('#b76665'))
    field = np.broadcast_to(paper, (*x.shape, 3)).copy()
    strength = .21 if variant == 'pastel' else .11
    field += strength * abs(y)[..., None] * (np.where((y > 0)[..., None], green, red) - paper)
    blue, pink = np.array(to_rgb('#7794a9')), np.array(to_rgb('#c37c9e'))
    if variant == 'pastel':
        field += .10 * abs(x)[..., None] * (np.where((x < 0)[..., None], blue, pink) - paper)
    else:
        edge = np.exp(-abs(y) * 12)
        field += .24 * (edge * abs(x))[..., None] * (np.where((x < 0)[..., None], blue, pink) - paper)
    ax.imshow(np.clip(field, 0, 1), extent=(-3, 3, -3, 3), origin='lower', zorder=-5)
    ax.plot([0, 0], [0, 3], color='#6b996a', alpha=.24, lw=10, zorder=-3)
    ax.plot([0, 0], [-3, 0], color='#c77f7e', alpha=.24, lw=10, zorder=-3)
    ax.plot([-3, 0], [0, 0], color='#7a98af', alpha=.24, lw=10, zorder=-3)
    ax.plot([0, 3], [0, 0], color='#ca8fab', alpha=.24, lw=10, zorder=-3)


def classic(ax, variant):
    background(ax, variant)
    vertices = [(-1., 3.), (-.52, 2.25), (-.32, 1.05), (-.28, .78),
                (-.20, .44), (-.44, .20), (-.78, .28), (-1.05, .32), (-2.25, .52), (-3., 1.)]
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        path = PlotPath(np.array(vertices) * [sx, sy], [PlotPath.MOVETO] + [PlotPath.CURVE4] * 9)
        ax.add_patch(PathPatch(path, fill=False, edgecolor='#4b4a43', lw=1.1, zorder=-1))
        ax.plot([sx * .44 / np.sqrt(2), sx * .4275], [sy * .44 / np.sqrt(2), sy * .4275], color='#4b4a43', lw=1.)
    theta = np.pi / 8 + np.arange(8) * np.pi / 4
    ax.add_patch(Polygon(np.c_[np.cos(theta), np.sin(theta)] * .44, facecolor='#faf7ed', edgecolor='#4b4a43', lw=1.2))
    ax.text(0, 0, 'NEUTRAL', ha='center', va='center', fontsize=10, fontfamily='DejaVu Sans', fontstretch='condensed')
    for x, y, text in ((-1.8, 1.22, 'LAWFUL GOOD'), (1.8, 1.22, 'CHAOTIC GOOD'),
                       (-1.8, -1.65, 'LAWFUL EVIL'), (1.8, -1.65, 'CHAOTIC EVIL'),
                       (-1.95, 0, 'LAWFUL NEUTRAL'), (1.95, 0, 'CHAOTIC NEUTRAL'),
                       (0, 1.8, 'NEUTRAL\nGOOD'), (0, -1.8, 'NEUTRAL\nEVIL')):
        ax.text(x, y, text, ha='center', va='center', fontsize=11 if x else 9, color='#55564e', fontfamily='DejaVu Sans', fontstretch='condensed')
    for x, y, word in ((-2.86, 2.85, 'Saintly'), (2.86, 2.85, 'Beatific'), (-2.86, -2.85, 'Diabolic'), (2.86, -2.85, 'Demoniac')):
        ax.text(x, y, word, ha='left' if x < 0 else 'right', va='top' if y > 0 else 'bottom', fontsize=12, fontfamily='DejaVu Serif', style='italic', color='#4b4a43')
    ax.set(xlim=(-3, 3), ylim=(-3, 3), aspect='equal', xticks=np.arange(-3, 4), yticks=np.arange(-3, 4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{-x:g}'))
    ax.set_xlabel('← Lawful  |  Chaotic →     (nats vs an even split)', fontsize=10)
    ax.set_ylabel('← Evil  |  Good →     (nats vs an even split)', fontsize=10)
    ax.tick_params(labelsize=9, colors='#66665e', length=3)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#4b4a43')
        spine.set_linewidth(1.3)


def model_markers(ax, rows):
    actual = [(-r['offset'][0], r['offset'][1]) for _, r in rows]
    for (name, _), point in zip(rows, actual):
        logo(ax, *point, 'DeepSeek repeat' if name == 'DeepSeek' else name, .065)
    return actual


def label_points(ax, rows, points):
    # Candidate labels avoid all markers and earlier labels in screen coordinates. -- PI/OpenAI
    ax.figure.canvas.draw()
    px = ax.transData.transform(points)
    boxes = []
    for i, (name, row) in enumerate(rows):
        candidates = [(dx, dy) for dy in (17, -23, 32, -38) for dx in (15, -15, 0, 35, -35)]
        best = None
        for dx, dy in candidates:
            anchor = px[i] + np.array([dx, dy]) * ax.figure.dpi / 72
            width = len(name) * 6.0 * ax.figure.dpi / 72
            x0 = anchor[0] if dx > 0 else anchor[0] - width if dx < 0 else anchor[0] - width / 2
            box = np.array([x0 - 3, anchor[1] - 6, x0 + width + 3, anchor[1] + 13])
            penalty = sum(2000 for b in boxes if not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]))
            penalty += sum(5000 for j, p in enumerate(px) if j != i and box[0] - 10 < p[0] < box[2] + 10 and box[1] - 10 < p[1] < box[3] + 10)
            penalty += 5000 * (box[0] < ax.bbox.x0 + 3 or box[2] > ax.bbox.x1 - 3 or box[1] < ax.bbox.y0 + 3 or box[3] > ax.bbox.y1 - 3)
            score = penalty + np.hypot(dx, dy)
            if best is None or score < best[0]:
                best = score, dx, dy, box
        _, dx, dy, box = best
        boxes.append(box)
        color = COLORS['DeepSeek repeat'] if name == 'DeepSeek' else COLORS[name]
        ax.annotate(name, points[i], xytext=(dx, dy), textcoords='offset points', color=color,
                    fontsize=10, ha='left' if dx > 0 else 'right' if dx < 0 else 'center', va='center',
                    arrowprops={'arrowstyle': '-', 'lw': .55, 'color': color, 'shrinkA': 2, 'shrinkB': 9}, zorder=7)


def bracket(ax, x, y, width, label, color):
    ax.plot([x - width / 2, x + width / 2], [y, y], color=color, lw=1.5)
    ax.plot([x - width / 2] * 2, [y - .025, y + .025], color=color, lw=1.1)
    ax.plot([x + width / 2] * 2, [y - .025, y + .025], color=color, lw=1.1)
    ax.text(x, y - .055, label, ha='center', va='top', fontsize=8, color=color)


def variance_key(source):
    name = source['representative_model']
    u = source['models'][name]['uncertainty']
    components = np.array([u[m]['variance'] for m in ('clusters_only', 'draws_only', 'cluster_map_only')])
    return {'representative_model': name, 'components': ['games_topics', 'draws_fixed_maps', 'cluster_map_refit_fixed_evaluation'],
            'normalized_diagnostic_variance_weights': (components / components.sum(0)).tolist(),
            'sum_variances_over_full_variance': (components.sum(0) / np.array(u['cluster_refit']['variance'])).tolist(),
            'interpretation': 'Normalized one-factor diagnostic weights, not identified independent contributions. The cross conditions on observed draws; its draw-resampling piece is a diagnostic weight only.'}


def underbrace(ax, left, width, y, color):
    x = np.array([0, .10, .03, .25, .44, .46, .5, .54, .56, .75, .97, .90, 1.]) * width + left
    yy = y + np.array([0, 0, -.035, -.035, -.035, -.035, -.060, -.035, -.035, -.035, -.035, 0, 0])
    ax.add_patch(PathPatch(PlotPath(np.c_[x, yy], [PlotPath.MOVETO] + [PlotPath.CURVE4] * 12), fill=False, ec=color, lw=1.))


def uncertainty_panel(ax, source, scale, source_key):
    name = source['representative_model']
    full = np.array(source['models'][name]['uncertainty']['cluster_refit']['width95'])
    weights = np.array(variance_key(source)['normalized_diagnostic_variance_weights'])[:, 0]
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.text(.5, 1., f'Uncertainty scale · {name}', ha='center', va='top', fontsize=10)
    ax.text(.5, .89, 'Nominal 95%, observed draws fixed', ha='center', fontsize=7.5, color='#666666')
    sx = scale
    sy = scale * ax.bbox.width / ax.bbox.height
    ax.errorbar(.5, .66, xerr=full[0] * sx / 2, yerr=full[1] * sy / 2, fmt='none', color='#373d40', capsize=4, lw=1.4)
    length = full[0] * sx
    left = .5 - length / 2
    for edge in (left, left + length):
        ax.plot([edge, edge], [.29, .66], color='#c7c7c2', lw=.55, ls=':', zorder=-1)
    ax.text(.5, .43, f'Lawful / good: {full[0]:.2f} / {full[1]:.2f} nats', ha='center', fontsize=8)
    ax.text(.5, .355, 'Lawful-axis variance weights', ha='center', fontsize=8, color='#555555')
    colors = ('#436b8b', '#9c6f32', '#886192')
    for j, (weight, color) in enumerate(zip(weights, colors)):
        width = length * weight
        ax.add_patch(Rectangle((left, .275), width, .035, fc=color, ec='none'))
        underbrace(ax, left, width, .25, color)
        ax.text(left + width / 2, .145 if j != 1 else .065, f'{weight:.0%}', ha='center', va='top', fontsize=8, color=color)
        left += width
    assert abs(left - (.5 + length / 2)) < 1e-12
    for x, label, color in zip((.14, .5, .86), ('Topics' if source_key == 'daily_dilemmas' else 'Games', 'Draws', 'Refit'), colors):
        ax.text(x, -.03, label, color=color, fontsize=8, ha='center')


def render(numbers, out, variant):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'savefig.facecolor': '#fffef9'})
    fig = plt.figure(figsize=(16, 11.5), facecolor='#fffef9')
    grid = fig.add_gridspec(2, 4, width_ratios=(1.7, 1., 1.7, 1.), height_ratios=(1.6, .85),
                          left=.055, right=.965, top=.91, bottom=.075, wspace=.38, hspace=.32)
    fig.suptitle('CHARACTER ALIGNMENT GRAPH', y=.977, fontsize=21, fontfamily='DejaVu Serif')
    scale = .74 / max(s['models'][s['representative_model']]['uncertainty']['cluster_refit']['width95'][0] for s in numbers['sources'].values())
    offsets = np.array([r['offset'] for s in numbers['sources'].values() for r in s['models'].values()])
    zoom_lim = max(.22, np.max(abs(offsets)) * 1.25)
    for col, (src, title) in zip((0, 2), SOURCES.items()):
        source = numbers['sources'][src]
        ax = fig.add_subplot(grid[0, col:col + 2])
        classic(ax, variant)
        counts = source['informative_questions']
        ax.set_title(f'{title}   ·   {source["questions"]} questions', fontsize=13, loc='left', pad=13)
        rows = sorted(source['models'].items(), key=lambda item: list(COLORS).index('DeepSeek repeat' if item[0] == 'DeepSeek' else item[0]))
        for name, row in rows:
            x, y = row['absolute']
            color = COLORS['DeepSeek repeat'] if name == 'DeepSeek' else COLORS[name]
            ax.plot(-x, y, 'o', ms=5.5, mfc=color, mec='white', mew=1.1, zorder=4)
        values = np.array([r['absolute'] for _, r in rows])
        lower, upper = values.min(0) - .11, values.max(0) + .11
        ax.add_patch(Rectangle((-upper[0], lower[1]), upper[0] - lower[0], upper[1] - lower[1], fill=False, ec='#76776f', lw=.8, ls=':', zorder=2))
        zoom = fig.add_subplot(grid[1, col])
        zoom.set(xlim=(-zoom_lim, zoom_lim), ylim=(-zoom_lim, zoom_lim))
        zoom.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{-x:+.1f}' if x else '0'))
        zoom.axhline(0, color='#aaa79f', lw=.8)
        zoom.axvline(0, color='#aaa79f', lw=.8)
        zoom.set_facecolor('#fffefa')
        avatar_positions = model_markers(zoom, rows)
        label_points(zoom, rows, avatar_positions)
        zoom.set_title('Zoom: calibrated offsets from DeepSeek', loc='left', fontsize=10, pad=10)
        zoom.set_xlabel('← More lawful | More chaotic →  (nats)', fontsize=9)
        zoom.set_ylabel('More good ↑  (nats)', fontsize=9)
        zoom.tick_params(labelsize=8)
        for side in ('top', 'right'):
            zoom.spines[side].set_visible(False)
        uncertainty_panel(fig.add_subplot(grid[1, col + 1]), source, scale, src)
    fig.text(.055, .018, 'github.com/wassname/virtue-choices', fontsize=8, color='#888888')
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    assert tight.x0 >= 0 and tight.y0 >= 0 and tight.x1 <= width and tight.y1 <= height, ('Figure text clipped', tight.bounds, width, height)
    path = out / f'dnd_classic_{variant}.png'
    fig.savefig(path, dpi=160, metadata={'Author': 'PI/OpenAI'})
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path, default=ROOT / 'outputs/20261027_random1000_v12/question_tables.jsonl')
    parser.add_argument('--out', type=Path, default=ROOT / '.local/choice_preview/random1000')
    parser.add_argument('--bootstrap', type=int, default=999)
    parser.add_argument('--reuse-numbers', action='store_true')
    args = parser.parse_args()
    args.tables = args.tables.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'dnd_classic.json'
    files = [args.tables, args.tables.parent / 'calibration_heldout/maps.json', args.tables.parent / 'calibration_heldout/folds.json', ROOT / 'scripts/evaluation/calibrate_probability_eval.py']
    signature = {str(p.relative_to(ROOT)): digest(p) for p in files}
    method_sha = hashlib.sha256(''.join(inspect.getsource(f) for f in (pack, transform, refit, estimate, bootstrap_model, compute, marginal_maps, target_sensitivity, quantile_knots, load_calibration, nats_core, calibrated_logp)).encode()).hexdigest()
    if args.reuse_numbers:
        numbers = json.loads(path.read_text())
        assert numbers['input_sha256'] == signature and numbers['method_sha256'] == method_sha and numbers['bootstrap_replicates'] == args.bootstrap
    else:
        numbers = compute(args.tables, args.bootstrap)
        numbers.update(input_sha256=signature, method_sha256=method_sha)
        path.write_text(json.dumps(numbers, indent=2, allow_nan=False) + '\n')
    for variant in ('pastel', 'axis_tints'):
        print(render(numbers, args.out, variant), flush=True)
    selected = args.out / 'dnd_classic_axis_tints.png'
    (args.out / 'dnd_classic.png').write_bytes(selected.read_bytes())
    print(path, flush=True)


if __name__ == '__main__':
    main()
