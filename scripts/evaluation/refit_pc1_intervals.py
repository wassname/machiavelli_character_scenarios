"""PC1 log-odds: conditional uncertainty and calibration-target sensitivity. -- PI/OpenAI

uv run --offline --with matplotlib --with scipy scripts/evaluation/refit_pc1_intervals.py
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import plot_dnd_classic as core
from plot_probability_eval import nats_core, virtue_basis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path, default=core.ROOT / 'outputs/20261027_random1000_v12/question_tables.jsonl')
    parser.add_argument('--bootstrap', type=int, default=999)
    args = parser.parse_args()
    args.tables = args.tables.resolve()
    out = args.tables.parent / 'calibration_heldout'
    groups_path = core.ROOT / 'rubrics/virtue_display_groups.json'
    lines, maps, _, folds = core.load_calibration(args.tables)
    basis = virtue_basis(lines, json.loads(groups_path.read_text())['groups'])
    models = [m for m in lines[1]['models'] if m != core.REFERENCE]
    common_ids = {q['source_id'] for q in lines[1:] if all(m in q['models'] and len(q['models'][m]['draws']) == 2 for m in models + [core.REFERENCE])}
    start = time.monotonic()
    report = {'author': 'PI/OpenAI', 'unit': 'pair-label-gap-weighted nats vs DeepSeek',
              'seed': core.SEED, 'bootstrap_replicates': args.bootstrap,
              'primary': 'Fixed-map and cluster-refit percentile intervals conditional on the observed two-draw records, fixed fold assignment, fixed PC1 basis and judge labels. Nominal95%, under an exchangeable-game/topic-cluster interpretation; coverage not validated.',
              'intervals': {'ci95_maps_fixed': 'Page-style clusters + empirical draws, maps frozen; potential response-variance double-counting and two-draw assumptions apply.',
                            'ci95_maps_fixed_clusters_only': 'Clusters only, maps frozen, observed draws retained.',
                            'ci95_refit': 'Clusters only, opposite-fold maps refitted, observed draws retained.',
                            'nested_refit_diagnostic': 'Clusters + empirical two-draw resampling + map refit. Central95% range only; systematic target drift means this is not a validated CI.'},
              'targets': {'point': 'Map fitted to each question\'s two-draw geometric pool and applied to that pool.',
                          'single_draw_point': 'Fit/apply separate opposite-fold maps for draw0vsreference0 and draw1vsreference1, then average the two scores.',
                          'single_draw_marginal_point': 'Fit one opposite-fold map to the marginal distribution of both individual draws, each weighted1/2 per question; apply to individual draws and average.',
                          'bootstrap_mixture_point': 'Fit to the1/4 individual0 +1/2 pooled +1/4 individual1 magnitude mixture and apply that same mixture; this is not a pure single-draw target.'},
              'warnings': ['Calibration-target sensitivity is substantive: negative pooled-target offsets need not remain negative under a single-draw target.',
                           'Same-sign point estimates across targets do not establish confidence-backed robustness.',
                           'Only17everyday topic clusters versus92game clusters; cluster interpretation and discrete-quantile regularity remain assumptions.',
                           'Draws rotate option order, so draw resampling also changes the original balanced-order design.',
                           'No interval adjustments for model comparisons or judge-label/basis uncertainty. No fresh draws or paid calls.'],
              'rows': [], 'basis': {}, 'input_sha256': {str(p.relative_to(core.ROOT)): core.digest(p) for p in (args.tables, out / 'maps.json', out / 'folds.json', groups_path)},
              'code_sha256': {str(p.relative_to(core.ROOT)): core.digest(p) for p in (Path(__file__).resolve(), Path(core.__file__), core.ROOT / 'scripts/evaluation/plot_probability_eval.py', core.ROOT / 'scripts/evaluation/calibrate_probability_eval.py')}}
    bank = {}
    for src in core.SOURCES:
        vector = basis['src'][src]['V'][0]
        label_fn = lambda q: (np.array([[q['option_scores'][o][i] for i in basis['direct']] for o in q['options']], float) * basis['fav']) @ vector[:, None]
        page_point, page_lo, page_hi, nq, info = nats_core(lines, src, label_fn, maps, folds)
        report['basis'][src] = {'virtue_ids': basis['ids'], 'favorable_space_weights': vector.tolist(), 'variance_share': float(basis['src'][src]['share'][0])}
        for j, model in enumerate(models):
            qs = [q for q in lines[1:] if q['source'] == src and model in q['models'] and len(q['models'][model]['draws']) == 2]
            data = core.pack(qs, model, label_fn, folds)
            evaluate = np.array([q['source_id'] in common_ids for q in qs])
            data['coeff'][~evaluate] = 0
            data['den'][~evaluate] = 0
            lm, lb = data['logs'][model], data['logs'][core.REFERENCE]
            fitted = core.transform(lm[:, 1], data['k'], data['fold'], maps[src][model])
            point = float(core.estimate(data['coeff'], data['den'], fitted - lb[:, 1])[0])
            assert abs(point - page_point[j, 0]) < 1e-12 and int((data['den'][:, 0] > 0).sum()) == info[0]
            recovered = core.refit(lm[:, 1], lb[:, 1], data['k'], data['fold'])
            for f in ('0', '1'):
                for key in ('x', 'y', 'end_slope'):
                    assert recovered[f][key] == maps[src][model][f][key]
            samples = core.bootstrap_model(data, model, maps[src][model], args.bootstrap)
            target = core.target_sensitivity(data, model)
            ci = lambda mode: np.quantile(samples[mode][:, 0], [.025, .975]).tolist()
            row = {'source': src, 'model': model, 'name': core.short(model), 'questions': nq, 'informative_questions': int(info[0]),
                   'calibration_questions': len(qs), 'clusters': len(data['members']), 'point': point,
                   'ci95_maps_fixed': ci('fixed_maps'), 'ci95_maps_fixed_clusters_only': ci('clusters_only'), 'ci95_refit': ci('cluster_refit'),
                   'page_ci95_maps_fixed_1000': [float(page_lo[j, 0]), float(page_hi[j, 0])],
                   'single_draw_point': target['single_draw_separately_fitted'][0],
                   'single_draw_separate_points': [v[0] for v in target['single_draw_separate_points']],
                   'single_draw_marginal_point': target['single_draw_marginal'][0], 'bootstrap_mixture_point': target['bootstrap_mixture'][0],
                   'nested_refit_diagnostic': {'quantile95': ci('full_refit'), 'mean': float(samples['full_refit'].mean()), 'mean_minus_point': float(samples['full_refit'].mean() - point), 'status': 'diagnostic_only; coverage_unvalidated'},
                   'bootstrap_mean_minus_point': {mode: float(v.mean() - point) for mode, v in samples.items()},
                   'verification': {'point_matches_page_error': abs(point - page_point[j, 0]), 'observed_maps_reproduce': True, 'fit_test_disjoint': True}}
            values = [point, row['single_draw_point'], row['single_draw_marginal_point']]
            row['same_point_sign_across_pooled_and_single_targets'] = len(set(np.sign(values))) == 1
            report['rows'].append(row)
            for mode, values in samples.items():
                bank[f'{src}__{core.short(model)}__{mode}'] = values
            print(src, core.short(model), 'point', round(point, 4), 'fixed', np.round(row['ci95_maps_fixed'], 4), 'cluster-refit', np.round(row['ci95_refit'], 4), 'single', round(row['single_draw_point'], 4), 'marginal', round(row['single_draw_marginal_point'], 4), flush=True)
    archive = out / 'pc1_refit_bootstrap.npz'
    np.savez_compressed(archive, **bank)
    report.update(runtime_seconds=time.monotonic() - start, bootstrap_bank={'path': str(archive.relative_to(core.ROOT)), 'sha256': core.digest(archive)})
    target = out / 'pc1_refit.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(target, flush=True)


if __name__ == '__main__':
    main()
