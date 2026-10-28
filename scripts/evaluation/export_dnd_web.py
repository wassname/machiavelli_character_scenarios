"""Export aggregate chart data without private question IDs or traces. -- PI/OpenAI"""

import argparse
import json
from pathlib import Path

import numpy as np

from plot_dnd_classic import ROOT, digest, variance_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / '.local/choice_preview/random1000/dnd_classic.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/20261027_random1000_v12/calibration_heldout/dnd_web.json')
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    assert data['display_interval'] == 'cluster_refit' and data['bootstrap_replicates'] == 999
    web = {'author': 'PI/OpenAI', 'axes': ['lawful', 'good'], 'unit': 'nats', 'reference': 'DeepSeek',
           'bootstrap_replicates': data['bootstrap_replicates'], 'geometry': data['geometry'],
           'intervals': {'ci95_fixed': 'Nominal marginal cluster-bootstrap percentiles, maps and observed answers fixed.',
                         'ci95_refit': 'Nominal marginal cluster-bootstrap percentiles, opposite-fold maps refitted, observed answers fixed.',
                         'ci95_fixed_answer_resampling': 'Page-style fixed-map cluster plus empirical answer resampling; separate diagnostic assumptions.'},
           'variance_key': 'Piece width = full cluster-refit interval width times diagnostic variance / sum of diagnostic variances. These overlapping one-factor weights are not independent variance contributions; the answer piece is diagnostic, not fresh-generation uncertainty in the conditional cross.',
           'sources': {}, 'source_sha256': digest(args.input)}
    for source, record in data['sources'].items():
        key = variance_key(record)
        shares = np.array(key['normalized_diagnostic_variance_weights'])
        assert np.allclose(shares.sum(0), 1, atol=1e-14) and np.all(shares >= 0)
        anchor = record['models']['DeepSeek']['absolute']
        out = {'deepseek_anchor': anchor, 'models': {}, 'key_model': record['representative_model'],
               'variance_shares': dict(zip(('questions', 'answers', 'calibration'), shares.tolist())),
               'sum_var_over_full': key['sum_variances_over_full_variance'],
               'questions': record['questions'], 'informative_questions': record['informative_questions']}
        for name, r in record['models'].items():
            if name == 'DeepSeek':
                continue
            assert np.max(abs(np.array(r['absolute']) - (np.array(anchor) + r['offset']))) < 1e-12
            u = r['uncertainty']
            out['models'][name] = {'model': r['model'], 'offset': r['offset'], 'absolute': r['absolute'],
                                   'ci95_fixed': u['clusters_only']['ci95'], 'ci95_refit': u['cluster_refit']['ci95'],
                                   'ci95_fixed_answer_resampling': u['fixed_maps']['ci95']}
            assert np.isfinite([out['models'][name][k] for k in ('offset', 'absolute')]).all()
            for k in ('ci95_fixed', 'ci95_refit', 'ci95_fixed_answer_resampling'):
                intervals = np.array(out['models'][name][k])
                assert intervals.shape == (2, 2) and np.isfinite(intervals).all() and np.all(intervals[:, 0] < intervals[:, 1])
        assert len(out['models']) == 5
        web['sources'][source] = out
    web['caption'] = 'Models favour lawful and good options relative to an even split; region boundaries are illustrative. Insets show calibrated offsets from DeepSeek. The uncertainty key normalizes overlapping diagnostic variances.'
    day, game = (web['sources'][s] for s in ('daily_dilemmas', 'machiavelli'))
    dr, gr = day['sum_var_over_full'], game['sum_var_over_full']
    extent = web['geometry']['extent_nats']
    radius = web['geometry']['neutral_octagon_circumradius_nats']
    web['footnote'] = ('Pair-gap-weighted log-odds, DeepSeek confidence scale; nominal cluster-bootstrap intervals condition on the two observed answers. '
                       f'Extent ±{extent:g} nats; octagon radius {radius:g}. Everyday/games: {day["questions"]}/{game["questions"]} questions; key models {day["key_model"]}/{game["key_model"]}. '
                       f'Sum of diagnostic variances/full variance, lawful/good: everyday {dr[0]:.2f}/{dr[1]:.2f}, games {gr[0]:.2f}/{gr[1]:.2f}. '
                       'Weights overlap; coverage is unvalidated.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(web, indent=2, allow_nan=False) + '\n')
    print(args.out.resolve())


if __name__ == '__main__':
    main()
