"""Unpaid native-reply, accounting and independent arithmetic checks for the pilot. -- PI/OpenAI"""

import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
from statistics import fmean
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation'))
from action_choice import behavior_eligible, parse_receipt
from eval_api import digest
from run_differentiated_pilot import trial_key
from select_differentiated_pilot import sha


def verify(run: Path, stage: str) -> dict:
    manifest = json.loads((run / 'manifest.json').read_text())
    for name, expected in manifest['frozen_sha256'].items():
        assert sha(run / name) == expected, name
    for name, expected in manifest['inference_code_sha256'].items():
        assert sha(run / 'inference_source' / name) == expected, name
    for name, expected in manifest['previous_smoke_inputs'].items():
        assert sha(Path(name)) == expected, name
    config = manifest['model_config']
    scenes = {s['source_id']: s for s in map(json.loads, (run / 'scenes.jsonl').open())}
    reviews = json.loads((run / 'inference_source/data/evaluation_eligibility.json').read_text())['cases']
    assert all(sid in reviews and behavior_eligible(s, reviews) for sid, s in scenes.items())
    packets = {trial_key(p): p for p in map(json.loads, (run / 'packets.jsonl').open())}
    where = run / 'smoke' if stage == 'smoke' else run
    if stage == 'smoke':
        packets = {key: p for key, p in packets.items() if key[0] in manifest['smoke_ids']}
    receipts = list(map(json.loads, (where / 'receipts.jsonl').open()))
    parsed = list(map(json.loads, (where / 'choices.jsonl').open()))
    assert len(receipts) == len(parsed) == len(packets) == (48 if stage == 'smoke' else 2400)
    assert {trial_key(r) for r in receipts} == packets.keys()
    assert parsed == [parse_receipt(r, config) for r in receipts], 'Saved parser results differ from native responses'
    assert all(r['status'] == 'ok' and not r['cache_hit'] for r in parsed)
    assert len({r['response']['id'] for r in receipts}) == len(receipts)
    for r in receipts:
        p = packets[trial_key(r)]
        assert r['request'] == p['request'] and r['metadata'] == p['metadata']
        assert r['request_key'] == digest(p['request'])
    counts = Counter(r['request_key'] for r in receipts)
    assert set(counts.values()) == {3}, 'Fresh repetitions are not byte-identical requests'
    ledger = list(map(json.loads, (ROOT / '.local/billing.jsonl').open()))
    receipt_ids = {r['response']['id'] for r in receipts}
    charges = {r['response_id']: r for r in ledger if 'response_id' in r and r['response_id'] in receipt_ids}
    assert charges.keys() == receipt_ids
    for r in receipts:
        charge = charges[r['response']['id']]
        assert charge['cost_usd'] == r['response']['usage']['cost']
        assert charge['request_key'] == r['request_key']
    stages = {(run / 'smoke').resolve(), (run / 'remaining').resolve()}
    stages.update(Path(p).parent.resolve() for p in manifest['previous_smoke_inputs'])
    all_attempt_cost = sum(r['cost_usd'] for r in ledger if 'run' in r and Path(r['run']).resolve() in stages)
    unknown = json.loads((ROOT / '.local/billing_reservations.json').read_text())
    unresolved = sum(r['reserved_usd'] for r in unknown if 'run' in r and Path(r['run']).resolve() in stages)
    result = {'author': 'PI/OpenAI', 'stage': stage, 'native_replies': len(receipts), 'unique_payloads': len(counts),
        'used_reply_cost_usd': sum(r['response']['usage']['cost'] for r in receipts), 'all_pilot_attempts_cost_usd': all_attempt_cost,
        'pilot_unknown_fee_reservations_usd': unresolved, 'model_reply_counts': dict(Counter(r['model'] for r in parsed)),
        'checks': ['Frozen source and inference code identities', 'All source-hash-bound eligibility reviews',
                   'Every planned payload and metadata matches its native receipt', 'Every native reply reparses identically',
                   'Three uncached distinct response IDs per exact payload', 'Every native charge matches the ledger']}
    if stage != 'smoke':
        analysis = json.loads((run / 'analysis.json').read_text())
        selection = {s['source_id']: s for s in json.loads((run / 'selection.json').read_text())['questions']}
        records = {(r['source_id'], r['model'], r['order'], r['repetition']): r for r in parsed}
        model_ids = list(config['models'])
        for q in analysis['questions']:
            sid = q['source_id']
            stable = {}
            for m in model_ids:
                choices = {records[sid, m, order, i]['original_choice_id'] for order in ('forward', 'reverse') for i in range(3)}
                if len(choices) == 1:
                    stable[config['models'][m]['display_name']] = choices.pop()
            assert stable == q['new_stable_models']
            assert sum(a != b for a, b in combinations(stable.values(), 2)) == q['new_stable_pair_disagreements']
            assert len(stable) * (len(stable) - 1) // 2 == q['new_stable_pairs']
            differences = []
            for order in ('forward', 'reverse'):
                p = {m: Counter(records[sid, m, order, i]['original_choice_id'] for i in range(3)) for m in model_ids}
                for a, b in combinations(model_ids, 2):
                    between = 1 - sum(p[a][c] * p[b][c] / 9 for c in scenes[sid]['choice_text'])
                    within = .75 * (2 - sum(v * v / 9 for v in p[a].values()) - sum(v * v / 9 for v in p[b].values()))
                    differences.append((between, within, between - within))
            assert np.allclose(np.mean(differences, axis=0), [q['between_model_disagreement'], q['within_model_disagreement'], q['excess_disagreement']], atol=1e-12)
        dims = analysis['dimensions']
        actions = {(r['source_id'], r['choice_id']): r['scores'] for r in pq.read_table(ROOT / 'outputs/20261025_individual_dimensions/action_scores.parquet').to_pylist() if r['source_id'] in scenes}
        lookup = {(r['source'], r['selection'], r['mode'], r['model'], r['item']): r for r in analysis['rows']}
        direct = [i for i, d in enumerate(dims) if d['family'] == 'virtues' and d['derived_from'] is None]
        signs = np.array([-1 if dims[i]['kind'] == 'vices' else 1 for i in direct])
        pca = json.loads((ROOT / 'outputs/20261026_pca_profiles/analysis.json').read_text())
        for source, parts in analysis['sources'].items():
            u = np.array([pca['sources'][source]['loadings'][dims[i]['id']] for i in direct])
            for stratum, info in parts['strata'].items():
                ids = [sid for sid in scenes if scenes[sid]['source'] == source and selection[sid]['selection'] == stratum]
                for model, settings in config['models'].items():
                    name = settings['display_name']
                    for i, d in enumerate(dims):
                        excluded = info['families'][d['family']]['excluded_missing_labels']
                        members = [sid for sid in ids if sid not in excluded]
                        actual = fmean(actions[sid, records[sid, model, order, rep]['original_choice_id']][i] for sid in members for order in ('forward', 'reverse') for rep in range(3))
                        reported = lookup[source, stratum, 'raw', name, d['id']]
                        assert abs(actual - reported['mean']) < 1e-12 and reported['scenes'] == len(members)
                        relative = lookup[source, stratum, 'versus_qwen', name, d['id']]['mean']
                        assert abs(relative - actual + lookup[source, stratum, 'raw', 'Qwen', d['id']]['mean']) < 1e-12
                    raw = np.array([lookup[source, stratum, 'versus_qwen', name, dims[i]['id']]['mean'] for i in direct])
                    expected = ((raw * signs) - np.dot(raw * signs, u) * u) * signs
                    observed = np.array([lookup[source, stratum, 'residual_vs_qwen', name, dims[i]['id']]['mean'] for i in direct])
                    assert np.allclose(expected, observed, atol=1e-12)
                    assert abs(np.dot(observed * signs, u)) < 1e-12
        for r in analysis['rows']:
            if r['model'] == 'Qwen' and r['mode'] in ('versus_qwen', 'residual_vs_qwen'):
                assert r['mean'] == 0 and r['ci95'] == [0, 0]
        result['checks'] += ['Independent probability-count disagreement formula', 'All-six-reply stability and differing stable pairs', 'Native choices to raw scores and exact family cohorts',
                             'Paired Qwen subtraction', 'Independent named-direction PC1 subtraction', 'Zero reference mean and intervals']
    (where / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/20261026_differentiated_pilot_v2')
    parser.add_argument('--stage', choices=['smoke', 'full'], default='full')
    args = parser.parse_args()
    verify(args.run, args.stage)
