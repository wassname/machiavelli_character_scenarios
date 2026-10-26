"""Freeze and execute the reviewed four-model pilot through the existing API pipeline. -- PI/OpenAI"""

import argparse
import asyncio
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import sys
import traceback

from loguru import logger

from action_choice import behavior_eligible, build_packet, ELIGIBILITY_PATH, parse_receipt, repeat_summary
from billing_reservations import outstanding
from eval_api import ROOT, digest, run_requests, SCOPE
from select_differentiated_pilot import sha

BUDGET = ROOT / '.local/differentiated_pilot_budget.json'
CONFIG = ROOT / 'data/models/differentiated_pilot.json'
SELECTION = ROOT / 'outputs/20261026_differentiated_pilot_selection'
OUT = ROOT / 'outputs/20261026_differentiated_pilot_v2'


def prepare(out: Path, previous: Path | None = None) -> None:
    out.mkdir(parents=True, exist_ok=False)
    selection = json.loads((SELECTION / 'selection.json').read_text())
    assert selection['selected_reviewed'] == len(selection['questions']) == 100
    for path, expected in selection['input_sha256'].items():
        assert sha(Path(path)) == expected, path
    config = json.loads(CONFIG.read_text())
    budget = json.loads(BUDGET.read_text())
    assert list(config['models']) == budget['models']
    reviews = json.loads(ELIGIBILITY_PATH.read_text())['cases']
    scenes = list(map(json.loads, (SELECTION / 'scenes.jsonl').open()))
    assert all(s['source_id'] in reviews and behavior_eligible(s, reviews) for s in scenes)
    packets = []
    for scene in scenes:
        for model in config['models']:
            for order in ('forward', 'reverse'):
                p = build_packet(scene, order, model, config | {'max_tokens': config['models'][model]['max_tokens']})
                for repetition in range(3):
                    packets.append({'request': p['request'], 'metadata': p['metadata'] | {'repetition': repetition}})
    random.Random(20261026).shuffle(packets)
    counts = Counter(digest(p['request']) for p in packets)
    assert len(packets) == 2400 and len(counts) == 800 and set(counts.values()) == {3}
    estimates = {}
    for model, settings in config['models'].items():
        subset = [p for p in packets if p['request']['model'] == model]
        def cost(chars_per_token: float, output_tokens: int) -> float:
            return sum((len(json.dumps(p['request'], ensure_ascii=False)) / chars_per_token * settings['input_usd_per_million']
                        + output_tokens * settings['output_usd_per_million']) / 1e6 for p in subset)
        estimates[model] = {'requests': len(subset), 'central_usd': cost(4, 1280), 'conservative_usd': cost(3, settings['max_tokens'])}
        assert cost(3, settings['max_tokens']) > 0
        assert max((len(json.dumps(p['request'], ensure_ascii=False)) / 3 * settings['input_usd_per_million'] + settings['max_tokens'] * settings['output_usd_per_million']) / 1e6 for p in subset) < .01, 'Per-request estimate exceeds API reservation'
    smoke_ids = [next(s['source_id'] for s in scenes if s['source'] == source) for source in ('daily_dilemmas', 'machiavelli')]
    for target, source in {'selection.json': SELECTION / 'selection.json', 'scenes.jsonl': SELECTION / 'scenes.jsonl',
                           'config.json': CONFIG, 'budget.json': BUDGET}.items():
        shutil.copyfile(source, out / target)
    (out / 'packets.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in packets))
    inherited, previous_inputs = [], {}
    if previous is not None:
        planned = {trial_key(p): p for p in packets if p['metadata']['source_id'] in smoke_ids}
        prior = previous / 'smoke/receipts.jsonl'
        previous_inputs[str(prior.resolve())] = sha(prior)
        for receipt in map(json.loads, prior.open()):
            p = planned[trial_key(receipt)]
            if receipt['request'] == p['request']:
                assert receipt['metadata'] == p['metadata']
                assert parse_receipt(receipt, config)['status'] == 'ok', 'Unchanged failing requests cannot be silently redrawn'
                inherited.append(receipt)
    (out / 'reused_smoke_receipts.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in inherited))
    files = ['scripts/evaluation/run_differentiated_pilot.py', 'scripts/evaluation/select_differentiated_pilot.py',
             'scripts/evaluation/action_choice.py', 'scripts/evaluation/eval_api.py', 'scripts/data/billing_reservations.py',
             'data/evaluation_eligibility.json', 'data/models/differentiated_pilot.json']
    code = {}
    for name in files:
        target = out / 'inference_source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        code[name] = sha(target)
    manifest = {'author': 'PI/OpenAI', 'requests': len(packets), 'unique_payloads': len(counts), 'repetitions': 3,
                'model_config': config, 'smoke_ids': smoke_ids, 'estimates': estimates,
                'central_total_usd': sum(e['central_usd'] for e in estimates.values()),
                'conservative_total_usd': sum(e['conservative_usd'] for e in estimates.values()),
                'estimate_method': 'Serialized request characters/4 plus 1280 completion tokens centrally; /3 plus each native output ceiling for conservative planning. Native usage is authoritative. The hard spend cap may stop expansion before completion.',
                'reused_smoke_replies': len(inherited), 'previous_smoke_inputs': previous_inputs,
                'budget': budget, 'inference_code_sha256': code,
                'frozen_sha256': {name: sha(out / name) for name in ('selection.json', 'scenes.jsonl', 'config.json', 'budget.json', 'packets.jsonl', 'reused_smoke_receipts.jsonl')},
                'control_note': 'Historical actors selected questions; all four new actors use identical current scenario prompts. Original and reversed order, three uncached exact repeats. Outcomes and virtue labels are not shown.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ('requests', 'unique_payloads', 'central_total_usd', 'conservative_total_usd')}, indent=2))
    print(out / 'manifest.json')


def trial_key(p: dict) -> tuple:
    m = p['metadata']
    return m['source_id'], m['model'], m['order'], m['repetition']


def run(out: Path, stage: str) -> None:
    manifest = json.loads((out / 'manifest.json').read_text())
    for name, expected in manifest['frozen_sha256'].items():
        assert sha(out / name) == expected, name
    for name, expected in manifest['inference_code_sha256'].items():
        assert sha(ROOT / name) == expected, f'Inference source changed after freezing: {name}'
    config, budget = manifest['model_config'], manifest['budget']
    packets = list(map(json.loads, (out / 'packets.jsonl').open()))
    smoke_ids = set(manifest['smoke_ids'])
    packets = [p for p in packets if (p['metadata']['source_id'] in smoke_ids) == (stage == 'smoke')]
    planned = packets
    inherited = list(map(json.loads, (out / 'reused_smoke_receipts.jsonl').open())) if stage == 'smoke' else []
    inherited_keys = {trial_key(r) for r in inherited}
    packets = [p for p in packets if trial_key(p) not in inherited_keys]
    if stage == 'remaining':
        smoke = list(map(json.loads, (out / 'smoke/choices.jsonl').open()))
        assert len(smoke) == 48 and all(r['status'] == 'ok' for r in smoke), 'Smoke must succeed before expansion'
    batch = out / stage
    batch.mkdir(exist_ok=False)
    logger.remove()
    logger.add(sys.stdout, format='{message}\n', level='INFO', colorize=False)
    logger.add(batch / 'run.md', format='{message}\n', level='DEBUG', colorize=False)
    logger.info(f'# Differentiated pilot: {stage}\n\n-- PI/OpenAI\n\nCommand: `{sys.argv}`\n\nSHOULD: {len(planned)} successful native replies: {len(packets)} new requests + {len(inherited)} byte-identical previous smoke observations; no cache hits or duplicate response IDs.\n\nTODO validate: selected cases distinguish new models more than random controls, beyond within-model variation.\n\nConfig: `{out / "config.json"}`; frozen inputs: `{out / "manifest.json"}`.')
    summary = {}
    try:
        with (ROOT / '.local/labelling.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            ledger = list(map(json.loads, (ROOT / '.local/billing.jsonl').open()))
            current = sum(r['cost_usd'] for r in ledger) + outstanding(ROOT)
            remaining = budget['additional_cap_usd'] - (current - budget['project_start_usd']) - budget['review_reserve_usd']
            assert remaining > 0
            logger.info(f'Additional-budget remaining for requests: ${remaining:.5f}. Native review reserve ${budget["review_reserve_usd"]:.2f}; default project caps unchanged.')
            summary = asyncio.run(run_requests(packets, batch, max_new_usd=min(.3, remaining) if stage == 'smoke' else remaining,
                concurrency=config['concurrency'], key_source=budget['key_source'], refresh=True, cache_write=False,
                project_cap_usd=budget['project_start_usd'] + budget['additional_cap_usd'] - budget['review_reserve_usd'],
                evaluation_cap_usd=budget['evaluation_start_usd'] + budget['additional_cap_usd'] - budget['review_reserve_usd']))
        receipts = list(map(json.loads, (batch / 'receipts.jsonl').open())) + inherited
        (batch / 'receipts.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in receipts))
        summary |= {'reused_native_replies': len(inherited), 'combined_completed': len(receipts)}
        parsed = [parse_receipt(r, config) for r in receipts]
        (batch / 'choices.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in parsed))
        status = Counter(r['status'] for r in parsed)
        summary['status_counts'] = dict(status)
        (batch / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        assert status == {'ok': len(planned)}, status
        assert {trial_key(p) for p in planned} == {trial_key(r) for r in receipts}
        ids = [r['response']['id'] for r in receipts]
        assert len(ids) == len(set(ids)) and all(not r['cache_hit'] for r in receipts)
        (batch / 'repeat_summary.json').write_text(json.dumps(repeat_summary(parsed, 3), indent=2) + '\n')
        if stage == 'remaining':
            all_receipts = list(map(json.loads, (out / 'smoke/receipts.jsonl').open())) + receipts
            all_parsed = smoke + parsed
            assert len(all_parsed) == manifest['requests'] == len({trial_key(r) for r in all_receipts})
            assert len({r['response']['id'] for r in all_receipts}) == manifest['requests']
            (out / 'choices.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in all_parsed))
            (out / 'receipts.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in all_receipts))
            summary['pilot_total_actor_cost_usd'] = sum(r['response']['usage']['cost'] for r in all_receipts)
            (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        logger.info('## Result\n\n```json\n' + json.dumps(summary, indent=2) + '\n```\n\n' + str(batch / 'run.md'))
    except Exception:
        logger.error('## Failed\n\n```text\n' + traceback.format_exc() + '\n```\n\nRaw attempts and charges remain saved.\n\n' + str(batch / 'run.md'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'smoke', 'remaining'])
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--allow-paid', action='store_true')
    parser.add_argument('--reuse-smoke-from', type=Path)
    args = parser.parse_args()
    if os.environ.get('BEARTYPE'):
        from beartype import beartype
        build_packet, parse_receipt = beartype(build_packet), beartype(parse_receipt)
    if args.stage == 'prepare':
        prepare(args.out, args.reuse_smoke_from)
    else:
        assert args.allow_paid, 'Execution requires --allow-paid'
        run(args.out, args.stage)
