"""Export stated-probability replies to a committed compact table and the local reply cache. -- PI/OpenAI"""

import argparse
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation'))
from eval_api import cache_key  # noqa: E402

CACHE = ROOT / 'data/interim/action_choice_cache'


def export(run: Path, name: str) -> Path:
    config = json.loads((run / 'config.json').read_text())
    receipts = [json.loads(l) for path in sorted(run.glob('*/receipts.jsonl')) for l in path.open()]
    choices = {c['request_key']: c for path in sorted(run.glob('*/choices.jsonl')) for c in map(json.loads, path.open())}
    rows, cached = [], 0
    for r in receipts:
        if 'followup_of' in r['metadata']:
            continue
        m = r['metadata']
        packet = {'request': r['request'], 'cache_scope': {'protocol': m['actor_prompt_protocol'], 'draw': m['draw']}}
        key = cache_key(packet)
        c = choices[r['request_key']]
        if c['status'] == 'ok':  # failed replies are never cached, so a rerun redraws them
            path = CACHE / (key + '.json.gz')
            if not path.exists():
                path.write_bytes(gzip.compress(json.dumps({'request_key': key, 'request': r['request'], 'http_status': 200, 'response': r['response']}, ensure_ascii=False).encode()))
                cached += 1
        usage = c.get('usage', {})
        rows.append({'source_id': m['source_id'], 'source': m['source'], 'model': m['model'], 'draw': m['draw'], 'level': m.get('level'),
                     'protocol': m['actor_prompt_protocol'], 'reasoning_request': r['request']['reasoning'], 'max_tokens': r['request']['max_tokens'],
                     'question_text_sha256': m['original_prompt_sha256'], 'displayed_to_original': m['displayed_to_original'],
                     'status': c['status'], 'choice': c.get('original_choice_id'), 'probabilities': c.get('original_probabilities'),
                     'followup_used': c.get('followup_used', False), 'cache_key': key,
                     'reasoning_tokens': (usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                     'completion_tokens': usage.get('completion_tokens'), 'cost_usd': usage.get('cost')})
    out = ROOT / 'data/probability_tables' / f'{name}.jsonl.gz'
    out.parent.mkdir(exist_ok=True)
    header = {'_meta': {'author': 'PI/OpenAI', 'source_run': str(run.relative_to(ROOT)), 'protocol': config['protocol'],
              'option_order_policy': config['option_order_policy'], 'rows': len(rows),
              'version_rule': 'cache_key = sha256(request payload + protocol + draw); the payload contains model, reasoning settings, system prompt and question text, so changing any of them is a cache miss.',
              'note': 'Stated percentages are self-reports, not logprobs. Reasoning text is not included; raw replies stay local.'}}
    out.write_bytes(gzip.compress(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in [header] + rows).encode(), mtime=0))
    print(json.dumps({'rows': len(rows), 'new_cache_entries': cached, 'table': str(out)}))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--name', required=True)
    a = parser.parse_args()
    export(a.run.resolve(), a.name)
