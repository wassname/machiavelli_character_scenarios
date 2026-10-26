"""Offline check: the stated-probability cache hits on identical requests and misses when any versioned field changes. -- PI/OpenAI"""

import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation'))
from eval_api import cache_key, run_requests  # noqa: E402
from run_tiny_eval import build  # noqa: E402

RUN = ROOT / 'outputs/20261026_probability_reference'


def main() -> None:
    config = json.loads((RUN / 'config.json').read_text())
    scene = json.loads((RUN / 'scenes.jsonl').open().readline())
    base = build(scene, 0, 'deepseek/deepseek-v4.1-flash', config)
    variants = {}
    variants['draw'] = build(scene, 2, 'deepseek/deepseek-v4.1-flash', config)  # same option order for 2-option questions
    variants['protocol'] = copy.deepcopy(base) | {'cache_scope': base['cache_scope'] | {'protocol': 'other'}}
    variants['model'] = build(scene, 0, 'qwen/qwen3.8-27b', config)
    changed = copy.deepcopy(scene)
    changed['text'] = changed['text'].replace('Current scene:', 'Current scene: ')
    variants['question_text'] = build(changed, 0, 'deepseek/deepseek-v4.1-flash', config)
    variants['reasoning'] = copy.deepcopy(base)
    variants['reasoning']['request']['reasoning'] = {'effort': 'low', 'exclude': False}
    keys = {name: cache_key(v) for name, v in variants.items()}
    assert len(scene['choice_text']) != 2 or variants['draw']['request'] == base['request'], 'expected identical payload for draw 0 vs 2 on a 2-option question'
    for name, k in keys.items():
        assert k != cache_key(base), f'{name} change did not bust the cache'
    # Real pipeline: the identical request is a free cache hit; any variant would need a paid call and is refused at a $0 cap.
    with tempfile.TemporaryDirectory() as tmp:
        result = asyncio.run(run_requests([base], Path(tmp), max_new_usd=0, concurrency=1, refresh=False, cache_write=False))
        assert result['cache_hits'] == 1 and result['new_cost_usd'] == 0, result
    with tempfile.TemporaryDirectory() as tmp:
        try:
            asyncio.run(run_requests([variants['draw']], Path(tmp), max_new_usd=0, concurrency=1, refresh=False, cache_write=False))
        except AssertionError as error:
            assert 'cap reached' in str(error), error  # refused before any network call
        else:
            raise AssertionError('changed draw was served from cache')
    print('PASS: identical request is a cache hit at $0; draw, protocol, model, question text and reasoning changes each give a new cache key; a changed draw is refused at a $0 cap (no cached reply).')


if __name__ == '__main__':
    main()
