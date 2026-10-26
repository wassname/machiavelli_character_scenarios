"""Per-question table (options, label scores, every model's stated draws) from a stated-probability run; no inference. -- PI/OpenAI"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def export(run: Path) -> Path:
    rows = [json.loads(l) for stage in ('smoke', 'rest') for l in (run / stage / 'choices.jsonl').open()]
    redraw = run.with_name(run.name + '_redraw') / 'rest/choices.jsonl'
    best = {(r['source_id'], r['model'], r['draw']): r for r in rows}
    if redraw.exists():  # a recorded new attempt replaces an invalid reply
        for r in map(json.loads, redraw.open()):
            k = (r['source_id'], r['model'], r['draw'])
            if best[k]['status'] != 'ok':
                best[k] = r | {'redraw_of_invalid': True}
    # A provider content-policy block is a recorded refusal, not a score: that draw is dropped and listed; any other failure stops.
    refused = sorted(k for k, r in best.items() if r['status'] == 'api_refusal' or (r['status'] == 'infrastructure_error' and 'PROHIBITED_CONTENT' in json.dumps(r.get('error'))))
    bad = [k for k, r in best.items() if r['status'] != 'ok' and k not in refused]
    assert not bad, bad
    best = {k: r for k, r in best.items() if k not in refused}
    scenes = {s['source_id']: s for s in map(json.loads, (run / 'scenes.jsonl').open())}
    dims = json.loads((ROOT / 'outputs/20261025_individual_dimensions/analysis.json').read_text())['dimensions']
    scores = {(r['source_id'], r['choice_id']): r['scores'] for r in pq.read_table(ROOT / 'outputs/20261025_individual_dimensions/action_scores.parquet').to_pylist() if r['source_id'] in scenes}
    by = defaultdict(lambda: defaultdict(list))
    for (sid, m, d), r in best.items():
        by[sid][m].append({'draw': d, 'probabilities': r['original_probabilities'], 'reasoning': r['reasoning'], 'setup_note': r['setup_note']})
    out = run / 'question_tables.jsonl'
    with out.open('w') as f:
        f.write(json.dumps({'_meta': {'author': 'PI/OpenAI', 'run': str(run), 'dimensions': dims, 'provider_refusals': [list(k) for k in refused]}}) + '\n')
        for sid, s in scenes.items():
            opts = list(s['choice_text'])
            models = {m: {'draws': sorted(v, key=lambda x: x['draw']),
                          'mean_stated_percent': {o: float(np.mean([x['probabilities'][o] for x in v])) for o in opts}} for m, v in by[sid].items()}
            f.write(json.dumps({'source_id': sid, 'source': s['source'], 'role': s.get('eval_role', 'random'), 'eval_cluster': s['eval_cluster'],
                                'text': s['text'], 'options': s['choice_text'], 'option_scores': {o: scores[sid, o] for o in opts}, 'models': models}, ensure_ascii=False) + '\n')
    print(out, len(scenes))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    export(parser.parse_args().run)
