"""Per-question table (options, label scores, every model's stated draws) from a stated-probability run; no inference. -- PI/OpenAI"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def export(run: Path, repairs: list[Path] = ()) -> Path:
    rows = [json.loads(l) for stage in ('smoke', 'rest') for l in (run / stage / 'choices.jsonl').open()]
    redraw = [f for stage in ('smoke', 'rest') if (f := run.with_name(run.name + '_redraw') / stage / 'choices.jsonl').exists()]
    best = {(r['source_id'], r['model'], r['draw']): r for r in rows}
    for f in redraw:  # a recorded new attempt replaces an invalid reply; small redraws can finish inside the smoke stage
        for r in map(json.loads, f.open()):
            k = (r['source_id'], r['model'], r['draw'])
            if best[k]['status'] != 'ok':
                best[k] = r | {'redraw_of_invalid': True}
    # A repair run re-asks whole questions with a repaired prompt (e.g. filled name placeholders): its replies replace every draw of those questions.
    replaced = set()
    for rep_run in repairs:
        new = [json.loads(l) for stage in ('smoke', 'rest') if (rep_run / stage / 'choices.jsonl').exists() for l in (rep_run / stage / 'choices.jsonl').open()]
        assert all(r['status'] == 'ok' for r in new), [r['status'] for r in new if r['status'] != 'ok']
        for r in new:
            assert (r['source_id'], r['model'], r['draw']) in best
            best[r['source_id'], r['model'], r['draw']] = r | {'repair_run': str(rep_run)}
            replaced.add(r['source_id'])
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
        f.write(json.dumps({'_meta': {'author': 'PI/OpenAI', 'run': str(run), 'dimensions': dims, 'provider_refusals': [list(k) for k in refused], 'repair_runs': [str(x) for x in repairs], 'repaired_questions': sorted(replaced)}}) + '\n')
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
    parser.add_argument('--repair', type=Path, nargs='*', default=[], help='runs that re-asked whole questions with a repaired prompt; their replies replace the main run\'s')
    a = parser.parse_args()
    export(a.run, a.repair)
