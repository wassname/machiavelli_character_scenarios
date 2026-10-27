"""Before/after check for the name-placeholder repair: on the re-asked questions, how many replies call the placeholder an unresolved artifact vs read it as an unknown name. -- PI/OpenAI"""

import argparse
import json
import re
from pathlib import Path

CONFUSED = re.compile(r'unresolved placeholder|placeholder (is|remains) unresolved|artifact|literal (placeholder|\[chosen)|template|bug|error in the (prompt|text)|user_input', re.I)
ANY = re.compile(r'placeholder|\[chosen name|user_input', re.I)


def count(run: Path, ids: set) -> tuple[int, int, int]:
    rows = [r for s in ('smoke', 'rest') if (run / s / 'choices.jsonl').exists() for r in map(json.loads, (run / s / 'choices.jsonl').open()) if r['status'] == 'ok' and r['source_id'] in ids]
    text = [(r['setup_note'] or '') + ' ' + r['reasoning'] for r in rows]
    return len(rows), sum(bool(ANY.search(t)) for t in text), sum(bool(CONFUSED.search(t)) for t in text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    a = parser.parse_args()
    ids = {json.loads(l)['source_id'] for l in (a.after / 'scenes.jsonl').open()}
    for name, run in (('before', a.before), ('after', a.after)):
        n, mention, confused = count(run, ids)
        print(f'{name}: {run} | {len(ids)} questions, {n} replies | mention placeholder {mention} | call it an unresolved artifact {confused}')
