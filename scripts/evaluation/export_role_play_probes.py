"""Role-play check for the page: per model, how much its stated choice changes when only the character's trait flips (total-variation distance, mirrored probe pairs). -- PI/OpenAI"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_role_probes import stated  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MIND = {'chess': 'Chess ambition', 'everest': 'Everest dream', 'gamble': 'Love of gambling', 'mission': 'Spy mission'}  # traits of the mind: should not carry over
BODY = {'mud': 'Loves eating mud', 'wetsocks': 'Likes wet socks'}  # bodily habits: allowed to carry over under the prompt


def main(runs: list[Path], out: Path) -> None:
    p = {}
    for run in runs:
        p |= stated(run)
    flips = {}
    for (sid, m), a in p.items():
        if not sid.endswith('_A'):
            continue
        b = p[sid[:-2] + '_B', m]
        pair = sid.split('_')[-2]
        flips.setdefault(m, {})[pair] = round(0.5 * sum(abs(a[o] - b[o]) for o in a), 4)
    rows = {m: {'mind': {MIND[k]: v[k] for k in MIND}, 'body': {BODY[k]: v[k] for k in BODY}, 'mind_mean': round(float(np.mean([v[k] for k in MIND])), 4)} for m, v in flips.items()}
    example = {m: {'dream': round(p['probe4_everest_A', m]['0'], 3), 'afraid': round(p['probe4_everest_B', m]['0'], 3)} for m in flips}
    out.write_text(json.dumps({'author': 'PI/OpenAI', 'runs': [str(r.relative_to(ROOT)) for r in runs], 'metric': 'total-variation distance between the A and B versions\' mean stated distributions (4 draws)',
                               'models': rows, 'everest_take_slot': example}, indent=1) + '\n')
    print(out, {m: r['mind_mean'] for m, r in rows.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    main([r.resolve() for r in a.runs], a.out)
