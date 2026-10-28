"""Two offline checks on a random-set run, in calibrated nats vs DeepSeek (same machinery as the page):
1. Rules vs virtue: on questions where the most lawful option is not the most virtuous one, does a model lean to the lawful (rule-following) option?
2. Virtue consistency: is a model's goodness-removed virtue profile the same on two random halves of the questions, and across everyday vs game questions?
-- PI/OpenAI"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_probability_eval import ROOT, SOURCES, load_calibration, nats_core, short, virtue_basis  # noqa: E402

DS = 'deepseek/deepseek-v4.1-flash'


def analyse(lines: list[dict], basis: dict, maps: dict, fold: dict) -> dict:
    # Returns {'rules': {source: (names, point, lo, hi, n_conflict)}, 'consistency': {model name: (halves everyday, halves games, everyday vs games)}}.
    dims = lines[0]['_meta']['dimensions']
    k_order = next(i for i, d in enumerate(dims) if d['id'] == 'order')
    models = [m for m in lines[1]['models'] if m != DS]
    names = [short(m) for m in models] + ['DeepSeek repeat']
    out, result = [], {}

    # 1. Rules vs virtue. goodness = PC1 of the 66 virtue labels (per source); lawful = D&D order label.
    rows = []
    for src, sname in SOURCES.items():
        V1 = basis['src'][src]['V'][0]
        good = lambda q: np.array([q['option_scores'][x] for x in q['options']], float)[:, basis['direct']] * basis['fav'] @ V1
        law = lambda q: np.array([q['option_scores'][x][k_order] for x in q['options']], float)

        def conflict(q):
            g, l = good(q), law(q)
            return np.isfinite(l).all() and np.ptp(l) > 0 and np.ptp(g) > 0 and np.argmax(l) != np.argmax(g)

        fn = lambda q: (np.stack([law(q), good(q)], 1) if conflict(q) else np.full((len(q['options']), 2), np.nan))
        point, lo, hi, nq, inf = nats_core(lines, src, fn, maps, fold)
        n_conf = int(inf[0])
        result.setdefault('rules', {})[src] = (names, point, lo, hi, n_conf)
        for j, n in enumerate(names):
            null = n == 'DeepSeek repeat'
            rows.append([sname, n, f'{point[j, 0]:+.3f} ({lo[j, 0]:+.3f} to {hi[j, 0]:+.3f})', f'{point[j, 1]:+.3f} ({lo[j, 1]:+.3f} to {hi[j, 1]:+.3f})', n_conf])
    out.append('## 1. Rules vs virtue\n\nQuestions where the most lawful option (D&D order label) is not the most virtuous one (first principal component of the 66 virtue labels). '
               'Calibrated nats vs DeepSeek toward the more lawful option, and toward the more virtuous option, on those questions only. '
               'For the DeepSeek repeat row the range is its own run-to-run range.\n\n'
               + tabulate(rows, headers=['Source', 'Model', 'Toward lawful', 'Toward virtuous', 'Conflict questions'], tablefmt='github'))

    # 2. Consistency of the goodness-removed (style) virtue profile.
    rng_split = lambda q: int(hashlib.sha256(('half:' + q['source_id']).encode()).hexdigest(), 16) % 2
    prof = {}
    for src in SOURCES:
        V1 = basis['src'][src]['V'][0]

        def style(q, half=None, V1=V1):
            S = np.array([q['option_scores'][x] for x in q['options']], float)[:, basis['direct']] * basis['fav']
            R = S - np.outer(S @ V1, V1)  # remove the overall-goodness part
            return R if half is None or rng_split(q) == half else np.full_like(R, np.nan)

        for half in (None, 0, 1):
            prof[src, half] = nats_core(lines, src, lambda q, h=half: style(q, h), maps, fold)[0]  # models+self x 66
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    result['consistency'] = {n: (corr(prof['daily_dilemmas', 0][j], prof['daily_dilemmas', 1][j]), corr(prof['machiavelli', 0][j], prof['machiavelli', 1][j]),
                                 corr(prof['daily_dilemmas', None][j], prof['machiavelli', None][j])) for j, n in enumerate(names)}
    rows = []
    for j, n in enumerate(names):
        rows.append([n] + [f'{corr(prof[s, 0][j], prof[s, 1][j]):+.2f}' for s in SOURCES]
                    + [f'{corr(prof["daily_dilemmas", None][j], prof["machiavelli", None][j]):+.2f}'])
    out.append('## 2. Virtue consistency\n\nEach model\'s profile = calibrated nats vs DeepSeek toward options with more of each of the 66 virtues, after removing overall goodness (as in the style spider). '
               'Numbers are correlations across the 66 virtues between two hash-random halves of the questions (within a source), and between everyday and game questions. '
               '1 = the same pattern of leanings; 0 = no shared pattern. The DeepSeek repeat row (DeepSeek against itself) shows what noise alone gives.\n\n'
               + tabulate(rows, headers=['Model', 'Halves, everyday', 'Halves, games', 'Everyday vs games'], tablefmt='github'))
    result['markdown'] = out
    return result


def main(tables: Path) -> None:
    lines, maps, identity, fold = load_calibration(tables)
    groups = json.loads((ROOT / 'rubrics/virtue_display_groups.json').read_text())['groups']
    out = analyse(lines, virtue_basis(lines, groups), maps, fold)['markdown']
    text = f'# Rules vs virtue, and virtue consistency\n\nData: `{tables.relative_to(ROOT)}`. Script: `scripts/evaluation/rules_and_consistency.py`.\n\n' + '\n\n'.join(out) + '\n\n-- PI/OpenAI\n'
    (tables.parent / 'rules_and_consistency.md').write_text(text)
    print(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path, default=ROOT / 'outputs/20261027_random1000_v12/question_tables.jsonl')
    main(parser.parse_args().tables.resolve())
