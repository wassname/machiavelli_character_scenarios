"""Scan every scenario's visible text (scene, options) for template artifacts left by the source games, and show how prompts repair them. -- PI/OpenAI"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/evaluation'))
from action_choice import repair_source_text  # noqa: E402

# Pattern -> what it would mean if present. Patterns are checked on the text a model sees.
PATTERNS = {
    'user_input (player-typed name)': re.compile(r'user_input', re.I),
    'undefined': re.compile(r'\bundefined\b'),
    'literal backslash-n': re.compile(r'\\n'),
    'template braces {..} or ${..}': re.compile(r'\$?\{[A-Za-z_][^}]*\}'),
    'NaN': re.compile(r'\bNaN\b'),
    'trailing ellipsis (cut text)': re.compile(r'(\.\.\.|\u2026)\s*$', re.M),
}


def visible(row: dict) -> str:
    return '\n'.join([row['decision_context'] or '', *json.loads(row['choice_text']).values()])


def main(parquet: Path, out: Path) -> None:
    rows = [r for r in pq.read_table(parquet, columns=['source', 'source_id', 'has_virtue_labels', 'decision_context', 'choice_text']).to_pylist()]
    hits, examples = defaultdict(Counter), defaultdict(list)
    after = Counter()
    for r in rows:
        text = visible(r)
        pool = 'labelled' if r['has_virtue_labels'] else 'unlabelled'
        for name, pat in PATTERNS.items():
            n = len(pat.findall(text))
            if n:
                hits[name][pool, 'questions'] += 1
                hits[name][pool, 'occurrences'] += n
                if len(examples[name]) < 3:
                    m = pat.search(text)
                    examples[name].append(f"{r['source_id']}: …{text[max(0, m.start() - 60):m.end() + 40]!r}")
                if pat.search(repair_source_text(text)):
                    after[name] += 1
    labelled = sum(bool(r['has_virtue_labels']) for r in rows)
    table = [[name, hits[name]['labelled', 'questions'], hits[name]['labelled', 'occurrences'], hits[name]['unlabelled', 'questions'] + hits[name]['labelled', 'questions'], after[name]] for name in PATTERNS]
    report = (f'# Template artifact scan\n\n`{parquet.relative_to(ROOT)}`: {len(rows):,} scenarios, {labelled:,} with virtue labels. Visible text = scene plus option texts.\n\n'
              + tabulate(table, headers=['Pattern', 'Labelled questions', 'Labelled occurrences', 'All questions', 'Questions still matching after prompt repair'], tablefmt='github')
              + '\n\n## First examples\n\n' + '\n'.join(f'- **{k}**: ' + '<br>'.join(v) for k, v in examples.items()) + '\n\n-- PI/OpenAI\n')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'template_artifacts.md').write_text(report)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parquet', type=Path, default=ROOT / 'data/processed/parquet/train.parquet')
    parser.add_argument('--out', type=Path, default=ROOT / 'outputs/20261027_template_scan')
    a = parser.parse_args()
    main(a.parquet, a.out)
