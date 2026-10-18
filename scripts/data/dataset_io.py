"""Pinned dataset loading and game-balanced sampling. PI/OpenAI."""

import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
SOURCES = json.loads((ROOT / "data/sources.json").read_text())


def published_rows() -> list[dict]:
    ref = SOURCES["original_prompts"]
    path = hf_hub_download(ref["repository"], ref["file"], repo_type="dataset", revision=ref["revision"])
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        for field in ("axes", "choice_text", "choice_labels", "agg_power_by_choice", "selection_axis_std", "selection_crude_axis_std", "selection_top_subtle_deltas", "selection_top_crude_deltas"):
            row[field] = json.loads(row[field])
    assert len(rows) == ref["rows"], (len(rows), ref["rows"])
    assert len({r["source_id"] for r in rows}) == len(rows)
    return rows


def source_rows(rows: list[dict]) -> dict[str, dict]:
    ref = SOURCES["source_dataset"]
    dataset = load_dataset(ref["repository"], revision=ref["revision"], split=ref["split"])
    wanted = pa.array([r["source_id"] for r in rows])
    matched = {}
    for batch in dataset.data.to_batches(max_chunksize=1000):
        ids = pc.binary_join_element_wise("machiavelli_", batch.column("f"), "_", batch.column("row_i"), "")
        mask = pc.is_in(ids, value_set=wanted)
        for row in batch.filter(mask).to_pylist():
            source_id = f"machiavelli_{row['f']}_{row['row_i']}"
            assert source_id not in matched, source_id
            matched[source_id] = row
    assert set(matched) == set(wanted.to_pylist()), "Missing source rows at the pinned revision"
    return matched


def input_rows(path: Path | None = None) -> list[dict]:
    if path is None:
        return published_rows()
    return [json.loads(line) for line in path.open()]


def sample_by_game(rows: list[dict], per_game: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    games = defaultdict(list)
    for row in rows:
        games[row["game_id"]].append(row)
    picked = {game: rng.sample(sorted(items, key=lambda r: r["source_id"]), min(per_game, len(items))) for game, items in sorted(games.items())}
    return [picked[game][depth] for depth in range(per_game) for game in picked if depth < len(picked[game])]


def select_rows(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    if limit is None:
        return rows
    assert limit > 0
    games = len({row["game_id"] for row in rows})
    return sample_by_game(rows, (limit + games - 1) // games, seed)[:limit]


def write_parquet(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0])
    assert all(set(row) == set(keys) for row in rows)
    encoded = [{key: json.dumps(row[key], ensure_ascii=False, sort_keys=True) if isinstance(row[key], (dict, list)) else row[key] for key in keys} for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(encoded), path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
