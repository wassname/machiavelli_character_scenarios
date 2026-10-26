"""Offline landmark calibration and label-wise log-probability contrasts. -- PI/OpenAI

Run with uv run --offline --with matplotlib --with scipy python <this file>.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import isotonic_regression
from scipy.special import logsumexp
from scipy.stats import kendalltau
import scipy
from tabulate import tabulate
from loguru import logger

from score_choice_dimensions import bootstrap

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = "deepseek/deepseek-v4.1-flash"
STRICT_SLOPE = 1e-6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(paths: list[Path], level: str | None = None, model: str | None = None) -> tuple[list[dict], dict]:
    rows, statuses, seen = [], Counter(), set()
    for path in paths:
        with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open()) as stream:
            for line_no, line in enumerate(stream, 1):
                r = json.loads(line)
                if "_meta" in r or (level is not None and str(r.get("level")) != level) or (model is not None and r['model'] != model):
                    continue
                statuses[r["status"]] += 1
                if r["status"] != "ok":
                    continue
                key = r["source_id"], r["model"], r["draw"]
                assert key not in seen, f"Duplicate draw or mixed reasoning levels: {key}"
                seen.add(key)
                p = r["original_probabilities"] if "original_probabilities" in r else r["probabilities"]
                assert len(p) >= 2 and all(type(v) is int and 0 <= v <= 100 for v in p.values())
                assert sum(p.values()) == 100
                rows.append(r | {"original_probabilities": p, "trace": f"{path}:{line_no}"})
    assert rows, f"No valid rows for level={level!r}, model={model!r}; missing levels are untagged, never inferred"
    return rows, dict(statuses)


def pooled_logs(rows: list[dict], scenes: dict, expected_draws: int) -> tuple[dict, list[dict]]:
    grouped = defaultdict(list)
    for r in rows:
        scene = scenes[r['source_id']]
        hash_field = 'question_text_sha256' if 'question_text_sha256' in r else 'original_prompt_sha256'
        assert r[hash_field] == sha_text(scene['text']), f"Changed story: {r['source_id']}"
        assert r['source'] == scene['source']
        grouped[r["model"], r["source_id"]].append(r)
    result, incomplete = {}, []
    for (model, sid), rs in grouped.items():
        if {r["draw"] for r in rs} != set(range(expected_draws)):
            incomplete.append({"model": model, "source_id": sid, "draws": sorted(r["draw"] for r in rs)})
            continue
        options = list(scenes[sid]["choice_text"])
        assert all(set(r["original_probabilities"]) == set(options) for r in rs), sid
        # Integer products preserve exact ties across permuted draw multisets. -- PI/OpenAI
        log_products = np.array([math.log(math.prod(r['original_probabilities'][o] + 1 for r in rs)) for o in options])
        result[model, sid] = log_products / len(rs) - math.log(100 + len(options))
    return result, incomplete


def normalize(logits: np.ndarray) -> np.ndarray:
    return logits - logsumexp(logits)


def fit_map(model: dict, reference: dict, ids: list[str], kind: str = "monotone", shrinkage: float = 0., extrapolation: str = "last") -> dict:
    if kind == 'identity':
        return {'x': [0., float(np.log(101))], 'y': [0., float(np.log(101))], 'end_slope': 1.,
                'kind': kind, 'landmark_ids': [], 'points': [], 'direction_disagreement_fraction': None,
                'note': 'Identity requires no fitted questions.'}
    points = []
    for sid in ids:
        x, y = model[sid], reference[sid]
        pairs = list(combinations(range(len(x)), 2))
        for i, j in pairs:
            points.append({"source_id": sid, "x": round(float(abs(x[i] - x[j])), 12),
                           "y": float(abs(y[i] - y[j])), "weight": 1 / len(pairs),
                           "same_direction": bool(np.sign(x[i] - x[j]) == np.sign(y[i] - y[j])),
                           "opposed": bool((x[i] - x[j]) * (y[i] - y[j]) < 0)})
    assert points
    positive = [p for p in points if p["x"] > 0]
    assert positive, "All landmark options tied; confidence map is unidentified"
    xs = np.array(sorted({p["x"] for p in positive}))
    weights = np.array([sum(p["weight"] for p in positive if p["x"] == x) for x in xs])
    ys = np.array([sum(p["weight"] * p["y"] for p in positive if p["x"] == x) for x in xs]) / weights
    if kind == "identity":
        fitted = xs.copy()
    elif kind in ("temperature", "orthogonal"):
        if kind == "temperature":
            slope = float(np.sum(weights * xs * ys) / np.sum(weights * xs * xs))
        else:
            sx = sum(p['weight'] * p['x'] ** 2 for p in points)
            sy = sum(p['weight'] * p['y'] ** 2 for p in points)
            sxy = sum(p['weight'] * p['x'] * p['y'] for p in points)
            assert sxy > 0, "Orthogonal slope unidentified"
            slope = float((sy - sx + np.sqrt((sy - sx) ** 2 + 4 * sxy ** 2)) / (2 * sxy))
        assert slope > 0, "Temperature slope unidentified or zero"
        fitted = slope * xs
    else:
        assert kind == "monotone" and 0 <= shrinkage <= 1
        fitted = isotonic_regression(ys, weights=weights).x
        fitted = (1 - shrinkage) * fitted + shrinkage * xs
        fitted = (1 - STRICT_SLOPE) * fitted + STRICT_SLOPE * xs
    knots_x, knots_y = np.r_[0., xs], np.r_[0., fitted]
    assert (np.diff(knots_y) > 0).all()
    return {"x": knots_x.tolist(), "y": knots_y.tolist(), "points": points,
            "landmark_ids": ids, "strict_identity_fraction": STRICT_SLOPE, "kind": kind,
            "identity_shrinkage": shrinkage, "extrapolation": extrapolation,
            "direction_disagreement_fraction": sum(p['weight'] * (not p['same_direction']) for p in points) / sum(p['weight'] for p in points),
            "opposed_pair_fraction": sum(p['weight'] * p['opposed'] for p in points) / sum(p['weight'] for p in points),
            "direction_disagreement_note": "Question-weighted fraction; includes a tie on only one side as disagreement; opposed_pair_fraction counts only strict opposite signs. Magnitudes still contribute to fitting.",
            "end_slope": 1. if extrapolation == "unit" and kind == "monotone" else float((knots_y[-1] - knots_y[-2]) / (knots_x[-1] - knots_x[-2])),
            "note": "Odd extension through origin; linear interpolation; specified end-slope continuation outside maximum x. Positive identity fraction prevents new ties. Abscissas rounded to 12 decimals to merge floating-point duplicate odds."}


def quantile_map(model: dict, reference: dict, ids: list[str]) -> dict:
    pairs = []
    for sid in ids:
        indices = list(combinations(range(len(model[sid])), 2))
        for i, j in indices:
            x, y = model[sid][i] - model[sid][j], reference[sid][i] - reference[sid][j]
            pairs.append({'source_id': sid, 'x': float(abs(x)), 'y': float(abs(y)),
                          'weight': 1 / len(indices), 'same_direction': bool(np.sign(x) == np.sign(y)), 'opposed': bool(x * y < 0)})
    levels = [.05, .25, .5, .75, .95]
    weights = np.array([p['weight'] for p in pairs])
    x = np.quantile([p['x'] for p in pairs], levels, weights=weights, method='inverted_cdf')
    y = np.quantile([p['y'] for p in pairs], levels, weights=weights, method='inverted_cdf')
    unique = sorted({round(float(v), 12) for v in x if v > 1e-12})
    assert unique, 'All quantile odds tied; map unidentified'
    targets = [float(np.mean([b for a, b in zip(x, y) if round(float(a), 12) == v])) for v in unique]
    targets = (1 - STRICT_SLOPE) * np.array(targets) + STRICT_SLOPE * np.array(unique)
    result = {'x': [0.] + unique, 'y': [0.] + targets.tolist(), 'end_slope': 1., 'kind': 'quantile',
              'identity_shrinkage': 0., 'strict_identity_fraction': STRICT_SLOPE, 'extrapolation': 'unit',
              'landmark_ids': [], 'fit_ids': ids, 'points': pairs,
              'quantile_levels': levels, 'model_quantiles': x.tolist(), 'reference_quantiles': y.tolist(),
              'reference_p95_at_integer_edge': bool(np.isclose(y[-1], np.log(101), atol=1e-10)),
              'model_p95_at_integer_edge': bool(np.isclose(x[-1], np.log(101), atol=1e-10)),
              'direction_disagreement_fraction': sum(p['weight'] * (not p['same_direction']) for p in pairs) / sum(weights),
              'opposed_pair_fraction': sum(p['weight'] * p['opposed'] for p in pairs) / sum(weights),
              'note': 'Transductive unpaired matching of weighted magnitude distributions, equal weight per question; origin anchor and linear origin-to-p05; unit slope above p95. Duplicate x quantiles merge by mean y; zero x constrained to zero. Positive identity fraction prevents new ties.'}
    assert (np.diff(result['y']) > 0).all()
    return result


def apply_map(logp: np.ndarray, mapping: dict) -> tuple[np.ndarray, int]:
    odds = logp[:, None] - logp[None, :]
    magnitude = np.abs(odds)
    x, y = np.asarray(mapping["x"]), np.asarray(mapping["y"])
    values = np.interp(magnitude, x, y)
    outside = magnitude > x[-1]
    values[outside] = y[-1] + (magnitude[outside] - x[-1]) * mapping["end_slope"]
    logits = (np.sign(odds) * values).mean(1)
    calibrated = normalize(logits)
    assert np.array_equal(np.sign(odds), np.sign(calibrated[:, None] - calibrated[None, :])), "Calibration changed option ranks/ties"
    return calibrated, int(np.triu(outside, 1).sum())


def choose_roles(ids: list[str], reference: dict, count: int) -> dict:
    entropy = {s: float(-np.sum(np.exp(normalize(reference[s])) * normalize(reference[s])) / np.log(len(reference[s]))) for s in ids}
    ranked = sorted(ids, key=lambda s: (entropy[s], sha_text(s)))
    n = min(count, len(ranked) // 2)
    assert n >= 2 and len(ranked) > n, "Need >=2 landmark and >=1 held-out questions per source"
    landmarks = [ranked[min(len(ranked) - 1, int(i))] for i in np.linspace(0, len(ranked) - 1, n)]
    assert len(set(landmarks)) == n
    return {"landmark": landmarks, "dev": [s for s in ids if s not in set(landmarks)], "control": [],
            "entropy": entropy, "note": "Offline sketch only: entropy-spanning landmarks among matched questions; remaining questions held out. This is not the frozen production selection."}


def uncertain_with_extremes(ids: list[str], reference: dict, scenes: dict) -> dict:
    entropy = {s: float(-np.sum(np.exp(normalize(reference[s])) * normalize(reference[s])) / np.log(len(reference[s]))) for s in ids}
    ranked = sorted(ids, key=lambda s: (-entropy[s], sha_text(s)))
    by_cluster = {}
    for sid in ranked:
        by_cluster.setdefault(scenes[sid]['eval_cluster'], sid)
    dev = list(by_cluster.values())
    for sid in ranked:
        if len(dev) >= max(len(by_cluster), round(.1 * len(ids))):
            break
        if sid not in dev:
            dev.append(sid)
    remaining = [s for s in ranked if s not in set(dev)]
    assert len(remaining) >= 20
    extremes = remaining[:10] + remaining[-10:]
    return {'landmark': [], 'dev': dev, 'control': [], 'extreme': extremes,
            'note': 'Top 10% normalized DeepSeek entropy, at least one per game/topic; add 10 highest and 10 lowest confidence questions outside dev. Selection is within the same fixed held-out-ID pool used by self-controls; no new inference.'}


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def score_labels(ids: list[str], nats: dict, models: list[str], scores: dict, dimensions: list[dict],
                 scenes: dict, replicates: int, include_derived: bool = False) -> tuple[list[dict], dict]:
    rows, coverage = [], {}
    for family in ("virtues", "behaviours", "dnd"):
        ix = [i for i, d in enumerate(dimensions) if d["family"] == family and (include_derived or d["derived_from"] is None)]
        if family == "virtues":
            assert len(ix) == (80 if include_derived else 66)
        complete = [s for s in ids if np.isfinite(scores[s][:, ix]).all()]
        coverage[family] = {"complete": len(complete), "excluded_missing": sorted(set(ids) - set(complete))}
        if not complete:
            continue
        numerators, denominators = [], []
        for sid in complete:
            x = scores[sid][:, ix]
            x = x - x.mean(0)
            delta = np.stack([nats[m, mode, sid] for m in models for mode in ("raw", "calibrated")])
            delta = delta - delta.mean(1, keepdims=True)
            numerators.append(np.einsum("kd,mk->md", x, delta) / len(x))
            denominators.append(np.broadcast_to((x * x).mean(0), (len(models) * 2, len(ix))))
        num, den = np.stack(numerators), np.stack(denominators)
        shape = num.shape[1:]
        sufficient = np.stack([num.reshape(len(complete), -1), den.reshape(len(complete), -1)], axis=1)
        groups = [scenes[s]["eval_cluster"] for s in complete]
        samples = bootstrap(sufficient, groups, replicates)
        boot = np.divide(samples[:, 0], samples[:, 1], out=np.full_like(samples[:, 0], np.nan), where=samples[:, 1] > 0).reshape(replicates, *shape)
        point = np.divide(num.sum(0), den.sum(0), out=np.full_like(num[0], np.nan), where=den.sum(0) > 0)
        for m, model in enumerate(models):
            for t, mode in enumerate(("raw", "calibrated")):
                for j, i in enumerate(ix):
                    b = boot[:, 2 * m + t, j]
                    valid = b[np.isfinite(b)]
                    informative = den[:, 2 * m + t, j] > 0
                    rows.append({"model": model, "mode": mode, "family": family, "dimension": dimensions[i],
                                 "beta_nats_per_label_unit": float(point[2 * m + t, j]) if np.isfinite(point[2 * m + t, j]) else None,
                                 "ci95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
                                 "bootstrap_se": float(valid.std(ddof=1)) if len(valid) > 1 else None,
                                 "unidentified_bootstrap_replicates": int(replicates - len(valid)),
                                 "questions": len(complete), "informative_questions": int(informative.sum()),
                                 "clusters": len(set(groups)), "informative_clusters": len({g for g, v in zip(groups, informative) if v}),
                                 "mean_within_question_score_variance": float(den[:, 2 * m + t, j].mean())})
    return rows, coverage


def read_selection(path: Path, extreme_role: str, exclude_ids: Path | None) -> dict:
    selection = json.loads(path.read_text())
    excluded = set(json.loads(exclude_ids.read_text())) if exclude_ids else set()
    questions = [q | {'role': 'extreme' if q['role'] == extreme_role else q['role']} for q in selection['questions'] if q['source_id'] not in excluded]
    assert len({q['source_id'] for q in questions}) == len(questions), 'Duplicate selected question'
    assert all(q['role'] in ('dev', 'control', 'landmark', 'extreme') for q in questions)
    return selection | {'questions': questions, 'excluded_ids': sorted(excluded), 'extreme_role_in_source': extreme_role}


def write_question_tables(result: dict, comparison_rows: list[dict], reference_rows: list[dict], scenes: dict,
                          scores: dict, dimensions: list[dict], out: Path, extreme_role: str = 'extreme') -> Path:
    records = defaultdict(list)
    for r in reference_rows + [r for r in comparison_rows if r['model'] != REFERENCE]:
        records[r['source_id'], r['model']].append(r)
    contrasts = defaultdict(dict)
    for q in result['question_nats']:
        contrasts[q['source_id']][q['model']] = q
    path = out / 'question_tables.jsonl'
    with path.open('w') as stream:
        stream.write(json.dumps({'_meta': {'author': 'PI/OpenAI', 'dimensions': dimensions,
                                          'probability_note': 'mean_stated_percent is arithmetic; geometric_probabilities are normalized exp(mean smoothed log p). Nats use the geometric pool. Native example is lowest draw, not effect-selected.'}}) + '\n')
        for sid, models in sorted(contrasts.items()):
            options = list(scenes[sid]['choice_text'])
            native = {}
            for model in [REFERENCE] + list(models):
                rows = sorted(records[sid, model], key=lambda r: r['draw'])
                assert rows
                item = {'mean_stated_percent': {o: float(np.mean([r['original_probabilities'][o] for r in rows])) for o in options},
                        'draws': [{'draw': r['draw'], 'probabilities': r['original_probabilities'],
                                   'choice': r['original_choice_id'] if 'original_choice_id' in r else r['choice'],
                                   'trace': r['trace']} for r in rows],
                        'native_example': {'trace': rows[0]['trace'], 'reply': rows[0]['content'] if 'content' in rows[0] else None,
                                           'displayed_to_original': rows[0]['displayed_to_original'],
                                           'id_note': 'Native reply IDs are displayed IDs; summary probabilities and choices use original IDs.'}}
                if model == REFERENCE:
                    baseline = next(iter(models.values()))['baseline_logp']
                    item['geometric_probabilities'] = dict(zip(options, np.exp(baseline).tolist()))
                else:
                    q = models[model]
                    item['map_kind'] = q['map_kind']
                    for key in ('raw_logp', 'calibrated_logp', 'raw_nats', 'calibrated_nats'):
                        item[key] = dict(zip(options, q[key]))
                    item['geometric_probabilities'] = dict(zip(options, np.exp(q['raw_logp']).tolist()))
                    item['calibrated_probabilities'] = dict(zip(options, np.exp(q['calibrated_logp']).tolist()))
                native[model] = item
            table = {'source_id': sid, 'source': scenes[sid]['source'], 'role': extreme_role if next(iter(models.values()))['role'] == 'extreme' else next(iter(models.values()))['role'],
                     'eval_cluster': scenes[sid]['eval_cluster'], 'text': scenes[sid]['text'],
                     'options': scenes[sid]['choice_text'],
                     'option_scores': {o: [float(v) if np.isfinite(v) else None for v in values] for o, values in zip(options, scores[sid])},
                     'models': native}
            stream.write(json.dumps(table, ensure_ascii=False, allow_nan=False) + '\n')
    return path


def analyze(name: str, model_logs: dict, reference: dict, scenes: dict, scores: dict, dimensions: list[dict],
            out: Path, landmark_count: int, replicates: int, selection: dict | None = None,
            role_overrides: dict | None = None, map_options: dict | None = None, include_derived: bool = False) -> dict:
    models = sorted(model_logs)
    matched = sorted(set(reference).intersection(*(set(model_logs[m]) for m in models)))
    assert matched, f"{name}: no DeepSeek overlap"
    maps, question_rows, label_rows, tau_rows, source_info = {}, [], [], [], {}
    for source in sorted({scenes[s]["source"] for s in matched}):
        ids = [s for s in matched if scenes[s]["source"] == source]
        if role_overrides is not None:
            roles = role_overrides[source]
        elif selection is None:
            roles = ({'landmark': [], 'dev': ids, 'control': []} if (map_options or {}).get('kind') == 'quantile'
                     else choose_roles(ids, reference, landmark_count))
        else:
            roles = {role: [q["source_id"] for q in selection["questions"] if q["source"] == source and q["role"] == role] for role in ("landmark", "dev", "control", "extreme")}
            assert all(set(v) <= set(ids) for v in roles.values()), "Missing frozen selected question/draw"
        assert not (set(roles["landmark"]) & (set(roles["dev"]) | set(roles["control"]) | set(roles.get('extreme', []))))
        assert not set(roles["dev"]) & (set(roles["control"]) | set(roles.get('extreme', [])))
        quantile_ids = roles['dev'] + roles.get('extreme', [])
        source_maps = {m: quantile_map(model_logs[m], reference, quantile_ids) if (map_options or {}).get('kind') == 'quantile'
                       else fit_map(model_logs[m], reference, roles["landmark"], **(map_options or {})) for m in models}
        maps[source] = source_maps
        nats = {}
        for sid in ids:
            baseline = normalize(reference[sid])
            for model in models:
                raw = normalize(model_logs[model][sid])
                cal, outside = apply_map(model_logs[model][sid], source_maps[model])
                nats[model, "raw", sid], nats[model, "calibrated", sid] = raw - baseline, cal - baseline
                question_rows.append({"source_id": sid, "source": source, "model": model,
                                      "role": next((r for r in ("landmark", "dev", "control", "extreme") if sid in roles.get(r, [])), "unused"),
                                      'map_kind': source_maps[model]['kind'],
                                      "option_ids": list(scenes[sid]["choice_text"]), "raw_logp": raw.tolist(),
                                      "baseline_logp": baseline.tolist(), "calibrated_logp": cal.tolist(),
                                      "raw_nats": (raw - baseline).tolist(), "calibrated_nats": (cal - baseline).tolist(),
                                      "extrapolated_pairs": outside, "pairs": len(raw) * (len(raw) - 1) // 2})
            for a, b in combinations(["DeepSeek reference"] + models, 2):
                x = reference[sid] if a == "DeepSeek reference" else model_logs[a][sid]
                y = reference[sid] if b == "DeepSeek reference" else model_logs[b][sid]
                tau = float(kendalltau(x, y).statistic)
                tau_rows.append({"source_id": sid, "source": source, "models": [a, b],
                                 "tau_b": tau if np.isfinite(tau) else None,
                                 "undefined_due_to_all_ties": not bool(np.isfinite(tau))})
        source_info[source] = {"matched": len(ids), "roles": roles, "cohorts": {}}
        cohorts = {r: roles[r] for r in ('dev', 'control', 'extreme') if roles.get(r)}
        if roles.get('extreme'):
            cohorts['scored'] = roles['dev'] + roles['extreme']
        for role, cohort_ids in cohorts.items():
            rows, coverage = score_labels(cohort_ids, nats, models, scores, dimensions, scenes, replicates, include_derived)
            label_rows.extend(r | {"source": source, "role": role,
                                        'map_kind': source_maps[r['model']]['kind'] if r['mode'] == 'calibrated' else 'identity'} for r in rows)
            source_info[source]["cohorts"][role] = coverage
    result = {"name": name, "models": models, "sources": source_info, "maps": maps,
              "rows": label_rows, "kendall_tau": tau_rows, "question_nats": question_rows}
    (out / f"{name}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def synthetic_check() -> dict:
    rng = np.random.default_rng(20261026)
    results = []
    for k in (2, 3, 4):
        probabilities = rng.dirichlet(np.ones(k) * 2, size=100)
        smoothed = (100 * probabilities + 1) / (100 + k)
        ref = {str(i): np.log(p) for i, p in enumerate(smoothed)}
        stretched = {s: normalize(2 * p) for s, p in ref.items()}
        mapping = fit_map(stretched, ref, [str(i) for i in range(20)])
        raw, corrected = [], []
        for sid in [str(i) for i in range(20, 100)]:
            cal, _ = apply_map(stretched[sid], mapping)
            raw.extend(np.abs(stretched[sid] - normalize(ref[sid])))
            corrected.extend(np.abs(cal - normalize(ref[sid])))
        assert max(corrected) < 2e-5, (k, max(corrected))
        results.append({"options": k, "landmarks": 20, "held_out": 80,
                        "raw_mean_abs_nats": float(np.mean(raw)), "calibrated_max_abs_nats": float(max(corrected))})
    x = np.array([-.8, -.8, -2., -4.])
    mapping = fit_map({"s": x}, {"s": x}, ["s"])
    identity, _ = apply_map(x, mapping)
    assert np.max(np.abs(identity - normalize(x))) < 1e-12
    permutation = np.array([3, 0, 2, 1])
    permuted, _ = apply_map(x[permutation], mapping)
    assert np.max(np.abs(permuted - identity[permutation])) < 1e-12
    label = np.array([-.6, .1, .7])
    beta = .43
    base = np.array([-.7, -1.2, -2.])
    delta = normalize(base + beta * label) - normalize(base)
    centered = label - label.mean()
    recovered = centered @ delta / (centered @ centered)
    assert abs(recovered - beta) < 1e-12
    regression = regression_check()
    nonlinear = nonlinear_check()
    return {"nonlinear": nonlinear, "temperature_x2": results, "identity_ties_and_permutation": "PASS", "known_beta": beta,
            "production_regression_checks": regression, "recovered_beta": float(recovered),
            "scope": "Temperature stretch is applied AFTER smoothing, without integer rounding. This isolates the calibrator, not quantization or real value differences."}


def nonlinear_check() -> list[dict]:
    rng = np.random.default_rng(20261027)
    results = []
    for k in (2, 3, 4):
        model = {str(i): np.log((100 * p + 1) / (100 + k)) for i, p in enumerate(rng.dirichlet(np.ones(k), 100))}
        target = {s: normalize(2 * np.tanh((x - x.mean()) / 2)) for s, x in model.items()}
        fit = fit_map(model, target, [str(i) for i in range(20)])
        errors, projection_errors = [], []
        for sid in [str(i) for i in range(20, 100)]:
            cal, _ = apply_map(model[sid], fit)
            errors.extend(np.abs(cal - target[sid]))
            desired_pairs = 2 * np.tanh(np.subtract.outer(model[sid], model[sid]) / 2)
            projected = desired_pairs.mean(1)
            projection_errors.extend(np.abs(np.subtract.outer(projected, projected) - desired_pairs).flat)
        results.append({'options': k, 'held_out': 80, 'calibrated_mean_abs_nats_to_coherent_target': float(np.mean(errors)),
                        'calibrated_max_abs_nats_to_coherent_target': float(max(errors)),
                        'max_pair_projection_residual_for_exact_tanh_map': float(max(projection_errors)),
                        'note': 'Coherent target compresses centered option logits with 2*tanh(z/2). Separately, applying exactly 2*tanh(pair_odds/2) produces generally non-transitive targets; row-mean projection cannot satisfy all pair targets for K>2.'})
        if k == 2:
            assert max(projection_errors) < 1e-12
    return results


def regression_check() -> dict:
    dimensions = [{"id": str(i), "family": "virtues", "derived_from": None} for i in range(66)]
    dimensions += [{"id": "constant", "family": "behaviours", "derived_from": None}]
    dimensions += [{"id": axis, "family": "dnd", "derived_from": None} for axis in ("order", "morality")]
    scores, nats = {}, {}
    scenes = {s: {"eval_cluster": s} for s in ("two", "four")}
    for sid, x, beta in (("two", np.array([-1., 1.]), .4), ("four", np.array([-1., 1., -1., 1.]), .8)):
        scores[sid] = np.column_stack([np.tile(x[:, None], (1, 66)), np.zeros(len(x)), x, np.zeros(len(x))])
        for model in ("injected", "null"):
            for mode in ("raw", "calibrated"):
                nats[model, mode, sid] = beta * x + 7 if model == "injected" else np.ones(len(x)) * 5
    scores["four"][:, -2:] = np.nan
    rows, coverage = score_labels(["two", "four"], nats, ["injected", "null"], scores, dimensions, scenes, 100)
    for r in rows:
        value = r["beta_nats_per_label_unit"]
        if r["family"] == "behaviours" or r["dimension"]["id"] == "morality":
            assert value is None
        else:
            expected = 0. if r["model"] == "null" else (.4 if r["family"] == "dnd" else .6)
            assert abs(value - expected) < 1e-12, r
    assert coverage["dnd"]["excluded_missing"] == ["four"]
    perm = np.array([3, 0, 2, 1])
    scores["four"] = scores["four"][perm]
    nats = {key: value[perm] if key[-1] == "four" else value for key, value in nats.items()}
    permuted, _ = score_labels(["two", "four"], nats, ["injected", "null"], scores, dimensions, scenes, 100)
    assert rows == permuted
    higher_variance = {s: x.copy() for s, x in scores.items()}
    higher_variance['four'][:, :66] *= 2
    higher_variance_nats = nats | {('injected', mode, 'four'): .8 * higher_variance['four'][:, 0] + 7 for mode in ('raw', 'calibrated')}
    weighted, _ = score_labels(['two', 'four'], higher_variance_nats, ['injected', 'null'], higher_variance, dimensions, scenes, 100)
    assert all(abs(r['beta_nats_per_label_unit'] - .72) < 1e-12 for r in weighted if r['model'] == 'injected' and r['family'] == 'virtues')
    tie_scene = {'q': {'text': 'Tie fixture', 'source': 'daily_dilemmas', 'choice_text': {'0': 'A', '1': 'B'}}}
    tie_rows = [{'model': 'fixture', 'source_id': 'q', 'source': 'daily_dilemmas', 'draw': d,
                 'original_prompt_sha256': sha_text('Tie fixture'), 'original_probabilities': {'0': p, '1': 100-p}}
                for d, p in enumerate([75, 30, 70, 25])]
    tie, missing = pooled_logs(tie_rows, tie_scene, 4)
    assert not missing and tie['fixture', 'q'][0] == tie['fixture', 'q'][1]
    return {"exact_permuted_draw_ties": 'PASS: permuting identical integer-percent draw multisets leaves pooled options exactly tied',
            "option_count_normalization": "PASS: equal-variance 2-option beta .4 and 4-option beta .8 pool to .6, not option-weighted .6667",
            "label_variance_weighting": "PASS: raising the second menu variance fourfold gives pooled beta .72, not equal-slope mean .6",
            "constant_question_offsets": "PASS: +7/+5 nats cancel; null model beta 0",
            "constant_scores": "PASS: unidentified coefficient and bootstrap stay null",
            "missing_dnd": "PASS: missing offered-action labels excluded, not zero",
            "permutation": "PASS: every coefficient and interval identical after option permutation"}


def draw_noise_analysis(comparison_rows: list[dict], reference_rows: list[dict], result: dict,
                        scenes: dict, replicates: int, extreme_role: str) -> dict:
    grouped = defaultdict(list)
    for r in reference_rows + [r for r in comparison_rows if r['model'] != REFERENCE]:
        grouped[r['model'], r['source_id']].append(r)
    models = [REFERENCE] + result['models']
    roles = {q['source_id']: q['role'] for q in result['question_nats']}
    questions = []
    for sid in sorted(roles):
        logs = {}
        options = list(scenes[sid]['choice_text'])
        for model in models:
            rows = grouped[model, sid]
            assert len(rows) >= 2, 'Draw-noise estimation needs at least two draws per model'
            logs[model] = np.log((np.array([[r['original_probabilities'][o] for o in options] for r in rows]) + 1) / (100 + len(options)))
        for name, subset in (('all_models', models), ('comparisons_only', result['models'])):
            means = np.stack([logs[m].mean(0) for m in subset])
            between = float(means.var(0, ddof=1).mean())
            mean_noise = float(np.mean([logs[m].var(0, ddof=1).mean() / len(logs[m]) for m in subset]))
            questions.append({'source_id': sid, 'source': scenes[sid]['source'],
                              'role': extreme_role if roles[sid] == 'extreme' else roles[sid],
                              'model_group': name, 'draws': {m: len(logs[m]) for m in subset},
                              'between_model_mean_logp_variance': between, 'mean_draw_noise_variance': mean_noise,
                              'between_over_draw_noise': between / mean_noise if mean_noise > 0 else None,
                              'zero_noise_status': None if mean_noise > 0 else 'positive_between_zero_noise' if between > 0 else 'both_zero'})
    summaries = []
    for source, role, model_group in sorted({(q['source'], q['role'], q['model_group']) for q in questions}):
        qs = [q for q in questions if (q['source'], q['role'], q['model_group']) == (source, role, model_group)]
        values = np.array([[q['between_model_mean_logp_variance'], q['mean_draw_noise_variance']] for q in qs])
        groups = [scenes[q['source_id']]['eval_cluster'] for q in qs]
        samples = bootstrap(values[:, None, :], groups, replicates)[:, 0]
        valid = samples[:, 1] > 0
        ratios = samples[valid, 0] / samples[valid, 1]
        point = values[:, 0].mean() / values[:, 1].mean() if values[:, 1].mean() > 0 else None
        finite_question_ratios = [q['between_over_draw_noise'] for q in qs if q['between_over_draw_noise'] is not None]
        summaries.append({'source': source, 'role': role, 'model_group': model_group, 'questions': len(qs),
                          'pooled_between_over_draw_noise': float(point) if point is not None else None,
                          'ci95': np.quantile(ratios, [.025, .975]).tolist() if len(ratios) else None,
                          'median_finite_question_ratio': float(np.median(finite_question_ratios)) if finite_question_ratios else None,
                          'zero_noise_questions': sum(q['zero_noise_status'] is not None for q in qs)})
    return {'author': 'PI/OpenAI', 'questions': questions, 'summaries': summaries,
            'method': 'Per model/question: mean add-one-smoothed log p across draws, before geometric-pool renormalization. Between = sample variance across model means, averaged over options. Noise of each model mean = within-model sample draw variance / its draw count, then averaged across options/models. Ratio 1 is the noise-only expectation under independent identical model distributions; ratio includes finite-sample noise and is not a significance test. Zero denominators stay null with explicit status. Pooled source/role ratio divides mean between variance by mean noise; intervals resample games/topics with fixed draws.'}


def export_scored_rows(results: list[dict], out: Path, dimensions: list[dict], extreme_role: str, primary_mode: str,
                       control_draws_a: list[int], control_draws_b: list[int]) -> None:
    comparisons = [r for r in results if r['name'] == 'comparison']
    if not comparisons:
        return
    main = comparisons[0]
    kinds = {m['kind'] for source in main['maps'].values() for m in source.values()}
    assert len(kinds) == 1
    kind = kinds.pop()
    for mode, filename, self_name in (('raw', 'scored_identity.json', 'self_identity'),
                                      ('calibrated', 'scored_quantile.json' if kind == 'quantile' else f'scored_calibrated_{kind}.json', 'self_quantile' if kind == 'quantile' else None)):
        rows = []
        for result in [main] + [r for r in results if r['name'] == self_name]:
            for r in result['rows']:
                if r['mode'] != mode or r['role'] not in ('dev', 'control', 'extreme', 'landmark'):
                    continue
                noise_floor = result is not main
                rows.append(r | {'model': 'DeepSeek self' if noise_floor else r['model'],
                                 'role': extreme_role if r['role'] == 'extreme' else r['role'],
                                 'derived': r['dimension']['derived_from'] is not None, 'noise_floor': noise_floor,
                                 'noise_floor_draws': {'model': control_draws_a, 'reference': control_draws_b} if noise_floor else None})
        source_metadata = {s: info | {'roles': {(extreme_role if role == 'extreme' else role): ids for role, ids in info['roles'].items()},
                                      'cohorts': {(extreme_role if role == 'extreme' else role): value for role, value in info['cohorts'].items()}}
                           for s, info in main['sources'].items()}
        artifact = {'author': 'PI/OpenAI', 'mode': mode, 'primary': mode == primary_mode, 'dimensions': dimensions,
                    'sources': source_metadata, 'rows': rows,
                    'beta_note': 'Pooled within-question slope: sum_q mean_options(centered label * centered nats) / sum_q mean_options(centered label squared). Menus with larger label variance weigh more. Each label fit is marginal, not a unique causal virtue effect.',
                    'noise_floor_note': 'DeepSeek self pools independent draw groups, not a zero-valued reference. Its two-draw-vs-two-draw noise is larger than would be expected for independent four-draw-vs-four-draw means. CIs fix maps and replies; derived virtues repeat their parent and are not independent evidence.'}
        (out / filename).write_text(json.dumps(artifact, indent=2, allow_nan=False) + '\n')


def plot_maps(result: dict, out: Path) -> Path:
    sources = list(result["maps"])
    models = result["models"]
    fig, axes = plt.subplots(len(sources), len(models), figsize=(max(6, 3.5 * len(models)), 3.4 * len(sources)), squeeze=False, constrained_layout=True)
    for i, source in enumerate(sources):
        for j, model in enumerate(models):
            ax = axes[i, j]
            mapping = result["maps"][source][model]
            max_obs = max(abs(r) for q in result["question_nats"] if q["source"] == source and q["model"] == model for r in (np.subtract.outer(q["raw_logp"], q["raw_logp"])).flat)
            end = max(mapping["x"][-1] * 1.15, max_obs, np.log(101))
            grid = np.linspace(0, end, 300)
            y = np.interp(grid, mapping["x"], mapping["y"])
            beyond = grid > mapping["x"][-1]
            y[beyond] = mapping["y"][-1] + (grid[beyond] - mapping["x"][-1]) * mapping["end_slope"]
            ax.plot(grid, grid, color=".75", linewidth=.8, label="Identity")
            pair_label = 'Matched question pairs' if mapping['kind'] == 'quantile' else 'Landmark option pairs'
            ax.scatter([p["x"] for p in mapping["points"]], [p["y"] for p in mapping["points"]], s=9, alpha=.25, color="#246891", label=pair_label)
            if mapping['kind'] == 'quantile':
                ax.scatter(mapping['model_quantiles'], mapping['reference_quantiles'], marker='D', s=28, color='#762a83', label='Unpaired quantile targets', zorder=4)
            ax.plot(grid[~beyond], y[~beyond], color="#b64b35", label="Fitted map")
            ax.plot(grid[beyond], y[beyond], color="#b64b35", linestyle="--", label="Extrapolation")
            ax.axvline(mapping["x"][-1], color=".5", linewidth=.7, linestyle=":", label='Last fit knot')
            ax.axvline(np.log(101), color='.65', linewidth=.7, linestyle='-.', label='Integer % edge: ln(101)')
            count = ('Identity: no fit or landmarks' if mapping['kind'] == 'identity'
                     else f"{len(mapping['fit_ids'])} scored questions; no landmarks" if mapping['kind'] == 'quantile'
                     else f"{len(mapping['landmark_ids'])} landmark questions")
            edge_note = '\nReference p95 = 100/0 integer edge' if mapping.get('reference_p95_at_integer_edge', False) else ''
            ax.set_title(model.split("/")[-1].replace('self_', '').replace('_', ' ') + f"\n{count}\nDots = option pairs\n" + source.replace("_", " ") + edge_note, fontsize=8)
            control = result['name'].startswith('self_')
            ax.set_xlabel((result.get('control_side_labels', ('DS draw 0', 'DS draw 1'))[0] if control else 'Model') + ' |log option odds| (nats)')
            ax.set_ylabel((result.get('control_side_labels', ('DS draw 0', 'DS draw 1'))[1] if control else 'DeepSeek') + ' |log option odds| (nats)')
            ax.spines[["top", "right"]].set_visible(False)
    legend = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    fig.legend(list(legend.values()), list(legend), loc="outside lower center", ncols=4, frameon=False, fontsize=8)
    toy = ' (two-landmark toy, not validation)' if result['name'] == 'comparison' and any(m['kind'] != 'quantile' and 0 < len(m['landmark_ids']) <= 2 for source in result['maps'].values() for m in source.values()) else ''
    fig.suptitle(result["name"] + ": confidence maps, positive half" + toy, fontsize=11)
    path = out / f"{result['name']}_maps.png"
    fig.canvas.draw()
    bounds = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    assert bounds.x0 >= -.02 and bounds.y0 >= -.02 and bounds.x1 <= width + .02 and bounds.y1 <= height + .02, 'Clipped figure text'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(args: argparse.Namespace) -> None:
    start = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, format="{message}", level="INFO")
    logger.add(args.out / "run.md", format="{message}", level="DEBUG", mode="w")
    logger.info("# Offline probability calibration\n\nAuthor: PI/OpenAI\n")
    logger.info("```json\n" + json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2) + "\n```\n")
    logger.info("SHOULD: an odd strictly increasing pairwise-odds map preserves every option ordering and tie; identity calibration changes nothing.\n\nSHOULD: a synthetic x2 temperature stretch calibrates to approximately zero held-out nats for 2, 3 and 4 options.\n")
    synthetic = synthetic_check()
    logger.info("## Synthetic checks\n\n" + tabulate([[r['options'], f"{r['calibrated_max_abs_nats']:.2g}", f"{r['raw_mean_abs_nats']:.3f}"] for r in synthetic['temperature_x2']], headers=["Options", "Calibrated max abs nats→0", "Raw mean abs nats"], tablefmt="pipe") + "\n")
    scenes = {r["source_id"]: r for r in map(json.loads, args.scenes.open())}
    metadata = json.loads((args.labels.parent / "analysis.json").read_text())
    dimensions = metadata["dimensions"]
    canonical_scenes_path = Path(metadata['inputs']['run']) / 'scenes.jsonl'
    assert sha(canonical_scenes_path) == metadata['input_sha256']['scenes'], 'Changed canonical label scenes'
    canonical_scenes = {r['source_id']: r for r in map(json.loads, canonical_scenes_path.open())}
    for sid, scene in scenes.items():
        assert all(scene[key] == canonical_scenes[sid][key] for key in ('text', 'choice_text', 'source', 'eval_cluster')), f'Changed labelled scene: {sid}'
    label_rows = pq.read_table(args.labels).to_pylist()
    labels = {(r["source_id"], r["choice_id"]): r["scores"] for r in label_rows}
    assert len(labels) == len(label_rows), "Duplicate action label"
    scores = {s: np.array([labels[s, c] for c in scene["choice_text"]]) for s, scene in scenes.items()}
    reference_rows, ref_status = read_rows(args.reference, args.reference_level, REFERENCE)
    assert {r["model"] for r in reference_rows} == {REFERENCE}
    reference_pooled, ref_incomplete = pooled_logs(reference_rows, scenes, args.reference_draws)
    reference = {sid: p for (model, sid), p in reference_pooled.items()}
    results = []
    noise_result = None
    comparison_status, comparison_incomplete = {}, []
    comparison_rows = []
    if args.comparisons:
        comparison_rows, comparison_status = read_rows(args.comparisons, args.level)
        pooled, comparison_incomplete = pooled_logs(comparison_rows, scenes, args.draws)
        comparison = {m: {s: p for (model, s), p in pooled.items() if model == m} for m in sorted({r["model"] for r in comparison_rows} - {REFERENCE})}
        selection = read_selection(args.selection, args.extreme_role, args.exclude_ids) if args.selection else None
        results.append(analyze("comparison", comparison, reference, scenes, scores, dimensions, args.out, args.landmarks, args.replicates, selection,
                               map_options={"kind": args.map_kind, "shrinkage": args.shrinkage, "extrapolation": args.extrapolation}, include_derived=args.all_virtues))
        write_question_tables(results[-1], comparison_rows, reference_rows, scenes, scores, dimensions, args.out, args.extreme_role)
        noise_result = draw_noise_analysis(comparison_rows, reference_rows, results[-1], scenes, args.replicates, args.extreme_role)
        (args.out / 'draw_noise.json').write_text(json.dumps(noise_result, indent=2, allow_nan=False) + '\n')
    control_rows, control_status = read_rows(args.control_reference, args.control_level, model=REFERENCE)
    assert {r['model'] for r in control_rows} == {REFERENCE}
    assert not set(args.control_draws_a) & set(args.control_draws_b), 'Self-control draws must be disjoint'
    draw_logs, control_incomplete = [], []
    for draws in (args.control_draws_a, args.control_draws_b):
        index = {draw: i for i, draw in enumerate(draws)}
        one, missing = pooled_logs([r | {'draw': index[r['draw']]} for r in control_rows if r['draw'] in index], scenes, len(draws))
        draw_logs.append({s: p for (_, s), p in one.items()})
        control_incomplete.append(missing)
    common = set(draw_logs[0]) & set(draw_logs[1])
    control_model = {"DeepSeek model-side": {s: draw_logs[0][s] for s in common}}
    control_base = {s: draw_logs[1][s] for s in common}
    reserved_roles = {}
    for source in sorted({scenes[s]['source'] for s in common}):
        ids = sorted(s for s in common if scenes[s]['source'] == source)
        if args.control_selection:
            frozen = read_selection(args.control_selection, args.extreme_role, args.exclude_ids)
            reserved_roles[source] = {role: [q['source_id'] for q in frozen['questions'] if q['source'] == source and q['role'] == role] for role in ('landmark', 'dev', 'control', 'extreme')}
            assert all(set(v) <= common for v in reserved_roles[source].values()), 'Missing frozen control question/draws'
        else:
            reserved_roles[source] = choose_roles(ids, control_base, 100)
    variants = [('identity', 20, {'kind': 'identity'}),
                ('temperature20', 20, {'kind': 'temperature'}),
                ('temperature100', 100, {'kind': 'temperature'}),
                ('orthogonal20', 20, {'kind': 'orthogonal'}),
                ('orthogonal100', 100, {'kind': 'orthogonal'}),
                ('quantile', 0, {'kind': 'quantile'}),
                ('shrunk_monotone20', 20, {'kind': 'monotone', 'shrinkage': .5, 'extrapolation': 'unit'}),
                ('shrunk_monotone100', 100, {'kind': 'monotone', 'shrinkage': .5, 'extrapolation': 'unit'}),
                ('flexible20', 20, {'kind': 'monotone'})]
    if args.control_selection:
        variants = [('identity', 0, {'kind': 'identity'}), ('quantile', 0, {'kind': 'quantile'})]
    for name, n, options in variants:
        roles = {}
        for source, reserve in reserved_roles.items():
            landmarks = [] if n == 0 else reserve['landmark'] if n == 100 else choose_roles(reserve['landmark'], control_base, n)['landmark']
            roles[source] = reserve | {'landmark': landmarks}
        results.append(analyze('self_' + name, control_model, control_base, scenes, scores, dimensions, args.out,
                               n, args.replicates, role_overrides=roles, map_options=options, include_derived=args.all_virtues))
    symmetric_selection_reference = {s: np.logaddexp(draw_logs[0][s], draw_logs[1][s]) - np.log(2) for s in common}
    scored_roles = {source: uncertain_with_extremes(reserve['dev'], symmetric_selection_reference, scenes) for source, reserve in reserved_roles.items()} if not args.control_selection else {}
    scored_variants = [('identity', {'kind': 'identity'}), ('shrunk100', {'kind': 'monotone', 'shrinkage': .5, 'extrapolation': 'unit'}), ('quantile', {'kind': 'quantile'})] if scored_roles else []
    for name, options in scored_variants:
        roles = {source: r | {'landmark': reserved_roles[source]['landmark'] if name != 'quantile' else []} for source, r in scored_roles.items()}
        results.append(analyze('scored_' + name, control_model, control_base, scenes, scores, dimensions, args.out,
                               100, args.replicates, role_overrides=roles, map_options=options, include_derived=args.all_virtues))
    inputs = args.reference + args.comparisons + args.control_reference + [args.labels, args.labels.parent / "analysis.json", args.scenes, canonical_scenes_path, Path(__file__), ROOT / 'scripts/evaluation/score_choice_dimensions.py', ROOT / 'uv.lock'] + ([args.selection] if args.selection else []) + ([args.control_selection] if args.control_selection else []) + ([args.exclude_ids] if args.exclude_ids else [])
    self_diagnostics = []
    control_results = [r for r in results if r['name'].startswith(('self_', 'scored_'))]
    for r in control_results:
        for source in reserved_roles:
            qs = [q for q in r['question_nats'] if q['source'] == source and q['role'] in ('dev', 'extreme')]
            changes = np.array([np.mean(np.abs(q['calibrated_nats'])) - np.mean(np.abs(q['raw_nats'])) for q in qs])
            groups = [scenes[q['source_id']]['eval_cluster'] for q in qs]
            intervals = np.quantile(bootstrap(changes[:, None, None], groups, args.replicates)[:, 0, 0], [.025, .975])
            dev = [q for q in qs if q['role'] == 'dev']
            dev_changes = np.array([np.mean(np.abs(q['calibrated_nats'])) - np.mean(np.abs(q['raw_nats'])) for q in dev])
            dev_groups = [scenes[q['source_id']]['eval_cluster'] for q in dev]
            dev_interval = np.quantile(bootstrap(dev_changes[:, None, None], dev_groups, args.replicates)[:, 0, 0], [.025, .975])
            self_diagnostics.append({'variant': r['name'], 'source': source, 'questions': len(qs),
                                     'cohort': 'uncertain_dev_plus_20_extremes' if r['name'].startswith('scored_') else 'full_fixed_heldout',
                                     'dev_only_questions': sum(q['role'] == 'dev' for q in qs),
                                     'dev_only_change_ci95': dev_interval.tolist(),
                                     'dev_only_raw_abs_nats': float(np.mean([np.mean(np.abs(q['raw_nats'])) for q in qs if q['role'] == 'dev'])),
                                     'dev_only_calibrated_abs_nats': float(np.mean([np.mean(np.abs(q['calibrated_nats'])) for q in qs if q['role'] == 'dev'])),
                                     'raw_mean_abs_nats': float(np.mean([np.mean(np.abs(q['raw_nats'])) for q in qs])),
                                     'calibrated_mean_abs_nats': float(np.mean([np.mean(np.abs(q['calibrated_nats'])) for q in qs])),
                                     'change': float(changes.mean()), 'change_ci95': intervals.tolist(),
                                     'raw_centered_nats_dot_baseline': float(np.mean([np.mean((np.array(q['raw_nats']) - np.mean(q['raw_nats'])) * (np.array(q['baseline_logp']) - np.mean(q['baseline_logp']))) for q in qs])),
                                     'calibrated_centered_nats_dot_baseline': float(np.mean([np.mean((np.array(q['calibrated_nats']) - np.mean(q['calibrated_nats'])) * (np.array(q['baseline_logp']) - np.mean(q['baseline_logp']))) for q in qs])),
                                     'does_not_increase_point_self_noise': bool(changes.mean() <= 1e-12)})
    clean_control = bool(args.control_selection and args.control_level == '10k' and len(args.control_draws_a) == len(args.control_draws_b) == 2)
    gate_rows = [r for r in self_diagnostics if r['variant'] == 'self_quantile']
    gate = {r['source']: r['dev_only_calibrated_abs_nats'] <= r['dev_only_raw_abs_nats'] + 1e-12 for r in gate_rows}
    primary_mode = 'calibrated' if not args.identity_primary and clean_control and args.map_kind == 'quantile' and len(gate) == 2 and all(gate.values()) else 'raw'
    export_scored_rows(results, args.out, dimensions, args.extreme_role, primary_mode, args.control_draws_a, args.control_draws_b)
    original_variants = [r for r in control_results if r['name'].startswith('self_') and r['name'] not in ('self_orthogonal20', 'self_orthogonal100', 'self_quantile')]
    combined = {'name': 'self_variants', 'models': [r['name'] for r in original_variants], 'maps': {}, 'question_nats': []}
    for source in reserved_roles:
        combined['maps'][source] = {r['name']: r['maps'][source]['DeepSeek model-side'] for r in original_variants}
    for r in control_results:
        combined['question_nats'].extend(q | {'model': r['name']} for q in r['question_nats'])
    plots = [str(plot_maps(r, args.out)) for r in results if not r['name'].startswith(('self_', 'scored_'))]
    side_labels = tuple('DS draws ' + '+'.join(map(str, d)) for d in (args.control_draws_a, args.control_draws_b))
    combined['control_side_labels'] = side_labels
    plots.append(str(plot_maps(combined, args.out)))
    alternatives = [r for r in control_results if r['name'] in ('self_identity', 'self_orthogonal20', 'self_orthogonal100', 'self_quantile', 'self_shrunk_monotone100')]
    alternative_plot = {'name': 'self_alternatives', 'models': [r['name'] for r in alternatives], 'maps': {}, 'question_nats': combined['question_nats']}
    for source in reserved_roles:
        alternative_plot['maps'][source] = {r['name']: r['maps'][source]['DeepSeek model-side'] for r in alternatives}
    alternative_plot['control_side_labels'] = side_labels
    plots.append(str(plot_maps(alternative_plot, args.out)))
    scored_plot = {'name': 'self_scored_domain', 'models': [r['name'] for r in control_results if r['name'].startswith('scored_')], 'maps': {}, 'question_nats': combined['question_nats']}
    for source in reserved_roles:
        scored_plot['maps'][source] = {r['name']: r['maps'][source]['DeepSeek model-side'] for r in control_results if r['name'].startswith('scored_')}
    if scored_plot['models']:
        scored_plot['control_side_labels'] = side_labels
        plots.append(str(plot_maps(scored_plot, args.out)))
    summary = {"author": "PI/OpenAI", "synthetic": synthetic, "reference_status": ref_status,
               "reference_incomplete_questions": ref_incomplete, "comparison_status": comparison_status,
               "comparison_incomplete_questions": comparison_incomplete,
               "control_condition_levels": sorted({str(r.get('level')) for r in control_rows}),
               "input_sha256": {str(p): sha(p) for p in inputs},
               "plots": plots, "replicates": args.replicates, "control_status": control_status,
               "control_draws": {'model_side': args.control_draws_a, 'reference_side': args.control_draws_b, 'incomplete_groups': control_incomplete, 'frozen_selection': str(args.control_selection), 'level': args.control_level},
               "scored_domain_selection": 'Symmetric mean of model/reference-side probabilities for exploratory control selection, or frozen prior-selected IDs when --control-selection is supplied.',
               "versions": {'numpy': np.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
               "self_control_diagnostics": self_diagnostics,
               "primary_mode": primary_mode, 'identity_primary_requested': args.identity_primary,
               'all_virtues_included': args.all_virtues,
               'label_family_counts': dict(Counter(d['family'] for d in dimensions if args.all_virtues or d['derived_from'] is None)),
               'between_over_draw_noise': noise_result['summaries'] if noise_result else [],
               "quantile_clean_dev_gate": {'clean_same_budget_two_vs_two': clean_control, 'per_source': gate,
                                           'rule': 'Calibrated primary only for quantile, frozen prior-selected IDs, same 10k level, independent two-draw groups, and no dev-only mean absolute self-noise increase in either source. This is a point-estimate gate, not proof of non-inferiority. Raw identity is always reported.'},
               "self_control_split": 'Frozen prior-selected dev/control/extreme IDs; quantile fits only dev+extreme distributions, identity fits nothing.' if args.control_selection else 'Reserve 100 confidence-spanning landmark questions/source; 20 is a spanning subset. Full-heldout variants share exactly the same held-out IDs. Scored-domain variants use an uncertainty-selected subset plus 20 extremes, selected symmetrically from both control sides.',
               "comparison_map_options": {"kind": args.map_kind, "shrinkage": args.shrinkage, "extrapolation": args.extrapolation},
               "method": "Average add-one-smoothed log p over draws; softmax gives geometric probability pool. Fit odd increasing isotonic-interpolated pairwise log-odds magnitude map, equal total weight per landmark question. Labels never enter calibration. Apply per source; mean mapped pair odds then softmax. Per-option nats = calibrated log p minus normalized DeepSeek pooled log p. Raw uses the same pooling. Each label is a separate marginal within-question regression: beta = sum_q mean_options(x_centered * nats_centered) / sum_q mean_options(x_centered squared). No question-variance normalization. Each question contributes mean option covariance and mean option variance. The final pooled within-question slope weights menus with larger label variance more; it is not an equal-weight average of per-question slopes. Vice polarity follows the named trait in source dimension metadata, not favorable polarity.",
               "uncertainty": "Game/topic-cluster bootstrap of held-out question sufficient statistics, reusing score_choice_dimensions.bootstrap. Draws and fitted landmark maps fixed; excludes calibration-fit uncertainty, label error, fresh-generation uncertainty, selection uncertainty and model-provider changes. Undefined coefficient/replicate remains null, never zero.",
               "limitations": ["Noisy-regressor attenuation: paired y-on-x landmark fits can shrink odds toward uniform and add an opposition-to-DeepSeek component to label slopes; identity shrinkage reduces but does not eliminate this concern.", "Confidence magnitude and genuine preference strength are not identifiable from different models' landmark answers; even disjoint landmarks can remove real preferences.", "Sketch landmarks use matched questions and are not a substitute for the frozen project-specific dev/control/landmark set.", "Sources remain separate; all label fits are marginal correlated-label associations, not unique causal virtue effects.", "Failed replies and incomplete draw groups are reported/excluded explicitly; no score is substituted.", "Kendall tau-b compares within-question option ranks, not between-model moral scores; all-tied ranks are undefined.",
                                "Quantile matching is transductive: it uses evaluated probability distributions themselves and removes any real overall decisiveness difference as well as confidence scale. It preserves direction, not preference strength.",
                                "Orthogonal/Deming through-origin fit assumes equal x/y measurement error variances; draw-to-draw noise need not satisfy this across different models.",
                                "Nonlinear mapped pair targets need not be transitive; row-mean projection preserves option order but generally cannot reproduce every fitted pair target for 3+ options.",
                                "A p95 knot at ln(101) is an integer-percentage quantisation boundary (100/0 answers), potentially driven by added extremes, not an estimate of a continuous latent confidence quantile."]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if comparison_rows:
        demo = comparison_rows[0]
        logger.debug("## First normal-path comparison row (file order, not selected for effect)\n\n" + demo['trace'] + "\n\n```json\n" + json.dumps(demo, ensure_ascii=False, indent=2) + "\n```\n")
    if noise_result:
        logger.info('## Between-model variance / draw noise\n\n' + noise_result['method'] + '\n\n' + tabulate([[r['source'], r['role'], r['model_group'], r['pooled_between_over_draw_noise'], r['ci95'], r['questions']] for r in noise_result['summaries']], headers=['Source', 'Role', 'Models', 'Pooled ratio↑', '95% interval', 'Questions'], tablefmt='pipe', floatfmt='.2f') + '\n')
    logger.info(f"## Primary score mode\n\n{primary_mode}: raw means identity/no calibration; calibrated means quantile. Gate: {summary['quantile_clean_dev_gate']}. Both versions remain in all label rows.\n")
    logger.info("## DeepSeek self-control: paired question comparisons\n\n" + summary['self_control_split'] + " Shrunk maps use 50% identity and unit-slope extrapolation. Changes are paired game/topic-bootstrap intervals, with fitted maps fixed.\n\n" + tabulate([[r['variant'], r['source'], f"{r['change']:+.3f}", str([round(x, 3) for x in r['change_ci95']]), r['does_not_increase_point_self_noise']] for r in self_diagnostics], headers=['Variant', 'Source', 'Change in abs nats↓', '95% interval', 'No point increase'], tablefmt='pipe') + "\n")
    logger.info("## Dev-only self-control changes\n\n" + tabulate([[r['variant'], r['source'], r['dev_only_calibrated_abs_nats'] - r['dev_only_raw_abs_nats'], r['dev_only_change_ci95'], r['dev_only_questions']] for r in self_diagnostics if r['variant'].startswith('scored_') or args.control_selection], headers=['Variant', 'Source', 'Dev-only abs nats change↓', '95% interval', 'Dev questions'], tablefmt='pipe', floatfmt='.4f') + "\n")
    logger.info("## Map direction and quantization checks\n\nAll maps retain signs/ties. Dots represent option pairs, not independent questions. The dash-dot ln(101) boundary is the maximum pairwise odds magnitude for add-one-smoothed integer percentages, including a 100/0 answer; repeated extremes create quantization spikes. Quantile targets use equal total question weight, preserve an exact origin, and use unit-slope continuation above p95.\n\n" + tabulate([[r['name'], source, model.split('/')[-1], m['direction_disagreement_fraction']] for r in results for source, source_maps in r['maps'].items() for model, m in source_maps.items()], headers=['Run', 'Source', 'Model', 'Question-weighted direction disagreement'], tablefmt='pipe', floatfmt='.3f') + "\n")
    logger.info("## Nonlinear synthetic and attenuation diagnostics\n\n" + json.dumps(synthetic['nonlinear'], indent=2) + "\n\n" + tabulate([[r['variant'], r['source'], r['raw_centered_nats_dot_baseline'], r['calibrated_centered_nats_dot_baseline']] for r in self_diagnostics], headers=['Variant', 'Source', 'Raw centered nats dot baseline', 'Calibrated centered nats dot baseline'], tablefmt='pipe', floatfmt='.4f') + "\n\nNegative alignment can already occur at identity because the baseline itself is noisy; a more negative calibrated contrast can add opposition to DeepSeek\'s preferences. Per-map question-weighted direction-disagreement fractions are saved in self_*.json and comparison.json.\n")
    logger.info("## Production regression controls\n\n" + "\n".join(synthetic['production_regression_checks'].values()) + "\n")
    for result in results:
        for source, info in result['sources'].items():
            if result['name'] != 'comparison':
                continue
            sample = [r for r in result['rows'] if r['source'] == source and r['mode'] == 'calibrated' and r['family'] == 'virtues' and r['dimension']['id'] == next(r['dimension']['id'] for r in result['rows'] if r['family'] == 'virtues')]
            rendered = [[r['model'].split('/')[-1], r['beta_nats_per_label_unit'], r['ci95'], r['informative_questions'], r['unidentified_bootstrap_replicates']] for r in sample]
            logger.info(f"## Example label fit: {source}\n\nFirst direct virtue in stored dimension order (not chosen for effect): {sample[0]['dimension']['id']}. Positive means more stated preference for the named trait than DeepSeek. This is a calibration diagnostic, not a validated model ranking.\n\n" + tabulate(rendered, headers=['Model', 'Nats / label unit', '95% interval', 'Informative questions', 'Unidentified resamples'], tablefmt='pipe', floatfmt='.3f') + "\n")
            tau = [r for r in result['kendall_tau'] if r['source'] == source and r['source_id'] in info['roles']['dev']]
            valid = [r['tau_b'] for r in tau if r['tau_b'] is not None]
            logger.info(f"Kendall tau-b: median {np.median(valid):.3f} over {len(valid)} defined held-out model-pair/question ranks; {len(tau)-len(valid)} undefined due to all ties. See comparison.json for every pair/question.\n")
    logger.info("## Method and limitations\n\n" + summary['method'] + "\n\n" + summary['uncertainty'] + "\n\n" + "\n".join("- " + x for x in summary['limitations']) + "\n")
    logger.info(f"Reference statuses: {ref_status}; incomplete reference groups: {len(ref_incomplete)}. Comparison statuses: {comparison_status}; incomplete groups: {len(comparison_incomplete)}.\n")
    table = []
    for result in results:
        for source, info in result["sources"].items():
            heldout = [q for q in result['question_nats'] if q['source'] == source and q['role'] == 'dev']
            for model in result['models']:
                qs = [q for q in heldout if q['model'] == model]
                table.append([result['name'] + '/' + source + '/' + model.split('/')[-1],
                              f"{np.mean([np.mean(np.abs(q['calibrated_nats'])) for q in qs]):.3f}",
                              f"{np.mean([np.mean(np.abs(q['raw_nats'])) for q in qs]):.3f}",
                              len(info['roles']['landmark']), len(qs), sum(q['extrapolated_pairs'] for q in qs)])
    logger.info("## Scored question diagnostics\n\nMean absolute nats is a size diagnostic, not a score to optimize; only synthetic/control contrasts have a zero target. Landmark fits and bootstrap label rows: comparison.json and self_*.json.\n\n" + tabulate(table, headers=["Run / source / model", "Calibrated abs nats", "Raw abs nats", "Landmarks", "Dev", "Extrapolated pairs"], tablefmt="pipe") + "\n")
    logger.info(f"All option orders preserved; synthetic and known-coefficient controls passed. Runtime {time.monotonic() - start:.1f}s; API calls 0.\n\n{args.out / 'run.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs="+", default=[ROOT / "data/probability_tables/deepseek_v4.1_flash_reference_medium_2draws.jsonl.gz"])
    parser.add_argument("--comparisons", type=Path, nargs="*", default=[ROOT / "outputs/20261026_tiny_eval_trial10" / stage / "choices.jsonl" for stage in ("smoke", "trial")])
    parser.add_argument("--control-reference", type=Path, nargs="+", default=[ROOT / "data/probability_tables/deepseek_v4.1_flash_reference_medium_2draws.jsonl.gz"])
    parser.add_argument('--control-level')
    parser.add_argument('--control-draws-a', type=int, nargs='+', default=[0])
    parser.add_argument('--control-draws-b', type=int, nargs='+', default=[1])
    parser.add_argument('--control-selection', type=Path)
    parser.add_argument("--map-kind", choices=("identity", "temperature", "orthogonal", "monotone", "quantile"), default="quantile")
    parser.add_argument("--shrinkage", type=float, default=.5)
    parser.add_argument("--extrapolation", choices=("unit", "last"), default="unit")
    parser.add_argument("--reference-level")
    parser.add_argument("--level")
    parser.add_argument("--reference-draws", type=int, default=2)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--landmarks", type=int, default=100)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument('--all-virtues', action='store_true', help='Include all80virtues and preserve derived_from metadata')
    parser.add_argument('--identity-primary', action='store_true', help='Keep raw identity primary regardless of the exploratory quantile gate')
    parser.add_argument("--selection", type=Path)
    parser.add_argument('--extreme-role', choices=('extreme', 'landmark'), default='extreme')
    parser.add_argument('--exclude-ids', type=Path)
    parser.add_argument("--scenes", type=Path, default=ROOT / "outputs/20261024_action_preferences_resumed/scenes.jsonl")
    parser.add_argument("--labels", type=Path, default=ROOT / "outputs/20261025_individual_dimensions/action_scores.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/20261026_calibration_sketch")
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception:
        logger.error("## FAILED\n\n```\n" + traceback.format_exc() + "```\n")
        raise
