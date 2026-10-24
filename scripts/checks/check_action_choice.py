"""Unpaid blinding, option mapping and parser regressions. -- PI/Sol"""

import asyncio
import base64
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import httpx
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
from action_choice import MODEL_PATH, ROOT, build_packet, embedded_overlay, parse_receipt, render, select_scenes, scene_spread, build_scope_packet, repeat_summary
from score_action_choice import uniform_baseline, cluster_interval, paired_summary, position_baseline
import eval_api
from eval_api import api_key, transient_provider_error
from billing_reservations import outstanding, unknown_fees, noncash_reviews


def main() -> None:
    config = json.loads(MODEL_PATH.read_text())
    model = next(iter(config["models"]))
    row = select_scenes(ROOT / "outputs/20261024_release21/parquet/train.parquet",
                        ROOT / "outputs/20261023_virtues_v2_contrast_2500_resumed/virtues.parquet", 1, 20261023, source="machiavelli")[0]
    forward = build_packet(row, "forward", model, config)
    reverse = build_packet(row, "reverse", model, config)
    contaminated = row | {"choice_virtue_labels": "SENTINEL_HIDDEN_VIRTUE_LABEL", "choice_labels": "SENTINEL_HIDDEN_OUTCOME"}
    assert build_packet(contaminated, "forward", model, config) == forward
    assert "SENTINEL_HIDDEN" not in json.dumps(forward["request"])
    assert forward["request"]["provider"]["allow_fallbacks"] is False
    assert set(reverse["metadata"]["displayed_to_original"].values()) == set(row["choice_text"])
    assert list(forward["metadata"]["displayed_to_original"].values()) == list(reverse["metadata"]["displayed_to_original"].values())[::-1]
    old = "Actions to choose:\n" + row["actions_text"] + "\n\n"
    prefix, suffix = row["text"].split(old)
    for order in ("forward", "reverse"):
        consumed, mapping = render(row, order)
        assert consumed.startswith(prefix) and consumed.endswith(suffix)
        assert all(consumed.count(f"{i}: {row['choice_text'][key]}") == 1 for i, key in mapping.items())
    response = {"model": model, "provider": "DeepInfra", "usage": {"cost": 0}, "choices": [{
        "finish_reason": "stop", "message": {"content": json.dumps({"choice": 0, "reasoning": "I choose this action. I accept its risk.", "setup_note": ""})}}]}
    record = {"metadata": reverse["metadata"], "request_key": "offline", "cache_hit": True, "http_status": 200, "response": response}
    parsed = parse_receipt(record, config)
    assert parsed["status"] == "ok" and parsed["original_choice_id"] == reverse["metadata"]["displayed_to_original"]["0"]
    scope = build_scope_packet(row, model, config)
    assert build_scope_packet(contaminated, model, config) == scope
    generic_scope = build_scope_packet(row | {"player_character": "You"}, model, config)
    assert "generic dataset placeholder" in generic_scope["request"]["messages"][1]["content"]
    assert "Decision-maker: You" not in generic_scope["request"]["messages"][1]["content"]
    scope_record = copy.deepcopy(record) | {"metadata": scope["metadata"], "request": scope["request"]}
    scope_answer = {"kind": "action_selection", "meanings_match_question": True,
                    "evidence_quote": next(iter(row["choice_text"].values())), "explanation": "The character chooses an act.", "setup_note": ""}
    scope_record["response"]["choices"][0]["message"]["content"] = json.dumps(scope_answer)
    assert parse_receipt(scope_record, config)["status"] == "ok"
    scope_answer["evidence_quote"] = "INVENTED SOURCE QUOTE"
    scope_record["response"]["choices"][0]["message"]["content"] = json.dumps(scope_answer)
    assert parse_receipt(scope_record, config)["status"] == "evidence_mismatch"
    length = copy.deepcopy(record)
    length["response"]["choices"][0]["finish_reason"] = "length"
    assert parse_receipt(length, config)["status"] == "length_error"
    refusal = copy.deepcopy(record)
    refusal["response"]["choices"][0]["message"]["content"] = "I cannot choose an action in this scene."
    assert parse_receipt(refusal, config)["status"] == "model_refusal"
    filtered = copy.deepcopy(record)
    filtered["response"]["choices"][0]["finish_reason"] = "content_filter"
    assert parse_receipt(filtered, config)["status"] == "api_refusal"
    invalid = copy.deepcopy(record)
    invalid["response"]["choices"][0]["message"]["content"] = json.dumps({"choice": True, "reasoning": "Text.", "setup_note": ""})
    assert parse_receipt(invalid, config)["status"] == "parse_error"
    assert transient_provider_error({"error": {"code": 504}})
    assert not transient_provider_error({"error": {"code": 403}})
    assert not transient_provider_error({"choices": []})
    labels = {str(i): {f"{j}|v": {"signed": value, "derived_from": None} for j, value in enumerate(vector)}
              for i, vector in enumerate([(1., 1.), (.8, .2), (.2, .8)])}
    geometry = scene_spread(labels)
    assert geometry["tradition_tradeoff_spread"] == 0 and geometry["pareto_frontier_choices"] == 1
    del labels["0"]
    assert abs(scene_spread(labels)["tradition_tradeoff_spread"] - .6) < 1e-12
    for name, settings in config["models"].items():
        packet = build_packet(row, "forward", name, config)
        assert all(packet["request"][key] == value for key, value in settings["sampling_parameters"].items())
        assert ("temperature" in packet["request"]) == ("temperature" in settings["sampling_parameters"])
    assert parsed["provider_message"] == response["choices"][0]["message"]
    repeats = [{"model": model, "source": "synthetic", "source_id": "one", "order": order, "repetition": repetition,
                "cache_hit": False, "request_key": order, "status": "ok", "original_choice_id": "a"}
               for order in ("forward", "reverse") for repetition in range(3)]
    assert repeat_summary(repeats, 3)[0]["same_prompt_disagreements"] == 0
    repeats[0]["original_choice_id"] = "b"
    repeated = repeat_summary(repeats, 3)[0]
    assert repeated["same_prompt_disagreements"] == 2 and repeated["same_prompt_pairs"] == 6
    assert repeated["order_disagreements"] == 1 and repeated["order_pairs"] == 3
    repeats[0]["status"] = "transport_error"
    repeated = repeat_summary(repeats, 3)[0]
    assert repeated["complete_conditions"] == 1 and repeated["same_prompt_pairs"] == 4 and repeated["order_pairs"] == 2
    assert repeated["status_counts"]["transport_error"] == 1
    values = {"a": [-1., .3], "b": [1., .5], "c": [.6, -.5]}
    assert position_baseline({"0": [0.], "1": [1.], "2": [0.]}, 0) == [0.]
    assert position_baseline({"0": [0.], "1": [1.], "2": [0.]}, 1) == [1.]
    baseline = uniform_baseline(values)
    assert all(abs(sum(vector[i] - baseline[i] for vector in values.values())) < 1e-12 for i in range(2))
    assert cluster_interval({"a": [0., 0.], "b": [0., 0.]}, {"a": "game1", "b": "game2"}, seed=1, replicates=100) == [(0., 0.), (0., 0.)]
    scenes = {"a": {"source": "one", "eval_cluster": "one:a"}, "b": {"source": "two", "eval_cluster": "two:b"}}
    points, bounds = paired_summary({"a": [2., 2.], "b": [8., 8.]}, scenes, set(scenes), {"one": .5, "two": .5}, replicates=100, ratios=((0, 1),))
    assert points[-1] == 1 and bounds[-1] == (1., 1.)
    points, bounds = paired_summary({"a": [0., 2.], "b": [0., 8.]}, scenes, set(scenes), {"one": .5, "two": .5}, replicates=100, ratios=((0, 1),))
    assert points[-1] == 0 and bounds[-1] == (0., 0.)
    points, bounds = paired_summary({"a": [0., 0.], "b": [0., 0.]}, scenes, set(scenes), {"one": .5, "two": .5}, replicates=100, ratios=((0, 1),))
    assert points[-1] is None and bounds[-1] == (None, None)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        combined = ROOT / "outputs/20261024_release21/parquet/train.parquet"
        overlay = embedded_overlay(combined, root / "virtues.parquet")
        selected = select_scenes(combined, overlay, 1, 20261023, source="machiavelli")[0]
        assert selected == row
        assert build_packet(selected, "forward", model, config) == forward
        original = pq.read_table(ROOT / "outputs/20261023_virtues_v2_contrast_2500_resumed/virtues.parquet").to_pylist()
        original_ids = {(r["source_id"], r["choice_id"]) for r in original}
        extracted = [r for r in pq.read_table(overlay).to_pylist() if (r["source_id"], r["choice_id"]) in original_ids]
        assert {(r["source_id"], r["choice_id"]): r["virtues"] for r in original} == {(r["source_id"], r["choice_id"]): r["virtues"] for r in extracted}
        (root / ".local").mkdir()
        (root / ".local/billing_reservations.json").write_text(json.dumps([
            {"response_id": "one", "scope": "eval", "reserved_usd": .01},
            {"request_key": "gateway", "scope": "virtues", "reserved_usd": .01}]))
        (root / ".local/review_usage_estimates.json").write_text(json.dumps([
            {"run_id": "native-review", "reported_cost_usd": {"costUsd": .5}}]))
        assert abs(unknown_fees(root) - .02) < 1e-12 and noncash_reviews(root) == .5
        assert abs(outstanding(root) - .52) < 1e-12 and outstanding(root, "eval") == .01
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-marker"}), patch.object(Path, "read_text", side_effect=AssertionError("Unexpected credential-file read")):
        assert api_key("env") == "offline-marker"
    client = httpx.AsyncClient
    network_calls = []
    def respond(request):
        network_calls.append(request)
        return httpx.Response(200, json={"id": "offline", "model": model, "usage": {"cost": .001}, "choices": []})
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        out = root / "out"
        out.mkdir()
        with patch.object(eval_api, "ROOT", root), patch.object(eval_api, "api_key", return_value="offline"), patch.object(eval_api.httpx, "AsyncClient", side_effect=lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs)):
            result = asyncio.run(eval_api.run_requests([forward, forward], out, max_new_usd=.1, concurrency=2))
            assert len(network_calls) == 1 and result["cache_hits"] == 1 and result["new_cost_usd"] == .001
            eval_api.reserve_unknown_fee({"request_key": "offline", "time": "one", "attempt": 0}, out)
            assert unknown_fees(root, eval_api.SCOPE) == eval_api.RESERVATION
            cache_path = next((root / "data/interim/action_choice_cache").glob("*.json.gz"))
            cached_bytes = cache_path.read_bytes()
            fresh_out = root / "fresh"
            fresh_out.mkdir()
            previous_calls = len(network_calls)
            fresh = asyncio.run(eval_api.run_requests([forward] * 3, fresh_out, max_new_usd=.1, concurrency=3, refresh=True, cache_write=False))
            assert len(network_calls) - previous_calls == fresh["new_requests"] == 3 and fresh.get("cache_hits", 0) == 0
            assert cache_path.read_bytes() == cached_bytes
    async def check_model_fairness():
        ready = asyncio.Event()
        async def respond_fair(request):
            name = json.loads(request.content)["model"]
            if name == "slow-model":
                await ready.wait()
            else:
                ready.set()
            return httpx.Response(200, json={"id": name, "model": name, "usage": {"cost": .001}, "choices": []})
        slow = copy.deepcopy(forward)
        slow["request"]["model"] = "slow-model"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / "out"
            out.mkdir()
            with patch.object(eval_api, "ROOT", root), patch.object(eval_api, "api_key", return_value="offline"), patch.object(eval_api.httpx, "AsyncClient", side_effect=lambda **kwargs: client(transport=httpx.MockTransport(respond_fair), **kwargs)):
                result = await asyncio.wait_for(eval_api.run_requests([slow, slow, forward], out, max_new_usd=.1, concurrency=2), timeout=2)
                assert result["completed"] == 3 and result["new_requests"] == 2
    asyncio.run(check_model_fairness())
    async def check_transport_recovery():
        class PartialTimeout(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"partial":'
                raise httpx.ReadTimeout("offline body timeout")
        for persistent in (False, True):
            attempts = []
            async def respond_transport(request):
                attempts.append(request)
                if persistent or len(attempts) == 1:
                    return httpx.Response(200, headers={"x-generation-id": f"partial-{len(attempts)}"}, stream=PartialTimeout())
                return httpx.Response(200, json=response | {"id": "recovered", "usage": {"cost": .001}})
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                out = root / "out"
                out.mkdir()
                with patch.object(eval_api, "ROOT", root), patch.object(eval_api, "api_key", return_value="offline"), patch.object(eval_api.httpx, "AsyncClient", side_effect=lambda **kwargs: client(transport=httpx.MockTransport(respond_transport), **kwargs)):
                    result = await eval_api.run_requests([forward], out, max_new_usd=.1, concurrency=1)
                    receipt = json.loads((out / "receipts.jsonl").read_text())
                    events = [json.loads(line) for line in (out / "events.jsonl").open()]
                    assert base64.b64decode(events[0]["partial_body_base64"]) == b'{"partial":'
                    assert events[0]["response_headers"]["x-generation-id"] == "partial-1"
                    if persistent:
                        assert len(attempts) == 4 and parse_receipt(receipt, config)["status"] == "infrastructure_error"
                        assert result["new_cost_usd"] == 0 and abs(unknown_fees(root) - .04) < 1e-12
                        assert not list((root / "data/interim/action_choice_cache").glob("*.gz"))
                    else:
                        assert len(attempts) == 2 and parse_receipt(receipt, config)["status"] == "ok"
                        assert result["new_cost_usd"] == .001 and unknown_fees(root) == .01
    asyncio.run(check_transport_recovery())
    print("PASS: bounded transport recovery preserves partial bytes, headers and uncertain fees; exhausted attempts stay missing and uncached; model-queue isolation prevents a stalled endpoint starving another model; Pareto-frontier tradeoffs, joint opportunity normalization including zero opportunity, concurrent exact-request cache and unknown-fee reserves; embedded labels match the original overlay and requests; caller environment key needs no Pi credentials; hidden labels excluded; reverse IDs map correctly; errors stay separate; uniform-null controls hold. No API calls. -- PI/Sol and PI/OpenAI")


if __name__ == "__main__":
    main()
