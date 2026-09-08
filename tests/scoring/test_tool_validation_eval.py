"""Manifest and leaderboard tool pass use the deterministic validator."""

from __future__ import annotations

import json

from ib.scoring.eval import MANIFEST_EVAL_POLICY_VERSION, manifest_eval


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_manifest_tool_pass_is_strict_even_when_fuzzy_diagnostic_exceeds_threshold(
    tmp_path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "cell_id": "c1",
                "logic_expected_tool_calls": [
                    {
                        "name": "send_message",
                        "arguments": {"recipient": "Maya", "message": "call first"},
                    },
                    {
                        "name": "update_calendar",
                        "arguments": {"time_window": "Friday 10 to noon"},
                    },
                ],
            }
        ],
    )
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "tool_calls": [
                    {
                        "name": "update_calendar",
                        "arguments": {"time_window": "Friday 10 to noon"},
                    },
                    {"name": "send_message", "arguments": {"recipient": "Maya"}},
                ],
            }
        ],
    )

    report = manifest_eval(manifest, predictions)
    summary = report["logic_tool_call_summary"]
    detail = summary["details"]["c1"]

    assert report["policy_version"] == MANIFEST_EVAL_POLICY_VERSION
    assert detail["tool_call_score"] > 0.8
    assert detail["validator_passed"] is False
    assert summary["tool_call_passed"] == 0


def test_manifest_tool_pass_accepts_declared_alternative_value(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        manifest,
        [
            {
                "cell_id": "c1",
                "logic_expected_tool_calls": [
                    {"name": "set_odor_control", "arguments": {"method": "citrus_peel"}}
                ],
                "logic_tool_validation": {
                    "call_policy": "exact_multiset",
                    "required_calls": [
                        {
                            "name": "set_odor_control",
                            "arguments": {
                                "method": {
                                    "op": "one_of",
                                    "values": ["citrus_peel", "bamboo_charcoal"],
                                }
                            },
                        }
                    ],
                },
            }
        ],
    )
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "tool_calls": [
                    {
                        "name": "set_odor_control",
                        "arguments": {"method": "bamboo_charcoal"},
                    }
                ],
            }
        ],
    )

    summary = manifest_eval(manifest, predictions)["logic_tool_call_summary"]

    assert summary["details"]["c1"]["validator_passed"] is True
    assert summary["tool_call_passed"] == 1
    assert summary["tool_call_pass_rate"] == 1.0


def test_explicit_only_no_calls_counts_without_fuzzy_diagnostic(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(
        manifest,
        [{"cell_id": "c1", "logic_tool_validation": {"call_policy": "no_calls"}}],
    )
    _write_jsonl(
        predictions,
        [{"cell_id": "c1", "predicted_action": "respond", "tool_calls": []}],
    )

    summary = manifest_eval(manifest, predictions)["logic_tool_call_summary"]
    detail = summary["details"]["c1"]

    assert summary["total"] == 1
    assert summary["tool_call_passed"] == 1
    assert summary["fuzzy_diagnostic_total"] == 0
    assert detail["tool_call_score"] is None
    assert detail["argument_score"] is None


def test_tool_calls_without_a_contract_are_not_scored(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write_jsonl(manifest, [{"cell_id": "c1"}])
    _write_jsonl(
        predictions,
        [
            {
                "cell_id": "c1",
                "predicted_action": "respond",
                "tool_calls": [{"name": "unscored", "arguments": {}}],
            }
        ],
    )

    summary = manifest_eval(manifest, predictions)["logic_tool_call_summary"]

    assert summary["total"] == 0
    assert summary["details"] == {}
