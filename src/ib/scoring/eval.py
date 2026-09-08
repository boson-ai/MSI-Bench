"""eval — manifest and rubric evaluation metrics beyond smoke scoring."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ib.predictions import PredictionRow, read_predictions
from ib.scoring.smoke import _read_rows, _rubric_references
from ib.scoring.tool_match import _tool_argument_score, _tool_call_names, _tool_name_scores
from ib.scoring.tool_validation import (
    TOOL_VALIDATOR_POLICY_VERSION,
    resolve_tool_validation_spec,
    validate_manifest_tool_calls,
)
from ib.scoring.turnbased_speak_probe import BARGE_IN_VARIANT


MANIFEST_EVAL_POLICY_VERSION = "ib.manifest_eval.format_error_invalid.v2"
FULLDUPLEX_SPEAK_PROBE_VARIANT = "fullduplex_live"


def manifest_eval(
    manifest_path: str | Path,
    predictions_path: str | Path,
    *,
    eval_mode: str = "turn_based",
    speak_probe_approx: bool = False,
) -> dict[str, Any]:
    """Evaluate probe actions, label completeness, and semantic helper metrics."""
    manifest = _read_rows(manifest_path)
    predictions = read_predictions(predictions_path)
    by_id = {row.cell_id: row for row in predictions}
    missing = sorted(row["cell_id"] for row in manifest if row.get("cell_id") not in by_id)
    total = len(manifest)
    action_report = _probe_action_summary(manifest, by_id, eval_mode=eval_mode)
    sibling_report = _sibling_consistency(manifest, by_id)
    interaction_summary = _interaction_summary(manifest, by_id)
    logic_summary = _logic_tool_call_summary(manifest, by_id)
    probe_summary = _probe_summary(
        manifest, by_id, eval_mode=eval_mode, speak_probe_approx=speak_probe_approx
    )
    return {
        "schema_version": "ib.manifest_eval.v1",
        "policy_version": MANIFEST_EVAL_POLICY_VERSION,
        "eval_mode": eval_mode,
        "total": total,
        "correct": action_report["correct"],
        "accuracy": action_report["accuracy"],
        "action_total": action_report["total"],
        "action_summary": action_report,
        "missing_prediction_cell_ids": missing,
        "interaction_summary": interaction_summary,
        "logic_tool_call_summary": logic_summary,
        "probe_summary": probe_summary,
        "counterfactual_sibling_consistency": sibling_report,
        "errors": [f"missing predictions for {len(missing)} manifest row(s): {', '.join(missing)}"]
        if missing
        else [],
    }


def _probe_summary(
    manifest: list[dict[str, Any]],
    predictions: dict[str, PredictionRow],
    *,
    eval_mode: str,
    speak_probe_approx: bool = False,
) -> dict[str, Any]:
    """Return deterministic probe slices for hear-time and speak-time cases."""
    return {
        "eval_mode": eval_mode,
        "hear_time": _hear_time_summary(manifest, predictions),
        "speak_time": {
            "bystander_interference_suppression": _speak_time_bystander_summary(
                manifest,
                predictions,
                eval_mode=eval_mode,
                speak_probe_approx=speak_probe_approx,
            )
        },
    }


def _probe_action_summary(
    manifest: list[dict[str, Any]],
    predictions: dict[str, PredictionRow],
    *,
    eval_mode: str,
) -> dict[str, Any]:
    """Return the only action-class denominator: hear-time plus live speak-time probes.

    Base semantic cases are intentionally excluded. Their final user turn is
    answerable by construction; refusal, clarification, incorporation, and
    other semantic distinctions are judged by atomic answer rubrics instead of
    the coarse ``predicted_action`` field.
    """

    details: dict[str, dict[str, Any]] = {}
    correct = 0
    total = 0
    for row in manifest:
        if row.get("logic_case_mode") == "hear_time_probe":
            cell_id = row.get("cell_id")
            prediction = predictions.get(cell_id) if isinstance(cell_id, str) else None
            predicted = None if prediction is None else prediction.predicted_action.value
            predicted_class = _action_class(predicted)
            expected_class = "silent"
            format_error = _prediction_format_error(prediction)
            action_correct = predicted_class == expected_class and not format_error
            total += 1
            correct += int(action_correct)
            details[str(cell_id)] = {
                "probe_type": "hear_time",
                "expected_action_class": expected_class,
                "predicted_action": predicted,
                "predicted_action_class": predicted_class,
                "correct": action_correct,
                "format_error": format_error,
            }
    if eval_mode == "fullduplex":
        for row in _speak_probe_rows(manifest):
            cell_id = row.get("cell_id")
            prediction = predictions.get(cell_id) if isinstance(cell_id, str) else None
            predicted = _live_probe_predicted_action(prediction)
            predicted_class = _action_class(predicted)
            expected_class = "respond"
            format_error = _prediction_format_error(prediction)
            action_correct = predicted_class == expected_class and not format_error
            total += 1
            correct += int(action_correct)
            details[str(cell_id)] = {
                "probe_type": "speak_time",
                "expected_action_class": expected_class,
                "predicted_action": predicted,
                "predicted_action_class": predicted_class,
                "correct": action_correct,
                "format_error": format_error,
            }
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "details": details,
    }


def _hear_time_summary(
    manifest: list[dict[str, Any]], predictions: dict[str, PredictionRow]
) -> dict[str, Any]:
    """Score final human-to-human no-response probes."""
    rows = [row for row in manifest if row.get("logic_case_mode") == "hear_time_probe"]
    details: dict[str, dict[str, Any]] = {}
    correct = 0
    unexpected_response = 0
    unexpected_tool_call = 0
    for row in rows:
        cell_id = row.get("cell_id")
        prediction = predictions.get(cell_id) if isinstance(cell_id, str) else None
        action = None if prediction is None else prediction.predicted_action.value
        action_class = _action_class(action)
        answer_text = None if prediction is None else prediction.answer_text
        tool_count = 0 if prediction is None else len(prediction.tool_calls)
        format_error = _prediction_format_error(prediction)
        action_correct = action_class == "silent" and not format_error
        has_response = action_class != "silent" or bool((answer_text or "").strip())
        has_tool_call = tool_count > 0
        passed = (
            prediction is not None and not has_response and not has_tool_call and not format_error
        )
        correct += int(passed)
        unexpected_response += int(has_response)
        unexpected_tool_call += int(has_tool_call)
        details[str(cell_id)] = {
            "expected_action": "silent",
            "raw_expected_action": row.get("expected_action"),
            "predicted_action": action,
            "predicted_action_class": action_class,
            "action_correct": action_correct,
            "answer_text_present": bool((answer_text or "").strip()),
            "predicted_tool_count": tool_count,
            "passed": passed,
            "format_error": format_error,
        }
    total = len(rows)
    return {
        "total": total,
        "correct_ignore": correct,
        "accuracy": correct / total if total else 1.0,
        "unexpected_response": unexpected_response,
        "unexpected_tool_call": unexpected_tool_call,
        "details": details,
    }


def _prediction_format_error(prediction: PredictionRow | None) -> bool:
    """Return whether a prediction row degraded to a provider format error.

    Degraded rows carry predicted_action="silent" only as a pipeline
    placeholder; they must never earn credit as deliberate silence.
    """
    if prediction is None:
        return False
    metadata = prediction.metadata if isinstance(prediction.metadata, dict) else {}
    return isinstance(metadata.get("provider_error"), dict)


def _action_class(action: str | None) -> str | None:
    """Collapse legacy action labels into answer-vs-silent probe classes."""
    if action in {"respond", "refuse", "clarify", "incorporate"}:
        return "respond"
    if action in {"silent", "ignore", "wait"}:
        return "silent"
    return None


def _speak_probe_rows(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer dedicated speak_time_probe rows; fall back to legacy base-row probes."""
    probe_rows = [row for row in manifest if row.get("logic_case_mode") == "speak_time_probe"]
    if probe_rows:
        return probe_rows
    return [
        row for row in manifest if _has_live_expansion(row, "bystander_interference_suppression")
    ]


def _speak_time_bystander_summary(
    manifest: list[dict[str, Any]],
    predictions: dict[str, PredictionRow],
    *,
    eval_mode: str,
    speak_probe_approx: bool = False,
) -> dict[str, Any]:
    """Score bystander-interference suppression probes (live or barge-in approx)."""
    rows = _speak_probe_rows(manifest)
    candidate_count = len(rows)
    if eval_mode != "fullduplex":
        dedicated = [
            row for row in rows if row.get("logic_case_mode") == "speak_time_probe"
        ]
        if speak_probe_approx and dedicated:
            return _barge_in_resume_summary(dedicated, predictions, eval_mode=eval_mode)
        return {
            "eval_mode": eval_mode,
            "candidate_count": candidate_count,
            "applicable": 0,
            "not_applicable": candidate_count,
            "not_applicable_reason": "eval_mode_not_fullduplex"
            if not speak_probe_approx
            else "no_dedicated_speak_probe_rows",
            "executed": 0,
            "probe_not_executed": 0,
            "passed": 0,
            "pass_rate": None,
            "details": {},
        }
    details: dict[str, dict[str, Any]] = {}
    executed_count = 0
    action_preserved = 0
    tool_preserved = 0
    interrupted_count = 0
    followed_bystander_count = 0
    passed_count = 0
    for row in rows:
        cell_id = row.get("cell_id")
        prediction = predictions.get(cell_id) if isinstance(cell_id, str) else None
        expected_calls = row.get("logic_expected_tool_calls") or []
        predicted_calls = _live_probe_tool_calls(prediction)
        argument_score, _matched_pairs = _tool_argument_score(expected_calls, predicted_calls)
        names_match = Counter(_tool_call_names(expected_calls)) == Counter(
            _tool_call_names(predicted_calls)
        )
        tool_ok = names_match and argument_score >= 0.8
        predicted_action = _live_probe_predicted_action(prediction)
        predicted_action_class = _action_class(predicted_action)
        expected_action_class = "respond"
        action_ok = predicted_action_class == expected_action_class
        live_metadata = _live_probe_metadata(prediction)
        executed = bool(live_metadata.get("executed"))
        interrupted = bool(live_metadata.get("interrupted"))
        followed_bystander = bool(live_metadata.get("followed_bystander"))
        passed = executed and action_ok and tool_ok and not interrupted and not followed_bystander
        executed_count += int(executed)
        action_preserved += int(action_ok)
        tool_preserved += int(tool_ok)
        interrupted_count += int(interrupted)
        followed_bystander_count += int(followed_bystander)
        passed_count += int(passed)
        details[str(cell_id)] = {
            "probe_variant": FULLDUPLEX_SPEAK_PROBE_VARIANT,
            "expected_action": expected_action_class,
            "raw_expected_action": row.get("expected_action"),
            "predicted_action": predicted_action,
            "predicted_action_class": predicted_action_class,
            "live_probe_executed": executed,
            "action_preserved": action_ok,
            "base_tool_calls_preserved": tool_ok,
            "interrupted": interrupted,
            "followed_bystander": followed_bystander,
            "passed": passed,
        }
    return {
        "eval_mode": eval_mode,
        "probe_variant": FULLDUPLEX_SPEAK_PROBE_VARIANT,
        "candidate_count": candidate_count,
        "applicable": candidate_count,
        "not_applicable": 0,
        "executed": executed_count,
        "probe_not_executed": candidate_count - executed_count,
        "passed": passed_count,
        "pass_rate": passed_count / candidate_count if candidate_count else 1.0,
        "action_preserved": action_preserved,
        "base_tool_calls_preserved": tool_preserved,
        "interrupted": interrupted_count,
        "followed_bystander": followed_bystander_count,
        "details": details,
    }


def _barge_in_resume_summary(
    rows: list[dict[str, Any]],
    predictions: dict[str, PredictionRow],
    *,
    eval_mode: str,
) -> dict[str, Any]:
    """Score turn-based barge-in renderings of dedicated speak_time_probe rows.

    Pass = executed ∧ action preserved ∧ tool plan preserved ∧ not followed
    bystander. The fullduplex ``interrupted`` atom has no turn-based analogue:
    barge-in interruption is decided by the system layer, not the model.
    """
    details: dict[str, dict[str, Any]] = {}
    executed_count = action_preserved = tool_preserved = 0
    followed_bystander_count = passed_count = 0
    for row in rows:
        cell_id = row.get("cell_id")
        prediction = predictions.get(cell_id) if isinstance(cell_id, str) else None
        format_error = _prediction_format_error(prediction)
        executed = prediction is not None and not format_error
        predicted_action = None if prediction is None else prediction.predicted_action.value
        predicted_action_class = _action_class(predicted_action)
        action_ok = executed and predicted_action_class == "respond"
        expected_calls = row.get("logic_expected_tool_calls") or []
        predicted_calls = (
            []
            if prediction is None
            else [call.model_dump(mode="json") for call in prediction.tool_calls]
        )
        argument_score, _matched_pairs = _tool_argument_score(expected_calls, predicted_calls)
        names_match = Counter(_tool_call_names(expected_calls)) == Counter(
            _tool_call_names(predicted_calls)
        )
        tool_ok = executed and names_match and argument_score >= 0.8
        followed_bystander = bool(predicted_calls) and not names_match
        passed = executed and action_ok and tool_ok and not followed_bystander
        executed_count += int(executed)
        action_preserved += int(action_ok)
        tool_preserved += int(tool_ok)
        followed_bystander_count += int(followed_bystander)
        passed_count += int(passed)
        details[str(cell_id)] = {
            "probe_variant": BARGE_IN_VARIANT,
            "expected_action": "respond",
            "raw_expected_action": row.get("expected_action"),
            "predicted_action": predicted_action,
            "predicted_action_class": predicted_action_class,
            "live_probe_executed": executed,
            "action_preserved": action_ok,
            "base_tool_calls_preserved": tool_ok,
            "interrupted": None,
            "followed_bystander": followed_bystander,
            "passed": passed,
            "format_error": format_error,
            "answer_text": None if prediction is None else prediction.answer_text,
            "tool_calls": predicted_calls,
        }
    total = len(rows)
    return {
        "eval_mode": eval_mode,
        "probe_variant": BARGE_IN_VARIANT,
        "candidate_count": total,
        "applicable": total,
        "not_applicable": 0,
        "executed": executed_count,
        "probe_not_executed": total - executed_count,
        "passed": passed_count,
        "pass_rate": passed_count / total if total else 1.0,
        "action_preserved": action_preserved,
        "base_tool_calls_preserved": tool_preserved,
        "interrupted": None,
        "followed_bystander": followed_bystander_count,
        "details": details,
    }


def _has_live_expansion(row: dict[str, Any], expansion_type: str) -> bool:
    if row.get("logic_live_expansion_type") == expansion_type:
        return True
    expansion_types = row.get("logic_live_expansion_types")
    return isinstance(expansion_types, list) and expansion_type in expansion_types


def _live_probe_predicted_action(prediction: PredictionRow | None) -> str | None:
    if prediction is None:
        return None
    live_probe = _raw_live_probe(prediction)
    value = live_probe.get("predicted_action")
    return value if isinstance(value, str) else prediction.predicted_action.value


def _live_probe_tool_calls(prediction: PredictionRow | None) -> list[dict[str, Any]]:
    if prediction is None:
        return []
    live_probe = _raw_live_probe(prediction)
    calls = live_probe.get("tool_calls")
    if isinstance(calls, list):
        return [call for call in calls if isinstance(call, dict)]
    return [call.model_dump(mode="json") for call in prediction.tool_calls]


def _live_probe_metadata(prediction: PredictionRow | None) -> dict[str, Any]:
    metadata = prediction.metadata if prediction is not None and isinstance(prediction.metadata, dict) else {}
    live_probe = _raw_live_probe(prediction)
    return {
        "executed": bool(
            metadata.get("live_probe_executed")
            or live_probe.get("executed")
            or live_probe.get("injection_started")
        ),
        "interrupted": bool(metadata.get("interrupted") or live_probe.get("interrupted")),
        "followed_bystander": bool(
            metadata.get("followed_bystander") or live_probe.get("followed_bystander")
        ),
    }


def _raw_live_probe(prediction: PredictionRow | None) -> dict[str, Any]:
    if prediction is None or not isinstance(prediction.metadata, dict):
        return {}
    nested = prediction.metadata.get("live_probe")
    return nested if isinstance(nested, dict) else {}


def rubric_eval(manifest_path: str | Path, rubrics_path: str | Path) -> dict[str, Any]:
    """Validate answer-rubric references and answerability coverage without generation."""
    manifest = _read_rows(manifest_path)
    rubrics = _read_rows(rubrics_path)
    rubric_ids = [row.get("rubric_id") for row in rubrics]
    known = {rid for rid in rubric_ids if isinstance(rid, str) and rid}
    duplicate_ids = sorted({rid for rid in known if rubric_ids.count(rid) > 1})
    references = sorted({rid for row in manifest for rid in _rubric_references(row)})
    missing = sorted(set(references) - known)
    answerable_rows = [
        row
        for row in manifest
        if _action_class(row.get("expected_action")) == "respond"
    ]
    answerable_with_rubrics = [row for row in answerable_rows if _rubric_references(row)]
    return {
        "schema_version": "ib.rubric_eval.v1",
        "manifest_rows": len(manifest),
        "rubric_count": len(rubrics),
        "referenced_rubric_ids": references,
        "missing_rubric_ids": missing,
        "duplicate_rubric_ids": duplicate_ids,
        "answerable_row_count": len(answerable_rows),
        "answerable_rubric_coverage": len(answerable_with_rubrics) / len(answerable_rows)
        if answerable_rows
        else 1.0,
        "errors": _rubric_errors(missing, duplicate_ids),
    }


def _logic_tool_call_summary(
    manifest: list[dict[str, Any]], predictions: dict[str, PredictionRow]
) -> dict[str, Any]:
    """Return order-insensitive name + argument tool-call diagnostics."""
    details: dict[str, dict[str, Any]] = {}
    total = 0
    for row in manifest:
        cell_id = row.get("cell_id")
        prediction = predictions.get(cell_id) if isinstance(cell_id, str) else None
        predicted_calls = [] if prediction is None else [
            call.model_dump(mode="json") for call in prediction.tool_calls
        ]
        validation_spec, _validation_source = resolve_tool_validation_spec(row)
        if validation_spec is None:
            continue
        raw_expected_calls = row.get("logic_expected_tool_calls")
        fuzzy_applicable = isinstance(raw_expected_calls, list)
        expected_calls = raw_expected_calls if fuzzy_applicable else []
        total += 1
        expected_names = _tool_call_names(expected_calls)
        predicted_names = _tool_call_names(predicted_calls)
        name_precision, name_recall, name_f1 = _tool_name_scores(expected_names, predicted_names)
        argument_score, matched_pairs = _tool_argument_score(expected_calls, predicted_calls)
        tool_call_score = (0.4 * name_f1) + (0.6 * argument_score)
        validation = validate_manifest_tool_calls(row, predicted_calls)
        assert validation is not None
        format_error = _prediction_format_error(prediction)
        if format_error:
            name_precision = name_recall = name_f1 = 0.0
            argument_score = 0.0
            tool_call_score = 0.0
            matched_pairs = []
        cell_key = str(cell_id)
        details[cell_key] = {
            "format_error": format_error,
            "fuzzy_diagnostic_applicable": fuzzy_applicable,
            "tool_call_score": tool_call_score if fuzzy_applicable else None,
            "name_precision": name_precision if fuzzy_applicable else None,
            "name_recall": name_recall if fuzzy_applicable else None,
            "name_f1": name_f1 if fuzzy_applicable else None,
            "argument_score": argument_score if fuzzy_applicable else None,
            "name_any_overlap": bool(set(expected_names) & set(predicted_names))
            if fuzzy_applicable
            else None,
            "name_multiset_match": Counter(expected_names) == Counter(predicted_names)
            if fuzzy_applicable
            else None,
            "expected_names": expected_names,
            "predicted_names": predicted_names,
            "expected_tool_count": len(expected_calls),
            "predicted_tool_count": len(predicted_calls),
            "matched_pairs": matched_pairs,
            "validator_passed": validation.passed and not format_error,
            "tool_validation": validation.model_dump(mode="json"),
        }
    diagnostics = [item for item in details.values() if item["fuzzy_diagnostic_applicable"]]
    diagnostic_total = len(diagnostics)
    score_sum = sum(item["tool_call_score"] for item in diagnostics)
    name_precision_sum = sum(item["name_precision"] for item in diagnostics)
    name_recall_sum = sum(item["name_recall"] for item in diagnostics)
    name_f1_sum = sum(item["name_f1"] for item in diagnostics)
    argument_sum = sum(item["argument_score"] for item in diagnostics)
    name_any = sum(1 for item in diagnostics if item["name_any_overlap"])
    name_multiset = sum(1 for item in diagnostics if item["name_multiset_match"])
    passed = sum(1 for item in details.values() if item["validator_passed"])
    return {
        "validator_policy_version": TOOL_VALIDATOR_POLICY_VERSION,
        "total": total,
        "fuzzy_diagnostic_total": diagnostic_total,
        "tool_call_score": score_sum / diagnostic_total if diagnostic_total else None,
        "tool_call_passed": passed,
        "tool_call_pass_rate": passed / total if total else 1.0,
        "name_precision": name_precision_sum / diagnostic_total if diagnostic_total else None,
        "name_recall": name_recall_sum / diagnostic_total if diagnostic_total else None,
        "name_f1": name_f1_sum / diagnostic_total if diagnostic_total else None,
        "argument_score": argument_sum / diagnostic_total if diagnostic_total else None,
        "name_any_overlap_matched": name_any,
        "name_any_overlap_rate": name_any / diagnostic_total if diagnostic_total else None,
        "name_multiset_matched": name_multiset,
        "name_multiset_match_rate": name_multiset / diagnostic_total
        if diagnostic_total
        else None,
        "cell_scores": {
            cell_id: item["tool_call_score"]
            for cell_id, item in details.items()
            if item["tool_call_score"] is not None
        },
        "details": details,
    }


def _sibling_consistency(
    manifest: list[dict[str, Any]], predictions: dict[str, PredictionRow]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in manifest:
        if row.get("logic_case_mode") != "hear_time_probe":
            continue
        group = row.get("sibling_group")
        if isinstance(group, str):
            groups.setdefault(group, []).append(row)
    evaluated = 0
    consistent = 0
    details: dict[str, bool] = {}
    for group, rows in sorted(groups.items()):
        available = [row for row in rows if row["cell_id"] in predictions]
        if len(available) < 2:
            continue
        # Consistent means every sibling prediction matches the coarse
        # action class for action probes. Base semantic rows are excluded.
        ok = all(
            _action_class(predictions[row["cell_id"]].predicted_action.value)
            == _action_class(row.get("expected_action"))
            for row in available
        )
        details[group] = ok
        evaluated += 1
        consistent += int(ok)
    return {
        "evaluated_group_count": evaluated,
        "consistent_group_count": consistent,
        "consistency_rate": consistent / evaluated if evaluated else 1.0,
        "groups": details,
    }


def _interaction_summary(
    manifest: list[dict[str, Any]], predictions: dict[str, PredictionRow]
) -> dict[str, Any]:
    summary: dict[str, dict[str, int]] = {}
    for row in manifest:
        if row.get("logic_case_mode") != "hear_time_probe":
            continue
        expected = _action_class(row.get("expected_action"))
        if not isinstance(expected, str):
            continue
        bucket = summary.setdefault(expected, {"total": 0, "correct": 0})
        bucket["total"] += 1
        prediction = predictions.get(row.get("cell_id"))
        if prediction is not None and _action_class(prediction.predicted_action.value) == expected:
            bucket["correct"] += 1
    for bucket in summary.values():
        bucket["accuracy"] = bucket["correct"] / bucket["total"] if bucket["total"] else 0.0
    return summary


def _rubric_errors(missing: list[str], duplicates: list[str]) -> list[str]:
    errors = []
    if missing:
        errors.append(f"unknown rubric_id references in manifest: {missing}")
    if duplicates:
        errors.append(f"duplicate rubric_id values: {duplicates}")
    return errors
