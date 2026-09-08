"""tool_match — fuzzy tool-call name/argument matching shared by eval summaries.

Calling spec:
    _tool_call_names(calls) -> list[str]
    _tool_name_scores(expected_names, predicted_names) -> (precision, recall, f1)
    _tool_argument_score(expected_calls, predicted_calls) -> (score, matched_pairs)

Inputs are raw manifest/prediction tool-call lists. Deterministic, no side
effects, independently testable; scoring policy (weights, containment credit,
token F1) is sealed here and reused verbatim by every eval summary.
"""

from __future__ import annotations

from collections import Counter
import json
from typing import Any


def _tool_call_names(calls: Any) -> list[str]:
    """Extract tool names from raw or validated tool-call objects."""
    if not isinstance(calls, list):
        return []
    names: list[str] = []
    for call in calls:
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            names.append(call["name"])
    return names


def _tool_name_scores(expected_names: list[str], predicted_names: list[str]) -> tuple[float, float, float]:
    overlap = _counter_overlap_count(Counter(expected_names), Counter(predicted_names))
    precision = overlap / len(predicted_names) if predicted_names else float(not expected_names)
    recall = overlap / len(expected_names) if expected_names else 1.0
    f1 = _f1(precision, recall)
    return precision, recall, f1


def _tool_argument_score(
    expected_calls: Any,
    predicted_calls: Any,
) -> tuple[float, list[dict[str, Any]]]:
    """Score gold arguments against same-name predicted calls, ignoring call order."""
    if not isinstance(predicted_calls, list):
        predicted_calls = []
    if not isinstance(expected_calls, list):
        expected_calls = []
    if not expected_calls:
        return (1.0 if not predicted_calls else 0.0), []
    candidates: list[tuple[float, int, int]] = []
    for expected_index, expected_call in enumerate(expected_calls):
        if not isinstance(expected_call, dict):
            continue
        for predicted_index, predicted_call in enumerate(predicted_calls):
            if not isinstance(predicted_call, dict):
                continue
            if expected_call.get("name") != predicted_call.get("name"):
                continue
            candidates.append(
                (
                    _single_call_argument_score(
                        expected_call.get("arguments", {}),
                        predicted_call.get("arguments", {}),
                    ),
                    expected_index,
                    predicted_index,
                )
            )
    used_expected: set[int] = set()
    used_predicted: set[int] = set()
    matched_pairs: list[dict[str, Any]] = []
    for score, expected_index, predicted_index in sorted(candidates, reverse=True):
        if expected_index in used_expected or predicted_index in used_predicted:
            continue
        used_expected.add(expected_index)
        used_predicted.add(predicted_index)
        matched_pairs.append(
            {
                "expected_index": expected_index,
                "predicted_index": predicted_index,
                "name": expected_calls[expected_index].get("name"),
                "argument_score": score,
            }
        )
    total_score = sum(pair["argument_score"] for pair in matched_pairs)
    denominator = max(len(expected_calls), len(predicted_calls), 1)
    return total_score / denominator, sorted(
        matched_pairs, key=lambda pair: pair["expected_index"]
    )


def _single_call_argument_score(expected_args: Any, predicted_args: Any) -> float:
    """Score required gold argument keys/values against one same-name predicted call."""
    if not isinstance(expected_args, dict) or not expected_args:
        return 1.0
    if not isinstance(predicted_args, dict):
        predicted_args = {}
    predicted_value_texts = [_argument_text(value) for value in predicted_args.values()]
    scores: list[float] = []
    for key, expected_value in expected_args.items():
        expected_text = _argument_text(expected_value)
        if key in predicted_args:
            value_score = _normalized_value_score(expected_text, _argument_text(predicted_args[key]))
            scores.append((0.35 * 1.0) + (0.65 * value_score))
            continue
        fallback = max(
            (_normalized_value_score(expected_text, predicted_text) for predicted_text in predicted_value_texts),
            default=0.0,
        )
        scores.append(0.5 * fallback)
    return sum(scores) / len(scores)


def _argument_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _normalized_value_score(expected: str, predicted: str) -> float:
    expected_compact = _compact_text(expected)
    predicted_compact = _compact_text(predicted)
    if not expected_compact:
        return 1.0 if not predicted_compact else 0.0
    if expected_compact == predicted_compact:
        return 1.0
    if expected_compact in predicted_compact or predicted_compact in expected_compact:
        return 0.9
    return _token_f1(_tokens(expected), _tokens(predicted))


def _compact_text(value: str) -> str:
    return "".join(char.casefold() for char in value if char.isalnum())


def _tokens(value: str) -> list[str]:
    tokens: list[str] = []
    ascii_buffer: list[str] = []
    for char in value:
        if char.isascii() and char.isalnum():
            ascii_buffer.append(char.casefold())
            continue
        if ascii_buffer:
            tokens.append("".join(ascii_buffer))
            ascii_buffer = []
        if char.isalnum():
            tokens.append(char.casefold())
    if ascii_buffer:
        tokens.append("".join(ascii_buffer))
    return tokens


def _token_f1(expected_tokens: list[str], predicted_tokens: list[str]) -> float:
    if not expected_tokens:
        return 1.0 if not predicted_tokens else 0.0
    if not predicted_tokens:
        return 0.0
    overlap = _counter_overlap_count(Counter(expected_tokens), Counter(predicted_tokens))
    return _f1(overlap / len(predicted_tokens), overlap / len(expected_tokens))


def _counter_overlap_count(left: Counter[str], right: Counter[str]) -> int:
    return sum(min(left[key], right[key]) for key in left.keys() & right.keys())


def _f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
