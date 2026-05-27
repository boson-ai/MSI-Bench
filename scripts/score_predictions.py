#!/usr/bin/env python3
"""Score Interaction-Bench predictions.

This scorer intentionally starts simple: action/engagement metrics plus normalized
exact match for answerable examples. Semantic answer judging can be layered on top
without changing the manifest contract.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ANSWER_ACTIONS = {"respond", "incorporate", "clarify"}
NON_ENGAGE_ACTIONS = {"ignore", "wait"}
ALLOWED_ACTIONS = ANSWER_ACTIONS | NON_ENGAGE_ACTIONS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} row must be an object")
            rows.append(row)
    return rows


def normalize_answer(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def safe_div(num: int | float, den: int | float) -> float | None:
    return None if den == 0 else num / den


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = {row["id"]: row for row in read_jsonl(args.manifest)}
    predictions = {row["id"]: row for row in read_jsonl(args.predictions)}

    missing = sorted(set(manifest) - set(predictions))
    extra = sorted(set(predictions) - set(manifest))

    total = 0
    action_correct = 0
    false_engagement = 0
    false_engagement_den = 0
    missed_engagement = 0
    missed_engagement_den = 0
    answer_exact = 0
    answer_den = 0
    by_slice: dict[str, Counter[str]] = defaultdict(Counter)

    for row_id, row in manifest.items():
        pred = predictions.get(row_id)
        if not pred:
            continue
        expected = row["task"]["expected_action"]
        predicted = str(pred.get("predicted_action", "")).strip().lower()
        if predicted not in ALLOWED_ACTIONS:
            predicted = "invalid"
        total += 1
        correct = expected == predicted
        action_correct += int(correct)

        if expected in NON_ENGAGE_ACTIONS:
            false_engagement_den += 1
            false_engagement += int(predicted in ANSWER_ACTIONS)
        if expected in {"respond", "incorporate"}:
            missed_engagement_den += 1
            missed_engagement += int(predicted in NON_ENGAGE_ACTIONS or predicted == "invalid")

        reference = row["task"].get("ground_truth")
        if expected in {"respond", "incorporate"} and reference not in (None, "", "SILENCE"):
            answer_den += 1
            answer_exact += int(normalize_answer(reference) == normalize_answer(pred.get("predicted_answer")))

        slice_values: list[tuple[str, Any]] = [
            ("expected_action", expected),
            ("interaction_class", row["scenario"].get("interaction_class")),
            ("domain", row["scenario"].get("domain")),
            ("source", row["source"].get("dataset")),
            ("noise_level", row["acoustics"].get("noise_level")),
            ("environment", row["acoustics"].get("environment")),
        ]
        for family in row["acoustics"].get("perturbation_families", []):
            slice_values.append(("perturbation_family", family))

        for slice_name, value in slice_values:
            key = f"{slice_name}:{value}"
            by_slice[key]["total"] += 1
            by_slice[key]["action_correct"] += int(correct)

    report = {
        "total_manifest_rows": len(manifest),
        "scored_rows": total,
        "missing_prediction_count": len(missing),
        "extra_prediction_count": len(extra),
        "engagement_accuracy": safe_div(action_correct, total),
        "false_engagement_rate": safe_div(false_engagement, false_engagement_den),
        "missed_engagement_rate": safe_div(missed_engagement, missed_engagement_den),
        "answer_exact_match": safe_div(answer_exact, answer_den),
        "counts": {
            "action_correct": action_correct,
            "false_engagement": false_engagement,
            "false_engagement_denominator": false_engagement_den,
            "missed_engagement": missed_engagement,
            "missed_engagement_denominator": missed_engagement_den,
            "answer_exact": answer_exact,
            "answer_denominator": answer_den,
        },
        "slices": {
            key: {
                "total": counts["total"],
                "engagement_accuracy": safe_div(counts["action_correct"], counts["total"]),
            }
            for key, counts in sorted(by_slice.items())
        },
        "missing_prediction_ids_sample": missing[:20],
        "extra_prediction_ids_sample": extra[:20],
    }

    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
