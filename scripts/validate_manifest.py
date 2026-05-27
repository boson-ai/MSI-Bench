#!/usr/bin/env python3
"""Validate an Interaction-Bench JSONL manifest with stdlib checks."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ALLOWED_ACTIONS = {"respond", "ignore", "wait", "clarify", "incorporate"}
ALLOWED_CLASSES = {"must_respond", "must_ignore", "must_wait", "must_clarify", "must_incorporate"}
ALLOWED_CONFIDENCE = {"gold", "source", "inferred", "synthetic"}
REQUIRED_TOP = ["id", "source", "audio", "scenario", "task", "acoustics", "labels", "split"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: row must be an object")
            row["__line_no"] = line_no
            rows.append(row)
    return rows


def candidate_roots(manifest: Path, audio_root: Path | None) -> list[Path]:
    """Return roots for resolving relative audio paths.

    Seed manifests are often generated from the workspace root but stored under
    ``Interaction-Bench/examples``.  Checking both the current directory
    and ancestor directories makes ``--check-audio`` work from either location
    without rewriting reproducible source-relative manifest paths.
    """
    roots: list[Path] = []
    if audio_root is not None:
        roots.append(audio_root)
    roots.append(Path.cwd())
    roots.extend(manifest.resolve().parents)

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            unique.append(root)
            seen.add(key)
    return unique


def resolve_existing_path(path_text: str, roots: list[Path]) -> Path | None:
    path = Path(path_text)
    if path.is_absolute():
        return path if path.exists() else None
    for root in roots:
        candidate = root / path
        if candidate.exists():
            return candidate
    return None


def validate_row(row: dict[str, Any], check_audio: bool, roots: list[Path]) -> list[str]:
    errors: list[str] = []
    prefix = f"line {row.get('__line_no', '?')} id={row.get('id', '<missing>')}: "
    for key in REQUIRED_TOP:
        if key not in row:
            errors.append(prefix + f"missing top-level key '{key}'")
    if not isinstance(row.get("id"), str) or not row.get("id"):
        errors.append(prefix + "id must be a non-empty string")

    source = row.get("source", {})
    if not isinstance(source, dict) or not source.get("dataset") or not source.get("license") or not source.get("path"):
        errors.append(prefix + "source requires dataset, license, and path")

    audio = row.get("audio", {})
    primary = audio.get("primary") if isinstance(audio, dict) else None
    if not isinstance(primary, str) or not primary:
        errors.append(prefix + "audio.primary must be a non-empty string")
    elif check_audio and resolve_existing_path(primary, roots) is None:
        errors.append(prefix + f"audio.primary does not exist under checked roots: {primary}")
    multichannel = audio.get("multichannel") if isinstance(audio, dict) else None
    if check_audio and multichannel and resolve_existing_path(str(multichannel), roots) is None:
        errors.append(prefix + f"audio.multichannel does not exist under checked roots: {multichannel}")

    scenario = row.get("scenario", {})
    interaction_class = scenario.get("interaction_class") if isinstance(scenario, dict) else None
    if interaction_class not in ALLOWED_CLASSES:
        errors.append(prefix + f"invalid scenario.interaction_class: {interaction_class}")

    task = row.get("task", {})
    expected_action = task.get("expected_action") if isinstance(task, dict) else None
    if expected_action not in ALLOWED_ACTIONS:
        errors.append(prefix + f"invalid task.expected_action: {expected_action}")
    if not isinstance(task.get("instruction") if isinstance(task, dict) else None, str):
        errors.append(prefix + "task.instruction must be a string")

    acoustics = row.get("acoustics", {})
    families = acoustics.get("perturbation_families") if isinstance(acoustics, dict) else None
    if not isinstance(families, list) or not all(isinstance(x, str) for x in families):
        errors.append(prefix + "acoustics.perturbation_families must be a list of strings")

    labels = row.get("labels", {})
    confidence = labels.get("label_confidence") if isinstance(labels, dict) else None
    if confidence not in ALLOWED_CONFIDENCE:
        errors.append(prefix + f"invalid labels.label_confidence: {confidence}")
    if not isinstance(labels.get("rationale") if isinstance(labels, dict) else None, str):
        errors.append(prefix + "labels.rationale must be a string")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check-audio", action="store_true", help="Verify referenced audio paths exist")
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="Optional root for relative audio paths; defaults to cwd plus manifest ancestors",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    roots = candidate_roots(args.manifest, args.audio_root)
    errors: list[str] = []
    seen: set[str] = set()
    ids = [row.get("id") for row in rows]
    for row in rows:
        row_id = row.get("id")
        if row_id in seen:
            errors.append(f"line {row.get('__line_no')}: duplicate id {row_id}")
        seen.add(row_id)
        errors.extend(validate_row(row, args.check_audio, roots))

    summary = {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "by_source": dict(Counter(row.get("source", {}).get("dataset") for row in rows)),
        "by_action": dict(Counter(row.get("task", {}).get("expected_action") for row in rows)),
        "errors": errors[:50],
        "error_count": len(errors),
    }
    if args.check_audio:
        summary["audio_roots_checked"] = [str(root) for root in roots]
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
