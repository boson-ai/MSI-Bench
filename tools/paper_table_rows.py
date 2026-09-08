#!/usr/bin/env python3
"""Aggregate ``ib eval`` outputs into the paper's per-model table rows.

Calling spec
------------
Usage:
    python tools/paper_table_rows.py --suite SUITE_DIR --runs ROOT [ROOT ...] [--json OUT.json]
Inputs:
  - suite: the suite dir the runs were evaluated on (``<entry>/plans/rubrics.jsonl``
    is read to count atoms of cases the judge never saw)
  - runs: one or more ``ib eval --output`` roots; every
    ``generated/<mode>/<provider>/<model>[/<variant>]`` dir holding
    ``predictions.jsonl`` is one target, merged across roots by ``model[/variant]``
Outputs: one line per target on stdout — n, APR with 95% Wilson half-width, ARS,
  Tool, BIR, PRR, per-pattern APR, format-error and unjudged counts; ``--json``
  writes the same numbers as a list of dicts.
Side effects: none (reads only).

Scoring policy (identical to the paper tables):
  - APR: fraction of base cases whose scored judge atoms all pass and whose
    deterministic tool-validator atom passes (``ib.eval_pattern_metrics.judge_by_cell``).
  - ARS: micro-average over all scored atoms of all base cases.
  - Tool: validator pass rate over base cases requiring at least one call.
  - A prediction with a provider/format error fails every atom.
  - PRR (lower is better): over selective-disclosure and background-speech-retrieval
    hear-probe rows whose prediction is a valid ``silent``/``respond``,
    1 - hear_time_pass; format-error rows are excluded.
  - BIR: pass rate of ``probe_summary.speak_time.bystander_interference_suppression``
    (speak-probe rows; ``--`` when no probe executed for the target).
Rounding: half-up to one decimal, as printed in the paper.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from ib.eval_pattern_metrics import judge_by_cell
from ib.scoring.tool_validation import validate_manifest_tool_calls

PATTERN_ORDER = ("auth", "disc", "prior", "seq", "scope", "retr")
PATTERN_LABEL = {
    "auth": "Auth", "disc": "Discl", "prior": "Prior", "seq": "Seq", "scope": "Scope", "retr": "Retr",
}
CASE_ID_RE = re.compile(r"msi-\w+-(\w+)-")
PROBE_MODES = {"hear_time_probe", "speak_time_probe"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def half_up(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def wilson_half_width(k: int, n: int, z: float = 1.96) -> float:
    p = k / n
    return 100 * (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def is_format_error(prediction: dict | None) -> bool:
    metadata = (prediction or {}).get("metadata") or {}
    return isinstance(metadata.get("provider_error"), dict)


def tool_validations(rows: list[dict], predictions: dict[str, dict]) -> dict[str, Any]:
    """Validate each base case once (ib.leaderboard.tool_records.tool_validations_by_cell)."""
    out = {}
    for row in rows:
        cell_id = str(row.get("cell_id") or row.get("id") or "")
        raw_calls = (predictions.get(cell_id) or {}).get("tool_calls")
        validation = validate_manifest_tool_calls(row, raw_calls if isinstance(raw_calls, list) else [])
        if validation is not None:
            out[cell_id] = validation
    return out


def validator_pass(validations: dict[str, Any], predictions: dict[str, dict]) -> dict[str, bool]:
    """One validator atom per case; format errors never earn it."""
    return {
        cell_id: bool(v.passed) and not is_format_error(predictions.get(cell_id))
        for cell_id, v in validations.items()
    }


def target_key(path: Path) -> str:
    parts = path.parts
    i = parts.index("generated")
    return "/".join(parts[i + 3 : -1])  # model[/variant], merged across modes/providers


def load_suite_rubrics(suite: Path) -> dict[str, dict]:
    rubrics: dict[str, dict] = {}
    for entry in sorted(suite.iterdir()):
        path = entry / "plans" / "rubrics.jsonl"
        if path.exists():
            for row in read_jsonl(path):
                rubrics[row["rubric_id"]] = row
    return rubrics


def new_target() -> dict[str, Any]:
    return {
        "cells": {},
        "hear": {"valid": 0, "fail": 0, "n": 0},
        "speak": {"executed": 0, "passed": 0},
    }


def collect(runs: list[Path], rubrics: dict[str, dict]) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = collections.defaultdict(new_target)
    for root in runs:
        for pred_path in sorted(root.rglob("predictions.jsonl")):
            if "_smoke" in pred_path.parts:
                continue
            target_dir = pred_path.parent
            target = targets[target_key(pred_path)]
            predictions = {r["cell_id"]: r for r in read_jsonl(pred_path)}
            manifest = read_jsonl(target_dir / "manifest_case_modes.jsonl")
            base_rows = [m for m in manifest if m.get("logic_case_mode") not in PROBE_MODES]
            judge_path = target_dir / "answer_judge.json"
            judged: dict[str, dict] = {}
            if base_rows and judge_path.exists():
                validations = tool_validations(base_rows, predictions)
                judged = judge_by_cell(
                    json.loads(judge_path.read_text(encoding="utf-8")),
                    validator_pass_by_cell=validator_pass(validations, predictions),
                )
            eval_path = target_dir / "manifest_eval.json"
            manifest_eval = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
            tool_details = (manifest_eval.get("logic_tool_call_summary") or {}).get("details") or {}
            for row in base_rows:
                cell_id = row["cell_id"]
                match = CASE_ID_RE.match(cell_id)
                if not match:
                    raise SystemExit(f"{cell_id}: not a release case id (expected msi-<lang>-<pattern>-...)")
                fmt_err = is_format_error(predictions.get(cell_id))
                tv = (tool_details.get(cell_id) or {}).get("tool_validation") or {}
                requires_call = (tv.get("required_call_count") or 0) > 0
                cell = judged.get(cell_id)
                unjudged = False
                if cell is None:
                    if not fmt_err and not judge_path.exists():
                        continue  # predictions exist but nothing was judged yet
                    refs = [rubrics[r] for r in row.get("rubric_ids", []) if r in rubrics]
                    atoms = sum(len(r.get("atomic_criteria", [])) for r in refs) + int(requires_call)
                    cell = {"passed": False, "atomic_count": atoms, "atomic_passed_count": 0}
                    unjudged = not fmt_err
                target["cells"][cell_id] = {
                    "pattern": match.group(1),
                    "apr": bool(cell.get("passed")),
                    "atoms_passed": cell["atomic_passed_count"],
                    "atoms": cell["atomic_count"],
                    "requires_call": requires_call,
                    "tool": bool(tv.get("passed", False)) if requires_call else None,
                    "format_error": fmt_err,
                    "unjudged": unjudged,
                }
            probes = manifest_eval.get("probe_summary") or {}
            hear = target["hear"]
            for cell_id, v in ((probes.get("hear_time") or {}).get("details") or {}).items():
                if not re.match(r"msi-\w+-(disc|retr)-", cell_id):
                    continue
                hear["n"] += 1
                if v.get("format_error") or v.get("predicted_action_class") not in ("silent", "respond"):
                    continue
                hear["valid"] += 1
                hear["fail"] += int(not v["passed"])
            speak = (probes.get("speak_time") or {}).get("bystander_interference_suppression") or {}
            target["speak"]["executed"] += int(speak.get("executed") or 0)
            target["speak"]["passed"] += int(speak.get("passed") or 0)
    return targets


def summarize(name: str, data: dict[str, Any]) -> dict[str, Any] | None:
    cells = data["cells"]
    n = len(cells)
    if not n:
        return None
    k = sum(1 for c in cells.values() if c["apr"])
    atoms_passed = sum(c["atoms_passed"] for c in cells.values())
    atoms = sum(c["atoms"] for c in cells.values())
    tool = [c["tool"] for c in cells.values() if c["requires_call"]]
    hear, speak = data["hear"], data["speak"]
    per_pattern = {}
    for p in PATTERN_ORDER:
        rows = [c for c in cells.values() if c["pattern"] == p]
        per_pattern[PATTERN_LABEL[p]] = half_up(100 * sum(c["apr"] for c in rows) / len(rows)) if rows else "--"
    return {
        "target": name,
        "n": n,
        "apr": half_up(100 * k / n),
        "apr_ci95": half_up(wilson_half_width(k, n)),
        "ars": half_up(100 * atoms_passed / atoms) if atoms else "--",
        "tool": half_up(100 * sum(tool) / len(tool)) if tool else "--",
        "tool_n": len(tool),
        "bir": half_up(100 * speak["passed"] / speak["executed"]) if speak["executed"] else "--",
        "bir_n": speak["executed"],
        "prr": half_up(100 * hear["fail"] / hear["valid"]) if hear["valid"] else "--",
        "prr_n": f"{hear['valid']}/{hear['n']}",
        "format_errors": sum(c["format_error"] for c in cells.values()),
        "unjudged": sum(c["unjudged"] for c in cells.values()),
        "per_pattern_apr": per_pattern,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    rubrics = load_suite_rubrics(Path(args.suite))
    targets = collect([Path(r) for r in args.runs], rubrics)
    rows = [s for name, data in sorted(targets.items()) if (s := summarize(name, data))]
    if not rows:
        sys.exit("no judged targets found under the given runs")
    for r in rows:
        pats = " ".join(f"{v:>5s}" for v in r["per_pattern_apr"].values())
        print(
            f"{r['target']:48s} n={r['n']:4d} APR={r['apr']:>5s}±{r['apr_ci95']} ARS={r['ars']:>5s} "
            f"Tool={r['tool']:>5s}[{r['tool_n']}] BIR={r['bir']:>5s}[{r['bir_n']}] "
            f"PRR={r['prr']:>5s}[{r['prr_n']}] fmt={r['format_errors']} unjudged={r['unjudged']} | {pats}"
        )
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
