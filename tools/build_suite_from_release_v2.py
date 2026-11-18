"""build_suite_from_release_v2 — adapt an MSI-Bench v2 release package for `ib eval`.

Calling spec:
    python tools/build_suite_from_release_v2.py --release <msi-bench-v2 dir> --output <suite>

Inputs:  a v2 release package (index/cases.jsonl, index/rubrics.jsonl, data/<lang>/<pat>/<scene>/<case_id>/
         {case.json, blocks/NN.wav}; schema in its docs/SCHEMA.md).
Outputs: <suite>/<lang>_<paper_pattern>/manifest/manifest.jsonl + plans/rubrics.jsonl, one entry per
         (language, pattern). Each manifest row replays the case turn by turn (audio_history, history_style
         text_turns: one audio message per room block, earlier assistant replies as text) and offers the
         case's Anthropic-format tools as `logic_tools`, in the key order of the eval.system_prompt listing
         (the training text; eval.tools has sorted keys). Block paths are relative to the entry dir, which
         ib eval uses as the run root, and resolve into the release package. Rubric rows are the package's
         rows with case_id renamed to cell_id; rubric_id is unchanged. Every manifest row carries
         prompt_version "v2" (the published v2 setup prompt, ib.scoring.prompt_version).
Side effects: writes the suite dir only; the release package is untouched. Exits non-zero on a duplicate
         rubric_id within one entry (the judge keys an entry's rubrics by rubric_id).

Then:  python -m ib eval --run <suite> --models <models.yaml> --output <eval dir> [--answer-judge ...]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from build_suite_from_release import PATTERN_TO_INTERNAL, read_jsonl


def audio_history(turns: list[dict], case_dir: Path, entry_dir: Path) -> list[dict]:
    """Map case turns to audio_history entries; an assistant entry carries the preceding block's index."""
    history, block = [], None
    for turn in turns:
        if turn["role"] == "user":
            block = 0 if block is None else block + 1
            history.append({"role": "user", "audio": os.path.relpath(case_dir / turn["audio"], entry_dir),
                            "block_index": block})
        elif block is None:
            raise SystemExit(f"{case_dir.name}: an assistant turn precedes the first room block")
        else:
            history.append({"role": "assistant", "text": turn["text"], "block_index": block})
    return history


def listed_tools(ev: dict) -> list[dict]:
    """Return the tools as eval.system_prompt lists them; abort if they differ from eval.tools."""
    tools = json.loads(ev["system_prompt"].split("Available functions:\n", 1)[1])
    if tools != ev["tools"]:
        raise SystemExit("eval.system_prompt lists other tools than eval.tools")
    return tools


def judge_line(line: dict) -> dict:
    """A heard line for the judge: floor acts are tagged by role so a backchannel is not read as a reply."""
    text = line["text"]
    if line.get("kind") == "floor":
        text = f"[{line.get('role', 'floor')}] {text}"
    elif line.get("truncated"):
        text = f"{text} [cut off]"
    return {"speaker": line["speaker"], "text": text}


def manifest_row(case: dict, release: Path, entry_dir: Path) -> dict:
    ev = case["eval"]
    history = audio_history(case["turns"], release / case["path"], entry_dir)
    audio_paths = [h["audio"] for h in history if h["role"] == "user"]
    return {
        "cell_id": case["case_id"],
        "prompt_version": "v2",
        "logic_case_mode": "base_semantic",
        "language": case["language"],
        "scene": case["scene"],
        "logic_pattern": PATTERN_TO_INTERNAL[case["pattern"]],
        "assistant_name": case["assistant_name"],
        "audio_paths": audio_paths,
        "mixed_testcase_audio_paths": audio_paths,
        "audio_history": history,
        "history_style": "text_turns",
        "visible_context_lines": [judge_line(t) for t in case["transcript"]],
        "logic_participants": case["participants"],
        "logic_scene_context": case["scene_context"],
        "logic_scene_date": case["scene_date"],
        "expected_action": ev["expected_action"],
        "expected_addressed_speaker": ev.get("expected_addressed_speaker"),
        "logic_standard_answer": ev.get("standard_answer"),
        "logic_expected_tool_calls": ev["tool_calls"],
        "logic_tool_validation": ev["tool_validation"],
        "logic_private_memory": ev.get("private_memory"),
        "logic_tools": listed_tools(ev),
        "rubric": ev.get("rubric"),
        "rubric_id": ev["rubric_ids"][0],
        "rubric_ids": ev["rubric_ids"],
        "answer_judge_inputs": ev.get("answer_judge_inputs"),
    }


def rubric_rows(rubrics: list[dict], case_ids: list[str]) -> list[dict]:
    """Return the package rubric rows of case_ids, in case order, with case_id renamed to cell_id."""
    by_case = defaultdict(list)
    for r in rubrics:
        by_case[r["case_id"]].append({"cell_id": r["case_id"], **{k: v for k, v in r.items() if k != "case_id"}})
    return [row for cid in case_ids for row in by_case[cid]]


def write_entry(entry_dir: Path, cases: list[dict], rubrics: list[dict], release: Path) -> tuple[int, int]:
    rows = [manifest_row(c, release, entry_dir) for c in cases]
    rub_rows = rubric_rows(rubrics, [c["case_id"] for c in cases])
    duplicated = sorted(rid for rid, n in Counter(r["rubric_id"] for r in rub_rows).items() if n > 1)
    if duplicated:
        raise SystemExit(f"{entry_dir.name}: duplicate rubric_id within one suite entry: {duplicated[:3]}")
    for sub, data in (("manifest/manifest.jsonl", rows), ("plans/rubrics.jsonl", rub_rows)):
        path = entry_dir / sub
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
    return len(rows), len(rub_rows)


def build_suite(release: Path, out: Path) -> dict[str, tuple[int, int]]:
    """Write every entry of the suite; return {entry: (cases, rubric rows)}."""
    cases = read_jsonl(release / "index" / "cases.jsonl")
    rubrics = read_jsonl(release / "index" / "rubrics.jsonl")
    groups = defaultdict(list)
    for c in cases:
        groups[f"{c['language']}_{c['pattern']}"].append(c)
    return {entry: write_entry(out / entry, members, rubrics, release) for entry, members in sorted(groups.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    for entry, (n_cases, n_rubrics) in build_suite(Path(args.release).resolve(), out).items():
        print(f"{entry}: {n_cases} cases, {n_rubrics} rubric rows")
    print(f"SUITE_READY {out}")


if __name__ == "__main__":
    main()
