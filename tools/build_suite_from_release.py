"""build_suite_from_release — adapt an MSI-Bench release package for `ib eval`.

Calling spec:
    python tools/build_suite_from_release.py --release <msi-bench dir> --output <suite dir>
        [--probes]

Inputs:  a release package (data/<lang>/<pattern>/<scene>/<case_id>/,
         index/cases.jsonl, index/rubrics.jsonl; with --probes also
         probes/speak_probe.jsonl, probes/hear_probe.jsonl, probes/audio_live/).
Outputs: <suite dir>/<entry>/manifest/manifest.jsonl + <entry>/plans/rubrics.jsonl,
         one entry per (language, pattern); audio paths are relative to the entry
         and resolve into the release package. Default emits base rows only;
         --probes additionally derives speak_time_probe rows (one per case) and
         hear_time_probe rows (the shipped subset) from the probe protocol files
         — no probe row references any audio outside the release package.
Side effects: writes the suite dir only; the release package is untouched.

Then:  ib eval --run <suite dir> --models <models.yaml> --output <eval dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

# paper snake_case pattern name -> internal manifest id
PATTERN_TO_INTERNAL = {
    "sequential_constraint_integration": "constraint_attribution_under_interleaving",
    "speaker_authority_constraint": "authority_gated_override",
    "selective_disclosure": "disclosure_clause_in_instruction",
    "scope_tracking": "distributed_parameters",
    "background_speech_retrieval": "eavesdropping",
    "constraint_prioritization": "hard_vs_soft_constraint",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def manifest_row(case: dict, release: Path, entry_dir: Path) -> dict:
    case_dir = release / case["path"]
    ev, turns = case["eval"], case["turns"]
    visible = [
        {
            "speaker": "assistant" if t["role"] == "assistant" else t["speaker"],
            "addressed_to": t["addressed_to"],
            "text": t["text"],
        }
        for t in turns
    ]
    audio_paths = [
        os.path.relpath(case_dir / t["audio"], entry_dir)
        for t in turns
        if t.get("audio")
    ]
    return {
        "cell_id": case["case_id"],
        "logic_case_mode": "base_semantic",
        "language": case["language"],
        "scene": case["scene"],
        "logic_pattern": PATTERN_TO_INTERNAL[case["pattern"]],
        "audio_paths": audio_paths,
        "visible_context_lines": visible,
        "logic_participants": case.get("participants"),
        "logic_scene_context": case.get("scene_context"),
        "logic_scene_date": case.get("scene_date"),
        "expected_action": ev["expected_action"],
        "expected_addressed_speaker": ev.get("expected_addressed_speaker"),
        "logic_standard_answer": ev.get("standard_answer"),
        "logic_available_functions": ev.get("available_functions"),
        "logic_tool_validation": ev.get("tool_validation"),
        "logic_private_memory": ev.get("private_memory"),
        "rubric": ev.get("rubric"),
        "rubric_id": (ev.get("rubric_ids") or [None])[0],
        "rubric_ids": ev.get("rubric_ids"),
        "answer_judge_inputs": ev.get("answer_judge_inputs"),
        "sibling_group": (case.get("provenance") or {}).get("sibling_group"),
    }


# --- probe-row derivation (--probes) ------------------------------------------

_MAX_TURN_INDEX = 8  # user_turn_1..8 field window; providers append the rest

_SPEAK_SAMPLING = (  # (field, salt, low, high) — mirrors ib.scoring.probe_sampling
    ("logic_speak_truncate_fraction", "speak_truncate", 0.1, 0.5),
    ("logic_speak_inject_offset_s", "speak_inject", 1.0, 2.0),
)


def _sampled(key: str, salt: str, low: float, high: float) -> float:
    digest = hashlib.md5(f"{salt}:{key}".encode("utf-8")).hexdigest()
    return low + (high - low) * int(digest[:8], 16) / 0x100000000


def _turn_fields(row: dict) -> dict:
    """Bind visible lines to the explicit turn fields (turn = line index + 1)."""
    fields: dict = {}
    audio = list(row["audio_paths"])
    for i, line in enumerate(row["visible_context_lines"][:_MAX_TURN_INDEX]):
        turn = i + 1
        if line["speaker"] == "assistant":
            fields[f"assistant_turn_{turn}_transcript"] = line["text"]
        else:
            fields[f"user_turn_{turn}_audio"] = audio.pop(0)
            fields[f"user_turn_{turn}_transcript"] = line["text"]
    return fields


def speak_row(base_row: dict, probe: dict, release: Path, entry_dir: Path) -> dict:
    """Derive a speak_time_probe manifest row: base row + live event fields."""
    row = dict(base_row)
    row["cell_id"] = base_row["cell_id"] + "s"
    row["logic_case_mode"] = "speak_time_probe"
    row["logic_live_events"] = [probe["live_event"]]
    row["logic_live_anchor"] = probe["anchor"]
    row["logic_live_expansion_type"] = probe["expansion_type"]
    row["logic_live_expansion_types"] = [probe["expansion_type"]]
    row["logic_expected_live_behavior"] = probe["expected_live_behavior"]
    row["logic_live_event_audio_paths"] = [
        os.path.relpath(release / probe["audio"], entry_dir)
    ]
    # Pin the per-case stimulus parameters to the values the internal campaign
    # sampled (probe_sampling hashes cell_id; release ids differ from internal).
    for field, salt, low, high in _SPEAK_SAMPLING:
        row[field] = _sampled(probe["source_cell_id"], salt, low, high)
    row.update(_turn_fields(base_row))
    return row


def hear_row(base_row: dict, probe: dict) -> dict:
    """Derive a hear_time_probe manifest row: base row truncated to the prefix."""
    visible = base_row["visible_context_lines"][: probe["n_lines"]]
    last, pl = visible[-1], probe["probe_line"]
    if (last["speaker"], last["text"]) != (pl["speaker"], pl["text"]):
        raise SystemExit(f"{base_row['cell_id']}: hear probe line is not the prefix end")
    kept_human = sum(1 for v in visible if v["speaker"] != "assistant")
    row = dict(base_row)
    row["cell_id"] = base_row["cell_id"] + "h"
    row["logic_case_mode"] = "hear_time_probe"
    row["visible_context_lines"] = visible
    row["audio_paths"] = base_row["audio_paths"][:kept_human]
    row["expected_action"] = "ignore"
    row["expected_addressed_speaker"] = None
    # Hear probes carry no rubric: scoring is the deterministic silence rule.
    for field in ("logic_standard_answer", "rubric", "rubric_id", "rubric_ids",
                  "answer_judge_inputs"):
        row[field] = None
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--probes", action="store_true",
                    help="also derive speak/hear probe rows from probes/*.jsonl")
    args = ap.parse_args()
    release = Path(args.release).resolve()
    out = Path(args.output).resolve()
    cases = read_jsonl(release / "index" / "cases.jsonl")
    rubrics = read_jsonl(release / "index" / "rubrics.jsonl")
    rub_by_case = defaultdict(list)
    for r in rubrics:
        row = {k: v for k, v in r.items() if k != "case_id"}
        row["cell_id"] = r["case_id"]
        rub_by_case[r["case_id"]].append(row)

    speak_by_case, hear_by_case = {}, {}
    if args.probes:
        for p in read_jsonl(release / "probes" / "speak_probe.jsonl"):
            speak_by_case[p["case_id"]] = p
        for p in read_jsonl(release / "probes" / "hear_probe.jsonl"):
            hear_by_case[p["case_id"]] = p

    groups = defaultdict(list)
    for c in cases:
        groups[f"{c['language']}_{c['pattern']}"].append(c)
    for entry, members in sorted(groups.items()):
        entry_dir = out / entry
        (entry_dir / "manifest").mkdir(parents=True, exist_ok=True)
        (entry_dir / "plans").mkdir(parents=True, exist_ok=True)
        rows = [manifest_row(c, release, entry_dir) for c in members]
        n_speak = n_hear = 0
        if args.probes:
            for base in list(rows):
                probe = speak_by_case.get(base["cell_id"])
                if probe is None:
                    raise SystemExit(f"{base['cell_id']}: no speak probe row")
                rows.append(speak_row(base, probe, release, entry_dir))
                n_speak += 1
                if base["cell_id"] in hear_by_case:
                    rows.append(hear_row(base, hear_by_case[base["cell_id"]]))
                    n_hear += 1
        (entry_dir / "manifest" / "manifest.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        rub_rows = [row for c in members for row in rub_by_case[c["case_id"]]]
        # ib eval's judge keys this entry's rubrics by rubric_id; release
        # rubric_ids collide ACROSS entries (sibling cases of other patterns/
        # languages reuse the source cell id), so a within-entry duplicate
        # would silently judge cases against the wrong rubric.
        duplicated = sorted(
            rid for rid, n in Counter(r["rubric_id"] for r in rub_rows).items() if n > 1
        )
        if duplicated:
            raise SystemExit(
                f"{entry}: duplicate rubric_id within one suite entry: {duplicated[:3]}"
            )
        (entry_dir / "plans" / "rubrics.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rub_rows), encoding="utf-8")
        probe_note = f", {n_speak} speak + {n_hear} hear probe rows" if args.probes else ""
        print(f"{entry}: {len(members)} cases, {len(rub_rows)} rubric rows{probe_note}")
    print(f"SUITE_READY {out}")


if __name__ == "__main__":
    main()
