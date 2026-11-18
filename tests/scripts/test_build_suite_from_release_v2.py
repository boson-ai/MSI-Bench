"""tools/build_suite_from_release_v2.py over a two-case fixture package."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ib.scoring.multiturn_audio import history_messages, setup_prompt_for
from ib.scoring.openai_understanding import _audio_paths_for

EXTRAS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(EXTRAS))
SPEC = importlib.util.spec_from_file_location("build_suite_from_release_v2", EXTRAS / "build_suite_from_release_v2.py")
v2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v2)

TOOL = {"name": "lookup", "description": "Look up.", "input_schema": {
    "type": "object", "properties": {"period": {"type": "string", "enum": ["now"]}}, "required": ["period"]}}
LISTING = "Available functions:\n" + json.dumps([TOOL], indent=2)


def _sorted_keys(value):
    return json.loads(json.dumps(value, sort_keys=True))


def _case(case_id: str, language: str, pattern: str, blocks: int) -> dict:
    turns, transcript = [], []
    for i in range(blocks):
        line = {"line_id": f"L{i + 1}", "speaker": "Ana", "text": f"line {i}", "kind": "main", "role": "target_user"}
        turns.append({"role": "user", "audio": f"blocks/{i:02d}.wav", "lines": [line]})
        transcript.append(line)
        if i < blocks - 1:
            turns.append({"role": "assistant", "text": f"reply {i}"})
            transcript.append({"line_id": f"L{i + 1}", "speaker": "assistant", "text": f"reply {i}"})
    path = f"data/{language}/x/scene/{case_id}"
    return {
        "case_id": case_id, "language": language, "pattern": pattern, "scene": "scene", "scene_date": "2026-08-27",
        "assistant_name": "Nova", "scene_context": "A room.", "participants": [{"name": "Ana", "voice_label": "<|S1|>"}],
        "turns": turns, "transcript": transcript, "path": path,
        "eval": {"expected_action": "respond", "expected_addressed_speaker": "Ana", "standard_answer": "ok",
                 "tool_calls": [{"name": "lookup", "arguments": {"period": "now"}}],
                 "tool_validation": {"call_policy": "no_calls"}, "tools": _sorted_keys([TOOL]),
                 "system_prompt": "Your name: Nova.\n" + LISTING, "private_memory": None,
                 "rubric": "Assess.", "rubric_ids": [f"{case_id}:logic"],
                 "answer_judge_inputs": [{"layer": "O1", "rubric_id": f"{case_id}:logic"}]},
    }


def _rubric(case_id: str, rubric_id: str | None = None) -> dict:
    return {"case_id": case_id, "rubric_id": rubric_id or f"{case_id}:logic", "judge_instruction": "Assess.",
            "atomic_criteria": [{"criterion": "Says ok.", "criterion_id": f"{case_id}:rubric:00",
                                 "criterion_index": 0, "dimension": "response.required_behavior"}]}


def _package(root: Path, cases: list[dict], rubrics: list[dict]) -> Path:
    release = root / "msi-bench-v2"
    (release / "index").mkdir(parents=True)
    for name, rows in (("cases", cases), ("rubrics", rubrics)):
        (release / "index" / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for case in cases:
        for turn in case["turns"]:
            if turn["role"] == "user":
                wav = release / case["path"] / turn["audio"]
                wav.parent.mkdir(parents=True, exist_ok=True)
                wav.write_bytes(b"RIFF")
    return release


@pytest.fixture
def built(tmp_path):
    cases = [_case("msi2-en-a", "en", "selective_disclosure", 2), _case("msi2-zh-b", "zh", "scope_tracking", 1)]
    release = _package(tmp_path, cases, [_rubric("msi2-en-a"), _rubric("msi2-zh-b")])
    suite = tmp_path / "suite"
    return release, suite, v2.build_suite(release, suite)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_one_entry_per_language_and_paper_pattern(built):
    _, suite, counts = built

    assert counts == {"en_selective_disclosure": (1, 1), "zh_scope_tracking": (1, 1)}
    [row] = _rows(suite / "en_selective_disclosure" / "manifest" / "manifest.jsonl")
    assert row["logic_pattern"] == "disclosure_clause_in_instruction"
    assert json.dumps(row["logic_tools"]) == json.dumps([TOOL])
    assert row["logic_expected_tool_calls"] == [{"name": "lookup", "arguments": {"period": "now"}}]
    assert row["visible_context_lines"][1] == {"speaker": "assistant", "text": "reply 0"}


def test_history_replays_every_block_through_the_real_loader(built):
    release, suite, _ = built
    manifest = suite / "en_selective_disclosure" / "manifest" / "manifest.jsonl"
    [row] = _rows(manifest)

    paths = _audio_paths_for(row, manifest)
    messages = history_messages(row, paths)

    case_dir = release / "data/en/x/scene/msi2-en-a"
    assert [p.resolve() for p in paths] == [case_dir / "blocks/00.wav", case_dir / "blocks/01.wav"]
    assert [(m.role, m.audio_path and m.audio_path.resolve(), m.assistant_text) for m in messages] == [
        ("user", case_dir / "blocks/00.wav", None),
        ("assistant", None, "reply 0"),
        ("user", case_dir / "blocks/01.wav", None),
    ]
    assert [m.final_user_turn for m in messages] == [False, False, True]
    assert [h["block_index"] for h in row["audio_history"]] == [0, 0, 1]
    assert LISTING in setup_prompt_for(row)


def test_rubric_rows_are_copied_with_cell_id(built):
    _, suite, _ = built

    [rubric] = _rows(suite / "zh_scope_tracking" / "plans" / "rubrics.jsonl")

    assert rubric == {"cell_id": "msi2-zh-b", **{k: v for k, v in _rubric("msi2-zh-b").items() if k != "case_id"}}


def test_tools_that_differ_from_the_system_prompt_listing_abort(tmp_path):
    case = _case("msi2-en-a", "en", "scope_tracking", 1)
    case["eval"]["tools"] = []
    release = _package(tmp_path, [case], [_rubric("msi2-en-a")])

    with pytest.raises(SystemExit, match="lists other tools"):
        v2.build_suite(release, tmp_path / "suite")


def test_duplicate_rubric_id_within_an_entry_aborts(tmp_path):
    cases = [_case("msi2-en-a", "en", "scope_tracking", 1), _case("msi2-en-b", "en", "scope_tracking", 1)]
    release = _package(tmp_path, cases, [_rubric("msi2-en-a", "same:logic"), _rubric("msi2-en-b", "same:logic")])

    with pytest.raises(SystemExit, match="en_scope_tracking: duplicate rubric_id"):
        v2.build_suite(release, tmp_path / "suite")
