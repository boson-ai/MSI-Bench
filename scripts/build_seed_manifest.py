#!/usr/bin/env python3
"""Build an Interaction-Bench seed manifest from local datasets.

The builder is intentionally stdlib-only. It normalizes local WearVox and
HumDial-FDBench examples into a shared audio-to-answer / speak-or-stay-silent
manifest. Seed labels imported from datasets not authored for this taxonomy are
marked as `source` or `inferred` so they can be audited before release.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ALLOWED_ACTIONS = {"respond", "ignore", "wait", "clarify", "incorporate"}

WEARVOX_LICENSE = "cc-by-nc-4.0"
HUMDIAL_LICENSE = "apache-2.0"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def bool_from_yes_no(value: Any) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    return None






def prompt_to_instruction(prompt: Any) -> str:
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for msg in prompt:
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                role = msg.get("role") or "message"
                parts.append(f"{role}: {msg['content']}")
        if parts:
            return "\n\n".join(parts)
    return "Answer only if the audio is addressed to the assistant."

def infer_wearvox_axis_a(metadata: dict[str, Any]) -> str:
    """Map WearVox real side-talk metadata to Axis A behavior-setting slugs.

    WearVox is a device/context source, not an Axis A class. The released
    metadata is coarse, so this function makes only conservative mappings and
    leaves weakly supported rows in a broad public/social bucket for audit.
    """
    environment = str(metadata.get("environment") or "").strip().lower()
    noise_type = str(metadata.get("_noise_type") or "").strip().lower()
    location = str(metadata.get("location") or "").strip().lower()

    if environment == "outdoors":
        return "public_civic_space"
    if "office" in noise_type:
        return "work_professional"
    if "music" in noise_type:
        return "leisure_media_social"
    if environment == "indoors" and any(token in location for token in ["room", "area"]):
        return "work_professional"
    return "personal_admin_communication"

def infer_perturbation_families(metadata: dict[str, Any]) -> list[str]:
    families: set[str] = set()
    joined = " ".join(str(metadata.get(k, "")) for k in ["environment", "noise_level", "_noise_type", "location"]).lower()

    if any(token in joined for token in ["conversation", "whisper", "bystander"]):
        families.add("overlap")
    if any(token in joined for token in ["car", "bus", "truck", "motorcycle", "road", "parking"]):
        families.add("continuous_noise")
    if any(token in joined for token in ["wind", "leaves", "construction", "vacuum", "music", "crowd", "street"]):
        families.add("continuous_noise")
    if any(token in joined for token in ["large room", "hallway", "medium room"]):
        families.add("reverb")
    if any(token in joined for token in ["phone", "notification", "alarm", "siren", "horn", "honk", "knock"]):
        families.add("transient_noise")
    if metadata.get("bystander__y_n_") == "Yes":
        families.add("overlap")
    return sorted(families) or ["naturalistic"]


def wearvox_rows(root: Path, limit: int | None) -> list[dict[str, Any]]:
    data_path = root / "data_public.json"
    metadata_path = root / "audio_metadata.json"
    if not data_path.exists():
        return []

    rows = load_json(data_path)
    metadata_by_id: dict[str, dict[str, Any]] = {}
    if metadata_path.exists():
        for item in load_json(metadata_path):
            metadata_by_id[str(item.get("id"))] = item.get("audio_metadata", {}) or {}

    out: list[dict[str, Any]] = []
    for item in rows:
        source_id = str(item.get("id", ""))
        task_name = str(item.get("task", "unknown"))
        if task_name != "non-assistant-directed":
            continue

        metadata = metadata_by_id.get(source_id, {})
        audio_primary = str(root / str(item.get("audio_query", "")))
        audio_mc = item.get("audio_query_mc")
        out.append(
            {
                "id": f"wearvox-{source_id}",
                "source": {
                    "dataset": "WearVox",
                    "license": WEARVOX_LICENSE,
                    "path": str(data_path),
                    "source_id": source_id,
                },
                "audio": {
                    "primary": audio_primary,
                    "multichannel": str(root / str(audio_mc)) if audio_mc else None,
                    "duration_s": None,
                    "sample_rate_hz": None,
                    "channels": None,
                },
                "scenario": {
                    "domain": infer_wearvox_axis_a(metadata),
                    "interaction_class": "must_ignore",
                    "base_id": source_id,
                    "speaker_roles": ["wearer", "assistant", "other_speaker"],
                    "semantic_perturbations": ["side_conversation"],
                },
                "task": {
                    "type": "side_conversation_rejection",
                    "instruction": prompt_to_instruction(item.get("text_prompt")),
                    "expected_action": "ignore",
                    "ground_truth": "SILENCE",
                    "transcript": item.get("gt_transcript"),
                },
                "acoustics": {
                    "environment": metadata.get("environment"),
                    "noise_level": metadata.get("noise_level"),
                    "noise_type": metadata.get("_noise_type"),
                    "location": metadata.get("location"),
                    "bystander": bool_from_yes_no(metadata.get("bystander__y_n_")),
                    "perturbation_families": infer_perturbation_families(metadata),
                },
                "labels": {
                    "label_confidence": "source",
                    "rationale": "WearVox non-assistant-directed task maps to Axis B side_conversation and must-ignore; QA/tool/translation rows are excluded.",
                },
                "split": "seed",
                "notes": "Imported from WearVox non-assistant-directed real data only; Axis A is inferred per row from audio metadata/transcript and should be audited.",
            }
        )
        if limit and len(out) >= limit:
            break
    return out


HUMDIAL_FOLDER_MAP = {
    "ask": ("must_respond", "respond", "single_user_addressed"),
    "repeat": ("must_respond", "respond", "single_user_addressed"),
    "deny": ("must_respond", "respond", "single_user_addressed"),
    "shift": ("must_respond", "respond", "single_user_addressed"),
    "backchannel": ("must_wait", "wait", "single_user_addressed"),
    "pause": ("must_wait", "wait", "single_user_addressed"),
    "wait": ("must_wait", "wait", "single_user_addressed"),
    "talk_to_others": ("must_ignore", "ignore", "side_conversation"),
    "others_talk_to_user_before": ("must_ignore", "ignore", "side_conversation"),
    "others_talk_to_user_after": ("must_ignore", "ignore", "side_conversation"),
}


def humdial_text_for(stem_path: Path) -> str | None:
    for suffix in ["_timestamp.json", "_add.json", ".json"]:
        candidate = stem_path.with_name(stem_path.name + suffix)
        if candidate.exists():
            try:
                data = load_json(candidate)
            except Exception:
                continue
            if isinstance(data, dict):
                if isinstance(data.get("text"), str):
                    return data["text"]
                segments = data.get("speech_segments")
                if isinstance(segments, list):
                    parts = [str(seg.get("text", "")).strip() for seg in segments if isinstance(seg, dict)]
                    parts = [p for p in parts if p]
                    if parts:
                        return " ".join(parts)
    return None


def humdial_base_stem(wav: Path) -> str:
    stem = wav.stem
    if stem.startswith("clean_"):
        stem = stem[len("clean_") :]
    if stem.endswith("_add"):
        stem = stem[: -len("_add")]
    return stem


def humdial_rows(root: Path, limit: int | None) -> list[dict[str, Any]]:
    if not root.exists():
        return []

    by_folder_count: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    wavs = sorted(root.rglob("*.wav"))
    for wav in wavs:
        folder = wav.parent.name
        if folder not in HUMDIAL_FOLDER_MAP:
            continue
        if limit and by_folder_count[folder] >= limit:
            continue
        interaction_class, expected_action, semantic = HUMDIAL_FOLDER_MAP[folder]
        rel_parts = wav.relative_to(root).parts
        language_split = rel_parts[0] if rel_parts else "unknown"
        base = humdial_base_stem(wav)
        transcript = humdial_text_for(wav.with_suffix(""))
        is_clean = wav.name.startswith("clean_")
        has_add = wav.stem.endswith("_add")
        perturbations = ["naturalistic"]
        if not is_clean:
            perturbations.append("overlap")
        if has_add:
            perturbations.append("semantic_interruption")

        source_id = "/".join(rel_parts)
        ground_truth = "SILENCE" if expected_action in {"ignore", "wait"} else None
        rationale = f"HumDial-FDBench folder '{folder}' maps to Axis B '{semantic}' and {interaction_class}/{expected_action}; Axis A transcript audit recommended."
        out.append(
            {
                "id": "humdial-" + source_id.replace("/", "-").replace(".wav", ""),
                "source": {
                    "dataset": "HumDial-FDBench",
                    "license": HUMDIAL_LICENSE,
                    "path": str(root),
                    "source_id": source_id,
                },
                "audio": {
                    "primary": str(wav),
                    "multichannel": None,
                    "duration_s": None,
                    "sample_rate_hz": None,
                    "channels": None,
                },
                "scenario": {
                    "domain": "personal_admin_communication",
                    "interaction_class": interaction_class,
                    "base_id": f"humdial-{language_split}-{folder}-{base}",
                    "speaker_roles": ["user", "assistant", "other_speaker"],
                    "semantic_perturbations": [semantic],
                },
                "task": {
                    "type": "full_duplex_interaction",
                    "instruction": "Decide whether the assistant should respond, stay silent, wait, or incorporate a speaker; answer only when engagement is appropriate.",
                    "expected_action": expected_action,
                    "ground_truth": ground_truth,
                    "transcript": transcript,
                },
                "acoustics": {
                    "environment": None,
                    "noise_level": None,
                    "noise_type": None,
                    "location": None,
                    "bystander": expected_action == "ignore",
                    "perturbation_families": sorted(set(perturbations)),
                },
                "labels": {
                    "label_confidence": "inferred",
                    "rationale": rationale,
                },
                "split": "seed",
                "notes": "Imported from HumDial-FDBench real test folder structure; Axis B is folder-derived, Axis A is a provisional transcript-level label pending audit.",
            }
        )
        by_folder_count[folder] += 1
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def count(path: tuple[str, ...]) -> dict[str, int]:
        c: Counter[str] = Counter()
        for row in rows:
            value: Any = row
            for key in path:
                value = value.get(key, {}) if isinstance(value, dict) else None
            c[str(value)] += 1
        return dict(sorted(c.items()))

    perturbations: Counter[str] = Counter()
    for row in rows:
        perturbations.update(row["acoustics"].get("perturbation_families", []))

    return {
        "total_rows": len(rows),
        "by_source": count(("source", "dataset")),
        "by_expected_action": count(("task", "expected_action")),
        "by_interaction_class": count(("scenario", "interaction_class")),
        "by_domain": count(("scenario", "domain")),
        "by_label_confidence": count(("labels", "label_confidence")),
        "by_perturbation_family": dict(sorted(perturbations.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wearvox", type=Path, help="Path to local WearVox directory")
    parser.add_argument("--humdial", type=Path, help="Path to HumDial-FDBench/test directory")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL manifest")
    parser.add_argument("--summary", type=Path, help="Optional JSON summary path")
    parser.add_argument("--limit-per-source", type=int, default=50, help="Max rows per source/folder seed slice; 0 means unlimited")
    args = parser.parse_args()

    limit = args.limit_per_source or None
    rows: list[dict[str, Any]] = []
    if args.wearvox:
        rows.extend(wearvox_rows(args.wearvox, limit))
    if args.humdial:
        rows.extend(humdial_rows(args.humdial, limit))

    rows.sort(key=lambda r: r["id"])
    count = write_jsonl(args.output, rows)
    summary = summarize(rows)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"written": count, "output": str(args.output), "summary": summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
