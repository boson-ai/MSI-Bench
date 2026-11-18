"""label_tool_side_effects — label every MSI-Bench v2 tool as read_only or state_changing.

Calling spec:
    python tools/label_tool_side_effects.py label \
        --provider <p> --model <m> --release-index <cases.jsonl> --output <labels.jsonl>
    python tools/label_tool_side_effects.py merge \
        --labels <labeler>=<labels.jsonl> --labels <labeler>=<labels.jsonl> --output <slim.jsonl>

label:  one LLM labels each distinct (name, description) of the release index's eval.tools. Output rows
        {"key", "name", "description", "label", "ambiguous"}; key = tool_key(name, description).
        40 tools per call, 8 calls at a time; finished batches are cached in <output>.batches/ so a
        rerun repeats only the missing ones. Side effects: LLM calls, writes <output> and its cache.
merge:  turns labeller outputs into the slim committed file, rows {"key", "name", "read_only",
        "labels": {<labeler>: label}, "ambiguous"} sorted by (name, key). read_only is true only when
        every labeler says read_only and none flags ambiguous. The labeller outputs must cover the
        same keys. Side effects: writes <output> only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ib.cli.environment import load_cli_environment
from ib.llm.service import complete_json
from ib.llm.types import LlmRequest

BATCH_SIZE = 40
PARALLEL_CALLS = 8
SYSTEM = """You label assistant tools (function definitions) by whether calling them has a side effect.

read_only: a call only retrieves, looks up, checks, searches, calculates, estimates or compares information and returns it. It changes nothing in the world: no record, booking, order, reminder, alarm, timer, note, list, message, payment, device or media state is created, changed, sent, started, stopped or cancelled. Calling it twice, or with wrong arguments, leaves no lasting effect.

state_changing: anything else. This includes sending or notifying, booking, ordering, paying, setting reminders or alarms or timers, saving or logging or recording, adding to a list, controlling a device, playing media, starting navigation, submitting a request, and cancelling or changing any of these.

When the description leaves open whether the call commits something (for example a "plan_..." or "prepare_..." tool that may save or activate the plan), label it state_changing and set ambiguous to true. Judge by the description first, the name second.

Reply with JSON only: {"labels": [{"id": <int>, "label": "read_only" | "state_changing", "ambiguous": true | false}, ...]} with one entry per tool, same ids."""


def tool_key(name: str, description: str) -> str:
    """Stable id of one tool definition: first 12 hex chars of md5(name + newline + description)."""
    return hashlib.md5((name + "\n" + description).encode()).hexdigest()[:12]


def distinct_tools(cases: list[dict]) -> list[dict]:
    """One entry per distinct tool definition in the cases' eval.tools, sorted by (name, key)."""
    seen: dict[str, dict] = {}
    for case in cases:
        for tool in case["eval"]["tools"]:
            key = tool_key(tool["name"], tool["description"])
            seen.setdefault(key, {
                "key": key, "name": tool["name"], "description": tool["description"],
                "params": sorted((tool.get("input_schema") or {}).get("properties", {})),
            })
    return sorted(seen.values(), key=lambda t: (t["name"], t["key"]))


def batch_listing(batch: list[dict]) -> str:
    return "\n".join(f'{i}. {t["name"]}({", ".join(t["params"])}): {t["description"]}'
                     for i, t in enumerate(batch))


def labelled_rows(reply: dict, batch: list[dict]) -> list[dict] | None:
    """The batch's label rows, or None when the reply does not label every id exactly once."""
    by_id = {int(x["id"]): x for x in reply.get("labels", [])
             if x.get("label") in ("read_only", "state_changing")}
    if set(by_id) != set(range(len(batch))):
        return None
    return [{"key": t["key"], "name": t["name"], "description": t["description"],
             "label": by_id[i]["label"], "ambiguous": bool(by_id[i].get("ambiguous"))}
            for i, t in enumerate(batch)]


def merge_labels(by_labeler: dict[str, list[dict]]) -> list[dict]:
    """Slim rows from labeller outputs; read_only only when every labeler agrees and none is unsure."""
    indexed = {labeler: {r["key"]: r for r in rows} for labeler, rows in by_labeler.items()}
    key_sets = {labeler: set(rows) for labeler, rows in indexed.items()}
    if len({frozenset(keys) for keys in key_sets.values()}) != 1:
        raise ValueError(f"labeller outputs cover different tool keys: { {k: len(v) for k, v in key_sets.items()} }")
    first = next(iter(indexed.values()))
    merged = []
    for key, row in first.items():
        votes = {labeler: rows[key] for labeler, rows in indexed.items()}
        ambiguous = any(v["ambiguous"] for v in votes.values())
        merged.append({
            "key": key,
            "name": row["name"],
            "read_only": not ambiguous and all(v["label"] == "read_only" for v in votes.values()),
            "labels": {labeler: v["label"] for labeler, v in votes.items()},
            "ambiguous": ambiguous,
        })
    return sorted(merged, key=lambda r: (r["name"], r["key"]))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def label_batch(provider: str, model: str, batch: list[dict], cache: Path) -> list[dict]:
    if cache.exists():
        return json.loads(cache.read_text())
    for _attempt in range(3):
        reply = complete_json(LlmRequest(provider=provider, model=model, max_tokens=6000, timeout_s=180, messages=[
            {"role": "system", "content": SYSTEM}, {"role": "user", "content": batch_listing(batch)}]))
        rows = labelled_rows(reply, batch)
        if rows is not None:
            cache.write_text(json.dumps(rows, ensure_ascii=False))
            return rows
    raise RuntimeError(f"batch {cache.name}: incomplete labels after 3 attempts")


def run_label(args: argparse.Namespace) -> None:
    load_cli_environment()
    cache_dir = Path(str(args.output) + ".batches")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tools = distinct_tools(read_jsonl(args.release_index))
    batches = [tools[i:i + BATCH_SIZE] for i in range(0, len(tools), BATCH_SIZE)]
    with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
        results = list(pool.map(
            lambda item: label_batch(args.provider, args.model, item[1], cache_dir / f"{item[0]:03d}.json"),
            enumerate(batches)))
    rows = [r for batch in results for r in batch]
    write_jsonl(args.output, rows)
    n = sum(r["label"] == "read_only" for r in rows)
    print(f"{args.model}: {len(rows)} tools, read_only {n}, state_changing {len(rows) - n}, "
          f"ambiguous {sum(r['ambiguous'] for r in rows)}")


def run_merge(args: argparse.Namespace) -> None:
    by_labeler = {}
    for item in args.labels:
        labeler, sep, path = item.partition("=")
        if not sep or not labeler or not path:
            raise SystemExit(f"--labels takes <labeler>=<file>, got {item!r}")
        by_labeler[labeler] = read_jsonl(Path(path))
    rows = merge_labels(by_labeler)
    write_jsonl(args.output, rows)
    print(f"{len(rows)} tools, read_only {sum(r['read_only'] for r in rows)}, "
          f"ambiguous {sum(r['ambiguous'] for r in rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    label = sub.add_parser("label")
    label.add_argument("--provider", required=True)
    label.add_argument("--model", required=True)
    label.add_argument("--release-index", type=Path, required=True)
    label.add_argument("--output", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--labels", action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"label": run_label, "merge": run_merge}[args.command](args)


if __name__ == "__main__":
    main()
