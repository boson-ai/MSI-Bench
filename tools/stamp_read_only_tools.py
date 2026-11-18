"""stamp_read_only_tools — write each row's read-only tool names into its tool-validation spec.

Calling spec:
    python tools/stamp_read_only_tools.py \
        --labels tools/data/tool_side_effects_v2.jsonl --manifest IN.jsonl --output OUT.jsonl

    stamp_row(row, read_only_by_key) -> row      pure; read_only_by_key = {tool_key: read_only}

Inputs:  a side-effect label file (rows {"key", "read_only", ...}) and a manifest whose rows carry
         `logic_tool_validation` and `logic_tools` (Anthropic tool definitions).
Outputs: OUT.jsonl, the same rows in the same order. A row whose spec has call_policy no_calls or
         exact_multiset gets `read_only_call_names` = sorted names of its tools labelled read_only,
         minus the spec's forbidden_call_names (the key is left out when that list is empty). Rows
         without `logic_tools`, without a spec, or with another policy pass through unchanged.
         Stamping a stamped row changes nothing.
Side effects: writes OUT only. Refuses when OUT is IN or OUT exists. A tool with no label is an
         error naming the cell and the tool; nothing is written then.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ib.scoring.tool_validation import resolve_tool_validation_spec
from label_tool_side_effects import read_jsonl, tool_key, write_jsonl

STAMPED_POLICIES = ("no_calls", "exact_multiset")


def stamp_row(row: dict, read_only_by_key: dict[str, bool]) -> dict:
    """Return the row with its spec's read_only_call_names recomputed from the labels."""
    spec = row.get("logic_tool_validation")
    tools = row.get("logic_tools")
    if not spec or not tools or spec.get("call_policy") not in STAMPED_POLICIES:
        return row
    read_only = set()
    for tool in tools:
        key = tool_key(tool["name"], tool.get("description", ""))
        if key not in read_only_by_key:
            raise ValueError(f"{row.get('cell_id')}: tool {tool['name']!r} (key {key}) has no side-effect label")
        if read_only_by_key[key]:
            read_only.add(tool["name"])
    names = sorted(read_only - set(spec.get("forbidden_call_names", [])))
    stamped = {k: v for k, v in spec.items() if k != "read_only_call_names"}
    if names:
        stamped["read_only_call_names"] = names
    # Fail while stamping, not at eval time, when the stamped spec does not load.
    resolve_tool_validation_spec({"logic_tool_validation": stamped})
    return {**row, "logic_tool_validation": stamped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.manifest.resolve():
        raise SystemExit("--output must not be the input manifest")
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a fresh output path")
    read_only_by_key = {r["key"]: r["read_only"] for r in read_jsonl(args.labels)}
    rows = read_jsonl(args.manifest)
    stamped = [stamp_row(row, read_only_by_key) for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, stamped)
    with_names = sum(bool((r.get("logic_tool_validation") or {}).get("read_only_call_names")) for r in stamped)
    print(f"{len(rows)} rows, {with_names} with read-only tool names -> {args.output}")


if __name__ == "__main__":
    main()
