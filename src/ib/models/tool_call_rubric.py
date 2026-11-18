"""tool_call_rubric — shared identity for the derived tool-call rubric.

Calling spec:
    dimension = TOOL_CALL_ARGS_DIMENSION

When enabled, the planner derives this rubric from gold tool calls and the
answer judge scores it semantically. The option defaults on. Deterministic tool
validation is a separate, always-on metric path.
Side effects: none.
"""

TOOL_CALL_ARGS_DIMENSION = "tool.call_args"
