"""
decision_agent_prompt.py -- the single canonical system prompt for the
final tool-choosing decision agent (this project's "A" role), matching
EXACTLY what masflow/run_029_full_benchmark.py uses to produce every
family-sweep result (a_only_hd, verbatim_hd/ba_hd, para_hd) -- the
numbers every other experiment in this project is compared against.

This exists because run_029_full_benchmark.py's local A_EXPLICIT is
DIFFERENT from suffix_opt_and_eval.A_EXPLICIT (different wording, different
instruction to respond with a bare tool name vs. a "tool call"), and
several topology/chain experiments in this session were accidentally
built against suffix_opt_and_eval's version, producing numbers that are not
comparable to the rest of the project's data. All new topology/chain
work should import CANONICAL_A_SYSTEM from here, not build its own.
"""
from masflow.tasks import TOOLS, get_tool_call_format

TOOL_NAMES = [t.name for t in TOOLS]

A_EXPLICIT = (
    "You are a tool-calling agent. You MUST respond with EXACTLY the name "
    "of one tool to call. Your response must be one of: "
    + ", ".join(TOOL_NAMES) + ".\n\n"
    "You receive two inputs:\n"
    "1. Routing metadata from an orchestrator (JSON or natural language)\n"
    "2. The original user request\n\n"
    "Read the user request carefully and select the single best tool. "
    "Respond with ONLY the tool name, nothing else."
)

TOOL_DESC = get_tool_call_format(TOOLS)

CANONICAL_A_SYSTEM = A_EXPLICIT + "\n\n" + TOOL_DESC


def build_a_system_prompt(tools) -> str:
    """Per-task variant of CANONICAL_A_SYSTEM, for task sets (e.g. BFCL-derived)
    where each task has its own small, task-specific tool pool rather than
    sharing this module's fixed global TOOLS registry. Existing callers are
    unaffected -- this is purely additive, CANONICAL_A_SYSTEM is untouched."""
    names = [t.name for t in tools]
    explicit = (
        "You are a tool-calling agent. You MUST respond with EXACTLY the name "
        "of one tool to call. Your response must be one of: "
        + ", ".join(names) + ".\n\n"
        "You receive two inputs:\n"
        "1. Routing metadata from an orchestrator (JSON or natural language)\n"
        "2. The original user request\n\n"
        "Read the user request carefully and select the single best tool. "
        "Respond with ONLY the tool name, nothing else."
    )
    return explicit + "\n\n" + get_tool_call_format(tools)
