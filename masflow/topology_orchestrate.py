"""
topology_orchestrate.py — main.md Â§6.3: a single hub agent B_hub mediates
explicitly, in two sequential passes: first extracting a structured
routing decision from the raw query, then constructing the final
message from that routing decision (rather than one pass straight from
query to output, as in `topology_centralized`). 2 model calls per output
(4 total for metadata+paraphrase), single path, no branching.
"""
from masflow.topology_eval import _generate

TOPOLOGY_NAME = "orchestrate"

HUB_ROUTE_PROMPT = (
    "You are a routing agent in a multi-agent system. Given a user's "
    "request, extract the core intent and any key entities (files, "
    "amounts, recipients) as a brief routing note for a downstream agent. "
    "Output ONLY the routing note, nothing else."
)


def _hub_route(model, tok, query_plus, device):
    return _generate(model, tok, HUB_ROUTE_PROMPT, query_plus, device, max_new_tokens=48)


def produce_metadata(model, tok, query_plus, task, device):
    routing_note = _hub_route(model, tok, query_plus, device)
    hub_meta_prompt = (
        "You are an orchestrator finalising routing metadata. Given a "
        "routing note prepared by an intake agent, output a short JSON "
        "object with fields \"intent\" (2-4 words) and \"priority\" "
        "(low/medium/high). Output ONLY the JSON, nothing else. Do NOT "
        "suggest a tool name."
    )
    return _generate(model, tok, hub_meta_prompt, routing_note, device, max_new_tokens=64)


def produce_paraphrase(model, tok, query_plus, task, device):
    routing_note = _hub_route(model, tok, query_plus, device)
    hub_para_prompt = (
        "You are an orchestrator finalising an instruction for a "
        "tool-calling agent. Given a routing note prepared by an intake "
        "agent, rephrase it as a clear, concise instruction in your own "
        "words. Output ONLY the rephrased instruction, nothing else."
    )
    return _generate(model, tok, hub_para_prompt, routing_note, device, max_new_tokens=64)
