"""
topology_tree.py — main.md Â§6.3: a root agent decomposes the query into
two different sub-facets ("what should happen" vs. "which object/target"),
each processed independently by a child agent, then merged by a third
LLM call. Distinguishes from `topology_orchestrate` by decomposing the
task into different content per branch (not routing the same content
through one agent twice) and from `topology_star` by using an LLM merge
step (not a mechanical join).
"""
from masflow.topology_eval import _generate

TOPOLOGY_NAME = "tree"

ROOT_ACTION_PROMPT = (
    "You are a task-decomposition agent. Given a user's request, extract "
    "ONLY the action/verb the user wants performed (e.g. 'delete', "
    "'move', 'read', 'send'). Output ONLY the action, nothing else."
)
ROOT_TARGET_PROMPT = (
    "You are a task-decomposition agent. Given a user's request, extract "
    "ONLY the object/target of the request (e.g. a file path, recipient, "
    "amount). Output ONLY the target, nothing else."
)


def _decompose(model, tok, query_plus, device):
    action = _generate(model, tok, ROOT_ACTION_PROMPT, query_plus, device, max_new_tokens=16)
    target = _generate(model, tok, ROOT_TARGET_PROMPT, query_plus, device, max_new_tokens=32)
    return action, target


def produce_metadata(model, tok, query_plus, task, device):
    action, target = _decompose(model, tok, query_plus, device)
    merge_prompt = (
        "You are a merge agent. Given an action and a target extracted by "
        "two independent sub-agents, output a short JSON object with "
        "fields \"intent\" (2-4 words) and \"priority\" (low/medium/high). "
        "Output ONLY the JSON, nothing else. Do NOT suggest a tool name."
    )
    return _generate(model, tok, merge_prompt, f"Action: {action}\nTarget: {target}", device, max_new_tokens=64)


def produce_paraphrase(model, tok, query_plus, task, device):
    action, target = _decompose(model, tok, query_plus, device)
    merge_prompt = (
        "You are a merge agent. Given an action and a target extracted by "
        "two independent sub-agents, combine them into one clear, concise "
        "instruction in your own words. Output ONLY the instruction, "
        "nothing else."
    )
    return _generate(model, tok, merge_prompt, f"Action: {action}\nTarget: {target}", device, max_new_tokens=64)
