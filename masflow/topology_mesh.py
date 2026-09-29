"""
topology_mesh.py — main.md Â§6.3: three peer agents in a sequential
relay, each conditioned on the PREVIOUS peer's output (genuine
peer-to-peer exchange, no central hub): B_p1(query) -> B_p2(query, p1_out)
-> B_p3(query, p2_out), and p3's output is final. Distinguishes from every
other topology here by making each hop depend on the prior hop's output
rather than the original query alone.
"""
from masflow.topology_eval import _generate

TOPOLOGY_NAME = "mesh"

PEER1_PROMPT = (
    "You are peer agent #1 in a peer-to-peer agent network. Given a "
    "user's request, give your initial read of what the user wants, in "
    "one short sentence. Output ONLY that sentence."
)
PEER_RELAY_PROMPT = (
    "You are a peer agent in a peer-to-peer agent network. You are given "
    "the original user request and a previous peer's read of it. Refine "
    "or correct it in one short sentence, in your own words. Output ONLY "
    "that sentence."
)


def _relay(model, tok, query_plus, device):
    p1 = _generate(model, tok, PEER1_PROMPT, query_plus, device, max_new_tokens=48)
    p2_input = f"Original request: {query_plus}\nPeer 1's read: {p1}"
    p2 = _generate(model, tok, PEER_RELAY_PROMPT, p2_input, device, max_new_tokens=48)
    p3_input = f"Original request: {query_plus}\nPeer 2's read: {p2}"
    p3 = _generate(model, tok, PEER_RELAY_PROMPT, p3_input, device, max_new_tokens=48)
    return p3


def produce_metadata(model, tok, query_plus, task, device):
    final = _relay(model, tok, query_plus, device)
    return '{"intent": "%s", "priority": "medium"}' % final[:60].replace('"', "'")


def produce_paraphrase(model, tok, query_plus, task, device):
    return _relay(model, tok, query_plus, device)
