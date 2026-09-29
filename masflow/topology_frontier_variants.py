"""
topology_frontier_variants.py -- frontier-API counterparts of
topology_depth_variants.py's 4 original-depth stage-generation functions
(centralized_d1_orig, orchestrate_d2_orig, tree_d2_orig, mesh_d3_orig) and
topology_star.py's 3-peer fan-out, using a FrontierClient instead of a
local (model, tokenizer) pair. Same role prompts, same stage structure --
only the generation backend changes.
"""
from masflow import suffix_opt_and_eval
from masflow.topology_orchestrate import HUB_ROUTE_PROMPT
from masflow.topology_depth_variants import (
    HUB_FINALIZE_PROMPT, HUB_REFINE_PROMPT, TREE_MERGE_PROMPT, STAR_REFINE_PROMPT,
)
from masflow.topology_mesh import PEER1_PROMPT, PEER_RELAY_PROMPT
from masflow.topology_tree import ROOT_ACTION_PROMPT, ROOT_TARGET_PROMPT
from masflow.topology_star import PEER_PROMPTS
from masflow.topology_deep_tree import LEAF_PROMPTS, MERGE_PROMPT


def _gen(client, system, user_input, max_tokens=64):
    return client.generate_chat(system, user_input, max_tokens=max_tokens)


def centralized_frontier(client, query_plus):
    s1 = _gen(client, suffix_opt_and_eval.B_PROMPT_METADATA, query_plus, max_tokens=suffix_opt_and_eval.INTERMEDIATE_LEN)
    return [s1]


def orchestrate_frontier(client, query_plus):
    s1 = _gen(client, HUB_ROUTE_PROMPT, query_plus, max_tokens=48)
    s2 = _gen(client, HUB_FINALIZE_PROMPT, s1, max_tokens=64)
    return [s1, s2]


def mesh_frontier(client, query_plus):
    p1 = _gen(client, PEER1_PROMPT, query_plus, max_tokens=48)
    p2_input = f"Original request: {query_plus}\nPeer 1's read: {p1}"
    p2 = _gen(client, PEER_RELAY_PROMPT, p2_input, max_tokens=48)
    p3_input = f"Original request: {query_plus}\nPeer 2's read: {p2}"
    p3 = _gen(client, PEER_RELAY_PROMPT, p3_input, max_tokens=48)
    return [p1, p2, p3]


def tree_frontier(client, query_plus):
    action = _gen(client, ROOT_ACTION_PROMPT, query_plus, max_tokens=16)
    target = _gen(client, ROOT_TARGET_PROMPT, query_plus, max_tokens=32)
    merged = _gen(client, TREE_MERGE_PROMPT, f"Action: {action}\nTarget: {target}", max_tokens=64)
    return [action, target, merged]


def fanout_frontier(client, query_plus):
    """3 independent peers (B, C, D), matching topology_star._fan_out."""
    return [_gen(client, p, query_plus, max_tokens=48) for p in PEER_PROMPTS]


# --- "+1" depth variants (orig depth -> orig+1), matching topology_depth_variants.py ---

def centralized_d2_frontier(client, query_plus):
    s1 = _gen(client, suffix_opt_and_eval.B_PROMPT_METADATA, query_plus, max_tokens=suffix_opt_and_eval.INTERMEDIATE_LEN)
    s2 = _gen(client, suffix_opt_and_eval.B_PROMPT_METADATA, s1, max_tokens=suffix_opt_and_eval.INTERMEDIATE_LEN)
    return [s1, s2]


def orchestrate_d3_frontier(client, query_plus):
    s1 = _gen(client, HUB_ROUTE_PROMPT, query_plus, max_tokens=48)
    s2 = _gen(client, HUB_REFINE_PROMPT, s1, max_tokens=48)
    s3 = _gen(client, HUB_FINALIZE_PROMPT, s2, max_tokens=64)
    return [s1, s2, s3]


def mesh_d4_frontier(client, query_plus):
    p1 = _gen(client, PEER1_PROMPT, query_plus, max_tokens=48)
    prev, outs = p1, [p1]
    for i in range(2):
        p_input = f"Original request: {query_plus}\nPeer {i+1}'s read: {prev}"
        prev = _gen(client, PEER_RELAY_PROMPT, p_input, max_tokens=48)
        outs.append(prev)
    return outs


# --- "+2" depth variants (orig depth -> orig+2) ---

def centralized_plus2_frontier(client, query_plus):
    s1 = _gen(client, suffix_opt_and_eval.B_PROMPT_METADATA, query_plus, max_tokens=suffix_opt_and_eval.INTERMEDIATE_LEN)
    s2 = _gen(client, suffix_opt_and_eval.B_PROMPT_METADATA, s1, max_tokens=suffix_opt_and_eval.INTERMEDIATE_LEN)
    s3 = _gen(client, suffix_opt_and_eval.B_PROMPT_METADATA, s2, max_tokens=suffix_opt_and_eval.INTERMEDIATE_LEN)
    return [s1, s2, s3]


def orchestrate_plus2_frontier(client, query_plus):
    s1 = _gen(client, HUB_ROUTE_PROMPT, query_plus, max_tokens=48)
    s2 = _gen(client, HUB_REFINE_PROMPT, s1, max_tokens=48)
    s3 = _gen(client, HUB_REFINE_PROMPT, s2, max_tokens=48)
    s4 = _gen(client, HUB_FINALIZE_PROMPT, s3, max_tokens=64)
    return [s1, s2, s3, s4]


def tree_plus2_frontier(client, query_plus):
    action = _gen(client, ROOT_ACTION_PROMPT, query_plus, max_tokens=16)
    target = _gen(client, ROOT_TARGET_PROMPT, query_plus, max_tokens=32)
    merged = _gen(client, TREE_MERGE_PROMPT, f"Action: {action}\nTarget: {target}", max_tokens=64)
    refine1 = _gen(client, STAR_REFINE_PROMPT, merged, max_tokens=64)
    refine2 = _gen(client, STAR_REFINE_PROMPT, refine1, max_tokens=64)
    return [action, target, merged, refine1, refine2]


def mesh_plus2_frontier(client, query_plus):
    p1 = _gen(client, PEER1_PROMPT, query_plus, max_tokens=48)
    prev, outs = p1, [p1]
    for i in range(4):
        p_input = f"Original request: {query_plus}\nPeer {i+1}'s read: {prev}"
        prev = _gen(client, PEER_RELAY_PROMPT, p_input, max_tokens=48)
        outs.append(prev)
    return outs


# --- tree's own "+1" depth variant: a genuinely recursive depth-3 binary
# tree (topology_deep_tree.py), dict-shaped rather than list-shaped, so
# it gets its own eval function (see topology_eval_frontier.py) instead
# of going through run_depth_variant_eval_frontier's list convention. ---

def deep_tree_frontier(client, query_plus):
    leaves = [_gen(client, p, query_plus, max_tokens=48) for p in LEAF_PROMPTS]
    mid_left = _gen(client, MERGE_PROMPT, f"Summary A: {leaves[0]}\nSummary B: {leaves[1]}", max_tokens=48)
    mid_right = _gen(client, MERGE_PROMPT, f"Summary A: {leaves[2]}\nSummary B: {leaves[3]}", max_tokens=48)
    root = _gen(client, MERGE_PROMPT, f"Summary A: {mid_left}\nSummary B: {mid_right}", max_tokens=48)
    return {"leaves": leaves, "mid_left": mid_left, "mid_right": mid_right, "root": root}


VARIANTS_FRONTIER = {
    "centralized": centralized_frontier,
    "orchestrate": orchestrate_frontier,
    "tree": tree_frontier,
    "mesh": mesh_frontier,
    "centralized_d2": centralized_d2_frontier,
    "orchestrate_d3": orchestrate_d3_frontier,
    "mesh_d4": mesh_d4_frontier,
    "centralized_plus2": centralized_plus2_frontier,
    "orchestrate_plus2": orchestrate_plus2_frontier,
    "tree_plus2": tree_plus2_frontier,
    "mesh_plus2": mesh_plus2_frontier,
}
