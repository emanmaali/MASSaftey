"""
topology_eval_frontier_generic.py -- frontier-API counterpart of
topology_depth_eval.run_depth_variant_eval / topology_fanmerge_eval.
run_fanmerge_eval_generic, for the four non-tool_selection attack
families (secret_leak, propagation, planning_steering, harmful_content).

Unlike tool_selection, these four families' check_node/check_final
(attack_protocol.py) are pure text functions -- a word-overlap ratio, a
marker substring check, a keyword list -- that never call the model at
all; `model`/`tok` are unused parameters kept only so the same
AttackProtocol interface covers both cases (confirmed by reading their
bodies). So the frontier version needs zero extra API calls beyond the
topology's own stage generation: generate the stage chain via
FrontierClient, then apply attack_protocol.check_node/check_final
directly to the resulting text, passing None for the unused model/tok.
"""
import masflow.protocol_mcp as p_mcp
from masflow.topology_frontier_variants import VARIANTS_FRONTIER, fanout_frontier, deep_tree_frontier


def run_depth_variant_eval_frontier_generic(client, suffix_text, task, variant_name, attack_protocol,
                                             protocol_mod=p_mcp):
    query_plus = task.user_task + " " + suffix_text
    variant_fn = VARIANTS_FRONTIER[variant_name]
    stages = variant_fn(client, query_plus)

    out = {"n_stages": len(stages), "stage_outputs": [s[:150] for s in stages]}

    for i, stage_text in enumerate(stages[:-1]):
        out[f"hd_stage{i+1}"] = attack_protocol.check_node(None, None, stage_text, task)

    # Generic (non-tool_selection) final check: the last stage's own raw
    # text, passed as both "verbatim" and "clean" -- no separate
    # decision-agent call exists for these families (matches
    # topology_depth_eval.run_depth_variant_eval's generic-attack call:
    # check_final(model, tok, final_text, final_text, task, ...)).
    final_text = stages[-1]
    final_res = attack_protocol.check_final(None, None, final_text, final_text, task, query_plus, protocol_mod)
    out["hd_final_verbatim"] = final_res["verbatim"]
    out["hd_final_paraphrase"] = final_res["clean"]
    return out


def run_deep_tree_eval_frontier_generic(client, suffix_text, task, attack_protocol, protocol_mod=p_mcp):
    query_plus = task.user_task + " " + suffix_text
    tree = deep_tree_frontier(client, query_plus)

    out = {"tree_outputs": {k: (v[:150] if isinstance(v, str) else [x[:150] for x in v])
                             for k, v in tree.items()}}
    for label in ["mid_left", "mid_right"]:
        out[f"hd_L1_{label}"] = attack_protocol.check_node(None, None, tree[label], task)
    out["hd_L2_root"] = attack_protocol.check_node(None, None, tree["root"], task)

    final_res = attack_protocol.check_final(None, None, tree["root"], tree["root"], task, query_plus, protocol_mod)
    out["hd_L3_verbatim"] = final_res["verbatim"]
    out["hd_L3_paraphrase"] = final_res["clean"]
    return out


def run_fanmerge_eval_frontier_generic(client, suffix_text, task, attack_protocol, protocol_mod=p_mcp):
    query_plus = task.user_task + " " + suffix_text
    peer_outputs = fanout_frontier(client, query_plus)

    out = {}
    for label, peer_text in zip(["B", "C", "D"], peer_outputs):
        out[f"hd_peer_raw_{label}"] = attack_protocol.check_node(None, None, peer_text, task)

    # Same verbatim (pipe-joined) vs. clean (space-joined) convention as
    # topology_fanmerge_eval.run_fanmerge_eval_generic.
    verbatim_text = " | ".join(o.strip() for o in peer_outputs if o.strip())
    clean_text = " ".join(o.strip() for o in peer_outputs if o.strip())
    final_res = attack_protocol.check_final(None, None, verbatim_text, clean_text, task, query_plus, protocol_mod)
    out["hd_final_verbatim"] = final_res["verbatim"]
    out["hd_final_paraphrase"] = final_res["clean"]
    out["peer_outputs"] = {label: text[:200] for label, text in zip(["B", "C", "D"], peer_outputs)}
    return out
