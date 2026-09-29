"""
topology_eval_frontier.py -- frontier-API counterpart of
topology_depth_eval.run_depth_variant_eval (per-stage hd + final hd,
generation-based only -- no ft/fu/top/ppl, since frontier chat-completion
APIs don't expose per-token logprobs for hypothetical continuations, same
limitation documented in frontier_eval.py) and a frontier counterpart of
topology_fanmerge_eval.run_fanmerge_eval covering the primary full-path
ASR (the entry->peer / peer->E diagnostic breakdown stays local-only for
now -- it would add ~11 extra API calls per task with no paper table
depending on it yet).

Reuses the exact same message-formatting helpers as the local eval
(protocol_mcp.format_message_for_a, suffix_opt_and_eval.build_paraphrase_a_input) --
those are pure string construction with no model dependency, so the
verbatim/paraphrase content fed to the frontier decision check is
byte-identical in shape to what the local pipeline builds.
"""
import masflow.protocol_mcp as p_mcp
from masflow import suffix_opt_and_eval
from masflow.frontier_eval import eval_a_content_frontier
from masflow.topology_frontier_variants import VARIANTS_FRONTIER, fanout_frontier, deep_tree_frontier

PEER_LABELS = ["B", "C", "D"]


def run_depth_variant_eval_frontier(client, suffix_text, task, variant_name, protocol_mod=p_mcp):
    query_plus = task.user_task + " " + suffix_text
    variant_fn = VARIANTS_FRONTIER[variant_name]
    stages = variant_fn(client, query_plus)

    out = {"n_stages": len(stages), "stage_outputs": [s[:150] for s in stages]}

    # Intermediate stages: raw output -> direct decision check.
    for i, stage_text in enumerate(stages[:-1]):
        out[f"hd_stage{i+1}"] = eval_a_content_frontier(client, stage_text, task)["hd"]

    # Final stage: protocol-wrapped, reaching the real decision agent.
    final_text = stages[-1]
    meta = '{"intent": "%s", "priority": "medium"}' % final_text[:80].replace('"', "'")
    a_verbatim = protocol_mod.format_message_for_a(query_plus, meta)
    a_paraphrase = suffix_opt_and_eval.build_paraphrase_a_input(meta, final_text.strip())
    v_res = eval_a_content_frontier(client, a_verbatim, task)
    p_res = eval_a_content_frontier(client, a_paraphrase, task)
    out["hd_final_verbatim"] = v_res["hd"]
    out["hd_final_paraphrase"] = p_res["hd"]
    return out


def run_deep_tree_eval_frontier(client, suffix_text, task, protocol_mod=p_mcp):
    """Frontier counterpart of topology_deep_tree_eval.run_deep_tree_eval:
    a genuinely recursive depth-3 binary tree (4 leaves -> 2 mid-merges ->
    1 root merge -> E), hd-only, mirroring its level naming (L1 = mid
    nodes, L2 = root, L3 = final protocol-wrapped check)."""
    query_plus = task.user_task + " " + suffix_text
    tree = deep_tree_frontier(client, query_plus)

    out = {"tree_outputs": {k: (v[:150] if isinstance(v, str) else [x[:150] for x in v])
                             for k, v in tree.items()}}

    for label in ["mid_left", "mid_right"]:
        out[f"hd_L1_{label}"] = eval_a_content_frontier(client, tree[label], task)["hd"]

    out["hd_L2_root"] = eval_a_content_frontier(client, tree["root"], task)["hd"]

    meta = '{"intent": "%s", "priority": "medium"}' % tree["root"][:80].replace('"', "'")
    a_verbatim = protocol_mod.format_message_for_a(query_plus, meta)
    a_paraphrase = suffix_opt_and_eval.build_paraphrase_a_input(meta, tree["root"].strip())
    out["hd_L3_verbatim"] = eval_a_content_frontier(client, a_verbatim, task)["hd"]
    out["hd_L3_paraphrase"] = eval_a_content_frontier(client, a_paraphrase, task)["hd"]
    return out


def run_fanmerge_eval_frontier(client, suffix_text, task):
    query_plus = task.user_task + " " + suffix_text
    peer_outputs = fanout_frontier(client, query_plus)

    joined_meta = '{"intent": "%s", "priority": "medium"}' % (
        " | ".join(o.strip() for o in peer_outputs if o.strip())[:80].replace('"', "'")
    )
    joined_para = " ".join(o.strip() for o in peer_outputs if o.strip())
    a_verbatim_full = p_mcp.format_message_for_a(query_plus, joined_meta)
    a_paraphrase_full = suffix_opt_and_eval.build_paraphrase_a_input(joined_meta, joined_para)
    v_res_full = eval_a_content_frontier(client, a_verbatim_full, task)
    p_res_full = eval_a_content_frontier(client, a_paraphrase_full, task)

    out = {
        "full_ba_hd": v_res_full["hd"],
        "full_para_hd": p_res_full["hd"],
        "peer_outputs": {label: text[:200] for label, text in zip(PEER_LABELS, peer_outputs)},
    }

    # Interim (node-level) hd: does THIS peer's own raw output, in
    # isolation, already carry the payload -- mirrors run_fanmerge_eval's
    # hd_peer_raw_{label} in topology_fanmerge_eval.py, the input
    # mean(interim) needs for the RQ2 estimator validation.
    for label, peer_text in zip(PEER_LABELS, peer_outputs):
        out[f"hd_peer_raw_{label}"] = eval_a_content_frontier(client, peer_text, task)["hd"]

    return out
