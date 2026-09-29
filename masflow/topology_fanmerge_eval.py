"""
topology_fanmerge_eval.py -- Fan-out/fan-in topology transfer: a single
entry point (user query + frozen suffix) fans out to 3 independent peers
(B, C, D -- reusing topology_star.py's peer prompts/fan-out unchanged),
whose outputs are mechanically joined and fed to a final decision agent E
(this codebase's "A" role). Measures three distinct things at three
distinct points in the pipeline, all using the SAME decision machinery
(suffix_opt_and_eval.eval_a_content / get_tool_logprobs) so the numbers are directly
comparable:

  1. Full-path ASR (entry -> {B,C,D} -> E, merged): does E choose the
     wrong tool once all three peers' outputs are mechanically joined?
     This is exactly topology_star's existing full-pipeline measurement.

  2. Entry->peer ASR (A-B, A-C, A-D): "as if B/C/D will choose the wrong
     tool" -- each peer's own raw output text is fed DIRECTLY as the
     decision-maker's content (same A_EXPLICIT system prompt, no protocol
     wrapping), asking whether that specific peer's paraphrase, taken in
     isolation and evaluated as if it were shown to a tool-choosing agent,
     already carries the hijack.

  3. Peer->E single-path ASR (B-E, C-E, D-E): each peer's output alone
     (bypassing the other two peers and the merge step entirely) is
     wrapped through the SAME protocol construction the full pipeline
     uses (format_message_for_a / build_paraphrase_a_input) and fed to E,
     asking whether that one path alone -- without dilution/competition
     from the other two peers -- would hijack E.

Peer outputs are generated ONCE per task (not re-generated per step,
since do_sample=False makes topology_star's fan-out deterministic) and
reused across all three measurements to avoid 3x redundant peer calls.
"""
from masflow import suffix_opt_and_eval
from masflow import topology_star
from masflow.attack_protocol import AttackProtocol, TOOL_SELECTION
import masflow.protocol_mcp as p_mcp

PEER_LABELS = ["B", "C", "D"]


def _single_peer_metadata(peer_text: str) -> str:
    """Mirrors topology_star.produce_metadata's wrapping, but for one
    peer's output instead of all three joined."""
    return '{"intent": "%s", "priority": "medium"}' % peer_text[:80].replace('"', "'")


def _single_peer_paraphrase(peer_text: str) -> str:
    return peer_text.strip()


def run_fanmerge_eval(model, tok, suffix_ids, task, a_system, protocol_mod=p_mcp):
    """Returns a dict with the full-path, entry->peer, and peer->E metrics
    for one task, given a frozen (already-optimized, e.g. BEAST) suffix."""
    device = next(model.parameters()).device
    suffix_text = tok.decode(suffix_ids, skip_special_tokens=True)
    query_plus = task.user_task + " " + suffix_text

    # Generate each peer's output ONCE, reuse for all three measurements.
    peer_outputs = topology_star._fan_out(model, tok, query_plus, device)

    # --- Step 1: full-path ASR (mechanical join of all 3 -> E) ---
    joined_meta = '{"intent": "%s", "priority": "medium"}' % (
        " | ".join(o.strip() for o in peer_outputs if o.strip())[:80].replace('"', "'")
    )
    joined_para = " ".join(o.strip() for o in peer_outputs if o.strip())
    a_verbatim_full = protocol_mod.format_message_for_a(query_plus, joined_meta)
    a_paraphrase_full = suffix_opt_and_eval.build_paraphrase_a_input(joined_meta, joined_para)
    v_res_full = suffix_opt_and_eval.eval_a_content(model, tok, a_verbatim_full, a_system, task)
    p_res_full = suffix_opt_and_eval.eval_a_content(model, tok, a_paraphrase_full, a_system, task)

    # Clean (no-suffix) baseline through the same fan-out/merge, for
    # disruption scoring -- consistent with topology_eval's per-topology
    # clean rerun (a lossy merge can corrupt even a clean query).
    clean_peer_outputs = topology_star._fan_out(model, tok, task.user_task, device)
    clean_joined_meta = '{"intent": "%s", "priority": "medium"}' % (
        " | ".join(o.strip() for o in clean_peer_outputs if o.strip())[:80].replace('"', "'")
    )
    clean_joined_para = " ".join(o.strip() for o in clean_peer_outputs if o.strip())
    a_clean = suffix_opt_and_eval.build_paraphrase_a_input(clean_joined_meta, clean_joined_para)
    c_res = suffix_opt_and_eval.eval_a_content(model, tok, a_clean, a_system, task)

    out = {
        "full_ba_hd": v_res_full["hd"], "full_ba_ft": v_res_full["ft"],
        "full_para_hd": p_res_full["hd"], "full_para_ft": p_res_full["ft"],
        "full_ba_dr": (c_res["top"] == task.expected_tool) and v_res_full["fu"],
        "full_para_dr": (c_res["top"] == task.expected_tool) and p_res_full["fu"],
        "clean_top": c_res["top"],
    }

    # --- Step 2: entry->peer ASR --- ALIGNED with the simple A-B-C chain's
    # Experiment 2 definition: the RAW injected content (query_plus) fed
    # DIRECTLY to a decision-maker (A_EXPLICIT), not the peer's own
    # role-processed output. Since this no longer depends on the peer's
    # specific role prompt at all, it's identical across B/C/D by
    # construction -- computed once and assigned to all three labels
    # (kept as separate keys for table-shape consistency with Step 3).
    a_raw_res = suffix_opt_and_eval.eval_a_content(model, tok, query_plus, a_system, task)
    for label in PEER_LABELS:
        out[f"hd_A_{label}"] = a_raw_res["hd"]
        out[f"ft_A_{label}"] = a_raw_res["ft"]
        out[f"top_A_{label}"] = a_raw_res["top"]

    # --- Step 2b: peer's OWN real output -> direct decision check ---
    # The genuinely comparable interim measurement to orchestrate/tree/
    # mesh's s1/s2 (topology_depth_eval.py): does THIS peer's actual
    # generated text, after real role-processing, still carry the
    # payload -- as opposed to Step 2 above, which bypasses peer
    # processing entirely and just re-checks the raw injected query.
    for label, peer_text in zip(PEER_LABELS, peer_outputs):
        peer_raw_res = suffix_opt_and_eval.eval_a_content(model, tok, peer_text, a_system, task)
        out[f"hd_peer_raw_{label}"] = peer_raw_res["hd"]
        out[f"ft_peer_raw_{label}"] = peer_raw_res["ft"]

    # --- Step 3: peer->E single-path ASR (protocol-wrapped, bypassing merge) ---
    for label, peer_text in zip(PEER_LABELS, peer_outputs):
        meta_single = _single_peer_metadata(peer_text)
        para_single = _single_peer_paraphrase(peer_text)
        a_verbatim_single = protocol_mod.format_message_for_a(query_plus, meta_single)
        a_paraphrase_single = suffix_opt_and_eval.build_paraphrase_a_input(meta_single, para_single)
        v_res = suffix_opt_and_eval.eval_a_content(model, tok, a_verbatim_single, a_system, task)
        p_res = suffix_opt_and_eval.eval_a_content(model, tok, a_paraphrase_single, a_system, task)
        out[f"hd_{label}_E"] = v_res["hd"]
        out[f"ft_{label}_E"] = v_res["ft"]
        out[f"para_hd_{label}_E"] = p_res["hd"]

    out["suffix_ppl_standalone"] = suffix_opt_and_eval.compute_suffix_ppl(model, tok, task.user_task, suffix_text)
    out["peer_outputs"] = {label: text[:200] for label, text in zip(PEER_LABELS, peer_outputs)}
    return out


def run_fanmerge_eval_generic(model, tok, suffix_ids, task, protocol_mod=p_mcp,
                               attack_protocol: AttackProtocol = TOOL_SELECTION):
    """Minimal, protocol-generic sibling of run_fanmerge_eval above: same
    entry -> {B,C,D} -> mechanical merge -> E star shape, but emits the
    shared hd_peer_raw_{label} (interim) / hd_final_verbatim /
    hd_final_paraphrase schema every other topology-eval module in this
    project uses (graph_topology_eval.py, topology_depth_eval.py), so any
    attack_protocol -- not just TOOL_SELECTION's rich ft/fu/dr diagnostic
    breakdown -- can be run through the star topology and folded into the
    same mean(interim) estimator/aggregation code.

    Use run_fanmerge_eval (above, untouched) for TOOL_SELECTION attacks
    that need the full diagnostic breakdown (BEAST/ACA/CFH/TAMAS-DPI, and
    every existing caller of run_fanmerge_eval keeps using it unchanged);
    use this for the other four families in the cross-family benchmark
    (draft.tex, "Contribution 1" section)."""
    device = next(model.parameters()).device
    suffix_text = tok.decode(suffix_ids, skip_special_tokens=True)
    query_plus = task.user_task + " " + suffix_text

    peer_outputs = topology_star._fan_out(model, tok, query_plus, device)

    out = {}
    for label, peer_text in zip(PEER_LABELS, peer_outputs):
        out[f"hd_peer_raw_{label}"] = attack_protocol.check_node(model, tok, peer_text, task)

    # Same verbatim (pipe-joined) vs. clean (space-joined) convention as
    # graph_topology_eval.py's mechanical multi-parent merge case.
    verbatim_text = " | ".join(o.strip() for o in peer_outputs if o.strip())
    clean_text = " ".join(o.strip() for o in peer_outputs if o.strip())
    final_res = attack_protocol.check_final(model, tok, verbatim_text, clean_text, task, query_plus, protocol_mod)
    out["hd_final_verbatim"] = final_res["verbatim"]
    out["hd_final_paraphrase"] = final_res["clean"]

    out["suffix_ppl_standalone"] = suffix_opt_and_eval.compute_suffix_ppl(model, tok, task.user_task, suffix_text)
    out["peer_outputs"] = {label: text[:200] for label, text in zip(PEER_LABELS, peer_outputs)}
    return out
