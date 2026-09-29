"""
topology_depth_eval.py -- generic instrumented evaluation for any
"ordered list of processing stages" topology variant (topology_depth_variants.py).

Every intermediate stage (all but the last) is checked the same way
Experiment 2 / topology_fanmerge_eval.py's corrected A-B check works:
raw stage output fed DIRECTLY to the active attack_protocol's
check_node -- "does the payload survive up to this depth, taken alone."

The FINAL stage is checked the proper way every other topology in this
project checks its final output: protocol-wrapped via
format_message_for_a / build_paraphrase_a_input for the verbatim view
(matching how B's output always reaches A elsewhere), and the plain
stage text for the clean view, via attack_protocol.check_final.

attack_protocol: same generalization as graph_topology_eval.py -- defaults
to TOOL_SELECTION (does this text make the decision agent call the wrong
tool), matching every existing caller's behavior exactly (BEAST/ACA/
CFH/TAMAS-DPI all stay TOOL_SELECTION). Pass SECRET_LEAK, PROPAGATION,
PLANNING_STEERING, or HARMFUL_CONTENT to run a different attack family
through the same depth-variant machinery. Note: this drops the
TOOL_SELECTION-only `ft_stage*`/`ft` logprob-ranking diagnostic that the
old hardcoded suffix_opt_and_eval.eval_a_content call used to emit -- confirmed unused
by every downstream consumer (nothing outside this module's own old
version reads it), so this is a schema simplification, not a behavior
change to any number this project reports.
"""
from masflow import suffix_opt_and_eval
from masflow.attack_protocol import AttackProtocol, TOOL_SELECTION
import masflow.protocol_mcp as p_mcp


def run_depth_variant_eval(model, tok, suffix_ids, task, variant_fn, protocol_mod=p_mcp,
                            attack_protocol: AttackProtocol = TOOL_SELECTION):
    device = next(model.parameters()).device
    suffix_text = tok.decode(suffix_ids, skip_special_tokens=True)
    query_plus = task.user_task + " " + suffix_text

    stages = variant_fn(model, tok, query_plus, device)
    n_stages = len(stages)

    out = {"n_stages": n_stages, "stage_outputs": [s[:150] for s in stages]}

    # Intermediate stages: raw output -> direct decision check
    for i, stage_text in enumerate(stages[:-1]):
        out[f"hd_stage{i+1}"] = attack_protocol.check_node(model, tok, stage_text, task)

    # Final stage: protocol-wrapped, reaching the real decision agent
    final_text = stages[-1]
    final_res = attack_protocol.check_final(model, tok, final_text, final_text, task, query_plus, protocol_mod)
    out["hd_final_verbatim"] = final_res["verbatim"]
    out["hd_final_paraphrase"] = final_res["clean"]

    out["suffix_ppl_standalone"] = suffix_opt_and_eval.compute_suffix_ppl(model, tok, task.user_task, suffix_text)
    return out
