"""
graph_topology_eval.py -- generic instrumented evaluation for any
TopologyGraph (graph_topology.py). Same convention as
topology_depth_eval.py: every non-sink node's raw output is checked
directly against the decision agent (interim signal), and the sink
node's output is checked the proper way -- protocol-wrapped, both
verbatim (includes MCP's raw-query passthrough) and paraphrase (clean).
"""
from masflow import suffix_opt_and_eval
from masflow.graph_topology import TopologyGraph, run_graph
from masflow.attack_protocol import AttackProtocol, TOOL_SELECTION
import masflow.protocol_mcp as p_mcp


def run_graph_eval(model, tok, suffix_ids, task, graph: TopologyGraph, protocol_mod=p_mcp,
                    node_models=None, attack_protocol: AttackProtocol = TOOL_SELECTION):
    """model/tok: the FIXED decision-agent (E) used for every interim and
    final check, regardless of what generated the content -- same
    fixed-E convention as topology_fanmerge_heterogeneous.py.
    node_models: optional {node_name: (model, tok)} to make individual
    graph nodes use a DIFFERENT model than E when generating their own
    content; nodes not listed fall back to (model, tok) (the homogeneous,
    single-model case -- fully backward compatible).
    attack_protocol: defines what "success" means at a node and at the
    sink (attack_protocol.py) -- defaults to TOOL_SELECTION (does this
    text make the decision agent call the wrong tool), matching every
    existing caller's behavior exactly. Pass SECRET_LEAK or PROPAGATION
    (or a new AttackProtocol) to test a different attack family through
    the same graph/estimator machinery."""
    device = next(model.parameters()).device
    suffix_text = tok.decode(suffix_ids, skip_special_tokens=True)
    query_plus = task.user_task + " " + suffix_text

    node_outputs = run_graph(model, tok, query_plus, graph, device, node_models=node_models)

    out = {"node_outputs": {k: v[:150] for k, v in node_outputs.items()}}

    for node in graph.nodes:
        if node.name == graph.sink:
            continue
        out[f"hd_{node.name}"] = attack_protocol.check_node(model, tok, node_outputs[node.name], task)

    sink_node = next(n for n in graph.nodes if n.name == graph.sink)
    final_text = node_outputs[graph.sink]
    if sink_node.mechanical and len(sink_node.parents) > 1:
        # matches topology_fanmerge_eval.py's joined_para convention: the
        # paraphrase/clean view of a mechanical multi-parent merge is a
        # plain space-joined natural-language read, not the "|"-delimited
        # text used for the verbatim view -- literal "|" characters read
        # as artificial structure, not a real single agent's paraphrase.
        clean_text = " ".join(node_outputs[p].strip() for p in sink_node.parents if node_outputs[p].strip())
    else:
        clean_text = final_text
    final_res = attack_protocol.check_final(model, tok, final_text, clean_text, task, query_plus, protocol_mod)
    out["hd_final_verbatim"] = final_res["verbatim"]
    out["hd_final_paraphrase"] = final_res["clean"]

    out["suffix_ppl_standalone"] = suffix_opt_and_eval.compute_suffix_ppl(model, tok, task.user_task, suffix_text)
    return out
