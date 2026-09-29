"""
graph_topology.py -- a general directed-acyclic-graph representation for
multi-agent routing topologies. Every topology this project has tested so
far (centralized, orchestrate, tree, star, mesh) is a specific instance
of this general shape:

    - a set of nodes, each with its own system prompt
    - edges: which nodes' outputs feed into which other node
    - source nodes (no parents) receive the raw query_plus directly
    - a single designated SINK node is the one whose output finally
      reaches the decision agent

Examples (see graph_examples.py):
    centralized = chain(depth=1)
    orchestrate = chain(depth=2)          # same shape, one more link
    tree        = diamond(n_branches=2)   # decompose-then-merge
    star        = diamond(n_branches=3)   # replicate-then-merge
    mesh        = relay(depth=3)          # sequential, each sees the last

This module is intentionally prompt-agnostic: a GraphNode just carries
whatever system prompt you give it. graph_examples.py's PLAIN_* graphs
use one neutral, uniform prompt for every node (no role flavor at all),
specifically to isolate "does this shape behave this way regardless of
prompt wording" from "was it something about our specific role prompts."
"""
from dataclasses import dataclass, field
from masflow.topology_eval import _generate


@dataclass
class GraphNode:
    name: str
    prompt: str
    parents: list[str] = field(default_factory=list)  # empty = source node
    max_new_tokens: int = 48
    mechanical: bool = False  # True: no LLM call, output = join of parent outputs (`prompt` ignored)
    join_style: str = "labeled"  # "labeled": "[parent]: text" per line; "plain": " | ".join(texts)


@dataclass
class TopologyGraph:
    name: str
    nodes: list[GraphNode]  # must be in topological order (parents before children)
    sink: str  # name of the node whose output is the final one reaching the decision agent

    def __post_init__(self):
        names = [n.name for n in self.nodes]
        assert len(names) == len(set(names)), "duplicate node names"
        for n in self.nodes:
            for p in n.parents:
                assert p in names[:names.index(n.name)], (
                    f"node '{n.name}' depends on '{p}', which must come earlier"
                )
        assert self.sink in names


def _join_parent_outputs(parents: list[str], outputs: dict, style: str = "labeled") -> str:
    if len(parents) == 1:
        return outputs[parents[0]]
    if style == "plain":
        return " | ".join(outputs[p].strip() for p in parents if outputs[p].strip())
    return "\n".join(f"[{p}]: {outputs[p]}" for p in parents)


def run_graph(model, tok, query_plus, graph: TopologyGraph, device, node_models=None):
    """Executes every node in topological order. Returns dict {node_name: output_text}.

    node_models: optional dict {node_name: (model, tok)} -- lets individual
    nodes use a DIFFERENT model instance than the default (model, tok).
    Nodes not present in node_models fall back to (model, tok), so this is
    fully backward compatible with every existing single-model caller
    (pass node_models=None or a partial dict for just the nodes you want
    to swap). A node with mechanical=True never calls a model at all --
    its output is just its parents' outputs joined (e.g. star's mechanical
    merge of B/C/D), so it is never a valid swap target."""
    node_models = node_models or {}
    outputs = {}
    for node in graph.nodes:
        content = query_plus if not node.parents else _join_parent_outputs(
            node.parents, outputs, style=node.join_style)
        if node.mechanical:
            outputs[node.name] = content
            continue
        node_model, node_tok = node_models.get(node.name, (model, tok))
        node_device = next(node_model.parameters()).device
        outputs[node.name] = _generate(node_model, node_tok, node.prompt, content, node_device,
                                        max_new_tokens=node.max_new_tokens)
    return outputs
