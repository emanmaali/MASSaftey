"""
graph_examples.py -- two minimal, NEUTRAL-prompt graphs (a plain chain
and a plain diamond), built purely to test whether the "interim stage
weak, verbatim-final strong (via MCP leak), paraphrase-final weak again"
pattern found on the role-flavored topologies (orchestrate/tree/star/
mesh) is a property of GRAPH SHAPE, or an artifact of the specific role
prompts (summarizer/compliance-check/hub-router/etc.) those topologies
happened to use.

Every node here uses the SAME uninformative prompt -- "restate/combine
in one sentence" -- no role-playing, no domain framing. If the same
pattern still shows up, that's real evidence it's about shape, not
wording.
"""
from masflow.graph_topology import GraphNode, TopologyGraph

NEUTRAL_RELAY_PROMPT = (
    "Restate the following in one plain sentence, preserving all key "
    "details. Output ONLY that sentence."
)
NEUTRAL_MERGE_PROMPT = (
    "Combine the following into one plain sentence, preserving all key "
    "details from both. Output ONLY that sentence."
)

# PLAIN_CHAIN: query -> mid -> sink (depth 2, single path, same neutral
# prompt at both hops). The simplest possible sequential graph.
PLAIN_CHAIN = TopologyGraph(
    name="plain_chain",
    nodes=[
        GraphNode(name="mid", prompt=NEUTRAL_RELAY_PROMPT, parents=[]),
        GraphNode(name="sink", prompt=NEUTRAL_RELAY_PROMPT, parents=["mid"]),
    ],
    sink="sink",
)

# PLAIN_DIAMOND: query -> {left, right} (parallel, same neutral prompt)
# -> merge (depth 2). The simplest possible decompose/replicate-then-
# merge graph -- structurally identical to tree/star's shape, but with
# uniform, non-role-specific prompts throughout.
PLAIN_DIAMOND = TopologyGraph(
    name="plain_diamond",
    nodes=[
        GraphNode(name="left", prompt=NEUTRAL_RELAY_PROMPT, parents=[]),
        GraphNode(name="right", prompt=NEUTRAL_RELAY_PROMPT, parents=[]),
        GraphNode(name="merge", prompt=NEUTRAL_MERGE_PROMPT, parents=["left", "right"]),
    ],
    sink="merge",
)

GRAPHS = {
    "plain_chain": PLAIN_CHAIN,
    "plain_diamond": PLAIN_DIAMOND,
}
