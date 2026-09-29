"""
protocol_registry.py -- maps a topology driver's --protocol CLI value to
the corresponding protocol_mod module (protocol_mcp.py and its four
siblings: protocol_a2a.py, protocol_acp.py, protocol_aitp.py,
protocol_raw.py). All five implement the identical interface
(get_b_system_prompt, get_a_system_prompt, format_message_for_a,
format_b_output_template), so any of them can be passed as the
protocol_mod argument every topology-eval module already accepts,
letting the same mean(interim)/calibration/budget machinery be tested
against a different agent-communication protocol with no other code
changes.

PRESERVES_QUERY distinguishes the two regimes: mcp/a2a/acp/aitp all
forward the raw user query to A unchanged (True) -- this is the protocol
artifact that motivates the verbatim-vs-clean distinction in the first
place (Section "Preliminaries" of draft.tex). raw is the one exception
(False): B paraphrases freely, so there is no verbatim-passthrough
channel at all, making it the natural test of whether mean(interim)
still holds when the artifact that motivated clean-vs-verbatim does not
exist.
"""
from masflow import protocol_mcp
from masflow import protocol_a2a
from masflow import protocol_acp
from masflow import protocol_aitp
from masflow import protocol_raw

PROTOCOLS = {
    "mcp": protocol_mcp,
    "a2a": protocol_a2a,
    "acp": protocol_acp,
    "aitp": protocol_aitp,
    "raw": protocol_raw,
}

ALL_PROTOCOL_CHOICES = sorted(PROTOCOLS.keys())


def resolve_protocol(name: str):
    return PROTOCOLS[name]
