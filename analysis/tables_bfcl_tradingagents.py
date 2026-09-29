#!/usr/bin/env python3
"""Recompute the real-BFCL and TradingAgents tables of the paper from
the saved result JSONs (stdlib only, no model/GPU code).

Tables / claims covered (labels as they appear in the paper):
  tab:realbfcl-attempt   Toy vs Real-BFCL(original target) vs Real-BFCL(fixed
                         auth_user target), BEAST, qwen-0.5b, 3 topologies.
  tab:tradingagents-leak Secret Leak per-node survival on TradingAgents.
  tab:tradingagents-all  All five attack families on TradingAgents (caption
                         "All five attack families tested against
                         TradingAgents"; it was referred to as `sec:frontier`
                         in the task brief, but its \\label in the paper is
                         tab:tradingagents-all -- sec:frontier is the label
                         of the following subsubsection).
  text:bfcl-null-suffix  Numbers quoted in the BFCL prose: null-suffix
                         baseline confusion 33--43% (qwen-0.5b), 3.3%
                         (qwen-1.5b), 0% after retargeting; star fan-out
                         hit-detection 0% vs logprob signal 33--37%.

Usage (from the package root):
  python3 analysis/tables_bfcl_tradingagents.py            # all tables
  python3 analysis/tables_bfcl_tradingagents.py --table tab:realbfcl-attempt
  python3 analysis/tables_bfcl_tradingagents.py --ta-leak-source propagation

Rounding: the paper rounds half away from zero (e.g. Star+2 real-BFCL
mean(interim) 20.5% -> "21%", error -18.5pp -> "-19pp"); the error is computed
from the UNROUNDED estimate/observed values and then rounded.
Exit status: 0 only if every requested table matches.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

PKG_DEFAULT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def rnd(x: float, nd: int = 0) -> float:
    """Round half away from zero (Decimal ROUND_HALF_UP is away-from-zero)."""
    q = Decimal(1).scaleb(-nd)
    return float(Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP))


def pct(x: float, nd: int = 0) -> str:
    v = rnd(x, nd)
    return f"{v:.{nd}f}%"


def pp(x: float) -> str:
    v = int(rnd(x, 0))
    return f"{'+' if v >= 0 else '-'}{abs(v)}pp"


def load_jsons(pattern: str):
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            d = json.load(fh)
        if "error" in d:  # same convention as results/scratch_bfcl/option1_*.py
            continue
        out.append(d)
    return out


def interim_of(d: dict):
    """Interim-node hit-detection flags, mirroring
    results/scratch_bfcl/option1_final_analysis.py (load_graph/load_depth)."""
    if "hd_mid" in d:  # graph_topology plain_chain
        return [d["hd_mid"]]
    if "hd_left" in d:  # graph_topology plain_diamond
        return [d["hd_left"], d["hd_right"]]
    return [d.get(f"hd_stage{i}", False) for i in range(1, d["n_stages"])]


def summarize(rows):
    """mean(interim) = average over interim-node positions of the per-node
    hit rate (== pooled mean, since every task in a file set has the same
    node count); final (clean) = hd_final_paraphrase rate."""
    n = len(rows)
    k = len(interim_of(rows[0]))
    node = [sum(bool(interim_of(r)[i]) for r in rows) / n * 100 for i in range(k)]
    mean_i = sum(node) / k
    final_c = sum(bool(r["hd_final_paraphrase"]) for r in rows) / n * 100
    return {"n": n, "nodes": node, "mean": mean_i, "final": final_c}


def compare(label, expected: dict, got: dict, notes=()):
    print("=" * 100)
    print(f"{label}")
    print("=" * 100)
    bad = 0
    w = max(len(str(k)) for k in expected)
    for key, exp in expected.items():
        g = got.get(key, "MISSING")
        ok = g == exp
        bad += not ok
        print(f"  {str(key):{w}s}  expected={exp!s:28s} recomputed={g!s:28s} {'ok' if ok else '<-- MISMATCH'}")
    for n in notes:
        print(f"  NOTE: {n}")
    status = "MATCH" if bad == 0 else f"MISMATCH ({bad} cells)"
    print(f"TABLE {label}: {status}")
    print()
    return bad == 0


# ---------------------------------------------------------------------------
# tab:realbfcl-attempt
# ---------------------------------------------------------------------------
EXPECTED_REALBFCL = {
    # (topology, condition): "est->obs (err)"
    ("Plain chain", "Toy"): "16%->12% (+4pp)",
    ("Plain chain", "Real-BFCL, original target"): "21%->40% (-19pp)",
    ("Plain chain", "Real-BFCL, fixed target"): "0%->0% (+0pp)",
    ("Mesh", "Toy"): "11%->8% (+3pp)",
    ("Mesh", "Real-BFCL, original target"): "21%->31% (-10pp)",
    ("Mesh", "Real-BFCL, fixed target"): "6%->1% (+5pp)",
    ("Star+2", "Toy"): "19%->12% (+7pp)",
    ("Star+2", "Real-BFCL, original target"): "21%->39% (-19pp)",
    ("Star+2", "Real-BFCL, fixed target"): "1%->1% (+0pp)",
}
EXPECTED_REALBFCL_N = {"Toy": 50, "Real-BFCL, original target": 100, "Real-BFCL, fixed target": 100}

# topology -> file stem per result directory
_TOPO_STEM = {
    "Plain chain": ("plain_chain_beast", "plain_chain_beast"),  # (toy/orig stem, injected stem)
    "Mesh": ("mesh_d3_orig", "mesh_d3_orig"),
    "Star+2": ("star_plus2", "star_plus2"),
}


def realbfcl_globs(root):
    r = os.path.join(root, "results")
    g = {}
    for topo in _TOPO_STEM:
        graph = topo == "Plain chain"
        stem = _TOPO_STEM[topo][0]
        g[(topo, "Toy")] = os.path.join(
            r, "run_graph_topology" if graph else "run_topology_depth_variants",
            "qwen_0.5b", f"qwen_0.5b_{stem}_task*.json")
        g[(topo, "Real-BFCL, original target")] = os.path.join(
            r, "run_graph_topology_bfcl" if graph else "run_topology_depth_variants_bfcl",
            "qwen_0.5b", f"qwen_0.5b_{stem}_task*.json")
        g[(topo, "Real-BFCL, fixed target")] = os.path.join(
            r, "run_beast_injected_pilot", "qwen_0.5b", f"qwen_0.5b_{_TOPO_STEM[topo][1]}_task*.json")
    return g


def tab_realbfcl_attempt(root):
    got, notes, ns = {}, [], {}
    for key, pattern in realbfcl_globs(root).items():
        rows = load_jsons(pattern)
        if not rows:
            got[key] = "NO DATA"
            continue
        s = summarize(rows)
        got[key] = f"{pct(s['mean'])}->{pct(s['final'])} ({pp(s['mean'] - s['final'])})"
        ns.setdefault(key[1], set()).add(s["n"])
        tgt = {r.get("target") for r in rows}
        notes.append(f"{key[0]:11s} | {key[1]:27s} N={s['n']:3d} target={'auth_user' if tgt == {'auth_user'} else ('mixed' if 'auth_user' in tgt else 'BFCL distractor')} "
                     f"nodes={[round(x, 1) for x in s['nodes']]} mean={s['mean']:.2f} final={s['final']:.2f} "
                     f"<- {os.path.relpath(pattern, root)}")
    for cond, exp_n in EXPECTED_REALBFCL_N.items():
        if ns.get(cond) != {exp_n}:
            notes.append(f"CAPTION N DISCREPANCY: '{cond}' caption says N={exp_n}, files give N={sorted(ns.get(cond, []))} "
                         f"(not counted as a cell).")
    return got, notes


# ---------------------------------------------------------------------------
# TradingAgents
# ---------------------------------------------------------------------------
TA_NODES = ["News Analyst (entry)", "Bull debate", "Bear debate", "Trader plan",
            "Aggressive risk debater", "Conservative risk debater", "Neutral risk debater"]
TA_SINK = "Final decision (sink)"
# JSON field names per results file
TA_FIELDS = {
    "secretleak": {"News Analyst (entry)": "interim_leak", "Bull debate": "bull_leak",
                   "Bear debate": "bear_leak", "Trader plan": "trader_leak",
                   "Aggressive risk debater": None, "Conservative risk debater": None,
                   "Neutral risk debater": None, TA_SINK: "final_leak"},
    "propagation": {"News Analyst (entry)": "news", "Bull debate": "bull", "Bear debate": "bear",
                    "Trader plan": "trader", "Aggressive risk debater": "aggressive",
                    "Conservative risk debater": "conservative", "Neutral risk debater": "neutral",
                    TA_SINK: "final"},
}

EXPECTED_TA_LEAK = {
    "News Analyst (entry)": "100%", "Bull debate": "0%", "Bear debate": "0%", "Trader plan": "0%",
    "Aggressive risk debater": "20%", "Conservative risk debater": "20%", "Neutral risk debater": "20%",
    TA_SINK: "0%", "mean(interim)": "22.9%", "AND bound": "0.0%",
    "mean(interim) - final (pp, caption)": "22.9", "AND - final (pp, text)": "0.0", "N": "5",
}


def _ta_file(root, name):
    return os.path.join(root, "results", "tradingagents", f"{name}_results.json")


def _ta_node_rates(root, source):
    rows = json.load(open(_ta_file(root, source)))
    rates = {}
    for node, field in TA_FIELDS[source].items():
        rates[node] = None if field is None else sum(bool(r[field]) for r in rows) / len(rows) * 100
    return rows, rates


def tab_tradingagents_leak(root, source="secretleak"):
    rows, rates = _ta_node_rates(root, source)
    got = {}
    for node in TA_NODES + [TA_SINK]:
        got[node] = "n/a (field absent)" if rates[node] is None else pct(rates[node])
    avail = [rates[n] for n in TA_NODES if rates[n] is not None]
    k = len(avail)
    mean_i = sum(avail) / k
    and_b = max(0.0, sum(avail) - (k - 1) * 100)  # Frechet/Boole, as option1_estimator_compare.py
    got["mean(interim)"] = pct(mean_i, 1)
    got["AND bound"] = pct(and_b, 1)
    got["mean(interim) - final (pp, caption)"] = f"{rnd(mean_i - rates[TA_SINK], 1):.1f}"
    got["AND - final (pp, text)"] = f"{rnd(and_b - rates[TA_SINK], 1):.1f}"
    got["N"] = str(len(rows))
    notes = [f"source file: {os.path.relpath(_ta_file(root, source), root)} ({len(rows)} rows; "
             f"tickers={[r['ticker'] for r in rows]}; dates={[r.get('date') for r in rows]}); "
             f"mean/AND over the {k} interior nodes present in this file.",
             "The paper's numbers (bear 0%, risk debaters 20%, mean 22.9%) are exactly those of "
             "results/tradingagents/propagation_results.json; secretleak_results.json has bear_leak=2/5 "
             "and no risk-debater fields. Re-run with --ta-leak-source propagation to see that match."]
    return got, notes


EXPECTED_TA_ALL = {
    "Secret leak: entry": "100%", "Secret leak: downstream range": "0-20%", "Secret leak: final": "0%",
    "Propagation: entry": "100%", "Propagation: final": "0%",
    "Tool selection: hijack successes": "0/5",
    "Planning steering": "Not tested",
    "Harmful content: entry": "100%", "Harmful content: trader": "0/5", "Harmful content: final": "0/5",
}


def tab_tradingagents_all(root, leak_source="secretleak"):
    got, notes = {}, []
    _, sl = _ta_node_rates(root, leak_source)
    down = [sl[n] for n in TA_NODES[1:] if sl[n] is not None]
    got["Secret leak: entry"] = pct(sl[TA_NODES[0]])
    got["Secret leak: downstream range"] = f"{rnd(min(down)):.0f}-{rnd(max(down)):.0f}%"
    got["Secret leak: final"] = pct(sl[TA_SINK])
    notes.append(f"Secret-leak row computed from {leak_source}_results.json")

    _, pr = _ta_node_rates(root, "propagation")
    got["Propagation: entry"] = pct(pr[TA_NODES[0]])
    got["Propagation: final"] = pct(pr[TA_SINK])
    notes.append("Propagation downstream nodes in propagation_results.json: "
                 + ", ".join(f"{n}={pct(pr[n])}" for n in TA_NODES[1:])
                 + " (the prose's '0% at all downstream nodes' does not hold for this file; the table cell only states entry->final).")

    ts = json.load(open(_ta_file(root, "toolselection")))
    got["Tool selection: hijack successes"] = f"{sum(bool(r['wrong_tool_called']) for r in ts)}/{len(ts)}"
    notes.append(f"Tool selection: topic_string_in_report={sum(bool(r['topic_string_in_report']) for r in ts)}/{len(ts)} "
                 "(not reported in the paper).")

    ps = glob.glob(os.path.join(root, "results", "tradingagents", "*plan*"))
    got["Planning steering"] = "Not tested" if not ps else f"FILES PRESENT: {ps}"

    hc = json.load(open(_ta_file(root, "harmfulcontent")))
    n = len(hc)
    got["Harmful content: entry"] = pct(sum(bool(r["harmful_in_news_report"]) for r in hc) / n * 100)
    got["Harmful content: trader"] = f"{sum(bool(r['harmful_in_trader_plan']) for r in hc)}/{n}"
    got["Harmful content: final"] = f"{sum(bool(r['harmful_in_final']) for r in hc)}/{n}"
    return got, notes


# ---------------------------------------------------------------------------
# BFCL prose: null-suffix controls and star fan-out bug
# ---------------------------------------------------------------------------
EXPECTED_NULL = {
    "qwen-0.5b null, orig target, plain chain (N=30): interim": "33%",
    "qwen-0.5b null, orig target, plain chain (N=30): final": "43%",
    "qwen-1.5b null, orig target, plain chain (N=30): final": "3.3%",
    "qwen-0.5b null, fixed target, Plain chain: interim": "0%",
    "qwen-0.5b null, fixed target, Plain chain: final": "0%",
    "qwen-0.5b null, fixed target, Mesh: interim": "0%",
    "qwen-0.5b null, fixed target, Mesh: final": "0%",
    "qwen-0.5b null, fixed target, Star+2: interim": "0%",
    "qwen-0.5b null, fixed target, Star+2: final": "0%",
    "Star (fan-out), orig target, qwen-0.5b BEAST: max hd over all nodes": "0%",
    "Star logprob ft range (entry hops A->B/C/D .. full back-attack)": "33-37%",
}


def tab_null_suffix_text(root):
    r = os.path.join(root, "results")
    got, notes = {}, []

    def sub(rows, ids):
        return [x for x in rows if x["task_id"] in ids]

    first30 = set(range(30))
    pc05 = load_jsons(os.path.join(r, "null_suffix_check/qwen_0.5b/qwen_0.5b_plain_chain_beast_task*.json"))
    s = summarize(sub(pc05, first30))
    got["qwen-0.5b null, orig target, plain chain (N=30): interim"] = pct(s["mean"])
    got["qwen-0.5b null, orig target, plain chain (N=30): final"] = pct(s["final"])
    for topo, stem in [("Plain chain", "plain_chain_beast"), ("Mesh", "mesh_d3_orig"), ("Star+2", "star_plus2")]:
        rows = load_jsons(os.path.join(r, f"null_suffix_check/qwen_0.5b/qwen_0.5b_{stem}_task*.json"))
        a, b = summarize(sub(rows, first30)), summarize(rows)
        notes.append(f"null orig-target qwen-0.5b {topo:11s}: tasks0-29 nodes={[round(x, 1) for x in a['nodes']]} final={a['final']:.1f} | "
                     f"N={b['n']} nodes={[round(x, 1) for x in b['nodes']]} final={b['final']:.1f}")
    pc15 = load_jsons(os.path.join(r, "null_suffix_check/qwen_1.5b/qwen_1.5b_plain_chain_beast_task*.json"))
    s15 = summarize(pc15)
    got["qwen-1.5b null, orig target, plain chain (N=30): final"] = pct(s15["final"], 1)
    notes.append(f"qwen-1.5b null plain chain N={s15['n']}: interim={s15['mean']:.1f} final={s15['final']:.1f}")

    for topo, stem in [("Plain chain", "plain_chain_beast"), ("Mesh", "mesh_d3_orig"), ("Star+2", "star_plus2")]:
        rows = load_jsons(os.path.join(r, f"null_suffix_check_injected/qwen_0.5b/qwen_0.5b_{stem}_task*.json"))
        s = summarize(rows)
        got[f"qwen-0.5b null, fixed target, {topo}: interim"] = pct(s["mean"])
        got[f"qwen-0.5b null, fixed target, {topo}: final"] = pct(s["final"])
        notes.append(f"null fixed-target {topo:11s} N={s['n']} nodes={s['nodes']} final={s['final']:.1f}")
    toy = summarize(load_jsons(os.path.join(r, "null_suffix_check_toy/qwen_0.5b/qwen_0.5b_plain_chain_beast_task*.json")))
    notes.append(f"toy null plain chain N={toy['n']}: interim={toy['mean']:.1f} final={toy['final']:.1f}")

    star = load_jsons(os.path.join(r, "run_topology_fanmerge_bfcl/qwen_0.5b/qwen_0.5b_fanmerge_task*.json"))
    n = len(star)
    hd_keys = [k for k in star[0] if k.startswith(("hd_", "para_hd_")) or k in ("full_ba_hd", "full_para_hd")]
    ft_keys = [k for k in star[0] if k.startswith("ft_") or k in ("full_ba_ft", "full_para_ft")]
    rate = {k: sum(bool(d[k]) for d in star) / n * 100 for k in hd_keys + ft_keys}
    got["Star (fan-out), orig target, qwen-0.5b BEAST: max hd over all nodes"] = pct(max(rate[k] for k in hd_keys))
    claim = [rate["ft_A_B"], rate["ft_A_C"], rate["ft_A_D"], rate["full_ba_ft"]]
    got["Star logprob ft range (entry hops A->B/C/D .. full back-attack)"] = f"{rnd(min(claim)):.0f}-{rnd(max(claim)):.0f}%"
    notes.append(f"star N={n}; all ft rates: " + ", ".join(f"{k}={rate[k]:.0f}" for k in ft_keys)
                 + f" (full range over every ft field: {min(rate[k] for k in ft_keys):.0f}-{max(rate[k] for k in ft_keys):.0f}%)")
    notes.append("INTERPRETATION: '33--43%' is reproduced as the first-30-task null run on plain chain "
                 "(interim 33.3%, final 43.3%; logs scratch_bfcl/null_suffix_*_05b.log ran tasks 0-29). "
                 "At N=100 the final-clean null rates are 37-43% and interim 22-27%. '33--37%' is reproduced "
                 "as ft on the entry hops (33%) up to the full-path back-attack ft (37%); the exact field set "
                 "used by the authors is not recorded.")
    return got, notes


# ---------------------------------------------------------------------------
TABLES = {
    "tab:realbfcl-attempt": (tab_realbfcl_attempt, EXPECTED_REALBFCL),
    "tab:tradingagents-leak": (tab_tradingagents_leak, EXPECTED_TA_LEAK),
    "tab:tradingagents-all": (tab_tradingagents_all, EXPECTED_TA_ALL),
    "text:bfcl-null-suffix": (tab_null_suffix_text, EXPECTED_NULL),
}
ALIASES = {"sec:frontier": "tab:tradingagents-all"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=PKG_DEFAULT, help="package root (default: %(default)s)")
    ap.add_argument("--table", default="all", help="one of: all, " + ", ".join(list(TABLES) + list(ALIASES)))
    ap.add_argument("--ta-leak-source", default="secretleak", choices=["secretleak", "propagation"],
                    help="TradingAgents JSON used for the Secret Leak cells (default: the file named "
                         "for that family, secretleak_results.json)")
    args = ap.parse_args(argv)
    labels = list(TABLES) if args.table == "all" else [ALIASES.get(args.table, args.table)]
    all_ok = True
    for label in labels:
        if label not in TABLES:
            ap.error(f"unknown table {label}")
        fn, exp = TABLES[label]
        if label in ("tab:tradingagents-leak", "tab:tradingagents-all"):
            got, notes = fn(args.root, args.ta_leak_source)
        else:
            got, notes = fn(args.root)
        all_ok &= compare(label, exp, got, notes)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
