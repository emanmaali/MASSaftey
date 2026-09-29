#!/usr/bin/env python3
"""
tables_families.py -- stdlib-only recomputation of the attack-family tables
of the paper from the per-task result JSONs under results/.

Tables covered (by \\label):
  tab:rq1-toolsel   mean(interim) -> final(clean), 3 Tool-Selection attacks
  tab:rq1-other     same, 5 non-Tool-Selection attacks
  tab:rq1-bestfit   best of {Mean, Frechet OR, Boole AND} vs final(clean)
  tab:rq1-full      all 8 attacks (union of rq1-toolsel + rq1-other)
  tab:rq2-crossfam  mean |error| of mean(interim) over the 11 topologies
  tab:toma-bounds   TOMA mean / OR / AND / final per topology

Conventions mirror results/scratch_bfcl/option1_estimator_compare.py and
aca_rq2_rq3_validation.py (the original analysis scripts):
  * per-node interim rate p_i = 100 * (#tasks with node hit) / N
  * mean   = sum(p_i) / k
  * OR     = min(100, sum p_i)                 (Frechet/Boole union bound)
  * AND    = max(0, sum p_i - (k-1)*100)       (Frechet intersection bound)
  * final  = 100 * (#tasks with clean/paraphrase final hit) / N
  * error  = estimate - final; cells printed with 1 decimal.
Topology -> file/field mapping:
  depth variants (run_topology_depth_variants): interim = hd_stage1..hd_stage{n_stages-1},
      final = hd_final_paraphrase
  Star (run_topology_fanmerge): interim = hd_peer_raw_{B,C,D};
      final = full_para_hd (Tool Selection) / hd_final_paraphrase (other families)
  plain_chain / plain_diamond (run_graph_topology): interim = hd_mid / hd_left,hd_right;
      final = hd_final_paraphrase
Attack -> result-file tag (ASR anchor used by the paper; inferred from data,
see NOTES_families.md): BEAST (no tag; '_beast' for graph files), CFH=cfhasr1,
TAMAS-DPI=tamasasr1, MASLEAK v2=masleakasr1, AgentLeak F1=agentleakf1asr1,
FlowSteer=flowsteerasr1, Prompt Infection=infectionasr0, TOMA=tomaasr0.

Usage:
  python3 analysis/tables_families.py [--root PKG] [--table tab:rq1-toolsel]
Exit status 0 iff every recomputed numeric cell equals the paper value.
"""
import argparse
import glob
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP

DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "qwen_0.5b"

# (display name, kind, key)  -- the 11 estimable topologies (Centralized
# has depth 1 / no interim node and carries no numbers in these tables).
TOPOS = [
    ("Orchestrate", "depth", "orchestrate_d2_orig"),
    ("Tree", "depth", "tree_d2_orig"),
    ("Mesh", "depth", "mesh_d3_orig"),
    ("Star", "fanmerge", None),
    ("Centralized+2", "depth", "centralized_plus2"),
    ("Orchestrate+2", "depth", "orchestrate_plus2"),
    ("Tree+2", "depth", "tree_plus2"),
    ("Mesh+2", "depth", "mesh_plus2"),
    ("Star+2", "depth", "star_plus2"),
    ("Plain chain", "graph", "plain_chain"),
    ("Plain diamond", "graph", "plain_diamond"),
]
TOPO_NAMES = [t[0] for t in TOPOS]

# display name -> (attack tag, is_tool_selection)
ATTACKS = {
    "BEAST": ("beast", True),
    "CFH": ("cfhasr1", True),
    "TAMAS-DPI": ("tamasasr1", True),
    "MASLEAK v2": ("masleakasr1", False),
    "AgentLeak F1": ("agentleakf1asr1", False),
    "FlowSteer": ("flowsteerasr1", False),
    "Prompt Infection": ("infectionasr0", False),
    "TOMA": ("tomaasr0", False),
}
TOOLSEL = ["BEAST", "CFH", "TAMAS-DPI"]
OTHER = ["MASLEAK v2", "AgentLeak F1", "FlowSteer", "Prompt Infection", "TOMA"]
ALL8 = TOOLSEL + OTHER


def r1(x):
    """Round half-up to 1 decimal (paper precision)."""
    if x is None:
        return None
    v = float(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    return 0.0 if v == 0 else v


# ---------------------------------------------------------------- loading

def _files(root, kind, key, tag):
    base = os.path.join(root, "results")
    if kind == "depth":
        t = "" if tag == "beast" else f"_{tag}"
        pat = f"{base}/run_topology_depth_variants/{MODEL}/{MODEL}_{key}{t}_task*.json"
    elif kind == "fanmerge":
        t = "" if tag == "beast" else f"_{tag}"
        pat = f"{base}/run_topology_fanmerge/{MODEL}/{MODEL}_fanmerge{t}_task*.json"
    else:
        pat = f"{base}/run_graph_topology/{MODEL}/{MODEL}_{key}_{tag}_task*.json"
    return sorted(glob.glob(pat))


def load_rows(root, kind, key, tag, tool_sel):
    """Returns list of (interim_bool_list, final_clean_bool)."""
    rows = []
    for f in _files(root, kind, key, tag):
        d = json.load(open(f))
        if "error" in d:
            continue
        if kind == "depth":
            interim = [bool(d.get(f"hd_stage{i}", False)) for i in range(1, d["n_stages"])]
            final = bool(d["hd_final_paraphrase"])
        elif kind == "fanmerge":
            interim = [bool(d[f"hd_peer_raw_{l}"]) for l in "BCD"]
            final = bool(d["full_para_hd"] if tool_sel else d["hd_final_paraphrase"])
        else:
            interim = [bool(d["hd_mid"])] if key == "plain_chain" else [bool(d["hd_left"]), bool(d["hd_right"])]
            final = bool(d["hd_final_paraphrase"])
        rows.append((interim, final))
    return rows


def stats(rows):
    n = len(rows)
    if n == 0:
        raise RuntimeError("no result rows")
    k = len(rows[0][0])
    p = [100.0 * sum(r[0][i] for r in rows) / n for i in range(k)]
    s = sum(p)
    return {
        "n": n, "k": k, "node_pcts": p,
        "mean": s / k,
        "or": min(100.0, s),
        "and": max(0.0, s - (k - 1) * 100.0),
        "final": 100.0 * sum(r[1] for r in rows) / n,
    }


_CACHE = {}


def get_stats(root, attack, topo):
    ck = (root, attack, topo)
    if ck not in _CACHE:
        tag, ts = ATTACKS[attack]
        _, kind, key = next(t for t in TOPOS if t[0] == topo)
        _CACHE[ck] = stats(load_rows(root, kind, key, tag, ts))
    return _CACHE[ck]


# ---------------------------------------------------------------- tables
# Each tab_* returns a list of cell dicts: {"row", "col", "value"}.

def _mean_final_err(root, attacks):
    out = []
    for topo in TOPO_NAMES:
        for a in attacks:
            s = get_stats(root, a, topo)
            for fld, v in (("mean", s["mean"]), ("final", s["final"]), ("err", s["mean"] - s["final"])):
                out.append({"row": topo, "col": f"{a}/{fld}", "value": v})
    return out


def tab_rq1_toolsel(root):
    return _mean_final_err(root, TOOLSEL)


def tab_rq1_other(root):
    return _mean_final_err(root, OTHER)


def tab_rq1_full(root):
    return _mean_final_err(root, ALL8)


def tab_rq1_bestfit(root):
    """Estimator closest to final among (M, O, A); ties -> first in M,O,A
    order (Python min, as in option1_estimator_compare.py)."""
    out = []
    for topo in TOPO_NAMES:
        for a in OTHER:
            s = get_stats(root, a, topo)
            cands = [("M", s["mean"]), ("O", s["or"]), ("A", s["and"])]
            lab, est = min(cands, key=lambda c: abs(c[1] - s["final"]))
            out.append({"row": topo, "col": f"{a}/est", "value": lab})
            out.append({"row": topo, "col": f"{a}/value", "value": est})
            out.append({"row": topo, "col": f"{a}/final", "value": s["final"]})
            out.append({"row": topo, "col": f"{a}/err", "value": est - s["final"]})
            # informational: improvement of chosen bound over the mean (for the green rule)
            out.append({"row": topo, "col": f"{a}/_gain_over_mean",
                        "value": abs(s["mean"] - s["final"]) - abs(est - s["final"]), "info": True})
    return out


def tab_rq2_crossfam(root):
    """Avg |mean(interim) - final| over the 11 estimable topologies,
    computed from unrounded per-topology values."""
    out = []
    for a in ["AgentLeak F1", "CFH", "MASLEAK v2", "BEAST", "FlowSteer", "TOMA", "TAMAS-DPI", "Prompt Infection"]:
        errs = [abs(get_stats(root, a, t)["mean"] - get_stats(root, a, t)["final"]) for t in TOPO_NAMES]
        out.append({"row": a, "col": "mae", "value": sum(errs) / len(errs)})
        # informational: MAE of the 1-decimal-rounded per-topology errors
        out.append({"row": a, "col": "_mae_from_rounded",
                    "value": sum(abs(r1(get_stats(root, a, t)["mean"] - get_stats(root, a, t)["final"]))
                                 for t in TOPO_NAMES) / len(TOPO_NAMES), "info": True})
    return out


def tab_toma_bounds(root):
    """Star+2: the paper reports OR/AND as n/a (only the post-join merged
    stage is instrumented in star_plus2, not the 3 branch marginals), so
    those two cells are emitted as None; the naive values from the merged
    stages are emitted as informational cells."""
    out = []
    for topo in TOPO_NAMES:
        s = get_stats(root, "TOMA", topo)
        out.append({"row": topo, "col": "mean", "value": s["mean"]})
        if topo == "Star+2":
            out.append({"row": topo, "col": "or", "value": None})
            out.append({"row": topo, "col": "and", "value": None})
            out.append({"row": topo, "col": "_or_naive", "value": s["or"], "info": True})
            out.append({"row": topo, "col": "_and_naive", "value": s["and"], "info": True})
        else:
            out.append({"row": topo, "col": "or", "value": s["or"]})
            out.append({"row": topo, "col": "and", "value": s["and"]})
        out.append({"row": topo, "col": "final", "value": s["final"]})
    return out


# ---------------------------------------------------------------- EXPECTED
# Transcribed from the paper. Triples are (mean, final, err).

_TOOLSEL_EXP = {  # BEAST, CFH, TAMAS-DPI
    "Orchestrate":   [(16.0, 14.0, 2.0), (22.0, 20.0, 2.0), (30.0, 16.0, 14.0)],
    "Tree":          [(17.0, 18.0, -1.0), (7.0, 4.0, 3.0), (13.0, 14.0, -1.0)],
    "Mesh":          [(11.0, 8.0, 3.0), (9.0, 6.0, 3.0), (14.0, 6.0, 8.0)],
    "Star":          [(20.0, 14.0, 6.0), (18.4, 22.0, -3.6), (23.3, 16.0, 7.3)],
    "Centralized+2": [(4.0, 4.0, 0.0), (6.0, 4.0, 2.0), (6.0, 4.0, 2.0)],
    "Orchestrate+2": [(17.0, 10.0, 7.0), (22.0, 22.0, 0.0), (27.3, 16.0, 11.3)],
    "Tree+2":        [(22.0, 0.0, 22.0), (8.0, 0.0, 8.0), (16.0, 0.0, 16.0)],
    "Mesh+2":        [(10.0, 10.0, 0.0), (9.5, 6.0, 3.5), (12.0, 6.0, 6.0)],
    "Star+2":        [(19.0, 12.0, 7.0), (23.0, 20.0, 3.0), (23.0, 18.0, 5.0)],
    "Plain chain":   [(16.0, 12.0, 4.0), (22.0, 20.0, 2.0), (28.0, 18.0, 10.0)],
    "Plain diamond": [(16.0, 14.0, 2.0), (22.0, 18.0, 4.0), (28.0, 22.0, 6.0)],
}
_OTHER_EXP = {  # MASLEAK v2, AgentLeak F1, FlowSteer, Prompt Infection, TOMA
    "Orchestrate":   [(8.0, 0.0, 8.0), (6.0, 0.0, 6.0), (8.0, 0.0, 8.0), (34.0, 0.0, 34.0), (4.0, 0.0, 4.0)],
    "Tree":          [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 0.0, 1.0), (29.0, 0.0, 29.0), (0.0, 0.0, 0.0)],
    "Mesh":          [(18.0, 18.0, 0.0), (3.0, 4.0, -1.0), (6.0, 6.0, 0.0), (37.0, 70.0, -33.0), (24.0, 26.0, -2.0)],
    "Star":          [(18.7, 34.0, -15.3), (2.7, 6.0, -3.3), (7.3, 14.0, -6.7), (40.7, 98.0, -57.3), (28.0, 70.0, -42.0)],
    "Centralized+2": [(7.0, 8.0, -1.0), (0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    "Orchestrate+2": [(4.7, 2.0, 2.7), (6.0, 0.0, 6.0), (6.0, 0.0, 6.0), (11.3, 0.0, 11.3), (4.0, 0.0, 4.0)],
    "Tree+2":        [(0.0, 4.0, -4.0), (0.5, 0.0, 0.5), (1.5, 4.0, -2.5), (14.5, 0.0, 14.5), (0.0, 0.0, 0.0)],
    "Mesh+2":        [(18.0, 18.0, 0.0), (3.5, 4.0, -0.5), (6.0, 6.0, 0.0), (38.5, 78.0, -39.5), (25.0, 26.0, -1.0)],
    "Star+2":        [(29.0, 22.0, 7.0), (5.0, 4.0, 1.0), (11.0, 6.0, 5.0), (39.0, 6.0, 33.0), (56.0, 40.0, 16.0)],
    "Plain chain":   [(14.0, 20.0, -6.0), (6.0, 6.0, 0.0), (8.0, 10.0, -2.0), (36.0, 10.0, 26.0), (0.0, 0.0, 0.0)],
    "Plain diamond": [(14.0, 14.0, 0.0), (6.0, 6.0, 0.0), (8.0, 8.0, 0.0), (36.0, 40.0, -4.0), (0.0, 0.0, 0.0)],
}
# tab:rq1-full is transcribed separately (it is a separate block of LaTeX);
# its values are identical to the two tables above.
_FULL_EXP = {
    "Orchestrate":   [(16.0, 14.0, 2.0), (22.0, 20.0, 2.0), (30.0, 16.0, 14.0), (8.0, 0.0, 8.0), (6.0, 0.0, 6.0), (8.0, 0.0, 8.0), (34.0, 0.0, 34.0), (4.0, 0.0, 4.0)],
    "Tree":          [(17.0, 18.0, -1.0), (7.0, 4.0, 3.0), (13.0, 14.0, -1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1.0, 0.0, 1.0), (29.0, 0.0, 29.0), (0.0, 0.0, 0.0)],
    "Mesh":          [(11.0, 8.0, 3.0), (9.0, 6.0, 3.0), (14.0, 6.0, 8.0), (18.0, 18.0, 0.0), (3.0, 4.0, -1.0), (6.0, 6.0, 0.0), (37.0, 70.0, -33.0), (24.0, 26.0, -2.0)],
    "Star":          [(20.0, 14.0, 6.0), (18.4, 22.0, -3.6), (23.3, 16.0, 7.3), (18.7, 34.0, -15.3), (2.7, 6.0, -3.3), (7.3, 14.0, -6.7), (40.7, 98.0, -57.3), (28.0, 70.0, -42.0)],
    "Centralized+2": [(4.0, 4.0, 0.0), (6.0, 4.0, 2.0), (6.0, 4.0, 2.0), (7.0, 8.0, -1.0), (0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
    "Orchestrate+2": [(17.0, 10.0, 7.0), (22.0, 22.0, 0.0), (27.3, 16.0, 11.3), (4.7, 2.0, 2.7), (6.0, 0.0, 6.0), (6.0, 0.0, 6.0), (11.3, 0.0, 11.3), (4.0, 0.0, 4.0)],
    "Tree+2":        [(22.0, 0.0, 22.0), (8.0, 0.0, 8.0), (16.0, 0.0, 16.0), (0.0, 4.0, -4.0), (0.5, 0.0, 0.5), (1.5, 4.0, -2.5), (14.5, 0.0, 14.5), (0.0, 0.0, 0.0)],
    "Mesh+2":        [(10.0, 10.0, 0.0), (9.5, 6.0, 3.5), (12.0, 6.0, 6.0), (18.0, 18.0, 0.0), (3.5, 4.0, -0.5), (6.0, 6.0, 0.0), (38.5, 78.0, -39.5), (25.0, 26.0, -1.0)],
    "Star+2":        [(19.0, 12.0, 7.0), (23.0, 20.0, 3.0), (23.0, 18.0, 5.0), (29.0, 22.0, 7.0), (5.0, 4.0, 1.0), (11.0, 6.0, 5.0), (39.0, 6.0, 33.0), (56.0, 40.0, 16.0)],
    "Plain chain":   [(16.0, 12.0, 4.0), (22.0, 20.0, 2.0), (28.0, 18.0, 10.0), (14.0, 20.0, -6.0), (6.0, 6.0, 0.0), (8.0, 10.0, -2.0), (36.0, 10.0, 26.0), (0.0, 0.0, 0.0)],
    "Plain diamond": [(16.0, 14.0, 2.0), (22.0, 18.0, 4.0), (28.0, 22.0, 6.0), (14.0, 14.0, 0.0), (6.0, 6.0, 0.0), (8.0, 8.0, 0.0), (36.0, 40.0, -4.0), (0.0, 0.0, 0.0)],
}
# (label, estimate, final, err) per OTHER attack
_BESTFIT_EXP = {
    "Orchestrate":   [("M", 8.0, 0.0, 8.0), ("M", 6.0, 0.0, 6.0), ("M", 8.0, 0.0, 8.0), ("M", 34.0, 0.0, 34.0), ("M", 4.0, 0.0, 4.0)],
    "Tree":          [("M", 0.0, 0.0, 0.0), ("A", 0.0, 0.0, 0.0), ("A", 0.0, 0.0, 0.0), ("A", 0.0, 0.0, 0.0), ("M", 0.0, 0.0, 0.0)],
    "Mesh":          [("M", 18.0, 18.0, 0.0), ("M", 3.0, 4.0, -1.0), ("M", 6.0, 6.0, 0.0), ("O", 74.0, 70.0, 4.0), ("M", 24.0, 26.0, -2.0)],
    "Star":          [("M", 18.7, 34.0, -15.3), ("O", 8.0, 6.0, 2.0), ("M", 7.3, 14.0, -6.7), ("O", 100.0, 98.0, 2.0), ("O", 84.0, 70.0, 14.0)],
    "Centralized+2": [("M", 7.0, 8.0, -1.0), ("M", 0.0, 0.0, 0.0), ("M", 2.0, 2.0, 0.0), ("M", 0.0, 0.0, 0.0), ("M", 0.0, 0.0, 0.0)],
    "Orchestrate+2": [("A", 0.0, 2.0, -2.0), ("A", 0.0, 0.0, 0.0), ("A", 0.0, 0.0, 0.0), ("A", 0.0, 0.0, 0.0), ("A", 0.0, 0.0, 0.0)],
    "Tree+2":        [("M", 0.0, 4.0, -4.0), ("A", 0.0, 0.0, 0.0), ("O", 6.0, 4.0, 2.0), ("A", 0.0, 0.0, 0.0), ("M", 0.0, 0.0, 0.0)],
    "Mesh+2":        [("M", 18.0, 18.0, 0.0), ("M", 3.5, 4.0, -0.5), ("M", 6.0, 6.0, 0.0), ("O", 100.0, 78.0, 22.0), ("M", 25.0, 26.0, -1.0)],
    "Star+2":        [("M", 29.0, 22.0, 7.0), ("M", 5.0, 4.0, 1.0), ("M", 11.0, 6.0, 5.0), ("A", 0.0, 6.0, -6.0), ("M", 56.0, 40.0, 16.0)],
    "Plain chain":   [("M", 14.0, 20.0, -6.0), ("M", 6.0, 6.0, 0.0), ("M", 8.0, 10.0, -2.0), ("M", 36.0, 10.0, 26.0), ("M", 0.0, 0.0, 0.0)],
    "Plain diamond": [("M", 14.0, 14.0, 0.0), ("M", 6.0, 6.0, 0.0), ("M", 8.0, 8.0, 0.0), ("M", 36.0, 40.0, -4.0), ("M", 0.0, 0.0, 0.0)],
}
# green cells as highlighted in the paper (for the informational highlight check)
_BESTFIT_GREEN = {("Mesh", "Prompt Infection"), ("Star", "AgentLeak F1"), ("Star", "Prompt Infection"),
                  ("Star", "TOMA"), ("Tree+2", "FlowSteer"), ("Mesh+2", "Prompt Infection")}
_CROSSFAM_EXP = {"AgentLeak F1": 1.8, "CFH": 3.1, "MASLEAK v2": 4.0, "BEAST": 4.9, "FlowSteer": 2.8,
                 "TOMA": 6.3, "TAMAS-DPI": 7.9, "Prompt Infection": 25.6}
_TOMA_EXP = {  # mean, or, and, final
    "Orchestrate": (4.0, 4.0, 4.0, 0.0), "Tree": (0.0, 0.0, 0.0, 0.0), "Mesh": (24.0, 48.0, 0.0, 26.0),
    "Star": (28.0, 84.0, 0.0, 70.0), "Centralized+2": (0.0, 0.0, 0.0, 0.0), "Orchestrate+2": (4.0, 12.0, 0.0, 0.0),
    "Tree+2": (0.0, 0.0, 0.0, 0.0), "Mesh+2": (25.0, 100.0, 0.0, 26.0), "Star+2": (56.0, None, None, 40.0),
    "Plain chain": (0.0, 0.0, 0.0, 0.0), "Plain diamond": (0.0, 0.0, 0.0, 0.0),
}


def _exp_mfe(table, attacks):
    out = {}
    for topo, triples in table.items():
        for a, (m, f, e) in zip(attacks, triples):
            out[(topo, f"{a}/mean")] = m
            out[(topo, f"{a}/final")] = f
            out[(topo, f"{a}/err")] = e
    return out


def _exp_bestfit():
    out = {}
    for topo, quads in _BESTFIT_EXP.items():
        for a, (lab, v, f, e) in zip(OTHER, quads):
            out[(topo, f"{a}/est")] = lab
            out[(topo, f"{a}/value")] = v
            out[(topo, f"{a}/final")] = f
            out[(topo, f"{a}/err")] = e
    return out


EXPECTED = {
    "tab:rq1-toolsel": _exp_mfe(_TOOLSEL_EXP, TOOLSEL),
    "tab:rq1-other": _exp_mfe(_OTHER_EXP, OTHER),
    "tab:rq1-bestfit": _exp_bestfit(),
    "tab:rq1-full": _exp_mfe(_FULL_EXP, ALL8),
    "tab:rq2-crossfam": {(a, "mae"): v for a, v in _CROSSFAM_EXP.items()},
    "tab:toma-bounds": {(t, c): v for t, vals in _TOMA_EXP.items()
                        for c, v in zip(("mean", "or", "and", "final"), vals)},
}

TABLES = {
    "tab:rq1-toolsel": tab_rq1_toolsel,
    "tab:rq1-other": tab_rq1_other,
    "tab:rq1-bestfit": tab_rq1_bestfit,
    "tab:rq1-full": tab_rq1_full,
    "tab:rq2-crossfam": tab_rq2_crossfam,
    "tab:toma-bounds": tab_toma_bounds,
}


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, str):
        return v
    return f"{r1(v):.1f}"


def _eq(exp, got):
    if exp is None or got is None:
        return exp is None and got is None
    if isinstance(exp, str) or isinstance(got, str):
        return exp == got
    return r1(got) == r1(exp)


def compare(label, root, verbose=True):
    cells = TABLES[label](root)
    exp = EXPECTED[label]
    got = {(c["row"], c["col"]): c["value"] for c in cells if not c.get("info")}
    info = [c for c in cells if c.get("info")]
    missing = set(exp) - set(got)
    extra = set(got) - set(exp)
    assert not missing and not extra, (label, missing, extra)
    mism = 0
    print("=" * 78)
    print(f"{label}")
    print(f"{'row':15s} {'column':30s} {'paper':>8s} {'recomp':>8s}")
    for (row, col), e in exp.items():
        g = got[(row, col)]
        ok = _eq(e, g)
        mism += not ok
        if verbose or not ok:
            print(f"{row:15s} {col:30s} {_fmt(e):>8s} {_fmt(g):>8s}  {'ok' if ok else '<-- MISMATCH'}")
    # informational extras (not counted)
    if label == "tab:rq1-bestfit":
        for c in info:
            topo, a = c["row"], c["col"].split("/")[0]
            green_rule = c["value"] > 5.0
            if green_rule != ((topo, a) in _BESTFIT_GREEN):
                print(f"  [info] green-highlight inconsistency {topo}/{a}: gain over mean = "
                      f"{c['value']:.1f}pp, paper {'green' if (topo, a) in _BESTFIT_GREEN else 'not green'}")
    elif info:
        for c in info:
            print(f"  [info] {c['row']:15s} {c['col']:22s} {_fmt(c['value'])}")
    print(f"TABLE {label}: " + ("MATCH" if mism == 0 else f"MISMATCH ({mism} cells)"))
    return mism


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=DEFAULT_ROOT, help="package root (contains results/)")
    ap.add_argument("--table", default=None, choices=list(TABLES), help="only this table")
    ap.add_argument("--quiet", action="store_true", help="print only mismatching cells")
    args = ap.parse_args()
    labels = [args.table] if args.table else list(TABLES)
    total = 0
    for lab in labels:
        total += compare(lab, args.root, verbose=not args.quiet)
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
