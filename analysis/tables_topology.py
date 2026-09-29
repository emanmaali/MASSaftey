#!/usr/bin/env python3
"""
tables_topology.py -- stdlib-only recomputation of four the paper tables
from the per-task result JSONs under <root>/results/.

  tab:topo-asr         verbatim ASR, 5 base topologies x 3 models (BEAST)
  tab:interim-full     per-node interim rates + 6 estimators + final ASR,
                       12 topologies, qwen-0.5b, BEAST
  tab:rq2-generalise   mean(interim) -> final(clean) (err), 12 topologies,
                       qwen-0.5b and qwen-1.5b, BEAST
  tab:protocol-gen     mean(interim) / clean / verbatim for Orchestrate and
                       Tree+2 under mcp / a2a / raw, qwen-0.5b, BEAST

Data sources (toy task set, tasks.TASKS; frozen BEAST suffixes replayed):
  depth-variant topologies  results/run_topology_depth_variants/<mk>/<mk>_<variant>[_<proto>]_task<id>.json
      interim = hd_stage1 .. hd_stage{n_stages-1}; verbatim = hd_final_verbatim;
      clean = hd_final_paraphrase
  Star (fan-out/fan-in)     results/run_topology_fanmerge/<mk>/<mk>_fanmerge_task<id>.json
      interim = hd_peer_raw_B/C/D; verbatim = full_ba_hd; clean = full_para_hd
  plain graphs              results/run_graph_topology/<mk>/<mk>_<graph>_beast_task<id>.json
      interim = hd_mid (chain) / hd_left, hd_right (diamond);
      verbatim = hd_final_verbatim; clean = hd_final_paraphrase

The loading / estimator logic mirrors the original analysis scripts
results/scratch_bfcl/option1_final_analysis.py and
results/scratch_bfcl/option1_estimator_compare.py (which point at the *_bfcl
directories; the paper tables use the toy-task directories above).

Rates are computed exactly (fractions.Fraction) and rounded for display with
round-half-away-from-zero, which is the only rule consistent with every cell
(see NOTES_topology.md; --rounding half_even shows the alternative).

Usage:
  python3 analysis/tables_topology.py [--root PKG] [--table tab:interim-full]
"""
import argparse
import glob
import json
import os
import re
import sys
from fractions import Fraction

# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------
ROUNDING = "half_away"


def rnd(x, nd=0):
    """Round a Fraction/number to nd decimals using the global ROUNDING rule."""
    if x is None:
        return None
    q = Fraction(x) * (10 ** nd)
    if ROUNDING == "half_even":
        r = round(q)  # Fraction.__round__ -> banker's rounding, exact
    else:
        sign = -1 if q < 0 else 1
        a = abs(q)
        r = sign * int(a + Fraction(1, 2))  # half away from zero
    r = Fraction(r, 10 ** nd)
    return int(r) if nd == 0 else float(r)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
TASK_RE = re.compile(r"_task\d+\.json$")

# (display name, paper depth, source, key)
TOPOS = [
    ("Centralized",   1, "depth",    "centralized_d1_orig"),
    ("Orchestrate",   2, "depth",    "orchestrate_d2_orig"),
    ("Tree",          2, "depth",    "tree_d2_orig"),
    ("Mesh",          3, "depth",    "mesh_d3_orig"),
    ("Star",          2, "fanmerge", None),
    ("Centralized+2", 3, "depth",    "centralized_plus2"),
    ("Orchestrate+2", 4, "depth",    "orchestrate_plus2"),
    ("Tree+2",        5, "depth",    "tree_plus2"),
    ("Mesh+2",        5, "depth",    "mesh_plus2"),
    ("Star+2",        4, "depth",    "star_plus2"),
    ("Plain chain",   2, "graph",    "plain_chain"),
    ("Plain diamond", 2, "graph",    "plain_diamond"),
]
TOPO = {t[0]: t for t in TOPOS}
BASE5 = ["Centralized", "Orchestrate", "Tree", "Mesh", "Star"]

# paper model label -> results sub-directory / filename model key
MODELS = {"qwen-0.5b": "qwen_0.5b", "qwen-1.5b": "qwen_1.5b", "phi3.5-mini": "phi35_mini"}


def _files(root, pattern):
    fs = sorted(f for f in glob.glob(os.path.join(root, pattern)) if TASK_RE.search(f))
    rows = []
    for f in fs:
        d = json.load(open(f))
        if "error" in d:
            continue
        rows.append(d)
    return rows


def load_rows(root, model_key, topo_name, protocol="mcp"):
    """Return list of (interim_bools, verbatim_bool, clean_bool), one per task."""
    _, _, src, key = TOPO[topo_name]
    ptag = "" if protocol == "mcp" else f"_{protocol}"
    out = []
    if src == "depth":
        pat = f"results/run_topology_depth_variants/{model_key}/{model_key}_{key}{ptag}_task*.json"
        for d in _files(root, pat):
            n = d["n_stages"]
            interim = [bool(d[f"hd_stage{i}"]) for i in range(1, n)]
            out.append((interim, bool(d["hd_final_verbatim"]), bool(d["hd_final_paraphrase"])))
    elif src == "fanmerge":
        if protocol != "mcp":
            raise ValueError("no non-mcp fanmerge runs")
        pat = f"results/run_topology_fanmerge/{model_key}/{model_key}_fanmerge_task*.json"
        for d in _files(root, pat):
            interim = [bool(d[f"hd_peer_raw_{l}"]) for l in ("B", "C", "D")]
            out.append((interim, bool(d["full_ba_hd"]), bool(d["full_para_hd"])))
    elif src == "graph":
        pat = f"results/run_graph_topology/{model_key}/{model_key}_{key}_beast{ptag}_task*.json"
        fields = ["hd_mid"] if key == "plain_chain" else ["hd_left", "hd_right"]
        for d in _files(root, pat):
            out.append(([bool(d[f]) for f in fields], bool(d["hd_final_verbatim"]),
                        bool(d["hd_final_paraphrase"])))
    return out


def summarize(rows):
    """Exact (Fraction, 0-100 scale) node rates, estimators and final rates."""
    n = len(rows)
    if n == 0:
        return None
    ks = {len(r[0]) for r in rows}
    assert len(ks) == 1, f"inconsistent interim-node count {ks}"
    k = ks.pop()
    node = [Fraction(100 * sum(r[0][i] for r in rows), n) for i in range(k)]
    s = {"n": n, "k": k, "nodes": node,
         "verbatim": Fraction(100 * sum(r[1] for r in rows), n),
         "clean": Fraction(100 * sum(r[2] for r in rows), n)}
    if k:
        srt = sorted(node)
        med = srt[k // 2] if k % 2 else (srt[k // 2 - 1] + srt[k // 2]) / 2
        tot = sum(node)
        s.update(mean=tot / k, median=med, min=min(node), max=max(node),
                 OR=min(Fraction(100), tot),                      # Frechet/Boole upper
                 AND=max(Fraction(0), tot - (k - 1) * 100))       # Frechet/Boole lower
    else:
        s.update(mean=None, median=None, min=None, max=None, OR=None, AND=None)
    return s


# ---------------------------------------------------------------------------
# Tables. Each returns list[dict]: {"row": label, "n": N, "cells": {col: value}}
# (values already rounded to the paper's precision; lists for interim rates)
# ---------------------------------------------------------------------------
def tab_topo_asr(root):
    out = []
    for label, mk in MODELS.items():
        cells, ns = {}, set()
        for t in BASE5:
            s = summarize(load_rows(root, mk, t))
            cells[t] = rnd(s["verbatim"]) if s else None
            ns.add(s["n"] if s else 0)
        out.append({"row": label, "n": "/".join(str(x) for x in sorted(ns)), "cells": cells})
    return out


def tab_interim_full(root):
    out = []
    for name, depth, _, _ in TOPOS:
        s = summarize(load_rows(root, "qwen_0.5b", name))
        c = {"Depth": depth}
        c["Interim"] = [rnd(p) for p in s["nodes"]] if s["k"] else None
        for col in ("mean", "median", "min", "max", "OR", "AND"):
            c[col] = rnd(s[col])
        c["Verbatim"] = rnd(s["verbatim"])
        c["Clean"] = rnd(s["clean"])
        out.append({"row": name, "n": s["n"], "cells": c})
    return out


def tab_rq2_generalise(root):
    out = []
    for name, depth, _, _ in TOPOS:
        c, ns = {"Depth": depth}, []
        for label, mk in (("qwen-0.5b", "qwen_0.5b"), ("qwen-1.5b", "qwen_1.5b")):
            s = summarize(load_rows(root, mk, name))
            ns.append(s["n"])
            c[f"{label} mean"] = rnd(s["mean"])
            c[f"{label} final"] = rnd(s["clean"])
            # error computed from the UNROUNDED mean, then rounded
            c[f"{label} err"] = rnd(s["mean"] - s["clean"]) if s["mean"] is not None else None
        out.append({"row": name, "n": "/".join(map(str, ns)), "cells": c})
    return out


def tab_protocol_gen(root):
    out = []
    for name in ("Orchestrate", "Tree+2"):
        for proto in ("mcp", "a2a", "raw"):
            s = summarize(load_rows(root, "qwen_0.5b", name, protocol=proto))
            out.append({"row": f"{name} / {proto}", "n": s["n"], "cells": {
                "Mean(interim)": rnd(s["mean"], 1),
                "Final (clean)": rnd(s["clean"], 1),
                "Verbatim": rnd(s["verbatim"], 1)}})
    return out


# ---------------------------------------------------------------------------
# Numbers transcribed from the paper
# ---------------------------------------------------------------------------
_ = None
EXPECTED = {
    "tab:topo-asr": {
        "qwen-0.5b":   dict(zip(BASE5, [78, 70, 68, 72, 74])),
        "qwen-1.5b":   dict(zip(BASE5, [82, 78, 86, 80, 66])),
        "phi3.5-mini": dict(zip(BASE5, [100, 100, 100, 80, 60])),
    },
    # Depth, Interim, mean, median, min, max, OR, AND, Verbatim, Clean
    "tab:interim-full": {r[0]: dict(zip(
        ["Depth", "Interim", "mean", "median", "min", "max", "OR", "AND", "Verbatim", "Clean"], r[1:])) for r in [
        ("Centralized",   1, None,             _,  _,  _,  _,  _,  _,  78, 4),
        ("Orchestrate",   2, [16],             16, 16, 16, 16, 16, 16, 70, 14),
        ("Tree",          2, [34, 0],          17, 17, 0,  34, 34, 0,  68, 18),
        ("Mesh",          3, [14, 8],          11, 11, 8,  14, 22, 0,  72, 8),
        ("Star",          2, [22, 12, 26],     20, 22, 12, 26, 60, 0,  74, 14),
        ("Centralized+2", 3, [4, 4],           4,  4,  4,  4,  8,  0,  76, 4),
        ("Orchestrate+2", 4, [16, 18, 16],     17, 16, 16, 18, 50, 0,  66, 10),
        ("Tree+2",        5, [34, 0, 40, 14],  22, 24, 0,  40, 88, 0,  72, 0),
        ("Mesh+2",        5, [14, 8, 8, 10],   10, 9,  8,  14, 40, 0,  72, 10),
        ("Star+2",        4, [16, 22],         19, 19, 16, 22, 38, 0,  74, 12),
        ("Plain chain",   2, [16],             16, 16, 16, 16, 16, 16, 74, 12),
        ("Plain diamond", 2, [16, 16],         16, 16, 16, 16, 32, 0,  70, 14),
    ]},
    # Depth, 0.5b mean, final, err, 1.5b mean, final, err
    "tab:rq2-generalise": {r[0]: dict(zip(
        ["Depth", "qwen-0.5b mean", "qwen-0.5b final", "qwen-0.5b err",
         "qwen-1.5b mean", "qwen-1.5b final", "qwen-1.5b err"], r[1:])) for r in [
        ("Centralized",   1, None, 4,  None, None, 8,  None),
        ("Orchestrate",   2, 16,   14, 2,    4,    0,  4),
        ("Tree",          2, 17,   18, -1,   10,   10, 0),
        ("Mesh",          3, 11,   8,  3,    6,    8,  -2),
        ("Star",          2, 20,   14, 6,    14,   10, 4),
        ("Centralized+2", 3, 4,    4,  0,    5,    0,  5),
        ("Orchestrate+2", 4, 17,   10, 7,    9,    4,  5),
        ("Tree+2",        5, 22,   0,  22,   9,    6,  3),
        ("Mesh+2",        5, 10,   10, 0,    7,    8,  -2),
        ("Star+2",        4, 19,   12, 7,    14,   8,  6),
        ("Plain chain",   2, 16,   12, 4,    24,   6,  18),
        ("Plain diamond", 2, 16,   14, 2,    24,   20, 4),
    ]},
    "tab:protocol-gen": {r[0]: dict(zip(["Mean(interim)", "Final (clean)", "Verbatim"], r[1:])) for r in [
        ("Orchestrate / mcp", 16.0, 14.0, 70.0),
        ("Orchestrate / a2a", 16.0, 14.0, 72.0),
        ("Orchestrate / raw", 16.0, 14.0, 20.0),
        ("Tree+2 / mcp",      22.0, 0.0,  72.0),
        ("Tree+2 / a2a",      22.0, 0.0,  76.0),
        ("Tree+2 / raw",      22.0, 0.0,  4.0),
    ]},
}

TABLES = {
    "tab:topo-asr": tab_topo_asr,
    "tab:interim-full": tab_interim_full,
    "tab:rq2-generalise": tab_rq2_generalise,
    "tab:protocol-gen": tab_protocol_gen,
}
# Structural (not data-derived) columns: printed, not counted as a data match
STRUCTURAL = {"Depth"}


def _fmt(v):
    if v is None:
        return "---"
    if isinstance(v, list):
        return ",".join(_fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def compare(label, root):
    rows = TABLES[label](root)
    exp = EXPECTED[label]
    print("=" * 100)
    print(f"{label}   (cells shown as expected|recomputed; '*' = mismatch)")
    print("=" * 100)
    mism = 0
    seen = set()
    for r in rows:
        e = exp.get(r["row"])
        seen.add(r["row"])
        parts = []
        for col, got in r["cells"].items():
            want = None if e is None else e.get(col)
            bad = (e is None or want != got) and col not in STRUCTURAL
            mism += bad
            parts.append(f"{col}={_fmt(want)}|{_fmt(got)}{'*' if bad else ''}")
        print(f"  {r['row']:<18s} N={r['n']:<6}  " + "  ".join(parts))
    for missing in set(exp) - seen:
        print(f"  {missing}: expected row not recomputed *")
        mism += len(exp[missing])
    status = "MATCH" if mism == 0 else f"MISMATCH ({mism} cells)"
    print(f"TABLE {label}: {status}\n")
    return mism


def main():
    global ROUNDING
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=os.path.dirname(here),
                    help="package root (directory containing results/)")
    ap.add_argument("--table", default=None, choices=list(TABLES), help="default: all")
    ap.add_argument("--rounding", default="half_away", choices=["half_away", "half_even"],
                    help="display rounding rule (paper = half_away)")
    a = ap.parse_args()
    ROUNDING = a.rounding
    labels = [a.table] if a.table else list(TABLES)
    total = sum(compare(l, a.root) for l in labels)
    sys.exit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
