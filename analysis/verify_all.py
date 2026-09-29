"""
verify_all.py -- regenerate every paper table from the result JSONs and
compare it with the numbers printed in the paper.

Runs the four per-section checkers in this directory and summarises them:

    python3 analysis/verify_all.py                 # shipped results/
    python3 analysis/verify_all.py --root reproduced  # your own re-run

Each checker prints one line per table, "TABLE <label>: MATCH" or
"TABLE <label>: MISMATCH (k cells)". Some paper cells are known not to match
the data. These are listed in KNOWN_DISCREPANCIES below and explained in
KNOWN_ISSUES.md. They are reported as KNOWN rather than as failures. The exit
code is 0 when every table either matches or mismatches by exactly its
documented number of cells.

Standard library only; Python >= 3.9.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)

CHECKERS = [
    ("Topology (section 3.2 / RQ2)", "tables_topology.py"),
    ("Attack families (RQ1, appendix)", "tables_families.py"),
    ("Real-BFCL + TradingAgents", "tables_bfcl_tradingagents.py"),
    ("Frontier + adaptive", "tables_frontier.py"),
]

# label -> number of cells in the paper that do not match the shipped data.
# Each entry is explained in KNOWN_ISSUES.md, section "Paper vs. data".
KNOWN_DISCREPANCIES = {
    "tab:rq1-toolsel": 4,
    "tab:rq1-full": 4,
    "tab:rq2-crossfam": 1,
    "tab:tradingagents-leak": 6,
    "tab:tradingagents-all": 1,
    "text:bfcl-null-suffix": 1,
}

LINE = re.compile(r"^TABLE (\S+): (MATCH|MISMATCH \((\d+) cells?\))")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=PKG_ROOT,
                    help="directory containing results/ (default: package root)")
    ap.add_argument("--verbose", action="store_true",
                    help="also print each checker's full expected-vs-recomputed output")
    args = ap.parse_args()

    rows, unexpected = [], 0
    for section, script in CHECKERS:
        proc = subprocess.run(
            [sys.executable, os.path.join(HERE, script), "--root", args.root],
            capture_output=True, text=True,
        )
        if args.verbose:
            print(proc.stdout)
        found, seen = [], set()
        for m in (LINE.match(l) for l in proc.stdout.splitlines()):
            # some checkers repeat their TABLE lines in a closing summary
            if m and m.group(1) not in seen:
                seen.add(m.group(1))
                found.append(m)
        if not found:
            unexpected += 1
            rows.append((section, script, "ERROR", "checker produced no TABLE lines:\n"
                         + (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]))
            continue
        for m in found:
            label, status, k = m.group(1), m.group(2), m.group(3)
            k = int(k) if k else 0
            if status == "MATCH":
                verdict = "MATCH"
            elif KNOWN_DISCREPANCIES.get(label) == k:
                verdict = f"KNOWN ({k} documented cell{'s' if k != 1 else ''})"
            else:
                verdict = f"UNEXPECTED MISMATCH ({k} cells)"
                unexpected += 1
            rows.append((section, label, verdict, ""))

    width = max(len(r[1]) for r in rows)
    current = None
    for section, label, verdict, note in rows:
        if section != current:
            print(f"\n{section}")
            current = section
        print(f"  {label:<{width}}  {verdict}")
        if note:
            print(f"      {note}")

    n_match = sum(r[2] == "MATCH" for r in rows)
    n_known = sum(r[2].startswith("KNOWN") for r in rows)
    print(f"\n{len(rows)} tables: {n_match} match exactly, {n_known} known paper discrepancies, "
          f"{unexpected} unexpected.")
    print("See KNOWN_ISSUES.md for what each KNOWN entry means.")
    return 0 if unexpected == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
