"""
compare_replay.py -- compare a (partial) re-run against the shipped results.

Every replay driver writes one JSON per (model, topology, attack, task). This
script pairs each file under <reproduced>/results/ with the file at the same
relative path under results/ and compares the attack-outcome fields: every
boolean key starting with "hd_" or "full_" (for example hd_stage1,
hd_final_paraphrase or full_ba_hd). Generated text is not compared, because
it can differ across GPUs, dtypes and library builds without changing an
outcome.

    python3 analysis/compare_replay.py                      # default: reproduced/
    python3 analysis/compare_replay.py --reproduced my_run --show-diffs

Standard library only; Python >= 3.9.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def outcome_fields(d: dict) -> dict:
    return {k: v for k, v in d.items()
            if (k.startswith("hd_") or k.startswith("full_")) and isinstance(v, bool)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reproduced", default=os.path.join(PKG_ROOT, "reproduced"),
                    help="directory containing your re-run's results/ tree")
    ap.add_argument("--shipped", default=PKG_ROOT,
                    help="directory containing the shipped results/ tree")
    ap.add_argument("--show-diffs", action="store_true")
    args = ap.parse_args()

    rep_root = os.path.join(args.reproduced, "results")
    if not os.path.isdir(rep_root):
        print(f"No re-run found at {rep_root}. Run one of scripts/*.sh first.")
        return 2

    n_files = n_fields = n_agree = n_missing = 0
    diffs = []
    for dp, _, fs in os.walk(rep_root):
        for name in sorted(fs):
            if not name.endswith(".json"):
                continue
            rel = os.path.relpath(os.path.join(dp, name), rep_root)
            ref = os.path.join(args.shipped, "results", rel)
            if not os.path.exists(ref):
                n_missing += 1
                continue
            a = outcome_fields(json.load(open(os.path.join(dp, name))))
            b = outcome_fields(json.load(open(ref)))
            n_files += 1
            for k in sorted(set(a) & set(b)):
                n_fields += 1
                if a[k] == b[k]:
                    n_agree += 1
                else:
                    diffs.append((rel, k, b[k], a[k]))

    if n_files == 0:
        print("No re-run files have a shipped counterpart to compare with.")
        return 2
    print(f"Compared {n_files} result files ({n_fields} outcome fields).")
    print(f"Agreement with shipped results: {n_agree}/{n_fields} "
          f"({100 * n_agree / n_fields:.1f}%)")
    if n_missing:
        print(f"{n_missing} re-run files had no shipped counterpart (ignored).")
    if diffs:
        print(f"{len(diffs)} outcome fields differ.")
        if args.show_diffs:
            for rel, k, shipped, mine in diffs:
                print(f"  {rel}  {k}: shipped={shipped} re-run={mine}")
        else:
            print("Re-run with --show-diffs to list them.")
    return 0 if not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
