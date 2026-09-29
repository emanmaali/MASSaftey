"""
convert_family_suffixes.py -- turn the attack-family suffix generators' raw
outputs into the per-task suffix files that the topology replay drivers read.

The generators (masflow.run_035_prompt_infection, run_039_agentleak_f1,
run_040_masleak_v2, run_041_flowsteer, run_043_toma, run_045_evil_geniuses)
write results/run_0NN/<model>_<name>_task<id>.json, each holding the
optimised per-task "suffix". The replay drivers read
<dir>/<model>_mcp_<family>asr1_task<id>.json = {"task_id", "suffix", "source"}.
This script does that conversion for ASR1 (optimised suffixes). ASR0 payloads
are the attacks' own fixed hand-written strings; those files are shipped
unchanged in results/suffixes/attack_families/.

This conversion step was not preserved from the original runs. This script
reimplements it. Run on the original generator outputs, it reproduces every
shipped ASR1 file exactly (verified for all 6 families x 50 tasks).

    python3 scripts/convert_family_suffixes.py --src results \
        --out reproduced/results/suffixes/attack_families/qwen_0.5b

Standard library only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

# family tag used by the replay drivers -> (generator run dir, raw file stem)
FAMILIES = {
    "infection":   ("run_035", "infection_asr1"),
    "agentleakf1": ("run_039", "agentleak_f1_asr1"),
    "masleak":     ("run_040", "masleak_v2_aca"),
    "flowsteer":   ("run_041", "flowsteer_asr1"),
    "toma":        ("run_043", "toma_asr1"),
    "eg":          ("run_045", "eg_asr1"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", default="results", help="directory containing run_0NN/ generator outputs")
    ap.add_argument("--out", required=True, help="output directory for the replay-ready suffix files")
    ap.add_argument("--model-key", default="qwen_0.5b")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    written = missing = 0
    for fam, (run_dir, stem) in FAMILIES.items():
        pattern = os.path.join(args.src, run_dir, f"{args.model_key}_{stem}_task*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"  {fam}: no generator output at {pattern}")
            missing += 1
            continue
        for f in files:
            d = json.load(open(f))
            if "error" in d or "suffix" not in d:
                continue
            tid = re.search(r"_task(\d+)\.json$", f).group(1)
            out = {"task_id": int(tid), "suffix": d["suffix"], "source": f"{fam}asr1_optimized"}
            path = os.path.join(args.out, f"{args.model_key}_mcp_{fam}asr1_task{tid}.json")
            with open(path, "w") as fh:
                json.dump(out, fh, indent=2)
            written += 1
        print(f"  {fam}: {len(files)} tasks")
    print(f"Wrote {written} ASR1 suffix files to {args.out}")
    return 1 if missing == len(FAMILIES) else 0


if __name__ == "__main__":
    sys.exit(main())
