import json, re
from collections import Counter, defaultdict

tasks = [json.loads(l) for l in open('.scratch_bfcl/multiple.json')]
answers = {json.loads(l)['id']: json.loads(l) for l in open('.scratch_bfcl/multiple_answers.json')}


def fix_schema_types(node):
    """BFCL uses 'dict' where JSON-schema/OpenAI function-calling expects
    'object'; recurse to fix nested parameter definitions too."""
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and v == "dict":
                out[k] = "object"
            else:
                out[k] = fix_schema_types(v)
        return out
    if isinstance(node, list):
        return [fix_schema_types(x) for x in node]
    return node


def cluster_key(name):
    return name.split(".")[0].split("_")[0]


# Domain-diversity sampling: cap per-cluster count so no single domain
# (e.g. "calculate.*", 53 tasks) dominates the 100-task sample.
by_cluster = defaultdict(list)
for t in tasks:
    fn_names = [f["name"] for f in t["function"]]
    key = cluster_key(fn_names[0])
    by_cluster[key].append(t)

TARGET_N = 100
CAP_PER_CLUSTER = 6
selected = []
clusters_sorted = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))
# round-robin across clusters, capped, until we hit TARGET_N
idx = 0
cluster_iters = {k: iter(v[:CAP_PER_CLUSTER]) for k, v in clusters_sorted}
active = list(cluster_iters.keys())
while len(selected) < TARGET_N and active:
    still_active = []
    for k in active:
        try:
            selected.append(next(cluster_iters[k]))
            still_active.append(k)
        except StopIteration:
            pass
        if len(selected) >= TARGET_N:
            break
    active = still_active

selected = selected[:TARGET_N]
print(f"Selected {len(selected)} tasks across {len(set(cluster_key(t['function'][0]['name']) for t in selected))} domain clusters")

converted = []
for i, t in enumerate(selected):
    a = answers[t["id"]]
    gt_name = list(a["ground_truth"][0].keys())[0]
    fn_by_name = {f["name"]: f for f in t["function"]}
    distractor_name = next(n for n in fn_by_name if n != gt_name)

    gt_args_raw = a["ground_truth"][0][gt_name]
    expected_args = {}
    for k, v in gt_args_raw.items():
        if isinstance(v, list) and v and not (v[0] == "" and len(v) > 1):
            expected_args[k] = v[0]

    tools_for_task = []
    for name in [gt_name, distractor_name]:
        f = fn_by_name[name]
        params = fix_schema_types(f["parameters"])
        tools_for_task.append({
            "name": f["name"],
            "description": f["description"],
            "parameters": params,
        })

    converted.append({
        "task_id": i,
        "user_task": t["question"][0][0]["content"],
        "expected_tool": gt_name,
        "expected_args": expected_args,
        "attack_target_tool": distractor_name,
        "tools": tools_for_task,
        "source_id": t["id"],
    })

json.dump(converted, open(".scratch_bfcl/converted_100.json", "w"), indent=2)
print(f"Wrote {len(converted)} converted tasks")
# sanity spot-check
print(json.dumps(converted[0], indent=2)[:900])
