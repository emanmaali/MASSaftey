"""
attack_family_registry.py -- maps a topology driver's --attack CLI value
to (task_list, attack_protocol), so the three topology drivers
(run_graph_topology.py, run_topology_depth_variants.py,
run_topology_fanmerge.py) can run any of the five attack families
through the same graph machinery, not just TOOL_SELECTION.

TOOL_SELECTION attacks (beast/aca/cfh*/tamas*) keep using
masflow.tasks.TASKS unchanged -- every other family supplies its own
task list here, since each family's task dataset has different content
per task_id (MASLEAK's secret system prompts, Prompt Infection's
documents, Evil Geniuses' role-play requests are not interchangeable
with the TOOL_SELECTION task set).

Adding a new family: register its (task_list, attack_protocol) pair in
ATTACK_REGISTRY under its <family>asr0/<family>asr1 attack names; nothing
in the three drivers needs to change.
"""
from masflow.tasks import TASKS
from masflow.tasks_bfcl import TASKS_BFCL
from masflow.masleak_tasks import MASLEAK_TASKS
from masflow.prompt_infection_tasks import INFECTION_TASKS
from masflow.evil_geniuses_tasks import EG_TASKS
from masflow.agentleak_f1_tasks import AGENTLEAK_F1_TASKS
from masflow.toma_tasks import TOMA_TASKS
from masflow.flowsteer_tasks import FLOWSTEER_TASKS
from masflow.attack_protocol import TOOL_SELECTION, SECRET_LEAK, PROPAGATION, HARMFUL_CONTENT, PLANNING_STEERING

TOOL_SELECTION_ATTACKS = {"beast", "aca", "cfhasr0", "cfhasr1", "tamasasr0", "tamasasr1"}

ATTACK_REGISTRY = {
    # attack_name: (task_list, attack_protocol)
    "masleakasr0": (MASLEAK_TASKS, SECRET_LEAK),
    "masleakasr1": (MASLEAK_TASKS, SECRET_LEAK),
    "infectionasr0": (INFECTION_TASKS, PROPAGATION),
    "infectionasr1": (INFECTION_TASKS, PROPAGATION),
    "egasr0": (EG_TASKS, HARMFUL_CONTENT),
    "egasr1": (EG_TASKS, HARMFUL_CONTENT),
    "agentleakf1asr0": (AGENTLEAK_F1_TASKS, SECRET_LEAK),
    "agentleakf1asr1": (AGENTLEAK_F1_TASKS, SECRET_LEAK),
    "tomaasr0": (TOMA_TASKS, PROPAGATION),
    "tomaasr1": (TOMA_TASKS, PROPAGATION),
    "flowsteerasr0": (FLOWSTEER_TASKS, PLANNING_STEERING),
    "flowsteerasr1": (FLOWSTEER_TASKS, PLANNING_STEERING),
    # BFCL-derived real-task variants of the TOOL_SELECTION attacks (100
    # tasks from BFCL v4's "multiple" category, Section on task-dataset
    # realism) -- additive only, the plain "beast"/"cfhasr*"/"tamasasr*"
    # names above are untouched and keep using tasks.TASKS as before.
    "beast_bfcl": (TASKS_BFCL, TOOL_SELECTION),
    "cfhasr0_bfcl": (TASKS_BFCL, TOOL_SELECTION),
    "cfhasr1_bfcl": (TASKS_BFCL, TOOL_SELECTION),
    "tamasasr0_bfcl": (TASKS_BFCL, TOOL_SELECTION),
    "tamasasr1_bfcl": (TASKS_BFCL, TOOL_SELECTION),
}

ALL_ATTACK_CHOICES = sorted(TOOL_SELECTION_ATTACKS | set(ATTACK_REGISTRY.keys()))


def resolve(attack_name: str):
    """Returns (task_list, attack_protocol) for any --attack value."""
    if attack_name in TOOL_SELECTION_ATTACKS:
        return TASKS, TOOL_SELECTION
    return ATTACK_REGISTRY[attack_name]
