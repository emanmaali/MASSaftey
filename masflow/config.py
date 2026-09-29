"""
config.py — Centralized configuration, model loading, and shared dataclasses.

All env-var reading, device/dtype selection, and model instantiation lives here.
No other file should import os.environ or load models directly.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
from torch import Tensor, LongTensor


# ---------------------------------------------------------------------------
# Dataclasses — model handles
# ---------------------------------------------------------------------------

@dataclass
class ModelHandle:
    """Abstraction over local-HF vs OpenAI-served model backends."""
    backend: Literal["local_hf", "openai"]
    model_name: str

    # local_hf fields
    device: str = "cuda"
    dtype: str = "bfloat16"

    # openai fields (ignored when backend == "local_hf")
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    def __post_init__(self):
        if self.backend == "openai":
            self.base_url = self.base_url or os.environ.get("OPENAI_BASE_URL")
            self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
            if not self.base_url:
                raise ValueError(
                    "OpenAI backend requires OPENAI_BASE_URL env var or explicit base_url"
                )


# ---------------------------------------------------------------------------
# Dataclasses — results
# ---------------------------------------------------------------------------

@dataclass
class GCGResult:
    """Result from gcg_attack()."""
    best_input_ids: LongTensor
    best_loss: float
    loss_history: list[float]
    target_achieved: bool


@dataclass
class STEResult:
    """Result from ste_gcg_attack()."""
    best_p_ids: LongTensor
    best_loss: float
    loss_history: list[float]
    induced_B_text: str
    induced_o: Optional[object]  # ToolCall or None
    hard_decode_success: bool


@dataclass
class BenchmarkReport:
    """Result from benchmark()."""
    protocol: str
    accuracy: float
    per_task: list[dict]
    confusion_matrix: dict
    passed_gate: bool  # True iff accuracy >= 0.9


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DTYPE = "bfloat16" if (
    torch.cuda.is_available() and torch.cuda.is_bf16_supported()
) else "float32"

SWEEP_MODELS = {
    "smol_135m": "HuggingFaceTB/SmolLM2-135M-Instruct",   # dev/debug
    "smol_1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",   # main sweep
    "qwen_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",            # main sweep
}

# ---------------------------------------------------------------------------
# Model roster — frozen paper scope (experimental_protocol.md Â§9.2)
# ---------------------------------------------------------------------------
# Ordered smallest-first so partial completion under compute/time pressure
# still yields full coverage at the cheap end of the Â§6.2 model-scale
# ablation. `gated` models require a Hugging Face access grant before use
# (Llama-3.2-3B-Instruct already failed to load under our HF token in
# Run 010 — see experiment_logs/run_010.md).
@dataclass
class RosterEntry:
    key: str
    hf_id: str
    params_b: float   # approx parameter count, in billions
    gated: bool = False
    proven: bool = False  # already used successfully somewhere in this repo


MODEL_ROSTER: list[RosterEntry] = [
    RosterEntry("smol_135m",  "HuggingFaceTB/SmolLM2-135M-Instruct", 0.135, proven=True),
    RosterEntry("smol_360m",  "HuggingFaceTB/SmolLM2-360M-Instruct", 0.36,  proven=True),
    RosterEntry("qwen_0.5b",  "Qwen/Qwen2.5-0.5B-Instruct",          0.5,   proven=True),
    RosterEntry("qwen3_0.6b", "Qwen/Qwen3-0.6B",                     0.6),
    RosterEntry("llama_1b",   "meta-llama/Llama-3.2-1B-Instruct",    1.0,   gated=True),
    RosterEntry("qwen_1.5b",  "Qwen/Qwen2.5-1.5B-Instruct",          1.5,   proven=True),
    RosterEntry("smol_1.7b",  "HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.7,   proven=True),
    RosterEntry("gemma2_2b",  "google/gemma-2-2b-it",                2.0),
    RosterEntry("qwen_3b",    "Qwen/Qwen2.5-3B-Instruct",            3.0,   proven=True),
    RosterEntry("llama_3b",   "meta-llama/Llama-3.2-3B-Instruct",    3.0,   gated=True),
    RosterEntry("phi35_mini", "microsoft/Phi-3.5-mini-instruct",     3.8),
]

MODEL_ROSTER_BY_KEY: dict[str, RosterEntry] = {m.key: m for m in MODEL_ROSTER}

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _patch_system_role_if_unsupported(tokenizer) -> None:
    """Some chat templates (Gemma's, notably) reject a leading
    {"role": "system"} message outright, raising at apply_chat_template
    time. Every attack script across this project constructs messages as
    [{"role": "system", ...}, {"role": "user", ...}, ...] -- 130+ call
    sites across 30+ files, all funneling through the tokenizer this
    function returns. Rather than special-case every call site, probe
    once here whether the native template actually supports a system
    turn; if not, transparently wrap apply_chat_template to fold the
    leading system message into the following user message before
    delegating to the original. Models whose template already supports
    system role (the common case) are left completely untouched."""
    probe = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    try:
        tokenizer.apply_chat_template(probe, tokenize=False)
        return
    except Exception:
        pass

    print("  [config] tokenizer's chat template rejects a system-role message "
          "(e.g. Gemma) -- patching apply_chat_template to fold system into "
          "the first user turn.")
    original_apply = tokenizer.apply_chat_template

    def patched_apply_chat_template(messages, *args, **kwargs):
        messages = list(messages)
        if messages and messages[0].get("role") == "system":
            sys_content = messages[0].get("content", "")
            rest = messages[1:]
            if rest and rest[0].get("role") == "user":
                merged = dict(rest[0])
                merged["content"] = sys_content + "\n\n" + rest[0].get("content", "")
                messages = [merged] + rest[1:]
            else:
                # No user turn to fold into -- demote system to a user turn.
                messages = [{"role": "user", "content": sys_content}] + rest
        return original_apply(messages, *args, **kwargs)

    tokenizer.apply_chat_template = patched_apply_chat_template


def load_local_model(
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    dtype: str = DEFAULT_DTYPE,
):
    """Load a HuggingFace CausalLM + tokenizer for white-box access.

    Returns (model, tokenizer).  Model is in eval mode with no grad by default.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = DTYPE_MAP.get(dtype, torch.float32)

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _patch_system_role_if_unsupported(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch_dtype,
    ).to(device)
    model.eval()
    print(f"  Loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, dtype={model.dtype}")

    return model, tokenizer


def get_openai_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """Return an OpenAI client configured from env vars or explicit args."""
    import openai

    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "no-key")

    return openai.OpenAI(base_url=base_url, api_key=api_key)


def get_anthropic_client(api_key: Optional[str] = None):
    """Return an Anthropic client configured from env vars or explicit args."""
    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=api_key)


def get_google_client(api_key: Optional[str] = None):
    """Return a Gemini (google-genai) client configured from env vars or explicit args."""
    import google.genai as genai

    api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 0):
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Default model : {DEFAULT_MODEL}")
    print(f"Default device: {DEFAULT_DEVICE}")
    print(f"Default dtype : {DEFAULT_DTYPE}")
    seed_everything(42)
    print("Seeding OK")
    print("config.py self-test passed ✓")
