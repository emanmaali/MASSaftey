"""
beast.py — BEAST (BEAm Search-based adversarial aTtack) for the B→A pipeline.

Based on: Sadasivan et al., "Fast Adversarial Attacks on Language Models
In One GPU Minute" (ICML 2024).

Key idea: gradient-free beam search over adversarial tokens.
At each step, expand beams by trying random token swaps, evaluate
with hard B.generate() → A.forward(), and keep the top-k beams.

This avoids the STE gradient problem entirely — no differentiable
path through B needed. Directly optimizes the hard discrete objective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import LongTensor
from typing import Optional
import time

from masflow.config import STEResult, seed_everything
from masflow.gcg import get_ascii_printable_tokens


# ---------------------------------------------------------------------------
# Helper: build A's (tool-caller's) input from pre-computed template + suffix
# ---------------------------------------------------------------------------

def build_a_input(a_prefix_ids, a_postfix_ids, suffix_ids):
    """Build A's full input IDs by inserting suffix into pre-computed template.

    Args:
        a_prefix_ids: LongTensor — tokens before suffix in A's input
        a_postfix_ids: LongTensor — tokens after suffix in A's input
        suffix_ids: LongTensor — current adversarial suffix tokens

    Returns:
        a_ids: LongTensor [seq_len] — A's full input
        suffix_positions_in_a: LongTensor [n_suffix] — positions of suffix in a_ids
    """
    a_ids = torch.cat([a_prefix_ids, suffix_ids, a_postfix_ids])
    suffix_start = a_prefix_ids.shape[0]
    suffix_positions_in_a = torch.arange(suffix_start, suffix_start + suffix_ids.shape[0])
    return a_ids, suffix_positions_in_a


def beast_expand_beams(
    beams: LongTensor,              # [beam_width, seq_len]
    optim_positions: LongTensor,    # [n_adv]
    allowed_tokens: LongTensor,     # [vocab_subset]
    n_expansions: int = 8,          # expansions per beam
) -> LongTensor:
    """Expand each beam by randomly swapping tokens at adversarial positions.

    For each beam, generate n_expansions variants by:
      - Picking a random adversarial position
      - Replacing it with a random token from allowed_tokens

    Returns:
        expanded: [beam_width * n_expansions, seq_len]
    """
    beam_width, seq_len = beams.shape
    n_adv = optim_positions.shape[0]
    expanded = []

    for b in range(beam_width):
        for _ in range(n_expansions):
            new_seq = beams[b].clone()
            # Pick random position to swap
            pos_idx = torch.randint(0, n_adv, (1,)).item()
            pos = optim_positions[pos_idx].item()
            # Pick random replacement token
            tok_idx = torch.randint(0, allowed_tokens.shape[0], (1,)).item()
            new_seq[pos] = allowed_tokens[tok_idx]
            expanded.append(new_seq)

    return torch.stack(expanded, dim=0)


def gradient_guided_expand_beams(
    B_model,
    A_model,
    beams: LongTensor,              # [beam_width, seq_len]
    optim_positions: LongTensor,    # [n_adv] — suffix positions in B's input
    target_tool_ids: LongTensor,    # [t] — target tool tokens for gradient
    allowed_tokens: LongTensor,     # [vocab_subset]
    n_expansions: int = 8,          # expansions per beam
    topk: int = 64,                 # gradient top-k for token selection
    grad_fraction: float = 0.5,     # fraction of expansions that use gradient
    a_prefix_ids: Optional[LongTensor] = None,  # A's input template prefix
    a_postfix_ids: Optional[LongTensor] = None, # A's input template postfix
) -> LongTensor:
    """Expand beams using the tool-caller's (A_model) gradient.

    Computes gradient on A's input at the suffix positions to find what
    suffix token changes would make A more likely to call the target tool.
    The suffix occupies known positions in A's input (for all protocols,
    the user query with suffix is included in A's formatted input).

    Requires a_prefix_ids/a_postfix_ids to construct A's input.
    Falls back to B_model gradient if A template not available.

    Returns:
        expanded: [beam_width * n_expansions, seq_len]
    """
    from masflow.gcg import token_gradients, sample_candidates

    beam_width, seq_len = beams.shape
    n_adv = optim_positions.shape[0]
    device = next(A_model.parameters()).device

    n_grad = max(1, int(n_expansions * grad_fraction))
    n_rand = n_expansions - n_grad

    use_a_gradient = (a_prefix_ids is not None and a_postfix_ids is not None)

    all_expanded = []

    for b in range(beam_width):
        beam = beams[b]

        # ── Gradient-guided expansions ──
        if n_grad > 0:
            try:
                if use_a_gradient:
                    # Build A's input with current suffix
                    suffix_ids = beam[optim_positions]
                    a_ids, a_suffix_pos = build_a_input(
                        a_prefix_ids, a_postfix_ids, suffix_ids
                    )
                    a_ids = a_ids.to(device)
                    a_suffix_pos = a_suffix_pos.to(device)

                    # Target slice: where A would generate (right after A's input)
                    a_len = a_ids.shape[0]
                    t_len = target_tool_ids.shape[0]
                    target_slice = slice(a_len - 1, a_len - 1 + t_len)

                    # Gradient on A (tool-caller) at suffix positions in A's input
                    grads = token_gradients(
                        A_model, a_ids, a_suffix_pos, target_tool_ids, target_slice
                    )  # [n_adv, V]
                else:
                    # Fallback: B_model gradient (legacy behavior)
                    prompt_len = beam.shape[0]
                    t_len = target_tool_ids.shape[0]
                    target_slice = slice(prompt_len - 1, prompt_len - 1 + t_len)
                    grads = token_gradients(
                        B_model, beam, optim_positions, target_tool_ids, target_slice
                    )

                # Apply candidates to B's beam at suffix positions
                grad_candidates = sample_candidates(
                    beam, optim_positions, grads,
                    search_width=n_grad, topk=topk,
                    allowed_token_ids=allowed_tokens,
                )
                all_expanded.append(grad_candidates)
            except Exception as e:
                # Fall back to random if gradient fails
                print(f"    [GBeast] gradient failed: {e}")
                n_rand += n_grad

        # ── Random expansions (diversity) ──
        for _ in range(n_rand):
            new_seq = beam.clone()
            pos_idx = torch.randint(0, n_adv, (1,)).item()
            pos = optim_positions[pos_idx].item()
            tok_idx = torch.randint(0, allowed_tokens.shape[0], (1,)).item()
            new_seq[pos] = allowed_tokens[tok_idx]
            all_expanded.append(new_seq.unsqueeze(0).to(device))

    return torch.cat(all_expanded, dim=0).cpu()  # beast_attack expects CPU candidates


def gumbel_guided_expand_beams(
    B_model,
    A_model,
    tokenizer,
    beams: LongTensor,              # [beam_width, seq_len]
    optim_positions: LongTensor,    # [n_adv]
    target_tool_ids: LongTensor,    # [t]
    tool_system_prompt: str,
    allowed_tokens: LongTensor,     # [vocab_subset]
    n_expansions: int = 8,
    topk: int = 64,
    grad_fraction: float = 0.5,
    *,
    intermediate_len: int = 32,
    step_frac: float = 0.0,        # for temperature annealing
    loss_mode: str = "ce",
    expected_tool_ids: Optional[LongTensor] = None,
    margin: float = 1.0,
    alternate_with_b: bool = False, # if True, alternate B-only and B→A gradients
    step: int = 0,                  # current step (for alternation)
) -> LongTensor:
    """Expand beams using FULL B→A gradient via Annealed Gumbel STE.

    Unlike gradient_guided_expand_beams which only uses B's gradient,
    this computes ∂L/∂x through the entire B→(Gumbel STE)→A pipeline.
    The gradient captures both how adversarial tokens affect B's output
    AND how B's output affects A's tool selection.

    When alternate_with_b=True, odd steps use B-only gradient and even
    steps use full B→A gradient, providing two complementary search
    directions.

    Returns:
        expanded: [beam_width * n_expansions, seq_len]
    """
    from masflow.gcg import sample_candidates, token_gradients
    from masflow.ste_gcg import ste_token_gradients
    from masflow.estimators import get_estimator

    beam_width, seq_len = beams.shape
    n_adv = optim_positions.shape[0]
    device = next(B_model.parameters()).device

    n_grad = max(1, int(n_expansions * grad_fraction))
    n_rand = n_expansions - n_grad

    # Choose gradient source
    use_b_only = alternate_with_b and (step % 2 == 0)

    all_expanded = []

    for b in range(beam_width):
        beam = beams[b]

        if n_grad > 0:
            try:
                if use_b_only:
                    # B-only gradient (same as current G-BEAST)
                    prompt_len = beam.shape[0]
                    t_len = target_tool_ids.shape[0]
                    target_slice = slice(prompt_len - 1, prompt_len - 1 + t_len)
                    grads = token_gradients(
                        B_model, beam, optim_positions,
                        target_tool_ids, target_slice,
                    )
                else:
                    # Full B→A gradient via Annealed Gumbel STE
                    estimator = get_estimator("annealed_gumbel")
                    decode_fn = estimator.decode_fn
                    est_kwargs = {
                        "tau_start": estimator.tau_start,
                        "tau_end": estimator.tau_end,
                        "topk": estimator.topk,
                        "n_samples": estimator.n_samples,
                        "step_frac": step_frac,
                    }

                    B_model.zero_grad()
                    A_model.zero_grad()

                    grads, _, _ = ste_token_gradients(
                        B_model, A_model, tokenizer,
                        beam, optim_positions, target_tool_ids,
                        tool_system_prompt,
                        intermediate_len=intermediate_len,
                        temperature=1.0,
                        decode_fn=decode_fn,
                        decode_kwargs=est_kwargs,
                        loss_mode=loss_mode,
                        expected_tool_ids=expected_tool_ids,
                        margin=margin,
                    )

                # Use gradient to propose candidates
                grad_candidates = sample_candidates(
                    beam, optim_positions, grads,
                    search_width=n_grad, topk=topk,
                    allowed_token_ids=allowed_tokens,
                )
                all_expanded.append(grad_candidates)
            except Exception as e:
                # Fall back to random if gradient fails
                n_rand += n_grad

        # ── Random expansions (diversity) ──
        for _ in range(n_rand):
            new_seq = beam.clone()
            pos_idx = torch.randint(0, n_adv, (1,)).item()
            pos = optim_positions[pos_idx].item()
            tok_idx = torch.randint(0, allowed_tokens.shape[0], (1,)).item()
            new_seq[pos] = allowed_tokens[tok_idx]
            all_expanded.append(new_seq.unsqueeze(0).to(device))

    return torch.cat(all_expanded, dim=0).cpu()


def beast_evaluate(
    B_model,
    A_model,
    tokenizer,
    candidates: LongTensor,         # [n_cands, seq_len]
    target_tool_ids: LongTensor,    # [t]
    tool_system_prompt: str,
    *,
    intermediate_len: int = 32,
    batch_size: int = 16,
    format_a_content=None,  # BUG-2 fix: callable (b_text) -> a_user_content
) -> tuple[list[float], list[str]]:
    """Evaluate candidates with hard B.generate() → A.forward().

    Returns:
        losses: list of CE losses (one per candidate)
        b_texts: list of B's output texts
    """
    device = next(B_model.parameters()).device
    n_cands = candidates.shape[0]
    losses = []
    b_texts = []

    # Process in batches
    for start in range(0, n_cands, batch_size):
        end = min(start + batch_size, n_cands)
        batch = candidates[start:end].to(device)

        with torch.no_grad():
            b_outputs = B_model.generate(
                batch,
                max_new_tokens=intermediate_len,
                do_sample=False,
            )

        prompt_len = candidates.shape[1]
        for i in range(batch.shape[0]):
            b_new = b_outputs[i, prompt_len:]
            b_text = tokenizer.decode(b_new, skip_special_tokens=True)
            b_texts.append(b_text)

            if not b_text.strip():
                losses.append(float("inf"))
                continue

            # Build A's input and compute loss
            a_user_content = format_a_content(b_text) if format_a_content else b_text
            messages_A = [
                {"role": "system", "content": tool_system_prompt},
                {"role": "user", "content": a_user_content},
            ]
            a_prompt = tokenizer.apply_chat_template(
                messages_A, tokenize=False, add_generation_prompt=True
            )
            a_ids = tokenizer.encode(a_prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                a_logits = A_model(a_ids).logits

            a_len = a_ids.shape[1]
            t_len = target_tool_ids.shape[0]
            target_logits = a_logits[0, a_len - 1: a_len - 1 + t_len, :]
            actual_t = min(target_logits.shape[0], t_len)
            loss = F.cross_entropy(
                target_logits[:actual_t],
                target_tool_ids[:actual_t].to(device),
            ).item()
            losses.append(loss)

    return losses, b_texts


# ---------------------------------------------------------------------------
# Single-candidate hard-decode evaluation
# ---------------------------------------------------------------------------

def _eval_single_candidate(
    B_model, A_model, tokenizer,
    candidate: LongTensor,          # [seq_len]
    target_tool_ids: LongTensor,
    tool_system_prompt: str,
    intermediate_len: int = 32,
    format_a_content=None,  # BUG-2 fix: callable (b_text) -> a_user_content
) -> tuple[float, str]:
    """Evaluate a single candidate with hard B.generate() → A.forward().

    Returns (loss, b_text).
    """
    device = next(B_model.parameters()).device
    with torch.no_grad():
        b_out = B_model.generate(
            candidate.unsqueeze(0).to(device),
            max_new_tokens=intermediate_len,
            do_sample=False,
        )
    b_new = b_out[0, candidate.shape[0]:]
    b_text = tokenizer.decode(b_new, skip_special_tokens=True)

    if not b_text.strip():
        return float("inf"), b_text

    a_user_content = format_a_content(b_text) if format_a_content else b_text
    messages_A = [
        {"role": "system", "content": tool_system_prompt},
        {"role": "user", "content": a_user_content},
    ]
    a_prompt = tokenizer.apply_chat_template(
        messages_A, tokenize=False, add_generation_prompt=True
    )
    a_ids = tokenizer.encode(a_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        a_logits = A_model(a_ids).logits

    a_len = a_ids.shape[1]
    t_len = target_tool_ids.shape[0]
    target_logits = a_logits[0, a_len - 1: a_len - 1 + t_len, :]
    actual_t = min(target_logits.shape[0], t_len)
    loss = F.cross_entropy(
        target_logits[:actual_t],
        target_tool_ids[:actual_t].to(device),
    ).item()

    return loss, b_text


# ---------------------------------------------------------------------------
# Hard-decode helper (BUG-1 fix: uses protocol-aware formatting for A)
# ---------------------------------------------------------------------------

def _hard_decode_validate(
    B_model, A_model, tokenizer, best_ids, target_tool_name,
    tool_system_prompt, intermediate_len, format_a_content=None,
):
    """Generate from B, then from A, checking if A produces the target tool.

    Args:
        format_a_content: Optional callable (b_text: str) -> str.
            If provided, transforms B's raw output into A's user message
            (e.g. prepending the original user query per protocol).
            If None, A sees only B's raw output (legacy behavior).
    """
    device = next(B_model.parameters()).device
    p_final = best_ids.to(device)
    with torch.no_grad():
        b_out = B_model.generate(
            p_final.unsqueeze(0), max_new_tokens=intermediate_len, do_sample=False
        )
    b_new = b_out[0, p_final.shape[0]:]
    final_b_text = tokenizer.decode(b_new, skip_special_tokens=True)

    # BUG-1 fix: use protocol-aware formatting if available
    a_user_content = format_a_content(final_b_text) if format_a_content else final_b_text

    messages_A = [
        {"role": "system", "content": tool_system_prompt},
        {"role": "user", "content": a_user_content},
    ]
    a_prompt = tokenizer.apply_chat_template(
        messages_A, tokenize=False, add_generation_prompt=True
    )
    a_ids = tokenizer.encode(a_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        a_out = A_model.generate(a_ids, max_new_tokens=128, do_sample=False)
    a_text = tokenizer.decode(a_out[0, a_ids.shape[1]:], skip_special_tokens=True)
    hard_success = target_tool_name.lower() in a_text.lower()

    print(f"  Hard-decode B(p): '{final_b_text[:80]}'")
    print(f"  Hard-decode A output: '{a_text[:80]}'")
    print(f"  Hard-decode success: {hard_success}")

    return final_b_text, hard_success


# ---------------------------------------------------------------------------
# Online hill-climbing attack (μ+1)-ES
# ---------------------------------------------------------------------------

def online_hillclimb_attack(
    B_model,
    A_model,
    tokenizer,
    p_ids: LongTensor,
    optim_positions: LongTensor,
    target_tool_ids: LongTensor,
    target_tool_name: str,
    tool_system_prompt: str,
    *,
    intermediate_len: int = 32,
    num_steps: int = 100,
    pool_size: int = 4,
    allowed_token_ids: Optional[LongTensor] = None,
    seed: int = 0,
    grad_fraction: float = 0.9,     # 90% gradient-guided, 10% random
    use_gumbel_grad: bool = False,   # use B→A Gumbel gradient
    alternate_grads: bool = True,    # alternate B-only and B→A gradients
    grad_topk: int = 64,
    loss_mode: str = "ce",
    expected_tool_ids: Optional[LongTensor] = None,
    margin: float = 1.0,
    format_a_content=None,  # BUG-1 fix: callable (b_text) -> a_user_content
) -> STEResult:
    """Online (μ+1)-ES hill-climbing attack.

    Algorithm:
        1. Initialise pool with K copies of p (evaluated)
        2. For each step:
           a. Sample a parent from the pool (weighted by inverse rank)
           b. Mutate: swap 1 adversarial token (90% gradient-guided, 10% random)
           c. Evaluate child with hard B.generate() → A.forward()
           d. If child beats pool's worst → insert and evict worst
        3. Return the best candidate from the pool

    Key property: every evaluation is on a child of a known-good parent.
    No evaluation budget is wasted on candidates from bad parents.

    Args:
        pool_size: number of candidates to maintain (top-K buffer)
        grad_fraction: fraction of mutations that use gradient guidance (0.9 = 90%)
        use_gumbel_grad: use full B→A Gumbel gradient (vs B-only)
        alternate_grads: alternate B-only and B→A gradient each step
        grad_topk: top-k tokens for gradient-guided selection
    """
    from masflow.gcg import token_gradients, sample_candidates
    from masflow.ste_gcg import ste_token_gradients
    from masflow.estimators import get_estimator

    seed_everything(seed)
    device = next(B_model.parameters()).device

    if allowed_token_ids is None:
        allowed_token_ids = get_ascii_printable_tokens(tokenizer)

    n_adv = optim_positions.shape[0]

    # ── Initialise pool ──
    # Evaluate the initial prompt
    init_loss, init_text = _eval_single_candidate(
        B_model, A_model, tokenizer,
        p_ids, target_tool_ids, tool_system_prompt,
        intermediate_len=intermediate_len,
        format_a_content=format_a_content,
    )

    # Pool: list of (loss, p_ids, b_text), sorted by loss (best first)
    pool = [(init_loss, p_ids.clone(), init_text)] * pool_size

    best_loss = init_loss
    best_ids = p_ids.clone()
    best_text = init_text
    loss_history = []

    # Cache gradient for current best (recompute periodically)
    cached_grads = {}  # parent_hash → grads
    grad_recompute_interval = 5  # recompute gradient every N steps

    t0 = time.time()

    for step in range(num_steps):
        step_frac = step / max(num_steps - 1, 1)

        # ── 1. Sample parent from pool (rank-weighted) ──
        # Better candidates sampled more often
        weights = torch.tensor([1.0 / (i + 1) for i in range(pool_size)])
        weights = weights / weights.sum()
        parent_idx = torch.multinomial(weights, 1).item()
        parent_loss, parent_ids, parent_text = pool[parent_idx]

        # ── 2. Decide mutation type ──
        use_gradient = torch.rand(1).item() < grad_fraction

        if use_gradient:
            # Determine gradient source
            use_b_only = alternate_grads and (step % 2 == 0) and not use_gumbel_grad

            try:
                if use_b_only or not (use_gumbel_grad or alternate_grads):
                    # B-only gradient
                    prompt_len = parent_ids.shape[0]
                    t_len = target_tool_ids.shape[0]
                    target_slice = slice(prompt_len - 1, prompt_len - 1 + t_len)
                    grads = token_gradients(
                        B_model, parent_ids, optim_positions,
                        target_tool_ids, target_slice,
                    )
                else:
                    # Full B→A gradient via Annealed Gumbel STE
                    estimator = get_estimator("annealed_gumbel")
                    decode_fn = estimator.decode_fn
                    est_kwargs = {
                        "tau_start": estimator.tau_start,
                        "tau_end": estimator.tau_end,
                        "topk": estimator.topk,
                        "n_samples": estimator.n_samples,
                        "step_frac": step_frac,
                    }

                    B_model.zero_grad()
                    A_model.zero_grad()

                    grads, _, _ = ste_token_gradients(
                        B_model, A_model, tokenizer,
                        parent_ids, optim_positions, target_tool_ids,
                        tool_system_prompt,
                        intermediate_len=intermediate_len,
                        temperature=1.0,
                        decode_fn=decode_fn,
                        decode_kwargs=est_kwargs,
                        loss_mode=loss_mode,
                        expected_tool_ids=expected_tool_ids,
                        margin=margin,
                    )

                # Sample ONE candidate from gradient
                child_batch = sample_candidates(
                    parent_ids, optim_positions, grads,
                    search_width=1, topk=grad_topk,
                    allowed_token_ids=allowed_token_ids,
                )
                child_ids = child_batch[0]

            except Exception:
                # Fall back to random mutation
                use_gradient = False

        if not use_gradient:
            # Random mutation: swap one random adversarial token
            child_ids = parent_ids.clone()
            pos_idx = torch.randint(0, n_adv, (1,)).item()
            pos = optim_positions[pos_idx].item()
            tok_idx = torch.randint(0, allowed_token_ids.shape[0], (1,)).item()
            child_ids[pos] = allowed_token_ids[tok_idx]

        # ── 3. Evaluate child (hard decode) ──
        child_loss, child_text = _eval_single_candidate(
            B_model, A_model, tokenizer,
            child_ids, target_tool_ids, tool_system_prompt,
            intermediate_len=intermediate_len,
            format_a_content=format_a_content,
        )

        # ── 4. Update pool ──
        worst_loss = pool[-1][0]  # pool sorted best-first
        if child_loss < worst_loss:
            pool[-1] = (child_loss, child_ids.clone(), child_text)
            pool.sort(key=lambda x: x[0])  # re-sort

        # Track global best
        if pool[0][0] < best_loss:
            best_loss = pool[0][0]
            best_ids = pool[0][1].clone()
            best_text = pool[0][2]

        loss_history.append(best_loss)

        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - t0
            grad_type = "B→A" if (use_gradient and not use_b_only) else ("B" if use_gradient else "rnd")
            adv_text = tokenizer.decode(best_ids[optim_positions])
            print(
                f"  HC Step {step + 1:4d}/{num_steps} | "
                f"loss={best_loss:.4f} | "
                f"pool=[{', '.join(f'{l:.2f}' for l,_,_ in pool)}] | "
                f"grad={grad_type} | "
                f"B='{best_text[:40]}' | "
                f"{elapsed:.1f}s"
            )

    # ── Final hard-decode validation ──
    print("\n  Running hard-decode validation...")
    final_b_text, hard_success = _hard_decode_validate(
        B_model, A_model, tokenizer, best_ids, target_tool_name,
        tool_system_prompt, intermediate_len, format_a_content,
    )

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s ({elapsed / max(num_steps, 1):.2f}s/step)")

    from masflow.pipeline import ToolCall
    induced_o = ToolCall(
        tool_name=target_tool_name if hard_success else "unknown",
        arguments={},
    )

    return STEResult(
        best_p_ids=best_ids,
        best_loss=best_loss,
        loss_history=loss_history,
        induced_B_text=final_b_text,
        induced_o=induced_o,
        hard_decode_success=hard_success,
    )


# ---------------------------------------------------------------------------
# MACA: Multi-Agent Communication Attack — two-phase gradient decomposition
# ---------------------------------------------------------------------------

def maca_expand_beams(
    B_model,
    A_model,
    tokenizer,
    beams: LongTensor,              # [beam_width, seq_len]
    optim_positions: LongTensor,    # [n_adv] — suffix positions in B's input
    target_tool_ids: LongTensor,    # [t] — target tool tokens
    allowed_tokens: LongTensor,     # [vocab_subset]
    n_expansions: int = 8,          # expansions per beam
    topk: int = 64,
    *,
    desired_c_ids: Optional[LongTensor] = None,   # desired communication tokens from Phase 1
    a_prefix_ids: Optional[LongTensor] = None,    # A's input template prefix
    a_postfix_ids: Optional[LongTensor] = None,   # A's input template postfix
) -> LongTensor:
    """Expand beams using two-phase dual gradients.

    Phase 1 (computed externally, result passed as desired_c_ids):
        Gradient through A (tool-caller) w.r.t. communication:
        ∂CE(A(comm), target_tool) / ∂comm
        → Tells us what the communication SHOULD look like to fool A.

    Phase 2 (Pool 1 — B-gradient targeting desired_c):
        Gradient through B (first agent) w.r.t. suffix:
        ∂CE(B(p), desired_c) / ∂p[suffix]
        → Tells us what suffix changes make B produce the desired communication.

    Pool 2 (A-gradient on suffix — direct path):
        Gradient through A w.r.t. suffix positions in A's input:
        ∂CE(A(a_input), target_tool) / ∂a_input[suffix]
        → Direct signal: what suffix changes fool A if A sees the suffix.

    Pool 3 (random): Random token swaps for diversity.

    Allocation: 40% B-phase2, 40% A-direct, 20% random.
    """
    from masflow.gcg import token_gradients, sample_candidates

    beam_width, seq_len = beams.shape
    n_adv = optim_positions.shape[0]
    device = next(B_model.parameters()).device

    # Allocate expansions: 40% B-phase2, 40% A-direct, 20% random
    n_b_phase2 = max(1, int(n_expansions * 0.4))
    n_a_direct = max(1, int(n_expansions * 0.4))
    n_rand = n_expansions - n_b_phase2 - n_a_direct

    use_a_template = (a_prefix_ids is not None and a_postfix_ids is not None)

    all_expanded = []

    for b in range(beam_width):
        beam = beams[b]

        prompt_len = beam.shape[0]

        # ── Pool 1: B-gradient targeting desired_c (Phase 2 of MACA) ──
        try:
            if desired_c_ids is not None and desired_c_ids.shape[0] > 0:
                # Target: make B's output match desired communication tokens
                c_len = desired_c_ids.shape[0]
                target_slice_B = slice(prompt_len - 1, prompt_len - 1 + c_len)
                grads_B = token_gradients(
                    B_model, beam, optim_positions, desired_c_ids, target_slice_B
                )
            else:
                # Fallback: target tool name directly (like G-BEAST on B)
                t_len = target_tool_ids.shape[0]
                target_slice_B = slice(prompt_len - 1, prompt_len - 1 + t_len)
                grads_B = token_gradients(
                    B_model, beam, optim_positions, target_tool_ids, target_slice_B
                )
            cands_B = sample_candidates(
                beam, optim_positions, grads_B,
                search_width=n_b_phase2, topk=topk,
                allowed_token_ids=allowed_tokens,
            )
            all_expanded.append(cands_B)
        except Exception as e:
            print(f"    [MACA] B-phase2 gradient failed: {e}")
            n_rand += n_b_phase2

        # ── Pool 2: A-gradient on suffix in A's input (direct path) ──
        try:
            if use_a_template:
                suffix_ids = beam[optim_positions]
                a_ids, a_suffix_pos = build_a_input(
                    a_prefix_ids, a_postfix_ids, suffix_ids
                )
                a_ids = a_ids.to(device)
                a_suffix_pos = a_suffix_pos.to(device)

                a_len = a_ids.shape[0]
                t_len = target_tool_ids.shape[0]
                target_slice_A = slice(a_len - 1, a_len - 1 + t_len)

                grads_A = token_gradients(
                    A_model, a_ids, a_suffix_pos, target_tool_ids, target_slice_A
                )
                cands_A = sample_candidates(
                    beam, optim_positions, grads_A,
                    search_width=n_a_direct, topk=topk,
                    allowed_token_ids=allowed_tokens,
                )
                all_expanded.append(cands_A)
            else:
                # No A template — add to random pool
                n_rand += n_a_direct
        except Exception as e:
            print(f"    [MACA] A-direct gradient failed: {e}")
            n_rand += n_a_direct

        # ── Pool 3: Random expansions (diversity) ──
        for _ in range(n_rand):
            new_seq = beam.clone()
            pos_idx = torch.randint(0, n_adv, (1,)).item()
            pos = optim_positions[pos_idx].item()
            tok_idx = torch.randint(0, allowed_tokens.shape[0], (1,)).item()
            new_seq[pos] = allowed_tokens[tok_idx]
            all_expanded.append(new_seq.unsqueeze(0).to(device))

    return torch.cat(all_expanded, dim=0).cpu()


def maca_attack(
    B_model,
    A_model,
    tokenizer,
    p_ids: LongTensor,
    optim_positions: LongTensor,
    target_tool_ids: LongTensor,
    target_tool_name: str,
    tool_system_prompt: str,
    *,
    intermediate_len: int = 32,
    num_steps: int = 100,
    beam_width: int = 4,
    expansions_per_beam: int = 8,
    batch_size: int = 16,
    allowed_token_ids: Optional[LongTensor] = None,
    seed: int = 0,
    grad_topk: int = 64,
    a_prefix_ids: Optional[LongTensor] = None,    # A's input template prefix
    a_postfix_ids: Optional[LongTensor] = None,    # A's input template postfix
    format_a_content=None,  # BUG-1 fix: callable (b_text) -> a_user_content
) -> STEResult:
    """MACA (Multi-Agent Communication Attack): two-phase gradient beam search.

    At each step:
        Phase 1: Compute desired_c — gradient through A (tool-caller) w.r.t.
            the communication. Tells us what B should output to fool A.
            ∂CE(A(comm), target_tool) / ∂comm → desired_c_tokens

        Phase 2: Generate suffix candidates — gradient through B (first agent)
            w.r.t. suffix positions, targeting desired_c from Phase 1.
            ∂CE(B(p), desired_c) / ∂p[suffix] → suffix candidates

        Also: Direct A-gradient on suffix in A's input (for protocols where
            suffix appears in A's input).

        Phase 3: Hard beam search — evaluate all candidates through full
            pipeline B.generate → format → A.forward, keep top-K.

    This decomposes the end-to-end gradient ∂Loss/∂suffix at the non-
    differentiable B.generate() boundary into two tractable gradients.
    """
    seed_everything(seed)

    device = next(B_model.parameters()).device
    if allowed_token_ids is None:
        allowed_token_ids = get_ascii_printable_tokens(tokenizer)

    beams = p_ids.unsqueeze(0).expand(beam_width, -1).clone()
    beam_losses = [float("inf")] * beam_width
    beam_texts = [""] * beam_width

    best_loss = float("inf")
    best_ids = p_ids.clone()
    best_text = ""
    loss_history = []

    t0 = time.time()
    label = "MACA-T"

    for step in range(num_steps):
        # ══════════════════════════════════════════════════
        # PHASE 1: Compute desired communication tokens
        # ══════════════════════════════════════════════════
        # Generate B's output for best beam, format for A, compute
        # ∂CE(A(formatted_comm), target_tool) / ∂comm → desired_c
        desired_c_ids = None
        try:
            with torch.no_grad():
                b_out = B_model.generate(
                    beams[0:1].to(device),
                    max_new_tokens=intermediate_len, do_sample=False
                )
            b_gen_ids = b_out[0, beams.shape[1]:]  # B's generated tokens

            # Build A's input from B's actual output
            b_text = tokenizer.decode(b_gen_ids, skip_special_tokens=True)
            if b_text.strip():
                a_user_content = format_a_content(b_text) if format_a_content else b_text
                messages_A = [
                    {"role": "system", "content": tool_system_prompt},
                    {"role": "user", "content": a_user_content},
                ]
                a_prompt = tokenizer.apply_chat_template(
                    messages_A, tokenize=False, add_generation_prompt=True
                )
                a_ids = torch.tensor(
                    tokenizer.encode(a_prompt), dtype=torch.long, device=device
                )

                # Gradient: ∂CE(A(comm), target) / ∂comm
                # We differentiate w.r.t. ALL communication positions
                # (B's output portion within A's input)
                a_len = a_ids.shape[0]
                t_len = target_tool_ids.shape[0]
                target_slice_A = slice(a_len - 1, a_len - 1 + t_len)

                # Find communication positions in A's input (positions
                # corresponding to B's output text, not the system prompt)
                # We use a simple heuristic: the last `len(b_gen_ids)`
                # tokens before the generation prompt are the communication.
                comm_len = min(b_gen_ids.shape[0], 32)  # limit to prevent OOM
                # Communication is the user-content portion of A's input
                # Use only the last `comm_len` positions before generation prompt
                comm_start = max(0, a_len - comm_len - 3)
                comm_end = min(a_len - 1, comm_start + comm_len)
                comm_positions = torch.arange(comm_start, comm_end, device=device)
                
                # Ensure target_slice is within bounds
                target_start = min(a_len - 1, a_len - 1)
                target_end = min(a_len - 1 + t_len, a_len + 10)
                target_slice_A = slice(target_start, target_end)
                
                if comm_positions.shape[0] > 0 and comm_positions.shape[0] < a_len:
                    from masflow.gcg import token_gradients
                    grads_comm = token_gradients(
                        A_model, a_ids, comm_positions,
                        target_tool_ids, target_slice_A
                    )  # [n_comm_pos, vocab_size]

                    # Desired c = for each comm position, the token that most
                    # reduces A's loss (most negative gradient)
                    desired_c_ids = grads_comm.argmin(dim=-1).to(device)
                    # Trim to reasonable length for B targeting
                    desired_c_ids = desired_c_ids[:intermediate_len]
        except Exception as e:
            if step == 0:
                print(f"    [MACA] Phase 1 gradient failed: {e}")

        # ══════════════════════════════════════════════════
        # PHASE 2: Expand beams with dual gradients
        # ══════════════════════════════════════════════════
        expanded = maca_expand_beams(
            B_model, A_model, tokenizer,
            beams, optim_positions, target_tool_ids,
            allowed_token_ids, expansions_per_beam,
            topk=grad_topk,
            desired_c_ids=desired_c_ids,
            a_prefix_ids=a_prefix_ids,
            a_postfix_ids=a_postfix_ids,
        )

        # ══════════════════════════════════════════════════
        # PHASE 3: Hard evaluation (beam search)
        # ══════════════════════════════════════════════════
        # Include current beams (elitism)
        all_candidates = torch.cat([beams, expanded], dim=0)

        # Evaluate all through full pipeline (B.generate → A.forward)
        losses, b_texts = beast_evaluate(
            B_model, A_model, tokenizer,
            all_candidates, target_tool_ids, tool_system_prompt,
            intermediate_len=intermediate_len,
            batch_size=batch_size,
            format_a_content=format_a_content,
        )

        # Select top beam_width
        loss_tensor = torch.tensor(losses)
        loss_tensor[loss_tensor.isinf()] = 1e6
        topk_vals, topk_idx = loss_tensor.topk(beam_width, largest=False)

        beams = all_candidates[topk_idx].clone()
        beam_losses = [losses[i] for i in topk_idx]
        beam_texts = [b_texts[i] for i in topk_idx]

        # Track global best
        if beam_losses[0] < best_loss:
            best_loss = beam_losses[0]
            best_ids = beams[0].clone()
            best_text = beam_texts[0]

        loss_history.append(best_loss)

        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - t0
            adv_text = tokenizer.decode(best_ids[optim_positions])
            dc_str = "yes" if desired_c_ids is not None else "no"
            print(
                f"  {label} Step {step + 1:4d}/{num_steps} | "
                f"loss={best_loss:.4f} | "
                f"B(p)='{best_text[:50]}' | "
                f"dc={dc_str} | "
                f"adv='{adv_text[:30]}' | "
                f"{elapsed:.1f}s"
            )

    # Final hard-decode validation
    print(f"\n  Running hard-decode validation...")
    final_b_text, hard_success = _hard_decode_validate(
        B_model, A_model, tokenizer, best_ids, target_tool_name,
        tool_system_prompt, intermediate_len, format_a_content,
    )

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s ({elapsed / max(step + 1, 1):.2f}s/step)")

    from masflow.pipeline import ToolCall
    induced_o = ToolCall(
        tool_name=target_tool_name if hard_success else "unknown",
        arguments={},
    )

    return STEResult(
        best_p_ids=best_ids,
        best_loss=best_loss,
        loss_history=loss_history,
        induced_B_text=final_b_text,
        induced_o=induced_o,
        hard_decode_success=hard_success,
    )


def beast_attack(
    B_model,
    A_model,
    tokenizer,
    p_ids: LongTensor,
    optim_positions: LongTensor,
    target_tool_ids: LongTensor,
    target_tool_name: str,
    tool_system_prompt: str,
    *,
    intermediate_len: int = 32,
    num_steps: int = 100,
    beam_width: int = 4,
    expansions_per_beam: int = 8,
    batch_size: int = 16,
    allowed_token_ids: Optional[LongTensor] = None,
    seed: int = 0,
    use_gradient: bool = False,
    grad_topk: int = 64,
    grad_fraction: float = 0.5,
    a_prefix_ids: Optional[LongTensor] = None,    # A's input template prefix
    a_postfix_ids: Optional[LongTensor] = None,   # A's input template postfix
    gradient_mode: str = "a_direct",  # 'a_direct' (G-BEAST) or 'gumbel' (DG-BEAST)
    format_a_content=None,  # BUG-1 fix: callable (b_text) -> a_user_content
) -> STEResult:
    """Run BEAST attack: beam search through B→A.

    When use_gradient=False (default): pure random expansion (original BEAST).
    When use_gradient=True and gradient_mode='a_direct': G-BEAST (A's gradient).
    When use_gradient=True and gradient_mode='gumbel': DG-BEAST (full B→A gradient).

    Algorithm:
        1. Initialize beam with the starting prompt
        2. For each step:
           a. Expand each beam (random or gradient-guided)
           b. Evaluate all expansions with hard B.generate() → A.forward()
           c. Keep top beam_width beams (lowest loss)
        3. Return the best beam

    Args:
        beam_width: number of beams to maintain (top-k)
        expansions_per_beam: candidates to try per beam per step
        batch_size: how many candidates to evaluate in one B.generate() call
        use_gradient: if True, use gradient-guided beam expansion (Option F)
        grad_topk: top-k for gradient-based token selection
        grad_fraction: fraction of expansions using gradient (rest are random)
    """
    seed_everything(seed)

    device = next(B_model.parameters()).device
    if allowed_token_ids is None:
        allowed_token_ids = get_ascii_printable_tokens(tokenizer)

    # Initialize beam with the starting prompt
    beams = p_ids.unsqueeze(0).expand(beam_width, -1).clone()  # [beam_width, seq]
    beam_losses = [float("inf")] * beam_width
    beam_texts = [""] * beam_width

    best_loss = float("inf")
    best_ids = p_ids.clone()
    best_text = ""
    loss_history = []

    if use_gradient:
        method_label = "DG-BEAST" if gradient_mode == "gumbel" else "GBEAST"
    else:
        method_label = "BEAST"
    t0 = time.time()

    for step in range(num_steps):
        # 1. Expand beams
        if use_gradient and gradient_mode == "gumbel":
            # DG-BEAST: full B→A gradient via Gumbel-Softmax estimator
            expanded = gumbel_guided_expand_beams(
                B_model, A_model, tokenizer, beams, optim_positions,
                target_tool_ids, tool_system_prompt, allowed_token_ids,
                n_expansions=expansions_per_beam, topk=grad_topk,
                grad_fraction=grad_fraction,
                intermediate_len=intermediate_len,
                step_frac=step / max(1, num_steps - 1),
            )
        elif use_gradient:
            # G-BEAST: A's direct gradient at suffix positions
            expanded = gradient_guided_expand_beams(
                B_model, A_model, beams, optim_positions, target_tool_ids,
                allowed_token_ids, expansions_per_beam,
                topk=grad_topk, grad_fraction=grad_fraction,
                a_prefix_ids=a_prefix_ids, a_postfix_ids=a_postfix_ids,
            )
        else:
            expanded = beast_expand_beams(
                beams, optim_positions, allowed_token_ids, expansions_per_beam
            )
        # Also include current beams in candidate pool (elitism)
        all_candidates = torch.cat([beams, expanded], dim=0)

        # 2. Evaluate all candidates
        losses, b_texts = beast_evaluate(
            B_model, A_model, tokenizer,
            all_candidates, target_tool_ids, tool_system_prompt,
            intermediate_len=intermediate_len,
            batch_size=batch_size,
            format_a_content=format_a_content,
        )

        # 3. Select top beam_width
        loss_tensor = torch.tensor(losses)
        # Handle inf values for sorting
        loss_tensor[loss_tensor.isinf()] = 1e6
        topk_vals, topk_idx = loss_tensor.topk(beam_width, largest=False)

        beams = all_candidates[topk_idx].clone()
        beam_losses = [losses[i] for i in topk_idx]
        beam_texts = [b_texts[i] for i in topk_idx]

        # Track global best
        if beam_losses[0] < best_loss:
            best_loss = beam_losses[0]
            best_ids = beams[0].clone()
            best_text = beam_texts[0]

        loss_history.append(best_loss)

        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - t0
            adv_text = tokenizer.decode(best_ids[optim_positions])
            print(
                f"  {method_label} Step {step + 1:4d}/{num_steps} | "
                f"loss={best_loss:.4f} | "
                f"B(p)='{best_text[:50]}' | "
                f"adv='{adv_text[:40]}' | "
                f"{elapsed:.1f}s"
            )

    # Final hard-decode validation
    print("\n  Running hard-decode validation...")
    final_b_text, hard_success = _hard_decode_validate(
        B_model, A_model, tokenizer, best_ids, target_tool_name,
        tool_system_prompt, intermediate_len, format_a_content,
    )

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s ({elapsed / max(step + 1, 1):.2f}s/step)")

    from masflow.pipeline import ToolCall
    induced_o = ToolCall(
        tool_name=target_tool_name if hard_success else "unknown",
        arguments={},
    )

    return STEResult(
        best_p_ids=best_ids,
        best_loss=best_loss,
        loss_history=loss_history,
        induced_B_text=final_b_text,
        induced_o=induced_o,
        hard_decode_success=hard_success,
    )


# ---------------------------------------------------------------------------
# Batched A evaluation (pads variable-length A prompts)
# ---------------------------------------------------------------------------

def beast_evaluate_batched(
    B_model,
    A_model,
    tokenizer,
    candidates: LongTensor,         # [n_cands, seq_len]
    target_tool_ids: LongTensor,    # [t]
    tool_system_prompt: str,
    *,
    intermediate_len: int = 32,
    batch_size: int = 16,
    format_a_content=None,  # BUG-2 fix: callable (b_text) -> a_user_content
) -> tuple[list[float], list[str]]:
    """Evaluate candidates with hard B.generate() → batched A.forward().

    Unlike beast_evaluate, this batches BOTH B.generate() and A.forward()
    by padding A's variable-length inputs to a common length within each
    batch.

    Returns:
        losses: list of CE losses (one per candidate)
        b_texts: list of B's output texts
    """
    device = next(B_model.parameters()).device
    n_cands = candidates.shape[0]
    all_losses = []
    all_b_texts = []

    for start in range(0, n_cands, batch_size):
        end = min(start + batch_size, n_cands)
        batch = candidates[start:end].to(device)

        # --- Batch B.generate() ---
        with torch.no_grad():
            b_outputs = B_model.generate(
                batch,
                max_new_tokens=intermediate_len,
                do_sample=False,
            )

        prompt_len = candidates.shape[1]

        # --- Collect B texts and build A inputs ---
        b_texts_batch = []
        a_id_list = []
        valid_indices = []  # track which candidates have non-empty B output

        for i in range(batch.shape[0]):
            b_new = b_outputs[i, prompt_len:]
            b_text = tokenizer.decode(b_new, skip_special_tokens=True)
            b_texts_batch.append(b_text)

            if not b_text.strip():
                continue

            a_user_content = format_a_content(b_text) if format_a_content else b_text
            messages_A = [
                {"role": "system", "content": tool_system_prompt},
                {"role": "user", "content": a_user_content},
            ]
            a_prompt = tokenizer.apply_chat_template(
                messages_A, tokenize=False, add_generation_prompt=True
            )
            a_ids = tokenizer.encode(a_prompt, return_tensors="pt").squeeze(0)
            a_id_list.append(a_ids)
            valid_indices.append(i)

        # --- Evaluate A individually (no padding/attention mask) ---
        # Using attention_mask with padding changes logit values due to
        # different attention patterns, causing loss mismatch vs sequential eval.
        # Individual evaluation is the only way to get consistent losses.
        batch_losses = [float("inf")] * len(b_texts_batch)

        for j, orig_idx in enumerate(valid_indices):
            a_ids_j = a_id_list[j].unsqueeze(0).to(device)
            with torch.no_grad():
                a_logits_j = A_model(a_ids_j).logits

            a_len = a_id_list[j].shape[0]
            t_len = target_tool_ids.shape[0]
            target_logits = a_logits_j[0, a_len - 1: a_len - 1 + t_len, :]
            actual_t = min(target_logits.shape[0], t_len)
            loss = F.cross_entropy(
                target_logits[:actual_t],
                target_tool_ids[:actual_t].to(device),
            ).item()
            batch_losses[orig_idx] = loss

        all_losses.extend(batch_losses)
        all_b_texts.extend(b_texts_batch)

    return all_losses, all_b_texts


# ---------------------------------------------------------------------------
# Iterated G-BEAST: R rounds of (gradient + N beam steps)
# ---------------------------------------------------------------------------

def iterated_gbeast_attack(
    B_model,
    A_model,
    tokenizer,
    p_ids: LongTensor,
    optim_positions: LongTensor,
    target_tool_ids: LongTensor,
    target_tool_name: str,
    tool_system_prompt: str,
    *,
    intermediate_len: int = 32,
    num_rounds: int = 10,           # R — number of gradient recomputation rounds
    steps_per_round: int = 10,      # N — beam search steps per round
    beam_width: int = 4,            # K — beams to maintain
    expansions_per_beam: int = 8,   # E — candidates per beam per step
    batch_size: int = 16,
    allowed_token_ids: Optional[LongTensor] = None,
    seed: int = 0,
    grad_topk: int = 64,
    grad_fraction: float = 0.5,
    # Gumbel gradient options
    use_gumbel_grad: bool = True,
    gumbel_n_samples: int = 1,      # N gradient samples to average
    tau_start: float = 2.0,
    tau_end: float = 0.05,
    loss_mode: str = "ce",
    expected_tool_ids: Optional[LongTensor] = None,
    margin: float = 1.0,
    format_a_content=None,  # BUG-1 fix: callable (b_text) -> a_user_content
) -> STEResult:
    """Iterated G-BEAST: alternating gradient computation and beam search.

    Algorithm:
        For round r = 1..R:
            1. Compute gradient at current best beam(s)
               — Either B-only or full B→A via Annealed Gumbel STE
            2. Run N steps of beam search using that gradient
               — Expand beams (gradient-guided + random)
               — Evaluate ALL with hard B.generate() → batched A.forward()
               — Keep top-K beams
            3. The new top-K beams become the starting point for round r+1

    Total steps: R × N
    Total gradient computations: R (much cheaper than R×N)
    Total hard-decode evaluations: R × N × (K×E + K)

    Args:
        num_rounds: R — how many times to recompute the gradient
        steps_per_round: N — beam search steps per gradient
        beam_width: K — number of beams
        expansions_per_beam: E — expansions per beam per step
        use_gumbel_grad: if True, use full B→A Gumbel gradient
        tau_start/tau_end: Gumbel temperature annealing range
        loss_mode: 'ce', 'margin', or 'combined' for gradient loss
    """
    from masflow.gcg import token_gradients, sample_candidates
    from masflow.ste_gcg import ste_token_gradients
    from masflow.estimators import get_estimator

    seed_everything(seed)
    device = next(B_model.parameters()).device

    if allowed_token_ids is None:
        allowed_token_ids = get_ascii_printable_tokens(tokenizer)

    total_steps = num_rounds * steps_per_round

    # Initialize beams
    beams = p_ids.unsqueeze(0).expand(beam_width, -1).clone()
    beam_losses = [float("inf")] * beam_width
    beam_texts = [""] * beam_width

    best_loss = float("inf")
    best_ids = p_ids.clone()
    best_text = ""
    loss_history = []

    t0 = time.time()
    global_step = 0

    for rnd in range(num_rounds):
        # ── 1. Compute gradient at current best beam ──
        step_frac = rnd / max(num_rounds - 1, 1)

        if use_gumbel_grad:
            try:
                estimator = get_estimator("annealed_gumbel")
                decode_fn = estimator.decode_fn
                est_kwargs = {
                    "tau_start": tau_start,
                    "tau_end": tau_end,
                    "topk": estimator.topk,
                    "n_samples": estimator.n_samples,
                    "step_frac": step_frac,
                }

                # Average N Gumbel gradient samples to reduce variance
                grad_accum = None
                for _gs in range(gumbel_n_samples):
                    B_model.zero_grad()
                    A_model.zero_grad()

                    g, grad_loss, _ = ste_token_gradients(
                        B_model, A_model, tokenizer,
                        best_ids, optim_positions, target_tool_ids,
                        tool_system_prompt,
                        intermediate_len=intermediate_len,
                        temperature=1.0,
                        decode_fn=decode_fn,
                        decode_kwargs=est_kwargs,
                        loss_mode=loss_mode,
                        expected_tool_ids=expected_tool_ids,
                        margin=margin,
                    )
                    if grad_accum is None:
                        grad_accum = g
                    else:
                        grad_accum = grad_accum + g

                grads = grad_accum / gumbel_n_samples
                grad_type = "B→A(×{})".format(gumbel_n_samples)
            except Exception as e:
                # Fall back to B-only gradient
                prompt_len = best_ids.shape[0]
                t_len = target_tool_ids.shape[0]
                target_slice = slice(prompt_len - 1, prompt_len - 1 + t_len)
                grads = token_gradients(
                    B_model, best_ids, optim_positions,
                    target_tool_ids, target_slice,
                )
                grad_type = "B(fallback)"
        else:
            prompt_len = best_ids.shape[0]
            t_len = target_tool_ids.shape[0]
            target_slice = slice(prompt_len - 1, prompt_len - 1 + t_len)
            grads = token_gradients(
                B_model, best_ids, optim_positions,
                target_tool_ids, target_slice,
            )
            grad_type = "B"

        elapsed = time.time() - t0
        print(
            f"  Round {rnd + 1:3d}/{num_rounds} | "
            f"grad={grad_type} | "
            f"best_loss={best_loss:.4f} | "
            f"{elapsed:.1f}s"
        )

        # ── 2. Run N steps of beam search using this gradient ──
        for step in range(steps_per_round):
            global_step += 1
            n_adv = optim_positions.shape[0]

            # Expand beams using cached gradient
            n_grad = max(1, int(expansions_per_beam * grad_fraction))
            n_rand = expansions_per_beam - n_grad

            all_expanded = []
            for b in range(beam_width):
                beam = beams[b]

                # Gradient-guided expansions using cached gradient
                if n_grad > 0:
                    try:
                        grad_candidates = sample_candidates(
                            beam, optim_positions, grads,
                            search_width=n_grad, topk=grad_topk,
                            allowed_token_ids=allowed_token_ids,
                        )
                        all_expanded.append(grad_candidates)
                    except Exception:
                        n_rand += n_grad

                # Random expansions
                for _ in range(n_rand):
                    new_seq = beam.clone()
                    pos_idx = torch.randint(0, n_adv, (1,)).item()
                    pos = optim_positions[pos_idx].item()
                    tok_idx = torch.randint(0, allowed_token_ids.shape[0], (1,)).item()
                    new_seq[pos] = allowed_token_ids[tok_idx]
                    all_expanded.append(new_seq.unsqueeze(0).to(device))

            expanded = torch.cat(all_expanded, dim=0).cpu()
            all_candidates = torch.cat([beams, expanded], dim=0)

            # Evaluate with batched hard decode
            losses, b_texts_step = beast_evaluate_batched(
                B_model, A_model, tokenizer,
                all_candidates, target_tool_ids, tool_system_prompt,
                intermediate_len=intermediate_len,
                batch_size=batch_size,
                format_a_content=format_a_content,
            )

            # Select top-K
            loss_tensor = torch.tensor(losses)
            loss_tensor[loss_tensor.isinf()] = 1e6
            topk_vals, topk_idx = loss_tensor.topk(beam_width, largest=False)

            beams = all_candidates[topk_idx].clone()
            beam_losses = [losses[i] for i in topk_idx]
            beam_texts = [b_texts_step[i] for i in topk_idx]

            # Track global best
            if beam_losses[0] < best_loss:
                best_loss = beam_losses[0]
                best_ids = beams[0].clone()
                best_text = beam_texts[0]

            loss_history.append(best_loss)

            if global_step % 10 == 0:
                elapsed = time.time() - t0
                adv_text = tokenizer.decode(best_ids[optim_positions])
                print(
                    f"    Step {global_step:4d}/{total_steps} (R{rnd+1}.S{step+1}) | "
                    f"loss={best_loss:.4f} | "
                    f"B='{best_text[:40]}' | "
                    f"{elapsed:.1f}s"
                )

    # ── Final hard-decode validation ──
    print("\n  Running hard-decode validation...")
    final_b_text, hard_success = _hard_decode_validate(
        B_model, A_model, tokenizer, best_ids, target_tool_name,
        tool_system_prompt, intermediate_len, format_a_content,
    )

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.1f}s | {num_rounds} rounds × {steps_per_round} steps")

    from masflow.pipeline import ToolCall
    induced_o = ToolCall(
        tool_name=target_tool_name if hard_success else "unknown",
        arguments={},
    )

    return STEResult(
        best_p_ids=best_ids,
        best_loss=best_loss,
        loss_history=loss_history,
        induced_B_text=final_b_text,
        induced_o=induced_o,
        hard_decode_success=hard_success,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from masflow.config import load_local_model
    from masflow.gcg import build_sequence
    from masflow.tasks import TOOLS, TASKS, get_tool_call_format

    model, tokenizer = load_local_model()
    task = TASKS[0]

    print(f"Task: '{task.user_task}'")
    print(f"Target: {task.attack_target_tool}")

    template_parts = [
        f"Please help the user with this request: {task.user_task}\n",
        "\nProcess this request carefully."
    ]
    adv_init = ["x " * 10]
    p_ids, optim_pos_list = build_sequence(template_parts, adv_init, tokenizer)
    optim_positions = optim_pos_list[0]

    target_tool_ids = torch.tensor(
        tokenizer.encode(task.attack_target_tool, add_special_tokens=False),
        dtype=torch.long,
    )
    tool_system_prompt = (
        "You are a tool-calling agent. Based on the instruction you receive, "
        "select the most appropriate tool and provide the required arguments.\n\n"
        + get_tool_call_format(TOOLS)
    )

    result = beast_attack(
        model, model, tokenizer,
        p_ids, optim_positions, target_tool_ids,
        target_tool_name=task.attack_target_tool,
        tool_system_prompt=tool_system_prompt,
        intermediate_len=32,
        num_steps=100,
        beam_width=4,
        expansions_per_beam=8,
        batch_size=16,
        seed=42,
    )

    print(f"\nBEAST Result:")
    print(f"  Final loss: {result.best_loss:.4f}")
    print(f"  Success: {result.hard_decode_success}")
