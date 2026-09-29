"""
gcg.py — Infix-capable GCG (Greedy Coordinate Gradient) attack.

Operates over a *single* white-box HF model.  The key generalization is that
adversarial positions (``optim_positions``) can sit at **arbitrary infix
locations** in the sequence, not just as a suffix.

Public helpers ``token_gradients``, ``sample_candidates`` are importable by
``ste_gcg.py`` so it can reuse candidate-proposal logic with its own loss.

No model loading, no pipeline logic, no API calls.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor

from masflow.config import GCGResult, seed_everything

# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def get_ascii_printable_tokens(tokenizer) -> LongTensor:
    """Return token ids whose decoded form is printable ASCII (no leading-space ambiguity).

    Result is cached on the tokenizer object for reuse.
    """
    cache_attr = "_ascii_printable_ids"
    if hasattr(tokenizer, cache_attr):
        return getattr(tokenizer, cache_attr)

    ids = []
    for i in range(tokenizer.vocab_size):
        try:
            decoded = tokenizer.decode([i])
        except Exception:
            continue
        # Keep tokens that are non-empty, printable ASCII, no leading space ambiguity
        if (
            decoded
            and all(32 <= ord(c) <= 126 for c in decoded)
            and len(decoded) == len(decoded.strip()) or decoded == " "
        ):
            ids.append(i)

    result = torch.tensor(ids, dtype=torch.long)
    setattr(tokenizer, cache_attr, result)
    return result


def build_sequence(
    template_parts: list[str],
    adv_init_strings: list[str],
    tokenizer,
) -> tuple[LongTensor, list[LongTensor]]:
    """Tokenize a prompt with adversarial regions, keeping position masks exact.

    The full sequence is ``template_parts[0] + adv[0] + template_parts[1] + adv[1] + ...``
    with ``len(template_parts) == len(adv_init_strings) + 1``.

    Returns:
        input_ids: [seq_len] concatenated token ids.
        optim_positions_list: list of LongTensors, one per adversarial region,
            containing the absolute positions in ``input_ids``.
    """
    assert len(template_parts) == len(adv_init_strings) + 1, (
        "template_parts must have exactly one more element than adv_init_strings"
    )

    all_ids: list[int] = []
    optim_positions_list: list[LongTensor] = []

    for i, part in enumerate(template_parts):
        # Tokenize the template segment
        if part:
            part_ids = tokenizer.encode(part, add_special_tokens=False)
            all_ids.extend(part_ids)

        # Tokenize the adversarial segment (if not the last template part)
        if i < len(adv_init_strings):
            adv_str = adv_init_strings[i]
            adv_ids = tokenizer.encode(adv_str, add_special_tokens=False)

            # Record positions
            start = len(all_ids)
            all_ids.extend(adv_ids)
            end = len(all_ids)
            positions = torch.arange(start, end, dtype=torch.long)
            optim_positions_list.append(positions)

            # Stability check: decode → re-encode should be stable
            round_trip = tokenizer.encode(
                tokenizer.decode(adv_ids), add_special_tokens=False
            )
            if round_trip != adv_ids:
                warnings.warn(
                    f"Adversarial init '{adv_str}' is not stable under "
                    f"decode→re-encode (got {len(round_trip)} vs {len(adv_ids)} tokens). "
                    f"Position mask may drift after the first GCG step — but GCG "
                    f"operates in id-space, so this only affects the init."
                )

    input_ids = torch.tensor(all_ids, dtype=torch.long)
    return input_ids, optim_positions_list


# ---------------------------------------------------------------------------
# Core GCG primitives (importable by ste_gcg.py)
# ---------------------------------------------------------------------------


def token_gradients(
    model,
    input_ids: LongTensor,       # [seq]
    optim_positions: LongTensor, # [n_adv]
    target_ids: LongTensor,      # [t]
    target_slice: slice,
) -> Tensor:
    """Compute per-token gradients at optimizable positions via the one-hot trick.

    Returns grad of shape [n_adv, vocab_size].
    """
    device = next(model.parameters()).device
    embed_matrix = model.get_input_embeddings().weight  # [V, d]
    vocab_size, embed_dim = embed_matrix.shape

    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    optim_positions = optim_positions.to(device)

    # One-hot embedding for the full sequence
    one_hot = F.one_hot(input_ids, num_classes=vocab_size).float()  # [seq, V]
    one_hot = one_hot.to(embed_matrix.dtype).to(device)

    # Detach everything, then enable grad only at optimizable positions
    one_hot = one_hot.detach().requires_grad_(False)
    one_hot_adv = one_hot[optim_positions].detach().clone().requires_grad_(True)

    # Build full one-hot with grad-enabled adversarial slice
    one_hot_full = one_hot.clone()
    one_hot_full[optim_positions] = one_hot_adv

    # Embed
    input_embeds = one_hot_full @ embed_matrix  # [seq, d]
    input_embeds = input_embeds.unsqueeze(0)    # [1, seq, d]

    # Forward
    outputs = model(inputs_embeds=input_embeds)
    logits = outputs.logits  # [1, seq, V]

    # Loss: cross-entropy at target positions
    # target_slice indexes into the *output* logits; logits[0, i] predicts token at i+1
    target_logits = logits[0, target_slice, :]  # [t, V]
    loss = F.cross_entropy(target_logits, target_ids[:target_logits.shape[0]])

    loss.backward()

    return one_hot_adv.grad.clone()  # [n_adv, V]


def get_attention_weights(
    model,
    input_ids: LongTensor,       # [seq]
    optim_positions: LongTensor, # [n_adv]
) -> Tensor:
    """Extract average attention weight at optimizable positions.

    Returns [n_adv] tensor of mean attention intensity at each adversarial
    position, averaged across all heads and layers.  Higher values mean
    the model "relies more" on that position — substitutions there are
    more impactful.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    optim_positions = optim_positions.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            output_attentions=True,
        )
    # outputs.attentions: tuple of [1, n_heads, seq, seq], one per layer
    # Average across layers and heads, then sum attention *to* each adv position
    attn_sum = None
    n_layers = len(outputs.attentions)
    for layer_attn in outputs.attentions:
        # layer_attn: [1, n_heads, seq, seq]
        # Mean across heads: [seq, seq]  (attention from row i to column j)
        mean_attn = layer_attn[0].mean(dim=0)  # [seq, seq]
        # Sum of attention flowing *to* each adv position from all other positions
        pos_attn = mean_attn[:, optim_positions].sum(dim=0)  # [n_adv]
        if attn_sum is None:
            attn_sum = pos_attn
        else:
            attn_sum = attn_sum + pos_attn
    attn_sum = attn_sum / n_layers  # average across layers
    return attn_sum  # [n_adv]


def sample_candidates(
    input_ids: LongTensor,        # [seq]
    optim_positions: LongTensor,  # [n_adv]
    gradients: Tensor,            # [n_adv, V]
    search_width: int,
    topk: int,
    allowed_token_ids: Optional[LongTensor] = None,
    attention_weights: Optional[Tensor] = None,  # [n_adv] for AttnGCG
) -> LongTensor:
    """Sample ``search_width`` candidate sequences, each differing from
    ``input_ids`` in exactly one optimizable position.

    If ``attention_weights`` is provided (AttnGCG), positions are selected
    proportionally to their attention weight — high-attention positions get
    more substitution attempts.

    Returns [search_width, seq] tensor of candidate token ids.
    """
    n_adv = optim_positions.shape[0]
    vocab_size = gradients.shape[1]
    device = gradients.device

    # Move everything to the same device
    input_ids = input_ids.to(device)
    optim_positions = optim_positions.to(device)

    # Restrict to allowed tokens if specified
    if allowed_token_ids is not None:
        mask = torch.full((vocab_size,), float("inf"), device=device)
        mask[allowed_token_ids.to(device)] = 0.0
        gradients = gradients + mask.unsqueeze(0)  # inf out disallowed tokens

    # Top-k tokens per position (most negative gradient = most helpful)
    topk_vals = torch.topk(-gradients, topk, dim=1)  # largest -grad = smallest grad
    topk_indices = topk_vals.indices  # [n_adv, topk]

    # For each candidate: pick a position (uniform or attention-weighted)
    candidates = input_ids.unsqueeze(0).repeat(search_width, 1)  # [B, seq]

    if attention_weights is not None:
        # AttnGCG: sample positions proportional to attention weight
        attn_probs = F.softmax(attention_weights.to(device).float(), dim=0)
        pos_idx = torch.multinomial(
            attn_probs.expand(search_width, -1), num_samples=1
        ).squeeze(1)  # [B]
    else:
        pos_idx = torch.randint(0, n_adv, (search_width,), device=device)

    tok_idx = torch.randint(0, topk, (search_width,), device=device)

    # Gather the replacement token ids
    new_tokens = topk_indices[pos_idx, tok_idx]  # [B]

    # Apply substitutions
    seq_positions = optim_positions[pos_idx]  # absolute positions in the sequence
    candidates[torch.arange(search_width, device=device), seq_positions] = new_tokens

    return candidates


def eval_candidates(
    model,
    candidates: LongTensor,   # [B, seq]
    target_ids: LongTensor,   # [t]
    target_slice: slice,
    batch_size: int = 64,
) -> tuple[LongTensor, float]:
    """Evaluate candidates by target loss. Returns (best_ids, best_loss)."""
    device = next(model.parameters()).device
    target_ids = target_ids.to(device)
    num_candidates = candidates.shape[0]

    all_losses = []
    for i in range(0, num_candidates, batch_size):
        batch = candidates[i : i + batch_size].to(device)
        with torch.no_grad():
            outputs = model(input_ids=batch)
            logits = outputs.logits  # [b, seq, V]

            target_logits = logits[:, target_slice, :]  # [b, t, V]
            # Per-candidate loss
            t_len = target_logits.shape[1]
            target_expanded = target_ids[:t_len].unsqueeze(0).expand(
                target_logits.shape[0], -1
            )
            losses = F.cross_entropy(
                target_logits.reshape(-1, target_logits.shape[-1]),
                target_expanded.reshape(-1),
                reduction="none",
            ).reshape(target_logits.shape[0], -1).mean(dim=1)

            all_losses.append(losses)

    all_losses = torch.cat(all_losses)
    best_idx = all_losses.argmin().item()
    best_loss = all_losses[best_idx].item()
    best_ids = candidates[best_idx]

    return best_ids, best_loss


# ---------------------------------------------------------------------------
# Main GCG attack loop
# ---------------------------------------------------------------------------


def gcg_attack(
    model,
    tokenizer,
    input_ids: LongTensor,
    optim_positions: LongTensor,
    target_ids: LongTensor,
    target_slice: slice,
    *,
    num_steps: int = 250,
    search_width: int = 256,
    topk: int = 256,
    batch_size: int = 64,
    allowed_token_ids: Optional[LongTensor] = None,
    seed: int = 0,
) -> GCGResult:
    """Run the GCG attack with arbitrary infix optimizable positions.

    Args:
        model: HF CausalLM (white-box, eval mode).
        tokenizer: corresponding tokenizer.
        input_ids: [seq] full sequence including adversarial region(s).
        optim_positions: [n_adv] indices into input_ids that may be modified.
        target_ids: [t] target completion token ids.
        target_slice: slice into the *logits* sequence where target is scored.
            Typically ``slice(prompt_len - 1, prompt_len + len(target_ids) - 1)``
            because logit[i] predicts token[i+1].
        num_steps: GCG optimization steps.
        search_width: number of candidate substitutions per step.
        topk: top-k tokens per gradient coordinate.
        batch_size: candidate evaluation batch size.
        allowed_token_ids: vocabulary restriction (default: ASCII-printable).
        seed: random seed.

    Returns:
        GCGResult with best_input_ids, best_loss, loss_history, target_achieved.
    """
    seed_everything(seed)
    device = next(model.parameters()).device

    if allowed_token_ids is None:
        allowed_token_ids = get_ascii_printable_tokens(tokenizer)

    # Ensure no silent truncation: all optim positions must be valid
    assert optim_positions.max() < input_ids.shape[0], (
        f"Optimizable position {optim_positions.max()} >= sequence length {input_ids.shape[0]}"
    )
    assert optim_positions.min() >= 0, "Negative optimizable position"

    best_ids = input_ids.clone()
    best_loss = float("inf")
    loss_history: list[float] = []

    for step in range(num_steps):
        # 1. Compute gradients
        grads = token_gradients(
            model, best_ids, optim_positions, target_ids, target_slice
        )

        # 2. Sample candidates
        candidates = sample_candidates(
            best_ids, optim_positions, grads, search_width, topk, allowed_token_ids
        )

        # 3. Evaluate candidates
        step_best_ids, step_best_loss = eval_candidates(
            model, candidates, target_ids, target_slice, batch_size
        )

        # 4. Update best
        if step_best_loss < best_loss:
            best_loss = step_best_loss
            best_ids = step_best_ids.clone()

        loss_history.append(best_loss)

        # 5. Check for early stop: greedy decode produces target exactly
        target_achieved = _check_target_achieved(
            model, best_ids, target_ids, target_slice, device
        )

        if (step + 1) % 25 == 0 or step == 0 or target_achieved:
            adv_text = tokenizer.decode(best_ids[optim_positions])
            print(
                f"  Step {step + 1:4d}/{num_steps} | "
                f"loss={best_loss:.4f} | "
                f"achieved={target_achieved} | "
                f"adv='{adv_text[:60]}'"
            )

        if target_achieved:
            print(f"  ✓ Target achieved at step {step + 1}")
            break

    return GCGResult(
        best_input_ids=best_ids,
        best_loss=best_loss,
        loss_history=loss_history,
        target_achieved=target_achieved,
    )


def _check_target_achieved(
    model, input_ids: LongTensor, target_ids: LongTensor, target_slice: slice, device
) -> bool:
    """Check if greedy decoding at target_slice exactly produces target_ids."""
    with torch.no_grad():
        outputs = model(input_ids=input_ids.unsqueeze(0).to(device))
        logits = outputs.logits[0, target_slice, :]
        preds = logits.argmax(dim=-1)
        t_len = min(preds.shape[0], target_ids.shape[0])
        return preds[:t_len].cpu().equal(target_ids[:t_len])


# ---------------------------------------------------------------------------
# Self-test: infix GCG attack
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from masflow.config import load_local_model, DEFAULT_MODEL

    print(f"Loading model: {DEFAULT_MODEL}")
    model, tokenizer = load_local_model()
    device = next(model.parameters()).device

    # Build a sequence with an adversarial INFIX region
    # Template: "The capital of France is [ADV x 10] The answer is"
    # Target: " Paris" (the next tokens after "The answer is")
    template_parts = ["The capital of France is ", " The answer is"]
    adv_init = ["! ! ! ! ! ! ! ! ! !"]  # 10 junk tokens as infix

    input_ids, optim_pos_list = build_sequence(template_parts, adv_init, tokenizer)
    optim_positions = optim_pos_list[0]

    print(f"Sequence length: {input_ids.shape[0]}")
    print(f"Adversarial positions: {optim_positions.tolist()}")
    print(f"Initial prompt: '{tokenizer.decode(input_ids)}'")

    # Target: 5 tokens for " Paris"
    target_str = " Paris"
    target_ids = torch.tensor(
        tokenizer.encode(target_str, add_special_tokens=False), dtype=torch.long
    )
    print(f"Target tokens: {target_ids.tolist()} = '{target_str}'")

    # target_slice: logits at position (seq_len - 1) predict next token
    prompt_len = input_ids.shape[0]
    target_slice = slice(prompt_len - 1, prompt_len - 1 + target_ids.shape[0])

    result = gcg_attack(
        model,
        tokenizer,
        input_ids,
        optim_positions,
        target_ids,
        target_slice,
        num_steps=100,
        search_width=128,
        topk=128,
        batch_size=32,
        seed=42,
    )

    print(f"\nFinal loss: {result.best_loss:.4f}")
    print(f"Target achieved: {result.target_achieved}")
    final_prompt = tokenizer.decode(result.best_input_ids)
    print(f"Final prompt: '{final_prompt}'")

    # Greedy decode to verify
    with torch.no_grad():
        gen = model.generate(
            result.best_input_ids.unsqueeze(0).to(device),
            max_new_tokens=target_ids.shape[0],
            do_sample=False,
        )
        gen_text = tokenizer.decode(gen[0, prompt_len:])
        print(f"Greedy continuation: '{gen_text}'")

    if result.target_achieved:
        print("\n✓ gcg.py infix self-test PASSED")
    else:
        print("\n✗ gcg.py infix self-test FAILED (target not achieved)")
        print("  (This may happen with very small models — try more steps or a larger model)")
