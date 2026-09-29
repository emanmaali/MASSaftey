"""
ppl_regularized_attacks.py — PPL-regularized variants of GCG, BEAST, and
G-BEAST, for testing whether their raw B-A hard-decode advantage over ACA
v2 survives once they're forced to stay as fluent as ACA (rather than
being architecturally weaker searches that just happen to also be loud).

Mechanism: L_total = L_attack + lambda_ppl * L_ppl, where L_ppl is the
mean NLL of the suffix's own tokens under the model's own predictions --
computed from the SAME forward pass already computing L_attack (the
"zero overhead" trick; see eval_batch_with_ppl below), exactly the
pattern run_019_ppl_constrained.py validated for G-BEAST alone and
run_029_full_benchmark.py's eval_batch_aca already uses internally for
ACA. This module generalizes it to GCG and plain BEAST too, and makes it
model-agnostic (run_019's version was hardcoded to Qwen2.5-0.5B-Instruct).

Candidate SELECTION uses L_total; the reported "a_only_attack_loss" is
still the plain L_attack of whichever candidate that selection picked, so
results stay comparable to the non-PPL baselines already collected.

Deliberate simplification vs. the imported gcg_attack(): no early-stop on
exact target achievement. BEAST/G-BEAST in this project already run the
full step budget with no early-stop, so this keeps all three PPL variants
structurally consistent with each other; the asymmetry this introduces
against the (already-collected) non-PPL GCG baseline, which CAN stop
early via masflow.gcg.gcg_attack, is a compute-time asymmetry only
(continuing past an already-achieved target doesn't change the reported
best result), not a correctness one.

Suffix perplexity for REPORTING (as opposed to the internal ppl_loss used
for candidate selection) is computed the same way as every other attack
in this project: downstream, uniformly, via run_pipeline_eval's
suffix_ppl_standalone -- not duplicated here.
"""
import time

import torch
import torch.nn.functional as F

from masflow.gcg import token_gradients, sample_candidates
from masflow.beast import beast_expand_beams


def eval_batch_with_ppl(model, candidates, target_ids, target_slice, suffix_pos,
                         lambda_ppl, batch_size=32):
    """Score a batch of candidate sequences by L_total = L_attack + lambda_ppl * L_ppl.

    L_ppl is the mean NLL of the suffix's own tokens, read off the SAME
    logits already computed for L_attack -- no extra forward pass.

    Returns (all_total, all_atk, all_ppl) as plain Python lists, one entry
    per candidate, in input order.
    """
    device = next(model.parameters()).device
    t_ids_d = target_ids.to(device)
    suf_pos_d = suffix_pos.to(device)
    pred_pos = suf_pos_d - 1

    all_total, all_atk, all_ppl = [], [], []
    for s in range(0, candidates.shape[0], batch_size):
        batch = candidates[s: s + batch_size].to(device)
        b = batch.shape[0]

        with torch.no_grad():
            logits = model(input_ids=batch).logits

        t_logits = logits[:, target_slice, :]
        t_exp = t_ids_d[:t_logits.shape[1]].unsqueeze(0).expand(b, -1)
        atk = F.cross_entropy(
            t_logits.reshape(-1, t_logits.shape[-1]), t_exp.reshape(-1), reduction="none"
        ).reshape(b, -1).mean(1)

        ppl = F.cross_entropy(
            logits[:, pred_pos, :].reshape(-1, logits.shape[-1]),
            batch[:, suf_pos_d].reshape(-1), reduction="none"
        ).reshape(b, -1).mean(1)

        total = atk + lambda_ppl * ppl
        all_total.extend(total.cpu().tolist())
        all_atk.extend(atk.cpu().tolist())
        all_ppl.extend(ppl.cpu().tolist())

    return all_total, all_atk, all_ppl


def _finish(model, tok, best_ids, task_label, elapsed, total_history, atk_history,
            ppl_history, best_atk, best_ppl_loss, lambda_ppl, tag):
    device = next(model.parameters()).device
    with torch.no_grad():
        out = model.generate(best_ids.unsqueeze(0).to(device), max_new_tokens=64, do_sample=False)
    hd_text = tok.decode(out[0, best_ids.shape[0]:], skip_special_tokens=True)
    hd = task_label.lower() in hd_text.lower()
    print(f"  [{tag}] DONE: atk={best_atk:.4f} ppl_loss={best_ppl_loss:.4f} "
          f"HD={hd} | '{hd_text[:60]}' ({elapsed:.0f}s)")
    return {
        "a_only_hd": hd, "a_only_hd_text": hd_text[:200],
        "a_only_attack_loss": best_atk, "a_only_para_loss": None,
        "a_only_suffix_ppl": None,
        "a_only_ppl_loss": best_ppl_loss, "lambda_ppl_reg": lambda_ppl,
        "best_ids": best_ids,
        "attack_history": [round(x, 4) for x in atk_history],
        "total_loss_history": [round(x, 4) for x in total_history],
        "ppl_loss_history": [round(x, 4) for x in ppl_history],
        "time_phase1": elapsed,
    }


def gcg_attack_ppl(model, tok, a_ids, suffix_pos, target_ids, target_slice,
                    allowed_tokens, task_label, steps, *,
                    search_width=128, topk=128, batch_size=16, lambda_ppl=1.0, seed=42):
    """GCG: single-trajectory greedy search, gradients from the current
    best_ids each step (mirrors masflow.gcg.gcg_attack's structure),
    candidate selection by L_total instead of L_attack alone."""
    torch.manual_seed(seed)
    t0 = time.time()
    device = next(model.parameters()).device

    best_ids = a_ids.clone()
    best_total, best_atk, best_ppl_loss = float("inf"), float("inf"), float("inf")
    total_history, atk_history, ppl_history = [], [], []

    for step in range(steps):
        grads = token_gradients(
            model, best_ids.to(device), suffix_pos.to(device), target_ids.to(device), target_slice
        )
        candidates = sample_candidates(
            best_ids, suffix_pos, grads.cpu(),
            search_width=search_width, topk=topk, allowed_token_ids=allowed_tokens,
        )

        totals, atks, ppls = eval_batch_with_ppl(
            model, candidates, target_ids, target_slice, suffix_pos, lambda_ppl, batch_size
        )
        totals_t = torch.tensor(totals)
        step_best_idx = totals_t.argmin().item()

        if totals[step_best_idx] < best_total:
            best_total = totals[step_best_idx]
            best_atk = atks[step_best_idx]
            best_ppl_loss = ppls[step_best_idx]
            best_ids = candidates[step_best_idx].clone()

        total_history.append(best_total)
        atk_history.append(best_atk)
        ppl_history.append(best_ppl_loss)

        if (step + 1) % 50 == 0 or step == 0:
            adv = tok.decode(best_ids[suffix_pos])
            print(f"  [GCG-PPL(λ={lambda_ppl})] Step {step+1:4d}/{steps} | "
                  f"atk={best_atk:.4f} ppl_loss={best_ppl_loss:.4f} | "
                  f"adv='{adv[:40]}' | {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    return _finish(model, tok, best_ids, task_label, elapsed, total_history,
                    atk_history, ppl_history, best_atk, best_ppl_loss, lambda_ppl,
                    tag=f"GCG-PPL(λ={lambda_ppl})")


def beast_attack_ppl(model, tok, a_ids, suffix_pos, target_ids, target_slice,
                      allowed_tokens, task_label, steps, *,
                      beam_width=4, beast_batch=32, lambda_ppl=1.0, seed=42):
    """BEAST: random-candidate beam search (mirrors run_beast's structure),
    candidate selection by L_total instead of L_attack alone."""
    torch.manual_seed(seed)
    t0 = time.time()

    beams = a_ids.unsqueeze(0).expand(beam_width, -1).clone()
    best_ids = a_ids.clone()
    best_total, best_atk, best_ppl_loss = float("inf"), float("inf"), float("inf")
    total_history, atk_history, ppl_history = [], [], []

    for step in range(steps):
        expanded = beast_expand_beams(beams, suffix_pos, allowed_tokens, beam_width * 8)
        expanded = torch.cat([expanded, beams], dim=0)

        totals, atks, ppls = eval_batch_with_ppl(
            model, expanded, target_ids, target_slice, suffix_pos, lambda_ppl, beast_batch
        )
        totals_t = torch.tensor(totals)
        topk_vals, topk_idx = totals_t.topk(beam_width, largest=False)
        beams = expanded[topk_idx].clone()

        step_best = topk_idx[0].item()
        if totals[step_best] < best_total:
            best_total = totals[step_best]
            best_atk = atks[step_best]
            best_ppl_loss = ppls[step_best]
            best_ids = beams[0].clone()

        total_history.append(best_total)
        atk_history.append(best_atk)
        ppl_history.append(best_ppl_loss)

        if (step + 1) % 50 == 0 or step == 0:
            adv = tok.decode(best_ids[suffix_pos])
            print(f"  [BEAST-PPL(λ={lambda_ppl})] Step {step+1:4d}/{steps} | "
                  f"atk={best_atk:.4f} ppl_loss={best_ppl_loss:.4f} | "
                  f"adv='{adv[:40]}' | {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    return _finish(model, tok, best_ids, task_label, elapsed, total_history,
                    atk_history, ppl_history, best_atk, best_ppl_loss, lambda_ppl,
                    tag=f"BEAST-PPL(λ={lambda_ppl})")


def gbeast_attack_ppl(model, tok, a_ids, suffix_pos, target_ids, target_slice,
                       allowed_tokens, task_label, steps, *,
                       beam_width=4, beast_batch=32, grad_topk=128,
                       lambda_ppl=1.0, seed=42):
    """G-BEAST: gradient candidates + random-expansion beam search (mirrors
    run_gbeast's structure), candidate selection by L_total instead of
    L_attack alone."""
    torch.manual_seed(seed)
    t0 = time.time()
    device = next(model.parameters()).device

    beams = a_ids.unsqueeze(0).expand(beam_width, -1).clone()
    best_ids = a_ids.clone()
    best_total, best_atk, best_ppl_loss = float("inf"), float("inf"), float("inf")
    total_history, atk_history, ppl_history = [], [], []

    for step in range(steps):
        try:
            grads = token_gradients(
                model, beams[0].to(device), suffix_pos.to(device),
                target_ids.to(device), target_slice
            )
            cands = sample_candidates(
                beams[0], suffix_pos, grads.cpu(),
                search_width=beam_width * 8, topk=grad_topk,
                allowed_token_ids=allowed_tokens,
            )
        except Exception:
            cands = beast_expand_beams(beams, suffix_pos, allowed_tokens, beam_width * 8)

        candidates = torch.cat([cands, beams], dim=0)

        totals, atks, ppls = eval_batch_with_ppl(
            model, candidates, target_ids, target_slice, suffix_pos, lambda_ppl, beast_batch
        )
        totals_t = torch.tensor(totals)
        topk_vals, topk_idx = totals_t.topk(beam_width, largest=False)
        beams = candidates[topk_idx].clone()

        step_best = topk_idx[0].item()
        if totals[step_best] < best_total:
            best_total = totals[step_best]
            best_atk = atks[step_best]
            best_ppl_loss = ppls[step_best]
            best_ids = beams[0].clone()

        total_history.append(best_total)
        atk_history.append(best_atk)
        ppl_history.append(best_ppl_loss)

        if (step + 1) % 50 == 0 or step == 0:
            adv = tok.decode(best_ids[suffix_pos])
            print(f"  [G-BEAST-PPL(λ={lambda_ppl})] Step {step+1:4d}/{steps} | "
                  f"atk={best_atk:.4f} ppl_loss={best_ppl_loss:.4f} | "
                  f"adv='{adv[:40]}' | {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    return _finish(model, tok, best_ids, task_label, elapsed, total_history,
                    atk_history, ppl_history, best_atk, best_ppl_loss, lambda_ppl,
                    tag=f"G-BEAST-PPL(λ={lambda_ppl})")
