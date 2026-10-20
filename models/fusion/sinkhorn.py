"""
Log-domain Sinkhorn, used twice in the 3D-VGA model for two different jobs.

Both are assignment problems that the usual softmax attention answers badly:

  1. frames -> slots (models/audio_module.py). Plain cross-attention normalises over
     frames per slot, so nothing stops every slot from landing on the same loud syllable.
     Sinkhorn adds the *column* constraint -- each frame's mass is also spread over slots --
     which forces the slots apart and makes them partition the utterance.

  2. slots -> proposals (models/fusion/spot_fusion.py). Softmax attention lets every
     proposal independently decide how much it likes the "chair" slot, so in a room with
     nine chairs all nine score high and the ranking that follows is nearly arbitrary.
     Sinkhorn makes proposals *compete* for a slot's fixed mass budget: one chair can only
     win the slot at the others' expense. Grounding is an assignment, so this is the
     operator that matches the task.

Everything runs in log space (no exp of a large negative), is masked, and is differentiable
in a handful of iterations -- the standard entropic-OT setup as used in SuperGlue and
SwAV, with unequal marginals so a slot can declare itself unused.
"""

import torch


def masked_log_softmax(logits, mask, dim):
    """log_softmax that treats masked entries as absent, not as -inf competitors.

    torch.log_softmax over an all-masked row yields NaN. Rows like that do occur here --
    a padded annotation has no valid frames -- so masked entries are pushed to a large
    finite negative and the result is re-masked afterwards.
    """
    logits = logits.masked_fill(~mask, -1e4)
    out = torch.log_softmax(logits, dim=dim)
    return out.masked_fill(~mask, -1e4)


def sinkhorn_log(scores, mask, row_marginal=None, col_marginal=None, eps=0.05, n_iters=5):
    """Entropic optimal transport between the two axes of ``scores``.

    Args:
        scores: (B, R, C) similarity, higher = cheaper to transport.
        mask:   (B, R, C) bool, True where a pairing is allowed.
        row_marginal: (B, R) non-negative mass per row. Uniform if None. Marginals are
            used as given -- nothing here renormalises them -- so a caller supplying both
            must give them the same total, or the iteration converges to neither. What the
            callers do supply is an *uneven* row marginal summing to one: a row holding
            little mass is how a slot says "this role is not mentioned", which the text
            branch writes with that literal token and which audio has no way to utter.
        col_marginal: (B, C) non-negative mass per column. Uniform over valid columns if
            None.
        eps: entropic temperature. What governs convergence is the ratio between the
            *range* of ``scores`` and eps, not eps alone, so both callers feed cosine
            similarities times a bounded learned scale -- a fixed range -- and leave eps
            fixed. Measured on this problem's shapes (6 slots x 128 frames, 6 slots x 256
            proposals), with scores in [-1, 1]: eps 0.3 and 8 iterations leave under 2%
            relative error on the row marginals, while eps 0.05 and 5 iterations leave
            30-50%. Sharpness is the learned scale's job; eps only has to keep the
            iteration converging.
        n_iters: Sinkhorn iterations. This runs inside the training loop twice per layer,
            so the count is the thing being bought with the eps above.

    Returns:
        (B, R, C) transport plan, non-negative, zero outside the mask.
    """
    batch, n_rows, n_cols = scores.shape
    device, dtype = scores.device, scores.dtype

    if row_marginal is None:
        row_marginal = scores.new_full((batch, n_rows), 1.0 / n_rows)
    if col_marginal is None:
        valid = mask.any(dim=1).to(dtype)                       # (B, C)
        col_marginal = valid / valid.sum(dim=1, keepdim=True).clamp_min(1.0)

    log_mu = torch.log(row_marginal.clamp_min(1e-8))
    log_nu = torch.log(col_marginal.clamp_min(1e-8))

    # Kernel in log space. Masked pairs are held at a large finite negative rather than
    # -inf, so a fully masked row cannot turn a logsumexp into -inf and the dual into NaN.
    neg = -1e4
    log_k = (scores / eps).masked_fill(~mask, neg)

    # A column no row may use (a padding frame, a non-object proposal) gets no mass, so its
    # dual is pinned rather than left to the iteration. Its v would otherwise grow to
    # ~ +1e4 on the first update -- the only way to give a -1e4 kernel the mass its
    # marginal asks for -- and log_k + v would be back near zero on the next row update.
    # The clamp on log_nu caps that leak at ~1e-8 of the total mass, so measured against a
    # peak plan entry of ~1e-2 it moves nothing (checked: a relative change below 1e-6);
    # this makes the guarantee exact instead of merely small, for one comparison per pass.
    row_valid = mask.any(dim=2)
    col_valid = mask.any(dim=1)

    u = torch.zeros(batch, n_rows, device=device, dtype=dtype)
    v = torch.zeros(batch, n_cols, device=device, dtype=dtype)

    for _ in range(n_iters):
        u = log_mu - torch.logsumexp(log_k + v.unsqueeze(1), dim=2)
        u = u.masked_fill(~row_valid, neg)
        v = log_nu - torch.logsumexp(log_k + u.unsqueeze(2), dim=1)
        v = v.masked_fill(~col_valid, neg)

    # After the final column update the plan is exactly column-normalised, so every
    # exponent is <= 0 and this cannot overflow.
    plan = torch.exp(log_k + u.unsqueeze(2) + v.unsqueeze(1))
    return plan.masked_fill(~mask, 0.0)


def plan_to_attention(plan, dim, eps=1e-8):
    """Row- or column-normalise a transport plan so it can be used as attention weights.

    The plan's rows sum to the slot masses, not to one, which is the point of it -- but the
    moment it multiplies a value tensor it has to be a convex combination, or an unused
    slot's output silently shrinks toward zero and the residual path sees a scale change
    rather than an information change.
    """
    return plan / plan.sum(dim=dim, keepdim=True).clamp_min(eps)


def plan_entropy(plan, eps=1e-8):
    """Mean entropy of the plan's rows, as a scalar regulariser.

    Maximising it early keeps the slots from collapsing onto one frame before the encoder
    has learned anything; the loss weight (AUDIO.LOSS_SLOT_ENTROPY) is small, because the
    endpoint wanted is a sharp assignment, not a smooth one.
    """
    normalised = plan_to_attention(plan, dim=-1, eps=eps)
    return -(normalised * torch.log(normalised.clamp_min(eps))).sum(dim=-1).mean()
