"""Non-Parametric PPO clipped surrogate verification gate (L_CLIP)."""

from __future__ import annotations

import numpy as np

from .models import PPOGateEvaluation


def compute_clipped_surrogate(
    old_log_probs: np.ndarray,
    new_log_probs: np.ndarray,
    step_advantages: np.ndarray,
    epsilon: float = 0.2,
    margin: float = 0.001,
    verifier_mode: str = "offline_proxy",
    relative_to_old: bool = False,
) -> PPOGateEvaluation:
    """Evaluate candidate skill off-policy via the clipped PPO surrogate objective.

    Computes importance ratios rho_t = exp(clip(new_logp - old_logp, -10, 10)),
    L_CLIP = mean(min(rho_t * A_t, clip(rho_t, 1 - eps, 1 + eps) * A_t)), and
    improvement = L_CLIP - mean(A_t) relative to the old skill (where rho_t = 1).
    Fails closed (-1e9, passed=False) on empty or non-finite inputs.
    """
    if (
        old_log_probs.size == 0
        or new_log_probs.size == 0
        or step_advantages.size == 0
        or not (old_log_probs.shape == new_log_probs.shape == step_advantages.shape)
    ):
        return PPOGateEvaluation(
            l_clip=-1e9,
            baseline_l_clip=0.0,
            improvement=-1e9,
            margin=margin,
            passed=False,
            clipped_fraction=0.0,
            mean_ratio=1.0,
            min_ratio=1.0,
            max_ratio=1.0,
            old_logprob_mean=0.0,
            new_logprob_mean=0.0,
            step_count=0,
            verifier_mode=verifier_mode,
            reason="Empty or mismatched logprob arrays; failing closed.",
        )

    if not (
        np.all(np.isfinite(old_log_probs))
        and np.all(np.isfinite(new_log_probs))
        and np.all(np.isfinite(step_advantages))
    ):
        return PPOGateEvaluation(
            l_clip=-1e9,
            baseline_l_clip=0.0,
            improvement=-1e9,
            margin=margin,
            passed=False,
            clipped_fraction=0.0,
            mean_ratio=1.0,
            min_ratio=1.0,
            max_ratio=1.0,
            old_logprob_mean=0.0,
            new_logprob_mean=0.0,
            step_count=int(old_log_probs.size),
            verifier_mode=verifier_mode,
            reason="Non-finite logprob or advantage detected; failing closed.",
        )

    log_ratio = np.clip(new_log_probs - old_log_probs, -10.0, 10.0)
    ratios = np.exp(log_ratio)

    surr1 = ratios * step_advantages
    clipped_ratios = np.clip(ratios, 1.0 - epsilon, 1.0 + epsilon)
    surr2 = clipped_ratios * step_advantages
    objectives = np.minimum(surr1, surr2)

    l_clip = float(np.mean(objectives))
    baseline_l_clip = float(np.mean(step_advantages))
    improvement = float(l_clip - baseline_l_clip)
    clipped_mask = (ratios < (1.0 - epsilon)) | (ratios > (1.0 + epsilon))
    clipped_fraction = float(np.mean(clipped_mask))

    score_to_check = improvement if relative_to_old else l_clip
    passed = bool(score_to_check > margin)
    metric_name = "Delta_L_CLIP" if relative_to_old else "L_CLIP"

    reason = (
        f"Accepted: {metric_name} ({score_to_check:.5f}) > margin ({margin:.5f})"
        if passed
        else f"Rejected: {metric_name} ({score_to_check:.5f}) <= margin ({margin:.5f})"
    )

    return PPOGateEvaluation(
        l_clip=l_clip,
        baseline_l_clip=baseline_l_clip,
        improvement=improvement,
        margin=margin,
        passed=passed,
        clipped_fraction=clipped_fraction,
        mean_ratio=float(np.mean(ratios)),
        min_ratio=float(np.min(ratios)),
        max_ratio=float(np.max(ratios)),
        old_logprob_mean=float(np.mean(old_log_probs)),
        new_logprob_mean=float(np.mean(new_log_probs)),
        step_count=int(old_log_probs.size),
        verifier_mode=verifier_mode,
        reason=reason,
    )
