"""Unit tests for the Non-Parametric PPO clipped surrogate verification gate."""

import numpy as np
import pytest

from skill_pro_rw.ppo_gate import compute_clipped_surrogate


def test_ppo_gate_accepts_beneficial_candidate() -> None:
    """Increasing probability on positive-advantage steps yields L_CLIP > margin."""
    old_lp = np.array([-1.2, -1.0, -1.4, -0.8], dtype=np.float64)
    new_lp = np.array([-0.8, -0.7, -1.1, -0.6], dtype=np.float64)
    adv = np.array([0.25, 0.25, 0.20, 0.20], dtype=np.float64)

    res = compute_clipped_surrogate(old_lp, new_lp, adv, epsilon=0.2, margin=0.001)

    assert res.passed is True
    assert res.l_clip > 0.001
    assert res.step_count == 4
    assert res.mean_ratio > 1.0


def test_ppo_gate_clips_extreme_ratios() -> None:
    """Importance ratios above 1 + epsilon are clipped in the pessimistic objective."""
    old_lp = np.array([-3.0, -3.0], dtype=np.float64)
    new_lp = np.array([-0.1, -0.1], dtype=np.float64)
    adv = np.array([1.0, 1.0], dtype=np.float64)

    res = compute_clipped_surrogate(old_lp, new_lp, adv, epsilon=0.2, margin=0.001)

    assert res.clipped_fraction == pytest.approx(1.0)
    assert res.l_clip == pytest.approx(1.2, rel=1e-5)


def test_ppo_gate_rejects_harmful_candidate() -> None:
    """Decreasing probability on winning steps and increasing on losing steps fails."""
    old_lp = np.array([-0.6, -1.2], dtype=np.float64)
    new_lp = np.array([-1.8, -0.4], dtype=np.float64)
    adv = np.array([0.5, -0.5], dtype=np.float64)

    res = compute_clipped_surrogate(old_lp, new_lp, adv, epsilon=0.2, margin=0.001)

    assert res.passed is False
    assert res.l_clip < 0.0


def test_ppo_gate_relative_improvement_on_negative_baseline() -> None:
    """Relative improvement passes when candidate reduces probability of negative-advantage steps."""
    old_lp = np.array([-0.5, -0.5], dtype=np.float64)
    new_lp = np.array([-1.2, -1.2], dtype=np.float64)
    adv = np.array([-0.20, -0.20], dtype=np.float64)

    res = compute_clipped_surrogate(
        old_lp, new_lp, adv, epsilon=0.2, margin=0.001, relative_to_old=True
    )

    assert res.passed is True
    assert res.improvement > 0.001
    assert res.baseline_l_clip == pytest.approx(-0.20)


def test_ppo_gate_fails_closed_on_invalid_inputs() -> None:
    """Empty or non-finite logprobs fail closed with -1e9 and passed=False."""
    empty_res = compute_clipped_surrogate(
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
    )
    assert empty_res.passed is False
    assert empty_res.l_clip == -1e9

    nan_res = compute_clipped_surrogate(
        np.array([-1.0, np.nan], dtype=np.float64),
        np.array([-0.9, -0.8], dtype=np.float64),
        np.array([0.2, 0.2], dtype=np.float64),
    )
    assert nan_res.passed is False
    assert nan_res.l_clip == -1e9
