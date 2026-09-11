"""
Pure-numpy implementations of the ICCEA power metric and the robot's intrinsic
reward. Equation numbers refer to Heitzig & Potham (2025), arXiv:2508.00159v2.
"""

from typing import Dict, Optional

import numpy as np

PAPER_HYPERPARAMS: Dict[str, float] = {
    "zeta": 2.0,  # eq. (7): reliability preference / risk aversion
    "xi": 1.0,  # eq. (8): inter-human inequality aversion
    "eta": 1.1,  # eq. (8): intertemporal inequality aversion
    "gamma_h": 0.99,
    "gamma_r": 0.99,
}


def attainable_goals(
    v_e: np.ndarray, zeta: float, *, axis: int = -1, reduce: str = "sum"
) -> np.ndarray:
    """
    Eq. (7): X_h(s) = sum_g V^e_h(s, g)^zeta.

    ``v_e`` holds goal-attainment probabilities V^e_h(s, g) in [0, 1] with goals along
    ``axis``. With ``reduce="mean"`` the sum is replaced by the mean over goals, which
    rescales X_h by 1/|G_h|; the paper's design makes the robot policy invariant to
    such common rescalings, and the mean is what a sampled-goal estimator learns.
    """
    v_e = np.asarray(v_e, dtype=float)
    powered = v_e**zeta
    if reduce == "sum":
        return np.asarray(powered.sum(axis=axis))
    if reduce == "mean":
        return np.asarray(powered.mean(axis=axis))
    raise ValueError(f"unknown reduce {reduce!r}; expected 'sum' or 'mean'")


def power_bits(x: np.ndarray) -> np.ndarray:
    """Eq. (10): W_h(s) = log2 X_h(s), the ICCEA power in bits."""
    return np.log2(np.asarray(x, dtype=float))


def robot_reward(
    x: np.ndarray, xi: float, eta: float, *, human_axis: Optional[int] = None
) -> np.ndarray:
    """
    Eq. (8): U_r(s) = -( sum_h X_h(s)^(-xi) )^eta.

    If ``human_axis`` is None, ``x`` is a single human's X_h (any shape) and the sum
    over humans has one term. Otherwise humans are laid out along ``human_axis``.
    """
    x = np.asarray(x, dtype=float)
    inner = x ** (-xi)
    if human_axis is not None:
        inner = inner.sum(axis=human_axis)
    return -(inner**eta)


def soft_power_policy(
    q_r: np.ndarray, beta_r: float, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Eq. (5): pi_r(s)(a) proportional to (-Q_r(s, a))^(-beta_r), over the last axis.

    Q_r must be strictly negative (it is a discounted sum of negative U_r values).
    ``beta_r = inf`` gives the argmax policy with ties split uniformly; ``beta_r = 0``
    gives the uniform policy. ``mask`` (same shape as ``q_r``, True = valid) restricts
    the support.
    """
    q_r = np.asarray(q_r, dtype=float)
    if mask is None:
        mask = np.ones_like(q_r, dtype=bool)
    if np.any(q_r[mask] >= 0):
        raise ValueError("soft_power_policy requires strictly negative Q_r values")

    if np.isinf(beta_r):
        best = np.where(mask, q_r, -np.inf).max(axis=-1, keepdims=True)
        weights = ((q_r == best) & mask).astype(float)
    else:
        # Work in log space for numerical stability: log w = -beta * log(-q).
        log_w = -beta_r * np.log(-q_r)
        log_w = np.where(mask, log_w, -np.inf)
        log_w = log_w - log_w.max(axis=-1, keepdims=True)
        weights = np.exp(log_w)
    return np.asarray(weights / weights.sum(axis=-1, keepdims=True))
