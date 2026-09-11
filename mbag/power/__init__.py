"""
Long-term human power objective from Heitzig & Potham (2025),
"Model-Based Soft Maximization of Suitable Metrics of Long-Term Human Power",
arXiv:2508.00159v2.
"""

from .metrics import (
    PAPER_HYPERPARAMS,
    attainable_goals,
    power_bits,
    robot_reward,
    soft_power_policy,
)

__all__ = [
    "PAPER_HYPERPARAMS",
    "attainable_goals",
    "power_bits",
    "robot_reward",
    "soft_power_policy",
]
