"""
Long-term human power objective from Heitzig & Potham (2025),
"Model-Based Soft Maximization of Suitable Metrics of Long-Term Human Power",
arXiv:2508.00159v2.
"""

from .exact import (
    HumanModelParams,
    HumanPrior,
    PowerParams,
    Solution,
    TabularGame,
    solve,
    solve_human_prior,
    solve_robot,
)
from .metrics import (
    PAPER_HYPERPARAMS,
    attainable_goals,
    power_bits,
    robot_reward,
    soft_power_policy,
)

__all__ = [
    "PAPER_HYPERPARAMS",
    "HumanModelParams",
    "HumanPrior",
    "PowerParams",
    "Solution",
    "TabularGame",
    "attainable_goals",
    "power_bits",
    "robot_reward",
    "soft_power_policy",
    "solve",
    "solve_human_prior",
    "solve_robot",
]
