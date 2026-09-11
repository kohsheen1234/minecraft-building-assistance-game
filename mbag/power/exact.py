"""
Exact (fixed-point) solution of eqs. (1) to (9) of Heitzig & Potham (2025) for small
tabular stochastic games with one robot and n humans.

Conventions
-----------
* States are integers 0..S-1. ``terminal[s]`` marks absorbing states with no further
  reward.
* Joint human actions are flattened into a single index ``j`` in C order (first human
  varies slowest). ``TabularGame.joint_index`` / ``split_joint`` convert.
* Transitions are stored as ``K`` weighted outcomes per (s, a_r, j):
  ``next_states[s, a_r, j, k]`` with probability ``next_probs[s, a_r, j, k]``.
* Goals are events: ``goal_sets[h][g, s]`` is True when state s fulfils goal g of
  human h. Entering a goal state pays U_h = 1 and, for that goal's value, the
  continuation is zero (goal-absorbing). This realises the paper's requirement that
  the states in a goal be mutually unreachable.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .metrics import (
    PAPER_HYPERPARAMS,
    attainable_goals,
    power_bits,
    robot_reward,
    soft_power_policy,
)


@dataclass
class TabularGame:
    num_states: int
    num_robot_actions: int
    human_action_counts: List[int]
    next_states: np.ndarray  # (S, A_r, J, K) int
    next_probs: np.ndarray  # (S, A_r, J, K) float
    robot_action_mask: np.ndarray  # (S, A_r) bool
    human_action_masks: List[np.ndarray]  # per human (S, A_h) bool
    goal_sets: List[np.ndarray]  # per human (G_h, S) bool
    terminal: np.ndarray  # (S,) bool

    def __post_init__(self):
        S, A_r = self.num_states, self.num_robot_actions
        J = int(np.prod(self.human_action_counts))
        assert self.next_states.shape[:3] == (S, A_r, J), self.next_states.shape
        assert self.next_probs.shape == self.next_states.shape
        assert self.robot_action_mask.shape == (S, A_r)
        assert len(self.human_action_masks) == self.num_humans
        assert len(self.goal_sets) == self.num_humans
        for h, A_h in enumerate(self.human_action_counts):
            assert self.human_action_masks[h].shape == (S, A_h)
            assert self.goal_sets[h].shape[1] == S
        assert self.terminal.shape == (S,)
        row_sums = self.next_probs.sum(-1)
        assert np.allclose(row_sums, 1.0), "next_probs must sum to 1 over outcomes"

    @property
    def num_humans(self) -> int:
        return len(self.human_action_counts)

    @property
    def num_joint_human_actions(self) -> int:
        return int(np.prod(self.human_action_counts))

    def joint_index(self, actions: Sequence[int]) -> int:
        return int(
            np.ravel_multi_index(tuple(actions), tuple(self.human_action_counts))
        )

    def split_joint(self, j: int) -> Tuple[int, ...]:
        return tuple(
            int(i) for i in np.unravel_index(j, tuple(self.human_action_counts))
        )

    def expected_next(self, values: np.ndarray) -> np.ndarray:
        """
        E_{s' ~ P(s, a_r, j)} values[s'] for an array ``values`` of shape (S,) or
        (..., S). Returns shape (S, A_r, J) or (..., S, A_r, J).
        """
        gathered = values[..., self.next_states]  # (..., S, A_r, J, K)
        return (gathered * self.next_probs).sum(-1)


@dataclass
class HumanModelParams:
    nu: float
    beta_h: float
    gamma_h: float
    default_policy: Optional[List[np.ndarray]] = None  # per human (G_h, S, A_h)
    beliefs_about_others: Optional[List[np.ndarray]] = None  # per human (S, A_h)


@dataclass
class HumanPrior:
    q_m: List[np.ndarray] = field(default_factory=list)  # (G_h, S, A_h)
    pi_h: List[np.ndarray] = field(default_factory=list)  # (G_h, S, A_h)
    v_m: List[np.ndarray] = field(default_factory=list)  # (G_h, S)


def _masked_softmax(scores: np.ndarray, beta: float, mask: np.ndarray) -> np.ndarray:
    """beta-softmax over the last axis restricted to mask; beta=inf is argmax."""
    if np.isinf(beta):
        best = np.where(mask, scores, -np.inf).max(-1, keepdims=True)
        weights = ((scores == best) & mask).astype(float)
    else:
        logits = np.where(mask, beta * scores, -np.inf)
        logits = logits - logits.max(-1, keepdims=True)
        weights = np.exp(logits)
    return weights / weights.sum(-1, keepdims=True)


def _joint_policy_of_others(
    game: TabularGame, h: int, per_human_policies: List[np.ndarray]
) -> np.ndarray:
    """
    Weight of each joint action j contributed by humans other than h, placed at
    human h's own action within j. Returns (S, J, A_h) with entry
    [s, j, a] = prod_{o != h} pi_o(s, a_o(j)) if a == a_h(j) else 0.
    Used to marginalise over a_{-h} in eq. (1).
    """
    S = game.num_states
    J = game.num_joint_human_actions
    A_h = game.human_action_counts[h]
    out = np.zeros((S, J, A_h))
    for j in range(J):
        actions = game.split_joint(j)
        weight = np.ones(S)
        for other, a_other in enumerate(actions):
            if other == h:
                continue
            weight = weight * per_human_policies[other][:, a_other]
        out[:, j, actions[h]] = weight
    return out


def solve_human_prior(
    game: TabularGame,
    params: HumanModelParams,
    *,
    max_iters: int = 10_000,
    tol: float = 1e-10,
) -> HumanPrior:
    """Fixed-point iteration of eqs. (1) to (3) for every human and goal."""
    S = game.num_states
    prior = HumanPrior()
    for h in range(game.num_humans):
        A_h = game.human_action_counts[h]
        goals = game.goal_sets[h]  # (G_h, S)
        G_h = goals.shape[0]
        mask_h = game.human_action_masks[h]  # (S, A_h)

        if params.beliefs_about_others is None:
            others = [np.ones((S, A_o)) / A_o for A_o in game.human_action_counts]
        else:
            others = list(params.beliefs_about_others)
        others_weight = _joint_policy_of_others(game, h, others)  # (S, J, A_h)

        if params.default_policy is None:
            pi0 = np.broadcast_to(mask_h / mask_h.sum(-1, keepdims=True), (G_h, S, A_h))
        else:
            pi0 = params.default_policy[h]

        v_m = np.zeros((G_h, S))
        q_m = np.zeros((G_h, S, A_h))
        pi_h = np.array(pi0, dtype=float)
        # Continuation is zero in goal states and terminal states (goal-absorbing).
        continue_mask = (~goals) & (~game.terminal)[None, :]  # (G_h, S)
        u_h = goals.astype(float)  # U_h(s', g) = 1[s' in g]

        for _ in range(max_iters):
            target_values = u_h + params.gamma_h * v_m * continue_mask  # (G_h, S)
            expected = game.expected_next(target_values)  # (G_h, S, A_r, J)
            # eq. (1): min over robot actions (masked), then E over a_{-h}.
            expected = np.where(
                game.robot_action_mask[None, :, :, None], expected, np.inf
            )
            worst = expected.min(axis=2)  # (G_h, S, J)
            q_new = np.einsum("gsj,sja->gsa", worst, others_weight)
            # eq. (2)
            pi_new = params.nu * pi0 + (1 - params.nu) * _masked_softmax(
                q_new, params.beta_h, mask_h[None]
            )
            # eq. (3)
            v_new = (pi_new * q_new).sum(-1) * continue_mask
            delta = max(np.abs(q_new - q_m).max(), np.abs(v_new - v_m).max())
            q_m, pi_h, v_m = q_new, pi_new, v_new
            if delta < tol:
                break

        prior.q_m.append(q_m)
        prior.pi_h.append(pi_h)
        prior.v_m.append(v_m)
    return prior


@dataclass
class PowerParams:
    zeta: float
    xi: float
    eta: float
    beta_r: float
    gamma_r: float
    gamma_h: float

    @classmethod
    def paper(cls, beta_r: float = 5.0) -> "PowerParams":
        return cls(
            zeta=PAPER_HYPERPARAMS["zeta"],
            xi=PAPER_HYPERPARAMS["xi"],
            eta=PAPER_HYPERPARAMS["eta"],
            beta_r=beta_r,
            gamma_r=PAPER_HYPERPARAMS["gamma_r"],
            gamma_h=PAPER_HYPERPARAMS["gamma_h"],
        )


@dataclass
class Solution:
    q_r: np.ndarray  # (S, A_r)
    pi_r: np.ndarray  # (S, A_r)
    v_r: np.ndarray  # (S,)
    u_r: np.ndarray  # (S,)
    v_e: List[np.ndarray]  # per human (G_h, S)
    x_h: List[np.ndarray]  # per human (S,)
    w_h: List[np.ndarray]  # per human (S,)
    prior: HumanPrior
    iterations: int = 0
    converged: bool = False


def _joint_human_policy(game: TabularGame, per_human: List[np.ndarray]) -> np.ndarray:
    """Product of per-human (S, A_h) policies over joint actions. Returns (S, J)."""
    S, J = game.num_states, game.num_joint_human_actions
    joint = np.ones((S, J))
    for j in range(J):
        for h, a_h in enumerate(game.split_joint(j)):
            joint[:, j] *= per_human[h][:, a_h]
    return joint


def solve_robot(
    game: TabularGame,
    prior: HumanPrior,
    params: PowerParams,
    *,
    max_iters: int = 10_000,
    tol: float = 1e-10,
    damping: float = 0.5,
) -> Solution:
    """
    Fixed-point iteration of eqs. (4) to (9). ``damping`` mixes the new robot policy
    with the previous one to stabilise the coupled iteration (the paper notes the map
    is not a contraction in cyclic games).

    Goals of different humans are independent uniform draws (paper Section 2.2,
    "Aggregation across uncertainty"), so E_g factorises over humans.
    """
    S = game.num_states
    live = ~game.terminal

    pi_r = game.robot_action_mask / game.robot_action_mask.sum(-1, keepdims=True)
    v_r = np.zeros(S)
    v_e = [np.zeros(gs.shape) for gs in game.goal_sets]

    # Goal-averaged policy per human (S, A_h), used for E_g in eq. (4) and for the
    # other humans' goals in eq. (6).
    avg_policies = [pi.mean(axis=0) for pi in prior.pi_h]
    joint_avg = _joint_human_policy(game, avg_policies)  # (S, J)
    # Per human h and goal g: joint policy where h follows pi_h(., g) and others are
    # goal-averaged. Shape (G_h, S, J).
    joint_for_goal: List[np.ndarray] = []
    for h in range(game.num_humans):
        stacks = []
        for g in range(game.goal_sets[h].shape[0]):
            per_human = list(avg_policies)
            per_human[h] = prior.pi_h[h][g]
            stacks.append(_joint_human_policy(game, per_human))
        joint_for_goal.append(np.stack(stacks))

    q_r = np.zeros_like(pi_r)
    u_r = np.zeros(S)
    x_h: List[np.ndarray] = []
    converged = False
    iterations = 0
    for iterations in range(1, max_iters + 1):
        # eq. (6): V^e_h(s, g) under actual pi_r and derived pi_H.
        v_e_new = []
        for h in range(game.num_humans):
            goals = game.goal_sets[h]
            continue_mask = (~goals) & live[None, :]
            target = goals.astype(float) + params.gamma_h * v_e[h] * continue_mask
            expected = game.expected_next(target)  # (G_h, S, A_r, J)
            over_r = np.einsum("gsaj,sa->gsj", expected, pi_r)
            v_e_h = np.einsum("gsj,gsj->gs", over_r, joint_for_goal[h])
            v_e_new.append(v_e_h * continue_mask)
        # eq. (7), (8). Terminal states have X_h = 0; their U_r is masked to 0.
        x_h = [attainable_goals(v, params.zeta, axis=0) for v in v_e_new]
        with np.errstate(divide="ignore"):
            u_r = robot_reward(np.stack(x_h), params.xi, params.eta, human_axis=0)
        u_r = np.where(live, u_r, 0.0)
        # eq. (4): Q_r(s, a_r) = E_g E_{a_H} E_{s'} gamma_r V_r(s')
        expected_v = game.expected_next(params.gamma_r * v_r)  # (S, A_r, J)
        q_r = np.einsum("saj,sj->sa", expected_v, joint_avg)
        # eq. (5). Q_r must be strictly negative; continuations into terminal states
        # are exactly zero, so clamp them just below zero.
        q_r_for_policy = np.minimum(q_r, -1e-12)
        pi_r_new = soft_power_policy(
            q_r_for_policy, params.beta_r, game.robot_action_mask
        )
        pi_r_new = damping * pi_r_new + (1 - damping) * pi_r
        # eq. (9)
        v_r_new = np.where(live, u_r + (pi_r_new * q_r).sum(-1), 0.0)

        delta = max(
            np.abs(v_r_new - v_r).max(),
            np.abs(pi_r_new - pi_r).max(),
            max(np.abs(a - b).max() for a, b in zip(v_e_new, v_e)),
        )
        v_r, pi_r, v_e = v_r_new, pi_r_new, v_e_new
        if delta < tol:
            converged = True
            break

    with np.errstate(divide="ignore"):
        w_h = [power_bits(x) for x in x_h]  # -inf at terminal states (X_h = 0)
    return Solution(
        q_r=q_r,
        pi_r=pi_r,
        v_r=v_r,
        u_r=u_r,
        v_e=v_e,
        x_h=x_h,
        w_h=w_h,
        prior=prior,
        iterations=iterations,
        converged=converged,
    )


def solve(
    game: TabularGame,
    human_params: HumanModelParams,
    power_params: PowerParams,
    **kwargs,
) -> Solution:
    prior = solve_human_prior(game, human_params)
    return solve_robot(game, prior, power_params, **kwargs)
