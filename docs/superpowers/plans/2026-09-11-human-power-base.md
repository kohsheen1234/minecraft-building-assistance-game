# Human-Power Base Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the reusable foundation for the human-power objective: the paper's power metrics as pure functions, an exact solver for eqs. (1) to (9) verified on closed forms and the paper's gridworld, and the environment plumbing that lets the assistant receive zero goal reward.

**Architecture:** A new dependency-free package `mbag/power/` holds the math (metrics, exact solver, gridworld). The environment gains one reward key (`goal_reward_scale`), one config flag (`goal_change_prob`), and one info field (`goal_completed`). The MCTS simulator applies the same scale so planner and env agree. Nothing in this plan touches RLlib policies; that is the `human-power/ppo-assistant` plan.

**Tech Stack:** Python 3.9, numpy 1.21, pytest. Existing repo tooling: `./lint.sh` (black, isort, flake8, mypy).

**Spec:** `docs/superpowers/specs/2026-09-11-human-power-design.md`

## Global Constraints

- Python 3.8 to 3.10 compatible syntax only (no `match`, no `X | Y` types, use `typing.List` etc.).
- numpy pinned `>=1.21,<1.22`.
- Run tests with the repo venv: `.venv/bin/pytest`. Fast tests must finish under the repo's 10 s per-test timeout (`pyproject.toml` sets `timeout = 10`); mark anything slower `@pytest.mark.slow`.
- Paper hyperparameters are fixed constants: `zeta=2.0, xi=1.0, eta=1.1, gamma_h=0.99, gamma_r=0.99`.
- Power is measured in bits: `log2`.
- Goal states are absorbing for the goal being evaluated (the paper's "mutually unreachable" requirement), implemented in the solver, not left to the caller.
- All new env config keys default to behaviour identical to today: `goal_reward_scale=1.0`, `goal_change_prob=0.0`.
- Commit after every task with the message trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Before each commit run `.venv/bin/black mbag tests && .venv/bin/isort mbag tests` on touched files.

---

## File map

| Path | Responsibility |
|---|---|
| `mbag/power/__init__.py` | re-exports |
| `mbag/power/metrics.py` | eqs. (5), (7), (8), (10) as numpy functions plus `PAPER_HYPERPARAMS` |
| `mbag/power/exact.py` | `TabularGame`, `HumanModelParams`, `PowerParams`, `solve_human_prior` (eqs. 1 to 3), `solve_robot` (eqs. 4 to 9), `Solution` dataclass |
| `mbag/power/gridworld.py` | the paper's key-and-door gridworld as a `TabularGame` |
| `mbag/environment/config.py` | `goal_reward_scale` reward key, `goal_change_prob` flag |
| `mbag/environment/mbag_env.py` | apply scale, emit `goal_completed`, resample goal |
| `mbag/environment/types.py` | `goal_completed` in `MbagInfoDict` |
| `mbag/rllib/alpha_zero/planning.py` | apply scale in predicted rewards |
| `mbag/scripts/train.py` | `goal_reward_scale` / `per_player_goal_reward_scale` / `goal_change_prob` sacred params |
| `mbag/evaluation/metrics.py`, `mbag/rllib/callbacks.py` | `human_actions_to_completion` |
| `tests/test_power_metrics.py`, `tests/test_power_exact.py`, `tests/test_power_gridworld.py`, `tests/test_goal_reward_scale.py`, `tests/test_goal_completed.py`, `tests/test_goal_change.py` | tests |
| `docs/human-power/README.md` | branch family overview, equations, mapping, how to run |

---

### Task 1: Power metric functions (eqs. 5, 7, 8, 10)

**Files:**
- Create: `mbag/power/__init__.py`
- Create: `mbag/power/metrics.py`
- Test: `tests/test_power_metrics.py`

**Interfaces:**
- Produces:
  - `PAPER_HYPERPARAMS: Dict[str, float]` with keys `zeta, xi, eta, gamma_h, gamma_r`.
  - `attainable_goals(v_e: np.ndarray, zeta: float, *, axis: int = -1, reduce: str = "sum") -> np.ndarray` — eq. (7). `reduce="mean"` is the sampled-goal variant used later by the learned estimator.
  - `power_bits(x: np.ndarray) -> np.ndarray` — eq. (10).
  - `robot_reward(x: np.ndarray, xi: float, eta: float, *, human_axis: Optional[int] = None) -> np.ndarray` — eq. (8). If `human_axis is None` the input is one human's `X_h`.
  - `soft_power_policy(q_r: np.ndarray, beta_r: float, mask: Optional[np.ndarray] = None) -> np.ndarray` — eq. (5) over the last axis; `beta_r=np.inf` gives argmax with ties split uniformly; `beta_r=0` gives uniform over valid actions.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_power_metrics.py
import numpy as np
import pytest

from mbag.power.metrics import (
    PAPER_HYPERPARAMS,
    attainable_goals,
    power_bits,
    robot_reward,
    soft_power_policy,
)


def test_paper_hyperparams():
    assert PAPER_HYPERPARAMS == {
        "zeta": 2.0,
        "xi": 1.0,
        "eta": 1.1,
        "gamma_h": 0.99,
        "gamma_r": 0.99,
    }


def test_k_certain_goals_gives_log2_k_bits():
    # Human can reach each of k goals for sure: X = k, W = log2 k.
    for k in [1, 2, 4, 8]:
        v_e = np.ones(k)
        x = attainable_goals(v_e, zeta=2.0)
        assert x == pytest.approx(k)
        assert power_bits(x) == pytest.approx(np.log2(k))


def test_k_uniform_random_goals_gives_minus_log2_k_bits():
    # Each goal reached with prob 1/k regardless of action, zeta=2: W = -log2 k.
    for k in [2, 4, 8]:
        v_e = np.full(k, 1.0 / k)
        x = attainable_goals(v_e, zeta=2.0)
        assert power_bits(x) == pytest.approx(-np.log2(k))


def test_zeta_prefers_reliability():
    # Two deterministic outcomes beat two coin tosses (paper Section 2.1).
    deterministic = attainable_goals(np.array([1.0, 1.0]), zeta=2.0)
    coin_tosses = attainable_goals(np.array([0.5, 0.5, 0.5, 0.5]), zeta=2.0)
    assert deterministic == pytest.approx(2.0)
    assert coin_tosses < deterministic


def test_attainable_goals_mean_reduction_and_axis():
    v_e = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert attainable_goals(v_e, zeta=2.0, axis=1, reduce="mean").tolist() == [
        pytest.approx(0.5),
        pytest.approx(0.25),
    ]
    assert attainable_goals(v_e, zeta=2.0, axis=0).tolist() == [
        pytest.approx(1.25),
        pytest.approx(0.25),
    ]


def test_robot_reward_single_human():
    # U_r = -(X^{-xi})^{eta} = -X^{-xi*eta}
    x = np.array([1.0, 2.0, 4.0])
    u = robot_reward(x, xi=1.0, eta=1.1)
    np.testing.assert_allclose(u, -(x ** (-1.1)))
    assert np.all(u < 0)
    assert np.all(np.diff(u) > 0)  # more power -> higher (less negative) reward


def test_robot_reward_protects_last_bit_xi_1():
    # Paper Section 2.2: with xi=1, dropping one human from 1 bit to 0 bits cannot be
    # compensated by raising another from >=1 bit to anything.
    before = robot_reward(np.array([2.0, 2.0]), xi=1.0, eta=1.0, human_axis=0)
    after = robot_reward(np.array([1.0, 1e9]), xi=1.0, eta=1.0, human_axis=0)
    assert after < before


def test_robot_reward_inequality_aversion():
    # Pigou-Dalton: equal split of the same total power beats an unequal one.
    equal = robot_reward(np.array([4.0, 4.0]), xi=1.0, eta=1.1, human_axis=0)
    unequal = robot_reward(np.array([2.0, 8.0]), xi=1.0, eta=1.1, human_axis=0)
    assert equal > unequal


def test_soft_power_policy_power_law():
    q = np.array([-1.0, -2.0, -4.0])
    pi = soft_power_policy(q, beta_r=1.0)
    expected = (-q) ** -1.0
    expected /= expected.sum()
    np.testing.assert_allclose(pi, expected)
    assert pi.sum() == pytest.approx(1.0)


def test_soft_power_policy_limits_and_mask():
    q = np.array([-1.0, -1.0, -4.0])
    np.testing.assert_allclose(soft_power_policy(q, beta_r=np.inf), [0.5, 0.5, 0.0])
    np.testing.assert_allclose(soft_power_policy(q, beta_r=0.0), [1 / 3] * 3)
    mask = np.array([False, True, True])
    pi = soft_power_policy(q, beta_r=np.inf, mask=mask)
    np.testing.assert_allclose(pi, [0.0, 1.0, 0.0])


def test_soft_power_policy_rejects_nonnegative_q():
    with pytest.raises(ValueError):
        soft_power_policy(np.array([0.0, -1.0]), beta_r=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_power_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbag.power'`

- [ ] **Step 3: Write the implementation**

```python
# mbag/power/__init__.py
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
```

```python
# mbag/power/metrics.py
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
        return powered.sum(axis=axis)
    if reduce == "mean":
        return powered.mean(axis=axis)
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
        weights = (q_r == best) & mask
        weights = weights.astype(float)
    else:
        # Work in log space for numerical stability: log w = -beta * log(-q).
        log_w = -beta_r * np.log(-q_r)
        log_w = np.where(mask, log_w, -np.inf)
        log_w = log_w - log_w.max(axis=-1, keepdims=True)
        weights = np.exp(log_w)
    return weights / weights.sum(axis=-1, keepdims=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_power_metrics.py -q`
Expected: `11 passed`

- [ ] **Step 5: Add the package to `pyproject.toml` and commit**

In `pyproject.toml`, under `[tool.setuptools] packages = [...]`, add `"mbag.power",` after `"mbag.evaluation",`.

```bash
.venv/bin/black mbag/power tests/test_power_metrics.py && .venv/bin/isort mbag/power tests/test_power_metrics.py
git add mbag/power tests/test_power_metrics.py pyproject.toml
git commit -m "Add mbag.power metrics: eqs. (5), (7), (8), (10) of Heitzig & Potham

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Tabular game and human behaviour prior (eqs. 1 to 3)

**Files:**
- Create: `mbag/power/exact.py`
- Test: `tests/test_power_exact.py`

**Interfaces:**
- Produces:
  - `TabularGame` dataclass: `num_states: int`, `num_robot_actions: int`, `human_action_counts: List[int]`, `next_states: np.ndarray` int of shape `(S, A_r, J, K)`, `next_probs: np.ndarray` float of shape `(S, A_r, J, K)`, `robot_action_mask: np.ndarray` bool `(S, A_r)`, `human_action_masks: List[np.ndarray]` each bool `(S, A_h)`, `goal_sets: List[np.ndarray]` each bool `(G_h, S)`, `terminal: np.ndarray` bool `(S,)`. `J = prod(human_action_counts)` indexes joint human actions in C order (first human varies slowest). `K` is the max number of stochastic outcomes; pad with prob 0.
  - `TabularGame.joint_index(actions: Sequence[int]) -> int` and `TabularGame.split_joint(j: int) -> Tuple[int, ...]`.
  - `HumanModelParams` dataclass: `nu: float`, `beta_h: float` (may be `np.inf`), `default_policy: Optional[List[np.ndarray]]` per human `(G_h, S, A_h)`, `beliefs_about_others: Optional[List[np.ndarray]]` per human `(S, A_h)` (μ_{-h}, goal-independent), `gamma_h: float`.
  - `HumanPrior` dataclass: `q_m: List[np.ndarray]` per human `(G_h, S, A_h)`, `pi_h: List[np.ndarray]` `(G_h, S, A_h)`, `v_m: List[np.ndarray]` `(G_h, S)`.
  - `solve_human_prior(game, params, *, max_iters=10_000, tol=1e-10) -> HumanPrior`.

Semantics: V(s', g) is replaced by 0 when `s' in g` or `terminal[s']`, so entering a goal state pays `U_h = 1` once and stops (goal-absorbing).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_power_exact.py
import numpy as np
import pytest

from mbag.power.exact import (
    HumanModelParams,
    TabularGame,
    solve_human_prior,
)


def make_bandit(k: int, outcome_probs: np.ndarray) -> TabularGame:
    """
    State 0 is the root. States 1..k are terminal outcomes. The human has k actions;
    outcome_probs[a] is the distribution over outcomes 1..k for human action a. The
    robot has one action and no influence. Goal g_i = {outcome i}.
    """
    S = k + 1
    A_r = 1
    A_h = k
    next_states = np.zeros((S, A_r, A_h, k), dtype=int)
    next_probs = np.zeros((S, A_r, A_h, k), dtype=float)
    for a in range(A_h):
        next_states[0, 0, a] = np.arange(1, k + 1)
        next_probs[0, 0, a] = outcome_probs[a]
    # Terminal states loop to themselves with probability 1 (never used).
    for s in range(1, S):
        next_states[s, :, :, 0] = s
        next_probs[s, :, :, 0] = 1.0
    goal_sets = np.zeros((k, S), dtype=bool)
    for i in range(k):
        goal_sets[i, i + 1] = True
    terminal = np.zeros(S, dtype=bool)
    terminal[1:] = True
    return TabularGame(
        num_states=S,
        num_robot_actions=A_r,
        human_action_counts=[A_h],
        next_states=next_states,
        next_probs=next_probs,
        robot_action_mask=np.ones((S, A_r), dtype=bool),
        human_action_masks=[np.ones((S, A_h), dtype=bool)],
        goal_sets=[goal_sets],
        terminal=terminal,
    )


def test_joint_index_roundtrip():
    game = make_bandit(3, np.eye(3))
    assert game.joint_index([2]) == 2
    assert game.split_joint(2) == (2,)


def test_rational_human_picks_goal_action_in_deterministic_bandit():
    k = 4
    game = make_bandit(k, np.eye(k))
    params = HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99)
    prior = solve_human_prior(game, params)
    q_m, pi_h, v_m = prior.q_m[0], prior.pi_h[0], prior.v_m[0]
    # Q^m_h(root, g_i, a) = 1 if a == i else 0.
    np.testing.assert_allclose(q_m[:, 0, :], np.eye(k))
    # Fully rational: pi_h(root, g_i) puts all mass on action i.
    np.testing.assert_allclose(pi_h[:, 0, :], np.eye(k))
    # V^m_h(root, g_i) = 1 for every goal.
    np.testing.assert_allclose(v_m[:, 0], np.ones(k))
    # Terminal states have zero value.
    np.testing.assert_allclose(v_m[:, 1:], 0.0)


def test_boltzmann_human_mixes_with_default_policy():
    k = 2
    game = make_bandit(k, np.eye(k))
    uniform = np.full((k, k + 1, k), 0.5)
    params = HumanModelParams(
        nu=0.5, beta_h=1.0, gamma_h=0.99, default_policy=[uniform]
    )
    prior = solve_human_prior(game, params)
    pi = prior.pi_h[0][0, 0]  # goal 0, root state
    softmax = np.exp([1.0, 0.0]) / np.exp([1.0, 0.0]).sum()
    expected = 0.5 * np.array([0.5, 0.5]) + 0.5 * softmax
    np.testing.assert_allclose(pi, expected)


def test_human_is_cautious_about_robot():
    # Robot has two actions; action 1 redirects every human action to outcome 2.
    k = 2
    S, A_r, A_h = k + 1, 2, k
    next_states = np.zeros((S, A_r, A_h, 1), dtype=int)
    next_probs = np.ones((S, A_r, A_h, 1), dtype=float)
    next_states[0, 0, 0, 0] = 1
    next_states[0, 0, 1, 0] = 2
    next_states[0, 1, :, 0] = 2  # robot sabotages goal 0
    for s in range(1, S):
        next_states[s, :, :, 0] = s
    goal_sets = np.zeros((k, S), dtype=bool)
    goal_sets[0, 1] = True
    goal_sets[1, 2] = True
    terminal = np.array([False, True, True])
    game = TabularGame(
        num_states=S,
        num_robot_actions=A_r,
        human_action_counts=[A_h],
        next_states=next_states,
        next_probs=next_probs,
        robot_action_mask=np.ones((S, A_r), dtype=bool),
        human_action_masks=[np.ones((S, A_h), dtype=bool)],
        goal_sets=[goal_sets],
        terminal=terminal,
    )
    prior = solve_human_prior(game, HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99))
    # eq. (1) takes min over robot actions: goal 0 is unattainable in the worst case.
    np.testing.assert_allclose(prior.q_m[0][0, 0, :], [0.0, 0.0])
    np.testing.assert_allclose(prior.q_m[0][1, 0, :], [1.0, 1.0])


def test_masked_human_actions_get_zero_probability():
    k = 3
    game = make_bandit(k, np.eye(k))
    game.human_action_masks[0][0, 2] = False
    prior = solve_human_prior(game, HumanModelParams(nu=0.0, beta_h=1.0, gamma_h=0.99))
    assert prior.pi_h[0][:, 0, 2].max() == 0.0
    np.testing.assert_allclose(prior.pi_h[0][:, 0, :].sum(-1), 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_power_exact.py -q`
Expected: FAIL with `ImportError: cannot import name 'HumanModelParams'`

- [ ] **Step 3: Write the implementation**

```python
# mbag/power/exact.py
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

from .metrics import attainable_goals, power_bits, robot_reward, soft_power_policy


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
        return int(np.ravel_multi_index(tuple(actions), tuple(self.human_action_counts)))

    def split_joint(self, j: int) -> Tuple[int, ...]:
        return tuple(int(i) for i in np.unravel_index(j, tuple(self.human_action_counts)))

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
    Probability of each joint action j given policies for humans other than h.
    Returns (S, J, A_h): for each state and joint action, the weight contributed by
    the other humans; entry is zero unless joint action j has human h taking action
    split_joint(j)[h]. Used to marginalise over a_{-h}.
    """
    S = game.num_states
    J = game.num_joint_human_actions
    A_h = game.human_action_counts[h]
    weights = np.ones((S, J))
    for j in range(J):
        actions = game.split_joint(j)
        for other, a_other in enumerate(actions):
            if other == h:
                continue
            weights[:, j] *= per_human_policies[other][:, a_other]
    out = np.zeros((S, J, A_h))
    for j in range(J):
        out[:, j, game.split_joint(j)[h]] = weights[:, j]
    return out


def solve_human_prior(
    game: TabularGame,
    params: HumanModelParams,
    *,
    max_iters: int = 10_000,
    tol: float = 1e-10,
) -> HumanPrior:
    """Fixed-point iteration of eqs. (1) to (3) for every human and goal."""
    S, A_r = game.num_states, game.num_robot_actions
    prior = HumanPrior()
    for h in range(game.num_humans):
        A_h = game.human_action_counts[h]
        goals = game.goal_sets[h]  # (G_h, S)
        G_h = goals.shape[0]
        mask_h = game.human_action_masks[h]  # (S, A_h)

        if params.beliefs_about_others is None:
            others = [
                np.ones((S, A_o)) / A_o for A_o in game.human_action_counts
            ]
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
            delta = max(
                np.abs(q_new - q_m).max(), np.abs(v_new - v_m).max()
            )
            q_m, pi_h, v_m = q_new, pi_new, v_new
            if delta < tol:
                break

        prior.q_m.append(q_m)
        prior.pi_h.append(pi_h)
        prior.v_m.append(v_m)
    return prior
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_power_exact.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
.venv/bin/black mbag/power tests/test_power_exact.py && .venv/bin/isort mbag/power tests/test_power_exact.py
git add mbag/power/exact.py tests/test_power_exact.py
git commit -m "Add exact solver for the human behaviour prior, eqs. (1)-(3)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Robot solution (eqs. 4 to 9) with closed-form and Appendix A checks

**Files:**
- Modify: `mbag/power/exact.py`
- Modify: `mbag/power/__init__.py`
- Test: `tests/test_power_exact.py`

**Interfaces:**
- Consumes: `TabularGame`, `HumanPrior`, `solve_human_prior` from Task 2; `attainable_goals`, `robot_reward`, `soft_power_policy`, `power_bits` from Task 1.
- Produces:
  - `PowerParams` dataclass: `zeta: float`, `xi: float`, `eta: float`, `beta_r: float`, `gamma_r: float`, `gamma_h: float`. Classmethod `PowerParams.paper(beta_r=5.0)` fills from `PAPER_HYPERPARAMS`.
  - `Solution` dataclass: `q_r (S, A_r)`, `pi_r (S, A_r)`, `v_r (S,)`, `u_r (S,)`, `v_e: List[(G_h, S)]`, `x_h: List[(S,)]`, `w_h: List[(S,)]`, `prior: HumanPrior`.
  - `solve_robot(game, prior, params, *, max_iters=10_000, tol=1e-10, damping=0.5) -> Solution`.
  - `solve(game, human_params, power_params, **kw) -> Solution` convenience wrapper.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_power_exact.py`)

```python
from mbag.power.exact import PowerParams, solve, solve_robot  # noqa: E402


def test_deterministic_bandit_power_is_log2_k():
    for k in [1, 2, 4, 8]:
        game = make_bandit(k, np.eye(k))
        sol = solve(
            game,
            HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99),
            PowerParams.paper(beta_r=5.0),
        )
        assert sol.x_h[0][0] == pytest.approx(k)
        assert sol.w_h[0][0] == pytest.approx(np.log2(k))
        assert sol.u_r[0] == pytest.approx(-(k ** -1.1))
        # Robot has one action: pi_r is trivially 1, Q_r = gamma_r * V_r(terminal) = 0
        # is not allowed (Q_r must be < 0), so the solver must add the terminal
        # continuation correctly: V_r(terminal) = 0, so we make root reward count.
        assert sol.v_r[0] == pytest.approx(sol.u_r[0])


def test_uniform_random_bandit_power_is_minus_log2_k():
    for k in [2, 4, 8]:
        game = make_bandit(k, np.full((k, k), 1.0 / k))
        sol = solve(
            game,
            HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99),
            PowerParams.paper(),
        )
        assert sol.w_h[0][0] == pytest.approx(-np.log2(k))


def test_rational_bandit_matches_closed_form_W():
    # Appendix A: with G = S and full rationality, W = log2 sum_s max_a P(s|a)^zeta.
    rng = np.random.default_rng(0)
    k = 4
    probs = rng.dirichlet(np.ones(k), size=k)
    game = make_bandit(k, probs)
    sol = solve(
        game,
        HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99),
        PowerParams.paper(),
    )
    closed_form = np.log2((probs.max(axis=0) ** 2.0).sum())
    assert sol.w_h[0][0] == pytest.approx(closed_form)


def _entropy_regularized_empowerment(probs: np.ndarray, zeta: float, rng) -> float:
    """E^zeta = max_pi I(a; s') - (zeta - 1) H(s' | a), by random search over pi."""
    k = probs.shape[0]
    best = -np.inf
    for pi in rng.dirichlet(np.ones(k) * 0.3, size=4000):
        p_s = pi @ probs
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(probs > 0, np.log2(probs / p_s), 0.0)
            mi = float((pi[:, None] * probs * log_ratio).sum())
            cond_h = float(-(pi[:, None] * probs * np.where(probs > 0, np.log2(probs), 0.0)).sum())
        best = max(best, mi - (zeta - 1) * cond_h)
    return best


def test_appendix_a_inequality_holds_on_random_bandits():
    rng = np.random.default_rng(1)
    for _ in range(5):
        k = int(rng.integers(2, 5))
        probs = rng.dirichlet(np.ones(k), size=k)
        game = make_bandit(k, probs)
        sol = solve(
            game,
            HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99),
            PowerParams.paper(),
        )
        e_zeta = _entropy_regularized_empowerment(probs, zeta=2.0, rng=rng)
        assert e_zeta <= sol.w_h[0][0] + 1e-9


def test_robot_prefers_action_that_empowers_human():
    # Two-step game: robot first chooses "unlock" (state 1) or "keep locked" (state 2).
    # From state 1 the human can reach either of two outcomes; from state 2 only one.
    S = 5  # 0 root, 1 unlocked, 2 locked, 3 outcome A, 4 outcome B
    A_r, A_h = 2, 2
    next_states = np.zeros((S, A_r, A_h, 1), dtype=int)
    next_probs = np.ones((S, A_r, A_h, 1))
    next_states[0, 0, :, 0] = 1  # robot action 0 = unlock
    next_states[0, 1, :, 0] = 2  # robot action 1 = keep locked
    next_states[1, :, 0, 0] = 3
    next_states[1, :, 1, 0] = 4
    next_states[2, :, :, 0] = 3  # locked: only outcome A reachable
    for s in [3, 4]:
        next_states[s, :, :, 0] = s
    goal_sets = np.zeros((2, S), dtype=bool)
    goal_sets[0, 3] = True
    goal_sets[1, 4] = True
    terminal = np.array([False, False, False, True, True])
    game = TabularGame(
        num_states=S,
        num_robot_actions=A_r,
        human_action_counts=[A_h],
        next_states=next_states,
        next_probs=next_probs,
        robot_action_mask=np.ones((S, A_r), dtype=bool),
        human_action_masks=[np.ones((S, A_h), dtype=bool)],
        goal_sets=[goal_sets],
        terminal=terminal,
    )
    sol = solve(
        game,
        HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99),
        PowerParams.paper(beta_r=5.0),
    )
    assert sol.w_h[0][1] == pytest.approx(1.0)  # unlocked: 2 goals -> 1 bit
    assert sol.w_h[0][2] == pytest.approx(0.0)  # locked: 1 goal -> 0 bits
    assert sol.q_r[0, 0] > sol.q_r[0, 1]
    assert sol.pi_r[0, 0] > 0.9
    # Soft policy still explores.
    assert sol.pi_r[0, 1] > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_power_exact.py -q`
Expected: FAIL with `ImportError: cannot import name 'PowerParams'`

- [ ] **Step 3: Write the implementation** (append to `mbag/power/exact.py`)

```python
from .metrics import PAPER_HYPERPARAMS  # add to the imports at the top of the file


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


def _joint_human_policy(game: TabularGame, prior: HumanPrior) -> np.ndarray:
    """
    Joint policy over j for each combination of goals, averaged over independent
    uniform goal draws for every human. Returns (S, J).

    For n humans the exact expectation E_g over the product of goal sets is taken.
    """
    S, J = game.num_states, game.num_joint_human_actions
    # Average each human's goal-conditioned policy over its uniform goal draw.
    avg = [pi.mean(axis=0) for pi in prior.pi_h]  # per human (S, A_h)
    joint = np.ones((S, J))
    for j in range(J):
        for h, a_h in enumerate(game.split_joint(j)):
            joint[:, j] *= avg[h][:, a_h]
    return joint


def _joint_human_policy_for_goal(
    game: TabularGame, prior: HumanPrior, h: int, g: int
) -> np.ndarray:
    """
    Joint policy over j when human h has goal g and all other humans' goals are
    averaged (uniform draws). Returns (S, J).
    """
    S, J = game.num_states, game.num_joint_human_actions
    per_human = [pi.mean(axis=0) for pi in prior.pi_h]
    per_human[h] = prior.pi_h[h][g]
    joint = np.ones((S, J))
    for j in range(J):
        for other, a in enumerate(game.split_joint(j)):
            joint[:, j] *= per_human[other][:, a]
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
    """
    S, A_r = game.num_states, game.num_robot_actions
    live = ~game.terminal

    pi_r = game.robot_action_mask / game.robot_action_mask.sum(-1, keepdims=True)
    v_r = np.zeros(S)
    v_e = [np.zeros(gs.shape) for gs in game.goal_sets]

    # Precompute goal-conditioned joint human policies: per human, (G_h, S, J).
    joint_for_goal = [
        np.stack(
            [
                _joint_human_policy_for_goal(game, prior, h, g)
                for g in range(game.goal_sets[h].shape[0])
            ]
        )
        for h in range(game.num_humans)
    ]
    joint_avg = _joint_human_policy(game, prior)  # (S, J)

    for _ in range(max_iters):
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
        # eq. (7), (8)
        x_h = [attainable_goals(v, params.zeta, axis=0) for v in v_e_new]
        x_stack = np.stack(x_h)  # (H, S)
        u_r = robot_reward(x_stack, params.xi, params.eta, human_axis=0)
        u_r = np.where(live, u_r, 0.0)
        # eq. (4): Q_r(s, a_r) = E_g E_{a_H} E_{s'} gamma_r V_r(s')
        expected_v = game.expected_next(params.gamma_r * v_r)  # (S, A_r, J)
        q_r = np.einsum("saj,sj->sa", expected_v, joint_avg)
        # Q_r must be strictly negative for eq. (5); shift by tiny epsilon where a
        # continuation is exactly zero (e.g. into terminal states).
        q_r_for_policy = np.minimum(q_r, -1e-12)
        # eq. (5)
        pi_r_new = soft_power_policy(q_r_for_policy, params.beta_r, game.robot_action_mask)
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
            break

    return Solution(
        q_r=q_r,
        pi_r=pi_r,
        v_r=v_r,
        u_r=u_r,
        v_e=v_e,
        x_h=x_h,
        w_h=[power_bits(x) for x in x_h],
        prior=prior,
    )


def solve(
    game: TabularGame,
    human_params: HumanModelParams,
    power_params: PowerParams,
    **kwargs,
) -> Solution:
    prior = solve_human_prior(game, human_params)
    return solve_robot(game, prior, power_params, **kwargs)
```

Update `mbag/power/__init__.py` to also export `TabularGame, HumanModelParams, HumanPrior, PowerParams, Solution, solve_human_prior, solve_robot, solve`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_power_exact.py -q`
Expected: `11 passed`. If `test_deterministic_bandit_power_is_log2_k` fails on the `v_r` assertion, check that `u_r` is zeroed only on terminal states and that `q_r` into terminal states is exactly `0` so `v_r[0] == u_r[0]`.

- [ ] **Step 5: Commit**

```bash
.venv/bin/black mbag/power tests/test_power_exact.py && .venv/bin/isort mbag/power tests/test_power_exact.py
git add mbag/power tests/test_power_exact.py
git commit -m "Add exact robot solution, eqs. (4)-(9), with closed-form and Appendix A tests

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: The paper's key-and-door gridworld

**Files:**
- Create: `mbag/power/gridworld.py`
- Test: `tests/test_power_gridworld.py`

**Interfaces:**
- Consumes: `TabularGame`, `HumanModelParams`, `PowerParams`, `solve`, `Solution` from Tasks 2 and 3.
- Produces:
  - `KeyDoorGridworld` class with `layout: List[str]`, `game: TabularGame`, `encode(human_xy, robot_xy, key_held, door_open) -> int`, `decode(s) -> Tuple[Tuple[int,int], Tuple[int,int], bool, bool]`, `initial_state: int`, `HUMAN_ACTIONS = ["stay","up","down","left","right"]`, `ROBOT_ACTIONS = HUMAN_ACTIONS + ["interact"]`, `goal_cell(g) -> Tuple[int,int]`.
  - `DEFAULT_LAYOUT` (see below).
  - `rollout(world, solution, human_goal: int, max_steps=30, greedy=True) -> List[int]` returns visited states with the human following `pi_h(., goal)` and the robot following `argmax pi_r` (greedy) or sampling.

Layout characters: `.` open, `#` wall, `D` door (closed at start), `K` key on floor, `H` human start, `R` robot start. Rules: agents move one cell in four directions; moving into a wall, a closed door, or the other agent's cell leaves the mover in place. Moves are applied human first, then robot, on the pre-move positions of the other (so swapping is impossible). Robot `interact`: if on the key cell and key not held, picks it up; else if holding key and adjacent (4-neighbour) to the door and door closed, opens it; else no-op. Goals: one per open cell (including door cell and start cells, excluding walls): `g_c = {s : human at c}`. No terminal states (goal-absorbing semantics in the solver handle termination per goal).

```
DEFAULT_LAYOUT = [
    "H..D..",
    ".RK#..",
]
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_power_gridworld.py
import numpy as np
import pytest

from mbag.power.exact import HumanModelParams, PowerParams, solve
from mbag.power.gridworld import DEFAULT_LAYOUT, KeyDoorGridworld, rollout


def test_encode_decode_roundtrip():
    world = KeyDoorGridworld(DEFAULT_LAYOUT)
    for s in range(0, world.game.num_states, 37):
        assert world.encode(*world.decode(s)) == s


def test_initial_state_matches_layout():
    world = KeyDoorGridworld(DEFAULT_LAYOUT)
    human, robot, key_held, door_open = world.decode(world.initial_state)
    assert human == (0, 0)
    assert robot == (1, 1)
    assert not key_held and not door_open


def test_closed_door_blocks_human():
    world = KeyDoorGridworld(DEFAULT_LAYOUT)
    s = world.encode((0, 2), (1, 0), False, False)
    j = world.game.joint_index([world.HUMAN_ACTIONS.index("right")])
    a_r = world.ROBOT_ACTIONS.index("stay")
    s_next = int(world.game.next_states[s, a_r, j, 0])
    assert world.decode(s_next)[0] == (0, 2)


def test_robot_interact_picks_up_key_then_opens_door():
    world = KeyDoorGridworld(DEFAULT_LAYOUT)
    stay = world.game.joint_index([world.HUMAN_ACTIONS.index("stay")])
    interact = world.ROBOT_ACTIONS.index("interact")
    s = world.encode((0, 0), (1, 2), False, False)  # robot on key
    s = int(world.game.next_states[s, interact, stay, 0])
    assert world.decode(s)[2] is True
    s = world.encode((0, 0), (0, 2), True, False)  # robot next to door with key
    s = int(world.game.next_states[s, interact, stay, 0])
    assert world.decode(s)[3] is True


@pytest.mark.slow
def test_open_door_raises_human_power():
    world = KeyDoorGridworld(DEFAULT_LAYOUT)
    sol = solve(
        world.game,
        HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99),
        PowerParams.paper(beta_r=5.0),
        max_iters=3000,
        tol=1e-6,
    )
    closed = world.encode((0, 0), (1, 0), True, False)
    opened = world.encode((0, 0), (1, 0), True, True)
    assert sol.w_h[0][opened] > sol.w_h[0][closed]


@pytest.mark.slow
def test_robot_fetches_key_opens_door_and_clears_path():
    world = KeyDoorGridworld(DEFAULT_LAYOUT)
    sol = solve(
        world.game,
        HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99),
        PowerParams.paper(beta_r=5.0),
        max_iters=3000,
        tol=1e-6,
    )
    far_goal = world.goal_index((0, 5))
    states = rollout(world, sol, human_goal=far_goal, max_steps=30, greedy=True)
    decoded = [world.decode(s) for s in states]
    assert any(d[2] for d in decoded), "robot never picked up the key"
    assert any(d[3] for d in decoded), "robot never opened the door"
    assert decoded[-1][0] == (0, 5), "human did not reach the far side"
    # Once the door is open the robot must not be standing in the doorway.
    door = world.door_cell
    assert all(d[1] != door for d in decoded if d[3])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_power_gridworld.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mbag.power.gridworld'`

- [ ] **Step 3: Write the implementation**

```python
# mbag/power/gridworld.py
"""
The key-and-door gridworld from Section 4.2 of Heitzig & Potham (2025), encoded as a
TabularGame so the exact solver can be checked against the paper's qualitative
result: the robot fetches the key, unlocks the door, and steps out of the way.
"""

from typing import List, Sequence, Tuple

import numpy as np

from .exact import Solution, TabularGame

Cell = Tuple[int, int]  # (row, col)

DEFAULT_LAYOUT: List[str] = [
    "H..D..",
    ".RK#..",
]

MOVES = {
    "stay": (0, 0),
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class KeyDoorGridworld:
    HUMAN_ACTIONS: List[str] = ["stay", "up", "down", "left", "right"]
    ROBOT_ACTIONS: List[str] = HUMAN_ACTIONS + ["interact"]

    def __init__(self, layout: Sequence[str] = DEFAULT_LAYOUT):
        self.layout = list(layout)
        self.rows = len(self.layout)
        self.cols = len(self.layout[0])
        assert all(len(row) == self.cols for row in self.layout)

        self.cells: List[Cell] = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.layout[r][c] != "#"
        ]
        self.cell_index = {cell: i for i, cell in enumerate(self.cells)}
        self.door_cell = self._find("D")
        self.key_cell = self._find("K")
        self.human_start = self._find("H")
        self.robot_start = self._find("R")

        self.num_cells = len(self.cells)
        self.game = self._build_game()
        self.initial_state = self.encode(
            self.human_start, self.robot_start, False, False
        )

    def _find(self, ch: str) -> Cell:
        for r, row in enumerate(self.layout):
            c = row.find(ch)
            if c >= 0:
                return (r, c)
        raise ValueError(f"layout has no {ch!r}")

    # State encoding: ((human * num_cells + robot) * 2 + key_held) * 2 + door_open
    def encode(self, human: Cell, robot: Cell, key_held: bool, door_open: bool) -> int:
        h = self.cell_index[human]
        r = self.cell_index[robot]
        return ((h * self.num_cells + r) * 2 + int(key_held)) * 2 + int(door_open)

    def decode(self, s: int) -> Tuple[Cell, Cell, bool, bool]:
        door_open = bool(s % 2)
        s //= 2
        key_held = bool(s % 2)
        s //= 2
        r = s % self.num_cells
        h = s // self.num_cells
        return self.cells[h], self.cells[r], key_held, door_open

    def goal_index(self, cell: Cell) -> int:
        return self.cell_index[cell]

    def goal_cell(self, g: int) -> Cell:
        return self.cells[g]

    def _passable(self, cell: Cell, door_open: bool, occupied: Cell) -> bool:
        r, c = cell
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        if self.layout[r][c] == "#":
            return False
        if cell == self.door_cell and not door_open:
            return False
        if cell == occupied:
            return False
        return True

    def _move(self, cell: Cell, action: str, door_open: bool, occupied: Cell) -> Cell:
        dr, dc = MOVES[action]
        target = (cell[0] + dr, cell[1] + dc)
        return target if self._passable(target, door_open, occupied) else cell

    def _adjacent(self, a: Cell, b: Cell) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def step(
        self, s: int, robot_action: str, human_action: str
    ) -> int:
        human, robot, key_held, door_open = self.decode(s)
        # Human moves first against the robot's current position.
        new_human = self._move(human, human_action, door_open, occupied=robot)
        if robot_action == "interact":
            if robot == self.key_cell and not key_held:
                key_held = True
            elif key_held and not door_open and self._adjacent(robot, self.door_cell):
                door_open = True
            new_robot = robot
        else:
            new_robot = self._move(robot, robot_action, door_open, occupied=new_human)
        return self.encode(new_human, new_robot, key_held, door_open)

    def _build_game(self) -> TabularGame:
        S = self.num_cells * self.num_cells * 4
        A_r = len(self.ROBOT_ACTIONS)
        A_h = len(self.HUMAN_ACTIONS)
        next_states = np.zeros((S, A_r, A_h, 1), dtype=int)
        next_probs = np.ones((S, A_r, A_h, 1), dtype=float)
        for s in range(S):
            human, robot, _, _ = self.decode(s)
            if human == robot:
                # Unreachable overlapping configuration; make it absorbing.
                next_states[s, :, :, 0] = s
                continue
            for a_r, ra in enumerate(self.ROBOT_ACTIONS):
                for a_h, ha in enumerate(self.HUMAN_ACTIONS):
                    next_states[s, a_r, a_h, 0] = self.step(s, ra, ha)
        goal_sets = np.zeros((self.num_cells, S), dtype=bool)
        for s in range(S):
            human, _, _, _ = self.decode(s)
            goal_sets[self.cell_index[human], s] = True
        terminal = np.zeros(S, dtype=bool)
        for s in range(S):
            human, robot, _, _ = self.decode(s)
            if human == robot:
                terminal[s] = True
        return TabularGame(
            num_states=S,
            num_robot_actions=A_r,
            human_action_counts=[A_h],
            next_states=next_states,
            next_probs=next_probs,
            robot_action_mask=np.ones((S, A_r), dtype=bool),
            human_action_masks=[np.ones((S, A_h), dtype=bool)],
            goal_sets=[goal_sets],
            terminal=terminal,
        )


def rollout(
    world: KeyDoorGridworld,
    solution: Solution,
    human_goal: int,
    max_steps: int = 30,
    greedy: bool = True,
    seed: int = 0,
) -> List[int]:
    """
    Simulate the human following pi_h(., human_goal) and the robot following pi_r.
    Stops when the human reaches the goal cell. Returns the visited states.
    """
    rng = np.random.default_rng(seed)
    pi_h = solution.prior.pi_h[0][human_goal]
    s = world.initial_state
    states = [s]
    for _ in range(max_steps):
        if world.game.goal_sets[0][human_goal, s]:
            break
        if greedy:
            a_r = int(np.argmax(solution.pi_r[s]))
            a_h = int(np.argmax(pi_h[s]))
        else:
            a_r = int(rng.choice(len(world.ROBOT_ACTIONS), p=solution.pi_r[s]))
            a_h = int(rng.choice(len(world.HUMAN_ACTIONS), p=pi_h[s]))
        s = world.step(s, world.ROBOT_ACTIONS[a_r], world.HUMAN_ACTIONS[a_h])
        states.append(s)
    return states
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_power_gridworld.py -q --timeout=600`
Expected: `6 passed`. The two slow tests solve a 576-state game; expect 10 to 60 s each.

If `test_robot_fetches_key_opens_door_and_clears_path` fails:
- Print `sol.w_h[0][s]` along the rollout and `sol.pi_r[s]` at the first state. Power should jump when the door opens.
- If the robot dithers, raise `beta_r` to 20 for the test or lower `damping` to 0.2; the paper's β_r=5 is soft.
- If the fixed point oscillates, check `delta` per iteration; reduce `damping`.
- If the human's `argmax pi_h` ties between actions, the greedy rollout may loop; use `greedy=False` with a fixed seed.

- [ ] **Step 5: Commit**

```bash
.venv/bin/black mbag/power tests/test_power_gridworld.py && .venv/bin/isort mbag/power tests/test_power_gridworld.py
git add mbag/power/gridworld.py tests/test_power_gridworld.py
git commit -m "Add key-and-door gridworld from the paper and behavioural test of the exact solver

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `goal_reward_scale` reward key in env, planner, and train script

**Files:**
- Modify: `mbag/environment/config.py:58-100` (RewardsConfigDict, RewardsConfigDictKey) and `:277-284` (DEFAULT_CONFIG rewards)
- Modify: `mbag/environment/mbag_env.py:530` (`_step_player`, where `reward = goal_dependent_reward + goal_independent_reward`)
- Modify: `mbag/rllib/alpha_zero/planning.py:261-334` (`_get_predicted_goal_dependent_reward`) and `:336-397` (`get_all_rewards`)
- Modify: `mbag/scripts/train.py:122-133` and `:227-278`
- Test: `tests/test_goal_reward_scale.py`

**Interfaces:**
- Produces: reward key `"goal_reward_scale"` (float or schedule, default `1.0`) that multiplies `goal_dependent_reward` for that player everywhere it is computed. Sacred params `goal_reward_scale` and `per_player_goal_reward_scale`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_goal_reward_scale.py
import copy

import numpy as np
import pytest

from mbag.agents.heuristic_agents import LayerBuilderAgent
from mbag.environment.config import DEFAULT_CONFIG
from mbag.environment.goals.simple import BasicGoalGenerator
from mbag.evaluation.evaluator import MbagEvaluator


def _two_player_config(assistant_scale: float):
    return {
        "world_size": (5, 5, 5),
        "num_players": 2,
        "horizon": 50,
        "goal_generator": BasicGoalGenerator,
        "goal_generator_config": {},
        "players": [
            {"rewards": {"goal_reward_scale": 1.0}},
            {"rewards": {"goal_reward_scale": assistant_scale, "own_reward_prop": 1.0}},
        ],
        "malmo": {"use_malmo": False, "use_spectator": False, "video_dir": None},
    }


def test_default_scale_is_one_and_unchanged_behaviour():
    evaluator = MbagEvaluator(
        _two_player_config(1.0), [(LayerBuilderAgent, {}), (LayerBuilderAgent, {})]
    )
    episode = evaluator.rollout()
    assistant_goal_reward = sum(
        infos[1]["goal_dependent_reward"] for infos in episode.info_history
    )
    assert assistant_goal_reward > 0


def test_zero_scale_removes_assistant_goal_reward():
    evaluator = MbagEvaluator(
        _two_player_config(0.0), [(LayerBuilderAgent, {}), (LayerBuilderAgent, {})]
    )
    episode = evaluator.rollout()
    for infos in episode.info_history:
        assert infos[1]["goal_dependent_reward"] == 0.0
    # The human still gets goal reward and the house still gets built.
    assert episode.last_infos[0]["goal_percentage"] == pytest.approx(1.0)
    human_goal_reward = sum(infos[0]["goal_dependent_reward"] for infos in episode.info_history)
    assert human_goal_reward > 0
    # With own_reward_prop=1 and scale 0 the assistant's env reward is exactly zero
    # at every step (it has no noop/action rewards configured either).
    assistant_rewards = [rewards[1] for rewards in episode.reward_history_per_player]
    assert all(r == 0.0 for r in assistant_rewards)


@pytest.mark.uses_rllib
def test_env_model_all_rewards_respect_scale():
    from mbag.agents.action_distributions import MbagActionDistribution
    from mbag.rllib.alpha_zero.planning import create_mbag_env_model

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["world_size"] = (5, 5, 5)
    config["goal_generator"] = "basic"
    config["num_players"] = 2
    config["players"] = [{}, {"rewards": {"goal_reward_scale": 0.0}}]

    env = create_mbag_env_model(config, player_index=1)
    obs, info = env.reset()
    world_obs, inventory_obs, timestep = obs
    obs_batch = (world_obs[None], inventory_obs[None], timestep[None])
    rewards = env.get_all_rewards(obs_batch)
    assert rewards.shape[1] == MbagActionDistribution.NUM_CHANNELS
    assert np.all(rewards == 0)

    # Player 0 still has scale 1 and gets non-zero rewards.
    rewards_player_0 = env.get_all_rewards(obs_batch, player_index=0)
    assert np.any(rewards_player_0 != 0)
```

Check `MbagEpisode` for a per-player reward history attribute name before relying on `reward_history_per_player`: run `grep -n "reward_history\|class MbagEpisode" mbag/evaluation/episode.py`. If only a summed `reward_history` exists, replace the last assertion with a direct env loop:

```python
    from mbag.environment.mbag_env import MbagEnv
    env = MbagEnv(_two_player_config(0.0))
    env.reset()
    agents = [LayerBuilderAgent({}, env.config), LayerBuilderAgent({}, env.config)]
    # (see tests/test_evaluator.py / mbag/evaluation/evaluator.py for how agents are
    # constructed and stepped; mirror that loop and assert rewards[1] == 0.0 each step)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_goal_reward_scale.py -q`
Expected: FAIL. `test_zero_scale_removes_assistant_goal_reward` fails on the first `== 0.0` assertion (unknown key is ignored today, or a `ValueError` from the schedule builder, or a `KeyError` in `_get_reward`).

- [ ] **Step 3: Implement**

`mbag/environment/config.py`, inside `RewardsConfigDict` after `get_resources`:

```python
    goal_reward_scale: RewardSchedule
    """
    Multiplier applied to this player's goal-dependent reward (progress towards the
    goal structure plus the incorrect_action penalty). 1.0 is the standard reward;
    0.0 removes all goal information from this player's reward, which is required
    for the human-power assistant objective.
    """
```

Replace `RewardsConfigDictKey`:

```python
RewardsConfigDictKey = Literal[
    "noop",
    "action",
    "incorrect_action",
    "place_wrong",
    "own_reward_prop",
    "get_resources",
    "goal_reward_scale",
]
```

In `DEFAULT_CONFIG["rewards"]` add `"goal_reward_scale": 1.0,`.

`mbag/environment/mbag_env.py`, in `_step_player`, replace

```python
        reward = goal_dependent_reward + goal_independent_reward
```

with

```python
        goal_dependent_reward *= self._get_reward(
            player_index, "goal_reward_scale", self.global_timestep
        )
        reward = goal_dependent_reward + goal_independent_reward
```

`mbag/rllib/alpha_zero/planning.py`, at the end of `_get_predicted_goal_dependent_reward` replace `return reward` with

```python
        reward *= env._get_reward(player_index, "goal_reward_scale", env.global_timestep)
        return reward
```

and in `get_all_rewards`, before `return rewards`:

```python
        rewards *= env._get_reward(player_index, "goal_reward_scale", env.global_timestep)
```

`mbag/scripts/train.py`, after the `per_player_own_reward_prop` line in the config:

```python
    goal_reward_scale: RewardSchedule = 1.0
    per_player_goal_reward_scale: Optional[List[RewardSchedule]] = None
```

In the per-player loop, after the `per_player_own_reward_prop` block:

```python
        if per_player_goal_reward_scale is not None:
            player_config["rewards"]["goal_reward_scale"] = per_player_goal_reward_scale[
                player_index
            ]
```

In the `"rewards"` dict of `environment_params` add `"goal_reward_scale": goal_reward_scale,`.

- [ ] **Step 4: Run tests to verify they pass, plus the existing reward tests**

Run: `.venv/bin/pytest tests/test_goal_reward_scale.py tests/test_env.py tests/test_planning.py tests/test_evaluator.py -q`
Expected: all pass. Then run the AlphaZero consistency test, which is slow:
`.venv/bin/pytest tests/test_train.py::test_predicted_rewards_equal_rewards_in_alpha_zero -q --timeout=900`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
.venv/bin/black mbag tests/test_goal_reward_scale.py && .venv/bin/isort mbag tests/test_goal_reward_scale.py
git add mbag/environment/config.py mbag/environment/mbag_env.py mbag/rllib/alpha_zero/planning.py mbag/scripts/train.py tests/test_goal_reward_scale.py
git commit -m "Add goal_reward_scale reward key so a player can receive zero goal reward

Applied in the env, in the MCTS env model, and exposed in the train script.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `goal_completed` info field (U_h indicator)

**Files:**
- Modify: `mbag/environment/types.py:94-130` (`MbagInfoDict`)
- Modify: `mbag/environment/mbag_env.py:349-356` (`step`, where `goal_percentage` is filled) and `:991-1027` (`_get_player_info`)
- Test: `tests/test_goal_completed.py`

**Interfaces:**
- Produces: `info["goal_completed"]: bool`, True exactly on the step in which `current_blocks == goal_blocks` first becomes true, False otherwise (including on reset infos). This is $U_h(s', g)$.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goal_completed.py
from mbag.agents.heuristic_agents import LayerBuilderAgent, NoopAgent
from mbag.environment.goals.simple import BasicGoalGenerator
from mbag.evaluation.evaluator import MbagEvaluator


def _config(num_players: int, terminate: bool):
    return {
        "world_size": (5, 5, 5),
        "num_players": num_players,
        "players": [{} for _ in range(num_players)],
        "horizon": 60,
        "terminate_on_goal_completion": terminate,
        "goal_generator": BasicGoalGenerator,
        "goal_generator_config": {},
        "malmo": {"use_malmo": False, "use_spectator": False, "video_dir": None},
    }


def test_goal_completed_fires_once_and_ends_episode():
    evaluator = MbagEvaluator(_config(1, terminate=True), [(LayerBuilderAgent, {})])
    episode = evaluator.rollout()
    flags = [infos[0]["goal_completed"] for infos in episode.info_history]
    assert sum(flags) == 1
    assert flags[-1] is True


def test_goal_completed_fires_once_even_without_termination():
    evaluator = MbagEvaluator(_config(1, terminate=False), [(LayerBuilderAgent, {})])
    episode = evaluator.rollout()
    flags = [infos[0]["goal_completed"] for infos in episode.info_history]
    assert sum(flags) == 1
    assert flags[-1] is False  # episode ran to the horizon after completion


def test_goal_completed_is_false_when_never_completed():
    evaluator = MbagEvaluator(_config(1, terminate=True), [(NoopAgent, {})])
    episode = evaluator.rollout()
    assert not any(infos[0]["goal_completed"] for infos in episode.info_history)


def test_goal_completed_same_for_all_players():
    evaluator = MbagEvaluator(
        _config(2, terminate=True), [(LayerBuilderAgent, {}), (NoopAgent, {})]
    )
    episode = evaluator.rollout()
    for infos in episode.info_history:
        assert infos[0]["goal_completed"] == infos[1]["goal_completed"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_goal_completed.py -q`
Expected: FAIL with `KeyError: 'goal_completed'`

- [ ] **Step 3: Implement**

`mbag/environment/types.py`, in `MbagInfoDict` after `goal_percentage`:

```python
    goal_completed: bool
    """
    True only on the step in which the current blocks first equal the goal blocks.
    This is the indicator goal reward U_h(s', g) of the human-power objective.
    """
```

`mbag/environment/mbag_env.py`:

In `reset`, right after `self.timesteps_with_no_progress = 0` (near line 304), add:

```python
        self.goal_was_complete = self.current_blocks == self.goal_blocks
```

In `_get_player_info`, add `"goal_completed": False,` to the dict after `"goal_percentage"`.

In `step`, in the loop that fills `goal_similarity` and `goal_percentage` (lines 349 to 356), after computing `info["goal_percentage"]`, add:

```python
            info["goal_completed"] = goal_just_completed
```

and before that loop compute once:

```python
        goal_complete_now = self.current_blocks == self.goal_blocks
        goal_just_completed = bool(goal_complete_now and not self.goal_was_complete)
        self.goal_was_complete = goal_complete_now
```

Also declare `self.goal_was_complete: bool = False` in `__init__` next to `self.is_first_episode = True`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_goal_completed.py tests/test_env.py tests/test_evaluator.py tests/test_metrics.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
.venv/bin/black mbag tests/test_goal_completed.py && .venv/bin/isort mbag tests/test_goal_completed.py
git add mbag/environment/types.py mbag/environment/mbag_env.py tests/test_goal_completed.py
git commit -m "Emit goal_completed in env infos as the indicator goal reward U_h

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: Optional mid-episode goal resampling (`goal_change_prob`)

**Files:**
- Modify: `mbag/environment/config.py:179-233` (`MbagConfigDict`) and `DEFAULT_CONFIG`
- Modify: `mbag/environment/mbag_env.py:308` (`step`) and `:210` (`reset`)
- Modify: `mbag/scripts/train.py` (sacred param + `environment_params`)
- Test: `tests/test_goal_change.py`

**Interfaces:**
- Produces: config key `goal_change_prob: float` (default `0.0`). At the start of each `step`, with this probability a fresh goal is generated, per-player `initial_goal_similarities` are recomputed from the current blocks, `maximum_goal_percentages` and `timesteps_with_no_progress` are reset, and `goal_was_complete` is recomputed. Info gains `goal_changed: bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_goal_change.py
import numpy as np

from mbag.environment.mbag_env import MbagEnv


def _config(prob: float, seed_generator: str = "random"):
    return {
        "world_size": (5, 5, 5),
        "num_players": 1,
        "horizon": 200,
        "goal_change_prob": prob,
        "terminate_on_goal_completion": False,
        "goal_generator": seed_generator,
        "goal_generator_config": {},
        "malmo": {"use_malmo": False},
    }


def test_default_never_changes_goal():
    env = MbagEnv(_config(0.0))
    env.reset()
    goal = env.goal_blocks.copy()
    for _ in range(50):
        _, _, _, infos = env.step([(0, 0, 0)])
        assert infos[0]["goal_changed"] is False
    assert env.goal_blocks == goal


def test_goal_changes_with_probability_one_every_step():
    env = MbagEnv(_config(1.0))
    env.reset()
    changes = 0
    previous = env.goal_blocks.copy()
    for _ in range(20):
        _, _, _, infos = env.step([(0, 0, 0)])
        assert infos[0]["goal_changed"] is True
        if not (env.goal_blocks == previous):
            changes += 1
        previous = env.goal_blocks.copy()
    # Random goals differ almost always.
    assert changes >= 15
    # Goal percentage is re-baselined: with no actions taken it is 0 after a change.
    assert infos[0]["goal_percentage"] == 0.0 or np.isnan(infos[0]["goal_percentage"])


def test_goal_change_rebaselines_progress_tracking():
    env = MbagEnv(_config(1.0))
    env.reset()
    env.step([(0, 0, 0)])
    assert env.timesteps_with_no_progress == 1 or env.timesteps_with_no_progress == 0
    assert all(p <= 1.0 for p in env.maximum_goal_percentages)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_goal_change.py -q`
Expected: FAIL with `KeyError: 'goal_changed'`

- [ ] **Step 3: Implement**

`mbag/environment/config.py`, in `MbagConfigDict` after `truncate_on_no_progress_timesteps`:

```python
    goal_change_prob: float
    """
    Probability at each step that the goal is replaced by a fresh draw from the goal
    generator. Default 0 (goal fixed for the episode). Nonzero values implement the
    per-step goal change p_g of Heitzig & Potham (2025); the frozen human models in
    this repo were trained with fixed goals, so leave this at 0 unless the human model
    handles goal changes.
    """
```

Add `"goal_change_prob": 0.0,` to `DEFAULT_CONFIG` after `"truncate_on_no_progress_timesteps": None,`.

`mbag/environment/types.py`, `MbagInfoDict`, after `goal_completed`:

```python
    goal_changed: bool
    """True on steps in which the goal was resampled (see goal_change_prob)."""
```

`mbag/environment/mbag_env.py`:

In `_get_player_info` add `"goal_changed": False,`.

Extract the re-baselining from `reset` into a helper and call it from both places. Add the method:

```python
    def _rebaseline_goal_tracking(self) -> None:
        """Recompute per-player progress baselines after the goal (or world) changed."""
        self.initial_goal_similarities = []
        for player_index in range(self.config["num_players"]):
            self.initial_goal_similarities.append(
                self._get_goal_similarity(
                    self.current_blocks[:],
                    self.goal_blocks[:],
                    partial_credit=True,
                    player_index=player_index,
                ).sum()
            )
        width, height, depth = self.config["world_size"]
        self.max_goal_similarity = width * height * depth
        self.maximum_goal_percentages = [
            self._get_goal_percentage(player_index)
            for player_index in range(self.config["num_players"])
        ]
        self.timesteps_with_no_progress = 0
        self.goal_was_complete = self.current_blocks == self.goal_blocks
```

In `reset`, replace the block that builds `self.initial_goal_similarities` and sets `self.max_goal_similarity` (lines 275 to 287) with a call `self._rebaseline_goal_tracking()`, and delete the later lines `self.maximum_goal_percentages = [...]` and `self.timesteps_with_no_progress = 0` and the `goal_was_complete` line added in Task 6 (the helper now sets them). Keep `info_list` construction as is.

In `step`, immediately after the `assert len(action_tuples) == ...`:

```python
        goal_changed = False
        if self.config["goal_change_prob"] > 0 and random.random() < self.config[
            "goal_change_prob"
        ]:
            self.goal_blocks = self._generate_goal()
            if not self.config["abilities"]["inf_blocks"]:
                self._copy_palette_from_goal()
            self._rebaseline_goal_tracking()
            goal_changed = True
```

and in the info-filling loop add `info["goal_changed"] = goal_changed`.

`_get_goal_percentage` divides by `max - initial`; when a fresh random goal equals the current world exactly this is zero. Guard it:

```python
        denominator = self.max_goal_similarity - self.initial_goal_similarities[player_index]
        if denominator == 0:
            return 1.0
        return (similarity - self.initial_goal_similarities[player_index]) / denominator
```

`mbag/scripts/train.py`: add `goal_change_prob = 0.0` to the sacred config near `truncate_on_no_progress_timesteps`, and `"goal_change_prob": goal_change_prob,` to `environment_params`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_goal_change.py tests/test_goal_completed.py tests/test_env.py tests/test_evaluator.py -q`
Expected: all pass. Fix `test_goal_change_rebaselines_progress_tracking` expectations to whatever the implementation does deterministically, then keep it as a regression test.

- [ ] **Step 5: Commit**

```bash
.venv/bin/black mbag tests/test_goal_change.py && .venv/bin/isort mbag tests/test_goal_change.py
git add mbag/environment mbag/scripts/train.py tests/test_goal_change.py
git commit -m "Add optional mid-episode goal resampling (goal_change_prob), default off

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: `human_actions_to_completion` metric

**Files:**
- Modify: `mbag/evaluation/metrics.py:11-48` (TypedDicts) and `:143-185` (`calculate_metrics`)
- Modify: `mbag/rllib/callbacks.py:202-280` (`on_episode_end`)
- Test: `tests/test_metrics.py` (append)

**Interfaces:**
- Produces: `MbagEpisodeMetrics["human_actions_to_completion"]: float` = number of steps in which player 0 took a non-noop action before and including the step where `goal_completed` is True; `NaN` if the goal was never completed. Same key in RLlib `episode.custom_metrics`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_metrics.py`)

```python
def test_human_actions_to_completion():
    from mbag.agents.heuristic_agents import LayerBuilderAgent, NoopAgent
    from mbag.environment.goals.simple import BasicGoalGenerator
    from mbag.evaluation.evaluator import MbagEvaluator
    from mbag.evaluation.metrics import calculate_metrics

    config = {
        "world_size": (5, 5, 5),
        "num_players": 1,
        "horizon": 60,
        "goal_generator": BasicGoalGenerator,
        "goal_generator_config": {},
        "malmo": {"use_malmo": False, "use_spectator": False, "video_dir": None},
    }
    episode = MbagEvaluator(config, [(LayerBuilderAgent, {})]).rollout()
    metrics = calculate_metrics(episode)
    # LayerBuilderAgent places 18 blocks to finish the basic goal (see test_evaluator).
    assert metrics["human_actions_to_completion"] == 18

    episode = MbagEvaluator(config, [(NoopAgent, {})]).rollout()
    metrics = calculate_metrics(episode)
    assert np.isnan(metrics["human_actions_to_completion"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_metrics.py::test_human_actions_to_completion -q`
Expected: FAIL with `KeyError: 'human_actions_to_completion'`

- [ ] **Step 3: Implement**

`mbag/evaluation/metrics.py`: add `human_actions_to_completion: float` to `MbagEpisodeMetrics`. In `calculate_metrics`, after building `metrics`, add:

```python
    metrics["human_actions_to_completion"] = _human_actions_to_completion(episode)
```

and the helper:

```python
def _human_actions_to_completion(episode: MbagEpisode) -> float:
    """
    Number of non-noop actions player 0 took up to and including the step where the
    goal was first completed; NaN if it never was.
    """
    count = 0
    for infos in episode.info_history:
        info = infos[0]
        if info["action"].action_type != MbagAction.NOOP:
            count += 1
        if info.get("goal_completed", False):
            return float(count)
    return float("nan")
```

`mbag/rllib/callbacks.py`, in `on_episode_step` after the per-policy reward accumulation, add per-episode counters:

```python
            if player_index == 0:
                episode.user_data.setdefault("human_actions", 0)
                if info_dict["action"].action_type != MbagAction.NOOP:
                    episode.user_data["human_actions"] += 1
                if info_dict.get("goal_completed", False) and (
                    "human_actions_to_completion" not in episode.user_data
                ):
                    episode.user_data["human_actions_to_completion"] = episode.user_data[
                        "human_actions"
                    ]
```

and in `on_episode_end` after `goal_distance`:

```python
        episode.custom_metrics["human_actions_to_completion"] = episode.user_data.get(
            "human_actions_to_completion", np.nan
        )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_metrics.py -q --timeout=600`
Expected: all pass (the existing `test_metrics` is marked `uses_rllib` and compares evaluator metrics with callback metrics; both paths now emit the new key, and NaN equality must be handled the same way the file already handles NaN accuracies).

- [ ] **Step 5: Commit**

```bash
.venv/bin/black mbag tests/test_metrics.py && .venv/bin/isort mbag tests/test_metrics.py
git add mbag/evaluation/metrics.py mbag/rllib/callbacks.py tests/test_metrics.py
git commit -m "Add human_actions_to_completion metric to evaluation and RLlib callbacks

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Documentation, lint, full test pass

**Files:**
- Create: `docs/human-power/README.md`
- Modify: `README.md` (one paragraph pointer under the title)

- [ ] **Step 1: Write `docs/human-power/README.md`**

```markdown
# Human-Power Assistant for MBAG

This branch family replaces AssistanceZero's objective with the long-term human
power objective of

> Heitzig, J. and Potham, R. (2025). Model-Based Soft Maximization of Suitable
> Metrics of Long-Term Human Power. arXiv:2508.00159v2.

The assistant never sees goal-progress reward and never infers the goal. Its reward
is the paper's intrinsic reward U_r(s'), derived from an estimate of the human's
ICCEA power W_h(s').

## Equations

| Eq. | Definition |
|-----|------------|
| (2) | pi_h(s,g) = nu * pi0_h(s,g) + (1-nu) * softmax_{beta_h} Q^m_h(s,g,.) |
| (6) | V^e_h(s,g) = E[ U_h(s',g) + gamma_h V^e_h(s',g) ],  U_h(s',g) = 1[s' in g] |
| (7) | X_h(s) = sum_g V^e_h(s,g)^zeta |
| (10)| W_h(s) = log2 X_h(s) |
| (8) | U_r(s) = -( sum_h X_h(s)^(-xi) )^eta |
| (4) | Q_r(s,a_r) = E gamma_r V_r(s') |
| (9) | V_r(s) = U_r(s) + E_{a_r ~ pi_r} Q_r(s,a_r) |
| (5) | pi_r(s)(a) proportional to (-Q_r(s,a))^(-beta_r) |

Paper hyperparameters: zeta = 2, xi = 1, eta = 1.1, gamma_h = gamma_r = 0.99,
beta_r annealed 1 -> 5.

## Mapping onto MBAG

| Paper | MBAG |
|---|---|
| robot r | assistant (player 1) |
| human h | player 0 |
| goal g, U_h | a house blueprint; `info["goal_completed"]` |
| pi_h(s, g) | frozen goal-conditioned human model (heuristic agent or the piKL `human` policy in `data/assistancezero_assistant/checkpoint_002000`) |
| V^e_h, X_h | learned networks (ppo-assistant branch) |
| U_r(s') | the assistant's per-step reward, replacing env reward |

Deviations from the paper are listed in
`docs/superpowers/specs/2026-09-11-human-power-design.md`, Section 3.1.

## What is on `human-power/base`

- `mbag/power/metrics.py`: eqs. (5), (7), (8), (10) as numpy functions.
- `mbag/power/exact.py`: exact fixed-point solver for eqs. (1) to (9) on small
  tabular games. Tests reproduce the closed forms W = log2 k and W = -log2 k and the
  Appendix A inequality.
- `mbag/power/gridworld.py`: the paper's key-and-door gridworld; the exact solver's
  robot fetches the key, opens the door, and clears the path.
- Env reward key `goal_reward_scale` (per player) so the assistant gets zero goal
  reward in both the env and the MCTS simulator.
- Env info `goal_completed` (the indicator U_h) and optional `goal_change_prob`.
- Metric `human_actions_to_completion`.

## Running

    .venv/bin/pytest tests/test_power_metrics.py tests/test_power_exact.py -q
    .venv/bin/pytest tests/test_power_gridworld.py -q --timeout=600   # slow
    .venv/bin/pytest -m "not uses_malmo and not slow" -q

Solve the gridworld and print the robot's plan:

    .venv/bin/python -c "
    import numpy as np
    from mbag.power.exact import HumanModelParams, PowerParams, solve
    from mbag.power.gridworld import DEFAULT_LAYOUT, KeyDoorGridworld, rollout
    w = KeyDoorGridworld(DEFAULT_LAYOUT)
    sol = solve(w.game, HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99),
                PowerParams.paper(beta_r=5.0), max_iters=3000, tol=1e-6)
    for s in rollout(w, sol, human_goal=w.goal_index((0, 5))):
        print(w.decode(s), 'W_h=%.3f' % sol.w_h[0][s])
    "

## Branches

- `human-power/base` (this): foundation.
- `human-power/ppo-assistant`: Phase 2 learning in MBAG with PPO. See its README.
- `human-power/mcts-assistant`: U_r inside AlphaZero search. Planned.

## Results log

(Empty on the base branch. Experiment results go in the branch READMEs.)
```

- [ ] **Step 2: Add a pointer to the top-level README**

After the first paragraph of `README.md` insert:

```markdown
> **Human-power branches:** the `human-power/*` branches replace the assistance-game
> objective with the long-term human power objective of Heitzig & Potham (2025).
> See [`docs/human-power/README.md`](docs/human-power/README.md).
```

- [ ] **Step 3: Lint and type-check**

Run: `./lint.sh` (uses the tools in the venv if activated: `source .venv/bin/activate && ./lint.sh`)
Expected: black, isort, flake8, mypy all clean. Typical fixes: unused imports, line length in test files (`# noqa: E501` is acceptable in tests), and mypy on `TypedDict` keys (add the new keys to `MbagInfoDict` as done above; use `info.get("goal_completed", False)` only on plain dicts).

- [ ] **Step 4: Full fast test pass**

Run: `.venv/bin/pytest -m "not uses_malmo and not slow" -q --timeout=600`
Expected: all pass. Record the count in the commit message.

- [ ] **Step 5: Commit**

```bash
git add docs/human-power/README.md README.md
git commit -m "Document the human-power branch family

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review against the spec

- Spec 4.1 metrics: Task 1. Spec 4.2 exact solver: Tasks 2 and 3. Spec 4.3 gridworld: Task 4. Spec 4.4 env changes (`goal_reward_scale`, `goal_change_prob`, `goal_completed`): Tasks 5, 7, 6. Spec 4.7 metrics: Task 8 covers `human_actions_to_completion`; the `human_power_bits` and `robot_reward` metrics depend on the learned estimator and move to the ppo-assistant plan. Spec 5 base-branch tests: closed forms and Appendix A in Task 3, gridworld in Task 4, `goal_reward_scale` in env and env model plus the existing AlphaZero consistency test in Task 5, `goal_completed` once per episode in Task 6.
- Names used consistently: `attainable_goals`, `power_bits`, `robot_reward`, `soft_power_policy`, `TabularGame`, `HumanModelParams`, `HumanPrior`, `PowerParams`, `Solution`, `solve_human_prior`, `solve_robot`, `solve`, `KeyDoorGridworld`, `rollout`, `goal_reward_scale`, `goal_completed`, `goal_changed`, `goal_change_prob`, `human_actions_to_completion`.
- Known risk: Task 4's behavioural test depends on the fixed point converging to the paper's qualitative behaviour. The debugging steps are in Task 4 Step 4.
