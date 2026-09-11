import numpy as np
import pytest

from mbag.power.exact import HumanModelParams, TabularGame, solve_human_prior


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
    prior = solve_human_prior(
        game, HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99)
    )
    # eq. (1) takes min over robot actions: goal 0 is unattainable in the worst case.
    np.testing.assert_allclose(prior.q_m[0][0, 0, :], [0.0, 0.0])
    # For goal 1, human action 0 only succeeds if the robot sabotages; min says 0.
    np.testing.assert_allclose(prior.q_m[0][1, 0, :], [0.0, 1.0])


def test_masked_human_actions_get_zero_probability():
    k = 3
    game = make_bandit(k, np.eye(k))
    game.human_action_masks[0][0, 2] = False
    prior = solve_human_prior(game, HumanModelParams(nu=0.0, beta_h=1.0, gamma_h=0.99))
    assert prior.pi_h[0][:, 0, 2].max() == 0.0
    np.testing.assert_allclose(prior.pi_h[0][:, 0, :].sum(-1), 1.0)


from mbag.power.exact import PowerParams, solve  # noqa: E402


def test_deterministic_bandit_power_is_log2_k():
    for k in [1, 2, 4, 8]:
        game = make_bandit(k, np.eye(k))
        sol = solve(
            game,
            HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99),
            PowerParams.paper(beta_r=5.0),
        )
        assert sol.converged
        assert sol.x_h[0][0] == pytest.approx(k)
        assert sol.w_h[0][0] == pytest.approx(np.log2(k))
        assert sol.u_r[0] == pytest.approx(-(k**-1.1))
        # Terminal continuation is zero, so V_r(root) = U_r(root).
        assert sol.v_r[0] == pytest.approx(sol.u_r[0])
        assert np.all(sol.v_r[1:] == 0.0)


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
    log_probs = np.where(probs > 0, np.log2(np.where(probs > 0, probs, 1.0)), 0.0)
    for pi in rng.dirichlet(np.ones(k) * 0.3, size=4000):
        p_s = pi @ probs
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(probs > 0, log_probs - np.log2(p_s)[None, :], 0.0)
        mi = float((pi[:, None] * probs * log_ratio).sum())
        cond_h = float(-(pi[:, None] * probs * log_probs).sum())
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
    assert sol.converged
    assert sol.w_h[0][1] == pytest.approx(1.0)  # unlocked: 2 goals -> 1 bit
    assert sol.w_h[0][2] == pytest.approx(0.0)  # locked: 1 goal -> 0 bits
    assert sol.q_r[0, 0] > sol.q_r[0, 1]
    assert sol.pi_r[0, 0] > 0.9
    # Soft policy still explores.
    assert sol.pi_r[0, 1] > 0.0
