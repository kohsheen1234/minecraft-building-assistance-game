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
