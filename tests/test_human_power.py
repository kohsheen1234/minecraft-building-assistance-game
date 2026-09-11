import copy
from typing import Any, Dict

import numpy as np
import pytest

try:
    import torch
except ImportError:
    pass

from mbag.environment.config import DEFAULT_CONFIG
from mbag.environment.mbag_env import MbagEnv


def _env_and_obs(num_steps=8, build=False):
    """
    A 2-player env stepped ``num_steps`` times (or until done when ``build``).
    Returns (env, obs, next_obs, goal_completed, terminated) for the assistant's
    (player 1) flattened observations. With ``build`` the human is a
    LayerBuilderAgent so states differ and the goal is eventually completed.
    """
    from ray.rllib.models.preprocessors import get_preprocessor

    from mbag.agents.heuristic_agents import LayerBuilderAgent

    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "world_size": (5, 5, 5),
            "num_players": 2,
            "players": [{}, {}],
            "horizon": 60,
            "goal_generator": "basic",
            "abilities": {"teleportation": True, "flying": True, "inf_blocks": True},
        }
    )
    env = MbagEnv(config)
    prep = get_preprocessor(env.observation_space)(env.observation_space)
    obs_list, info_list = env.reset()
    human = LayerBuilderAgent({}, env.config)
    human.reset()
    flat_obs, flat_next, completed, terminated = [], [], [], []
    for _ in range(num_steps if not build else config["horizon"]):
        prev = obs_list[1]
        if build:
            human_action = human.get_action_with_info_and_env_state(
                obs_list[0], info_list[0], env.get_state()
            )
        else:
            human_action = (0, 0, 0)
        obs_list, _, dones, info_list = env.step([human_action, (0, 0, 0)])
        flat_obs.append(prep.transform(prev))
        flat_next.append(prep.transform(obs_list[1]))
        completed.append(float(info_list[1]["goal_completed"]))
        terminated.append(float(dones[1]))
        if dones[0]:
            break
    return (
        env,
        np.stack(flat_obs),
        np.stack(flat_next),
        np.array(completed, dtype=np.float32),
        np.array(terminated, dtype=np.float32),
    )


def _estimator(env, **overrides):
    from gymnasium import spaces

    from mbag.agents.action_distributions import MbagActionDistribution
    from mbag.rllib.human_power import PowerEstimator

    kwargs: Dict[str, Any] = dict(
        zeta=2.0,
        xi=1.0,
        eta=1.1,
        gamma_h=0.99,
        x_epsilon=0.05,
        lr=1e-3,
        hidden_size=8,
        num_layers=1,
        filter_size=3,
        target_update_freq=1,
    )
    kwargs.update(overrides)
    num_flat_actions = MbagActionDistribution.get_action_mapping(env.config).shape[0]
    return PowerEstimator(
        env.observation_space,
        spaces.Discrete(num_flat_actions),
        env.config,
        **kwargs,
    )


@pytest.mark.uses_rllib
def test_power_estimator_shapes_and_ranges():
    env, obs, next_obs, _, _ = _env_and_obs()
    est = _estimator(env)
    v = est.v_e(torch.as_tensor(obs))
    x = est.x(torch.as_tensor(obs))
    assert v.shape == (len(obs),) and torch.all((v > 0) & (v < 1))
    assert x.shape == (len(obs),) and torch.all(x > 0)
    r = est.rewards(next_obs)
    assert r.shape == (len(obs),) and np.all(r < 0)
    # More power -> higher reward, by construction of eq. (8).
    bits = est.power_bits(next_obs)
    order = np.argsort(bits)
    assert np.all(np.diff(r[order]) >= -1e-6)


@pytest.mark.uses_rllib
def test_power_estimator_td_update_learns_indicator():
    env, obs, next_obs, goal_completed, terminated = _env_and_obs(build=True)
    assert goal_completed.sum() == 1 and goal_completed[-1] == 1 and terminated[-1] == 1
    # With gamma_h = 0 the TD target is exactly the indicator U.
    est = _estimator(env, gamma_h=0.0, lr=3e-3)
    first = est.update(
        obs, next_obs, goal_completed, terminated, num_sgd_iter=1, minibatch_size=64
    )
    for _ in range(80):
        last = est.update(
            obs,
            next_obs,
            goal_completed,
            terminated,
            num_sgd_iter=1,
            minibatch_size=64,
        )
    assert last["v_e_loss"] < first["v_e_loss"]
    assert last["goal_completion_rate"] == pytest.approx(1 / len(obs))
    v = est.v_e(torch.as_tensor(obs)).detach().numpy()
    # The state just before completion is the only one whose target is 1.
    assert v[-1] > v[:-1].max()
    # X tracks V^e ** zeta.
    x = est.x(torch.as_tensor(obs)).detach().numpy()
    assert np.allclose(x, v**2, atol=0.1)


@pytest.mark.uses_rllib
def test_power_estimator_state_roundtrip():
    env, obs, _, _, _ = _env_and_obs(num_steps=2)
    a = _estimator(env)
    b = _estimator(env)
    b.load_state_dict(a.state_dict())
    t = torch.as_tensor(obs)
    assert torch.allclose(a.x(t), b.x(t)) and torch.allclose(a.v_e(t), b.v_e(t))


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_human_power_ppo_smoke():
    import tempfile

    from mbag.scripts.train import ex

    result = ex.run(
        config_updates={
            "run": "MbagHumanPowerPPO",
            "log_dir": tempfile.mkdtemp(),
            "width": 6,
            "height": 6,
            "depth": 6,
            "horizon": 20,
            "goal_generator": "random",
            # Same goal-filter settings as tests/test_train.py::default_config; the
            # default size filters loop forever on tiny random goals.
            "min_width": 0,
            "min_height": 0,
            "min_depth": 0,
            "extract_largest_cc": True,
            "extract_largest_cc_connectivity": 6,
            "area_sample": False,
            "num_players": 2,
            "heuristic": "lowest_block",
            "policies_to_train": ["assistant"],
            "mask_goal": True,
            "goal_loss_coeff": 0,
            "use_extra_features": False,
            "own_reward_prop": 1,
            "per_player_goal_reward_scale": [1, 0],
            "gamma": 0.99,
            "num_workers": 0,
            "num_training_iters": 2,
            "train_batch_size": 200,
            "sgd_minibatch_size": 50,
            "rollout_fragment_length": 20,
            "vf_share_layers": True,
            "hidden_size": 16,
            "num_layers": 1,
            "filter_size": 3,
            "power_hidden_size": 8,
            "power_num_layers": 1,
            "power_minibatch_size": 50,
            "evaluation_interval": None,
        }
    ).result
    assert result is not None
    stats = result["info"]["learner"]["assistant"]["learner_stats"]
    assert stats["power/robot_reward_mean"] < 0
    assert "power/human_power_bits_mean" in stats
    assert stats["power/v_e_loss"] >= 0
    # The assistant's env reward is exactly zero; goal-dependent reward is off.
    assert result["custom_metrics"]["assistant/goal_dependent_reward_mean"] == 0


@pytest.mark.uses_rllib
def test_human_power_trajectory_on_evaluator_episode():
    from mbag.agents.heuristic_agents import LowestBlockAgent, NoopAgent
    from mbag.environment.config import MbagConfigDict
    from mbag.environment.goals.simple import BasicGoalGenerator
    from mbag.evaluation.evaluator import MbagEvaluator
    from mbag.rllib.human_power import human_power_trajectory

    config: MbagConfigDict = {
        "world_size": (5, 5, 5),
        "num_players": 2,
        "players": [{}, {}],
        "horizon": 40,
        "goal_generator": BasicGoalGenerator,
        "goal_generator_config": {},
        "malmo": {"use_malmo": False, "use_spectator": False, "video_dir": None},
    }
    episode = MbagEvaluator(config, [(LowestBlockAgent, {}), (NoopAgent, {})]).rollout()
    from mbag.environment.mbag_env import MbagEnv

    env = MbagEnv(config)
    est = _estimator(env)
    bits = human_power_trajectory(est, episode.obs_history, player_index=1)
    assert bits.shape == (len(episode.obs_history),)
    assert np.all(np.isfinite(bits))
    # W_h = log2(X + eps_X) with X > 0, so it is bounded below by log2(eps_X).
    assert np.all(bits > np.log2(0.05) - 1e-6)
    assert est.bits_for_obs(episode.obs_history[0][1]) == pytest.approx(bits[0])
