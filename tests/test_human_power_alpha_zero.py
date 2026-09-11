import copy

import numpy as np
import pytest

from mbag.environment.config import DEFAULT_CONFIG


def _env_model():
    from mbag.rllib.alpha_zero.planning import create_mbag_env_model

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["world_size"] = (5, 5, 5)
    config["goal_generator"] = "basic"
    config["num_players"] = 2
    config["players"] = [
        {},
        {"rewards": {"goal_reward_scale": 0.0, "own_reward_prop": 1.0}},
    ]
    return create_mbag_env_model(config, player_index=1)


@pytest.mark.uses_rllib
def test_power_reward_fn_adds_to_node_reward():
    env = _env_model()
    env.reset()
    state = env.get_state()

    _, plain_reward, _, _, plain_info = env.step(0)
    assert plain_reward == 0.0
    assert plain_info.get("power_reward", 0.0) == 0.0

    env.set_state(state)
    env.power_reward_fn = lambda obs: -0.5
    _, reward, _, _, info = env.step(0)
    assert info["power_reward"] == -0.5
    assert reward == pytest.approx(plain_reward - 0.5)

    # Re-evaluating the node reward with predicted goal logits keeps the power term.
    obs = env.last_obs_dict[env.agent_id]
    world_obs = obs[0]
    goal_logits = np.zeros((256,) + world_obs.shape[1:], dtype=np.float32)
    recomputed = env.get_reward_with_other_agent_actions(obs, info, goal_logits)
    assert recomputed == pytest.approx(-0.5)


@pytest.mark.uses_rllib
def test_power_estimator_reward_for_structured_obs():
    from gymnasium import spaces

    from mbag.agents.action_distributions import MbagActionDistribution
    from mbag.environment.mbag_env import MbagEnv
    from mbag.rllib.human_power import PowerEstimator

    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update({"world_size": (5, 5, 5), "num_players": 2, "players": [{}, {}]})
    env = MbagEnv(config)
    obs_list, _ = env.reset()
    num_flat_actions = MbagActionDistribution.get_action_mapping(env.config).shape[0]
    est = PowerEstimator(
        env.observation_space,
        spaces.Discrete(num_flat_actions),
        env.config,
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
    r = est.reward_for_obs(obs_list[1])
    assert isinstance(r, float) and r < 0

    clone = PowerEstimator(
        env.observation_space,
        spaces.Discrete(num_flat_actions),
        env.config,
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
    clone.load_state_numpy(est.state_numpy())
    assert clone.reward_for_obs(obs_list[1]) == pytest.approx(r)
