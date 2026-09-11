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


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_human_power_alpha_zero_smoke():
    import tempfile

    from mbag.scripts.train import ex

    result = ex.run(
        config_updates={
            "run": "MbagHumanPowerAlphaZero",
            "log_dir": tempfile.mkdtemp(),
            "width": 6,
            "height": 6,
            "depth": 6,
            "horizon": 10,
            "goal_generator": "random",
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
            "use_extra_features": False,
            "use_goal_predictor": False,
            "goal_loss_coeff": 0,
            "own_reward_prop": 1,
            "per_player_goal_reward_scale": [1, 0],
            "gamma": 0.99,
            "model": "convolutional_alpha_zero",
            "vf_share_layers": True,
            "hidden_size": 16,
            "hidden_channels": 16,
            "num_layers": 1,
            "filter_size": 3,
            "interleave_lstm_every": -1,
            "num_simulations": 3,
            "use_replay_buffer": False,
            "num_workers": 0,
            "num_training_iters": 2,
            "sample_batch_size": 60,
            "train_batch_size": 1,
            "sgd_minibatch_size": 20,
            "num_sgd_iter": 1,
            "rollout_fragment_length": 10,
            "power_hidden_size": 8,
            "power_num_layers": 1,
            "power_minibatch_size": 30,
            "evaluation_interval": None,
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["assistant/goal_dependent_reward_mean"] == 0
    power_stats = result["info"]["learner"]["assistant"]["custom_metrics"]
    assert power_stats["power/robot_reward_mean"] < 0
    assert power_stats["power/v_e_loss"] >= 0
    # MCTS saw the (negative) power rewards while planning: every expected reward
    # metric for the assistant is negative.
    expected_keys = [
        key
        for key in result["custom_metrics"]
        if key.startswith("assistant/expected_reward")
    ]
    assert expected_keys
    assert all(result["custom_metrics"][key] < 0 for key in expected_keys)
