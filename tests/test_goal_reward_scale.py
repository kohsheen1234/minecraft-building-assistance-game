import copy

import numpy as np
import pytest

from mbag.agents.heuristic_agents import LayerBuilderAgent
from mbag.environment.config import DEFAULT_CONFIG
from mbag.environment.goals.simple import BasicGoalGenerator
from mbag.environment.mbag_env import MbagEnv
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
            {
                "rewards": {
                    "goal_reward_scale": assistant_scale,
                    "own_reward_prop": 1.0,
                }
            },
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
    human_goal_reward = sum(
        infos[0]["goal_dependent_reward"] for infos in episode.info_history
    )
    assert human_goal_reward > 0


def test_zero_scale_assistant_env_reward_is_exactly_zero_every_step():
    # With own_reward_prop=1 and goal_reward_scale=0 (and no noop/action rewards),
    # the assistant's env reward must be exactly 0 at every step.
    env = MbagEnv(_two_player_config(0.0))
    agents = [LayerBuilderAgent({}, env.config), LayerBuilderAgent({}, env.config)]
    for agent in agents:
        agent.reset()
    all_obs, all_infos = env.reset()
    done = False
    steps = 0
    while not done:
        env_state = env.get_state()
        actions = [
            agent.get_action_with_info_and_env_state(obs, info, env_state)
            for agent, obs, info in zip(agents, all_obs, all_infos)
        ]
        all_obs, rewards, dones, all_infos = env.step(actions)
        assert rewards[1] == 0.0
        done = dones[0]
        steps += 1
    assert steps > 1
    assert all_infos[0]["goal_percentage"] == pytest.approx(1.0)


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
