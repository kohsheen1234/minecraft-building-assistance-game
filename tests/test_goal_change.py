import numpy as np

from mbag.environment.mbag_env import MbagEnv


def _config(prob: float):
    return {
        "world_size": (5, 5, 5),
        "num_players": 1,
        "horizon": 200,
        "goal_change_prob": prob,
        "terminate_on_goal_completion": False,
        "goal_generator": "random",
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
    # Goal percentage is re-baselined against the current world after a change.
    assert infos[0]["goal_percentage"] == 0.0 or np.isnan(infos[0]["goal_percentage"])


def test_goal_change_rebaselines_progress_tracking():
    env = MbagEnv(_config(1.0))
    env.reset()
    _, _, _, infos = env.step([(0, 0, 0)])
    # After a change with no actions taken, progress tracking starts fresh: the
    # only no-progress step counted is the one just taken.
    assert env.timesteps_with_no_progress == 1
    assert env.maximum_goal_percentages == [0.0]
    assert infos[0]["goal_completed"] is False


def test_reset_info_has_goal_changed_false():
    env = MbagEnv(_config(1.0))
    _, infos = env.reset()
    assert infos[0]["goal_changed"] is False
