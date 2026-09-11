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


def test_reset_info_has_goal_completed_false():
    from mbag.environment.mbag_env import MbagEnv

    env = MbagEnv(_config(1, terminate=True))
    _, infos = env.reset()
    assert infos[0]["goal_completed"] is False
