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


@pytest.fixture(scope="module")
def solved_world():
    world = KeyDoorGridworld(DEFAULT_LAYOUT)
    sol = solve(
        world.game,
        # robot_epsilon matches the paper's Phase 1 eps-greedy robot (final eps 0.01).
        HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99, robot_epsilon=0.01),
        PowerParams.paper(beta_r=5.0),
        max_iters=3000,
        tol=1e-6,
    )
    return world, sol


@pytest.mark.slow
def test_open_door_raises_human_power(solved_world):
    world, sol = solved_world
    closed = world.encode((0, 0), (1, 0), True, False)
    opened = world.encode((0, 0), (1, 0), True, True)
    assert sol.w_h[0][opened] > sol.w_h[0][closed]


@pytest.mark.slow
def test_robot_fetches_key_opens_door_and_clears_path(solved_world):
    world, sol = solved_world
    far_goal = world.goal_index((0, 5))
    states = rollout(world, sol, human_goal=far_goal, max_steps=30, greedy=True)
    decoded = [world.decode(s) for s in states]
    assert any(d[2] for d in decoded), "robot never picked up the key"
    assert any(d[3] for d in decoded), "robot never opened the door"
    assert decoded[-1][0] == (0, 5), "human did not reach the far side"
    # Once the door is open the robot must not be standing in the doorway.
    door = world.door_cell
    assert all(d[1] != door for d in decoded if d[3])
