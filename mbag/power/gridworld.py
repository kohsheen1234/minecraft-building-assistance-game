"""
The key-and-door gridworld from Section 4.2 of Heitzig & Potham (2025), encoded as a
TabularGame so the exact solver can be checked against the paper's qualitative
result: the robot fetches the key, unlocks the door, and steps out of the way.

Layout characters: ``.`` open, ``#`` wall, ``D`` door (closed at start), ``K`` key on
the floor, ``H`` human start, ``R`` robot start. Agents move one cell in four
directions; moving into a wall, a closed door, or the other agent leaves the mover in
place. The human moves first against the robot's current position, then the robot
moves against the human's new position (so swapping is impossible). The robot's
``interact`` action picks up the key when standing on it, or opens the door when
holding the key and standing on a 4-neighbour of the door. Every open cell is a
potential human goal: g_c = {states with the human at c}.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .exact import Solution, TabularGame

Cell = Tuple[int, int]  # (row, col)

DEFAULT_LAYOUT: List[str] = [
    "H..D..",
    ".RK#..",
]

MOVES: Dict[str, Tuple[int, int]] = {
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
        self.cell_index: Dict[Cell, int] = {
            cell: i for i, cell in enumerate(self.cells)
        }
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

    @staticmethod
    def _adjacent(a: Cell, b: Cell) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def step(self, s: int, robot_action: str, human_action: str) -> int:
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
        S = self.num_cells * self.num_cells * 4  # noqa: N806
        A_r = len(self.ROBOT_ACTIONS)  # noqa: N806
        A_h = len(self.HUMAN_ACTIONS)  # noqa: N806
        next_states = np.zeros((S, A_r, A_h, 1), dtype=int)
        next_probs = np.ones((S, A_r, A_h, 1), dtype=float)
        terminal = np.zeros(S, dtype=bool)
        goal_sets = np.zeros((self.num_cells, S), dtype=bool)
        for s in range(S):
            human, robot, _, _ = self.decode(s)
            goal_sets[self.cell_index[human], s] = True
            if human == robot:
                # Unreachable overlapping configuration; make it absorbing.
                next_states[s, :, :, 0] = s
                terminal[s] = True
                continue
            for a_r, ra in enumerate(self.ROBOT_ACTIONS):
                for a_h, ha in enumerate(self.HUMAN_ACTIONS):
                    next_states[s, a_r, a_h, 0] = self.step(s, ra, ha)
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

    def render(self, s: int) -> str:
        human, robot, key_held, door_open = self.decode(s)
        rows = []
        for r in range(self.rows):
            chars = []
            for c in range(self.cols):
                cell = (r, c)
                if cell == human:
                    chars.append("H")
                elif cell == robot:
                    chars.append("R")
                elif self.layout[r][c] == "#":
                    chars.append("#")
                elif cell == self.door_cell:
                    chars.append("_" if door_open else "D")
                elif cell == self.key_cell and not key_held:
                    chars.append("K")
                else:
                    chars.append(".")
            rows.append("".join(chars))
        return "\n".join(rows)


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
