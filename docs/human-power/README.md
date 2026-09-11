# Human-Power Assistant for MBAG

This branch family replaces AssistanceZero's objective with the long-term human
power objective of

> Heitzig, J. and Potham, R. (2025). *Model-Based Soft Maximization of Suitable
> Metrics of Long-Term Human Power.* arXiv:2508.00159v2.

The assistant never sees goal-progress reward and never infers the goal. Its reward
is the paper's intrinsic reward U_r(s'), derived from an estimate of the human's
ICCEA power W_h(s'): the log of the effective number of goals the human can still
attain.

## Equations

Fully observed case, Table 1 of the paper. One human in MBAG, so the sum over h has
one term.

| Eq. | Definition |
|-----|------------|
| (1) | Q^m_h(s,g,a_h) = E_{a_-h ~ mu_-h} min_{a_r} E_{s'} [ U_h(s',g) + gamma_h V^m_h(s',g) ] |
| (2) | pi_h(s,g) = nu * pi0_h(s,g) + (1-nu) * softmax_{beta_h} Q^m_h(s,g,.) |
| (3) | V^m_h(s,g) = E_{a_h ~ pi_h} Q^m_h(s,g,a_h) |
| (4) | Q_r(s,a_r) = E_g E_{a_H ~ pi_H} E_{s'} gamma_r V_r(s') |
| (5) | pi_r(s)(a) proportional to (-Q_r(s,a))^(-beta_r) |
| (6) | V^e_h(s,g) = E[ U_h(s',g) + gamma_h V^e_h(s',g) ],  U_h(s',g) = 1[s' in g] |
| (7) | X_h(s) = sum_g V^e_h(s,g)^zeta |
| (8) | U_r(s) = -( sum_h X_h(s)^(-xi) )^eta |
| (9) | V_r(s) = U_r(s) + E_{a_r ~ pi_r} Q_r(s,a_r) |
| (10)| W_h(s) = log2 X_h(s) |

Phase 2 learning targets (Section 3.1 of the paper):

| Eq. | Target |
|-----|--------|
| (12) | v_r(s) <- gamma_r V_r(s') |
| (13) | v^e_h(s,g) <- U_h(s',g) + gamma_h V^e_h(s',g) |
| (14) | x_h(s) <- \|G_h\| V^e_h(s,g)^zeta, g the goal active in the sampled transition |

Paper hyperparameters (Tables 3 to 5): zeta = 2, xi = 1, eta = 1.1,
gamma_h = gamma_r = 0.99, beta_r annealed 1 -> 5, goal change probability
p_g = 0.01 per step. The per-transition robot reward is U_r(s') (Appendix F.3).

## Mapping onto MBAG

| Paper | MBAG |
|---|---|
| robot r | assistant, player index 1, policy id `assistant` |
| human h | player index 0, policy id `human` |
| state s | `MbagEnv` world state: blocks, inventories, positions |
| goal g, U_h(s', g) | a house blueprint; `info["goal_completed"]` is the indicator |
| pi_h(s, g) | a frozen goal-conditioned human model: a heuristic agent such as `LayerBuilderAgent`, or the piKL `human` policy stored in `data/assistancezero_assistant/checkpoint_002000` |
| V^e_h(s, g), X_h(s) | learned networks, `human-power/ppo-assistant` branch |
| U_r(s') | the assistant's per-step reward, replacing env reward |
| pi_r | entropy-regularised PPO (the paper's actor-critic variant) |

Deviations from the paper are listed in
`docs/superpowers/specs/2026-09-11-human-power-design.md`, Section 3.1. The exact
solver adds one knob the paper's learning procedure has implicitly:
`HumanModelParams.robot_epsilon` mixes the cautious min over robot actions in eq. (1)
with a uniform robot action, matching the paper's eps-greedy adversarial robot in
Phase 1 (final eps 0.01 in its Table 4). With eps = 0 a single blockable doorway
makes every goal behind it worth exactly zero to the human.

## What is on `human-power/base`

- `mbag/power/metrics.py`: eqs. (5), (7), (8), (10) as numpy functions and
  `PAPER_HYPERPARAMS`.
- `mbag/power/exact.py`: exact damped fixed-point solver for eqs. (1) to (9) on
  small tabular games (`TabularGame`, `solve_human_prior`, `solve_robot`, `solve`).
  Tests reproduce the closed forms W = log2 k (k certain goals) and W = -log2 k
  (k uniformly random outcomes, zeta = 2), the closed form of Appendix A, and the
  Appendix A inequality E^zeta <= W on random bandits.
- `mbag/power/gridworld.py`: the paper's key-and-door gridworld. The exact
  solver's robot walks to the key, picks it up, opens the door, and retreats to the
  far corner; human power rises from 2.83 to 3.28 bits along the way.
- Env reward key `goal_reward_scale` (per player) so the assistant gets zero goal
  reward in both the env and the MCTS simulator. Sacred params `goal_reward_scale`
  and `per_player_goal_reward_scale`.
- Env info `goal_completed` (the indicator U_h) and `goal_changed`, and the optional
  env flag `goal_change_prob` (default 0).
- Metric `human_actions_to_completion` in `calculate_metrics` and in the RLlib
  callbacks.

## Running

Fast tests:

    .venv/bin/pytest tests/test_power_metrics.py tests/test_power_exact.py \
        tests/test_power_gridworld.py tests/test_goal_reward_scale.py \
        tests/test_goal_completed.py tests/test_goal_change.py -q

Everything except Malmo (Ray needs a short temp dir on macOS):

    TMPDIR=/tmp .venv/bin/pytest -m "not uses_malmo and not slow" -q

Solve the gridworld and print the robot's plan:

    .venv/bin/python -c "
    import numpy as np
    from mbag.power.exact import HumanModelParams, PowerParams, solve
    from mbag.power.gridworld import DEFAULT_LAYOUT, KeyDoorGridworld, rollout
    w = KeyDoorGridworld(DEFAULT_LAYOUT)
    sol = solve(w.game,
                HumanModelParams(nu=0.0, beta_h=np.inf, gamma_h=0.99, robot_epsilon=0.01),
                PowerParams.paper(beta_r=5.0), max_iters=3000, tol=1e-6)
    for s in rollout(w, sol, human_goal=w.goal_index((0, 5))):
        print(w.render(s)); print('W_h = %.3f bits' % sol.w_h[0][s]); print()
    "

## Running in Minecraft (Malmo)

Training runs in the Python simulator (Malmo steps at 0.8 s, about 40,000x slower).
Everything human-facing runs in real Minecraft through Project Malmo: playing with a
trained human-power assistant, recording episodes, and logging the human's estimated
power W_h live.

Prerequisites: `pip install -e .[rllib,malmo]` and JDK 1.8.0_152 (on macOS Malmo
looks for exactly this version via `/usr/libexec/java_home -V`; Azul Zulu 8u152 works
and needs no Oracle login).

    # Terminal 1: two Minecraft instances, one for the human (sees the goal) and one
    # for the assistant. Ready when the log shows "CLIENT enter state: DORMANT".
    TMPDIR=/tmp .venv/bin/python -m malmo.minecraft launch --num_instances 2 --goal_visibility True False

    # Terminal 2: play with a human-power assistant and log W_h per step.
    TMPDIR=/tmp .venv/bin/python -m mbag.scripts.evaluate with human_with_power_assistant \
        assistant_run=MbagHumanPowerPPO \
        assistant_checkpoint=path/to/MbagHumanPowerPPO/.../checkpoint_000200

    # Or the MCTS variant (num_simulations controls planning time per step):
    TMPDIR=/tmp .venv/bin/python -m mbag.scripts.evaluate with human_with_power_assistant \
        assistant_run=MbagHumanPowerAlphaZero num_simulations=10 \
        assistant_checkpoint=path/to/MbagHumanPowerAlphaZero/.../checkpoint_000010

In game: Return enables movement, Fn+Backspace enables flying. The episode ends when
the house is complete or on Ctrl+C. The run directory then contains the usual episode
metrics plus `human_power_bits_first/last/mean/min/max/gain` and
`human_power_trajectories.json` with W_h at every step. `record_video=True` adds a
spectator instance (launch three) and records video.

The same `log_human_power=True` flag works in the simulator with a heuristic or
learned human, e.g.

    TMPDIR=/tmp .venv/bin/python -m mbag.scripts.evaluate with \
        runs='["lowest_block","MbagHumanPowerPPO"]' checkpoints='[None,"path/to/checkpoint"]' \
        policy_ids='[None,"assistant"]' algorithm_config_updates='[{},{}]' \
        num_episodes=10 use_malmo=False log_human_power=True out_dir=/tmp/eval

The Malmo tests (`pytest -m uses_malmo`) need the two instances running; the four in
`tests/test_human.py` wait for a real player and are not for unattended runs.

## Branches

- `human-power/base`: foundation (this document).
- `human-power/ppo-assistant`: Phase 2 learning in MBAG with PPO
  (`MbagHumanPowerPPO`, configs `iccea_power_assistant`, `iccea_power_cpu_smoke`).
  See `docs/human-power/ppo-assistant.md`.
- `human-power/mcts-assistant`: U_r inside AlphaZero search
  (`MbagHumanPowerAlphaZero`, configs `iccea_power_alphazero_assistant`,
  `iccea_power_alphazero_cpu_smoke`). See `docs/human-power/mcts-assistant.md`.

## Results log

Base branch, 2026-09-11, exact solver on the key-and-door gridworld
(576 states, paper hyperparameters, beta_r = 5, robot_epsilon = 0.01):

| step | robot | door | W_h (bits) |
|-----:|-------|------|-----------:|
| 0 | at start | closed | 2.831 |
| 2 | picks up key | closed | 2.988 |
| 4 | opens door | open | 3.176 |
| 6 | retreats left | open | 3.278 |
| 9 | in far corner, human through door | open | 3.241 |

Learned-estimator results belong in the branch READMEs.
