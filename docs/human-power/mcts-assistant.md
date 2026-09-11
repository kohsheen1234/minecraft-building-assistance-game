# `human-power/mcts-assistant`: planning with U_r inside MCTS

This branch puts the human-power reward of Heitzig & Potham (2025),
arXiv:2508.00159v2, inside the AlphaZero-style MBAG assistant. Where
`human-power/ppo-assistant` learns a model-free policy from U_r(s'), this branch
lets the assistant **plan** with it: every Monte Carlo tree search node is rewarded
with U_r of the state it leads to, so search maximises long-term human power rather
than goal progress. It builds on `human-power/ppo-assistant` (it reuses
`PowerEstimator`) and `human-power/base`.

## What is implemented

| Piece | Where |
|---|---|
| `MbagEnvModel.power_reward_fn` hook: the planning env adds `power_reward_fn(next_obs)` to each step's reward and stores it as `info["power_reward"]`, so re-evaluating a node's reward with new goal logits keeps it | `mbag/rllib/alpha_zero/planning.py` |
| `MbagHumanPowerAlphaZeroPolicy`: owns a `PowerEstimator`; installs `power_estimator.reward_for_obs` on every planning env it creates; replaces batch rewards with U_r(s') before AlphaZero computes value targets; carries the estimator inside `get_weights`/`set_weights` so rollout workers plan with the latest X_h | `mbag/rllib/human_power_alpha_zero.py` |
| `MbagHumanPowerAlphaZero`: trains the estimator on each fresh sample batch (targets 13 and 14) right after sampling, logs `power/*` under the assistant's learner custom metrics; weights sync at the end of the normal training step | same |
| Named configs `iccea_power_alphazero_assistant` (paper-sized, from `assistancezero_assistant` with the goal predictor and goal reward off) and `iccea_power_alphazero_cpu_smoke` | `mbag/scripts/train_configs.py` |

Relation to the paper: the robot's planning quantities Q_r and V_r (eqs. 4 and 9)
are what MCTS estimates by search with node rewards U_r(s'); the soft policy
pi_r (eq. 5) is replaced by AlphaZero's visit-count policy with temperature. The
estimator side (V^e, X_h, U_r) is identical to the PPO branch.

## Config knobs

Same `power_*` parameters as the PPO branch (see `docs/human-power/ppo-assistant.md`).
The relevant AlphaZero settings: `use_goal_predictor=False` (nothing to infer),
`goal_loss_coeff=0`, `prev_goal_kl_coeff=0`, `own_reward_prop=1`,
`per_player_goal_reward_scale=[1, 0]`, `gamma=0.99`.

Logged under `info/learner/assistant/custom_metrics`: `power/human_power_bits_mean|min|max`,
`power/robot_reward_mean`, `power/v_e_loss`, `power/x_loss`, `power/v_e_mean`,
`power/x_mean`, `power/goal_completion_rate`. The existing
`assistant/expected_reward_mean` (MCTS's own reward estimate) should be negative,
which confirms search saw the power rewards.

## Running

CPU smoke run:

    TMPDIR=/tmp .venv/bin/python -m mbag.scripts.train with \
        iccea_power_alphazero_assistant iccea_power_alphazero_cpu_smoke num_training_iters=10

Paper-sized (GPU box):

    python -m mbag.scripts.train with iccea_power_alphazero_assistant \
        checkpoint_to_load_policies=path/to/human/model/checkpoint checkpoint_name=name

Tests:

    TMPDIR=/tmp .venv/bin/pytest tests/test_human_power_alpha_zero.py -q --timeout=900

## Results log

### 2026-09-11, CPU smoke run (this Mac, 10 cores, no GPU)

Config: `iccea_power_alphazero_assistant iccea_power_alphazero_cpu_smoke
num_training_iters=10`. 6x6x6 random goals, `lowest_block` heuristic human,
horizon 150, 10 MCTS simulations per step, assistant net 2 conv layers x 32
channels, estimator nets 2 x 16, no replay buffer. 10 iterations x 1200 env steps
in 6 minutes. Full log: `docs/human-power/results/2026-09-11-alphazero-cpu-smoke-progress.csv`.

| iter | episode len | goal % | human actions to completion | MCTS expected reward (episode) | W_h (bits) | U_r | V^e mean | assistant place / break / noop |
|-----:|------:|------:|------:|------:|------:|------:|------:|:--|
| 1 | 79.7 | 0.992 | 79.1 | -105.5 | -2.48 | -6.63 | 0.488 | 31.1 / 6.0 / 42.6 |
| 5 | 80.5 | 0.989 | 79.4 | -328.5 | -1.94 | -4.38 | 0.473 | 30.7 / 6.4 / 43.5 |
| 10 | 70.1 | 1.000 | 69.7 | -301.1 | -1.99 | -4.56 | 0.457 | 19.9 / 6.2 / 44.0 |

What this shows:

- Search really plans on U_r: the assistant's MCTS expected reward per episode is
  about -300, which is ~80 steps times U_r of about -4, and
  `assistant/goal_dependent_reward_mean` is exactly 0 throughout.
- As in the PPO smoke run, V^e has not left its initialisation after so few
  updates (completions are 0.6% of transitions), so W_h is nearly flat and the
  reward carries little state information yet. The drop in assistant placements
  (31 -> 20) and in human actions to completion (79 -> 70) is consistent with
  "interfere less" but is not yet attributable to the objective.
- Iteration 1's smaller expected reward (-105) is the first batch, collected before
  the estimator's first update with a freshly initialised X_h.

### Next experiments

1. 100+ iterations of the smoke config with `power_num_sgd_iter=16` to get V^e
   calibrated, then compare `human_actions_to_completion` against `MbagAlphaZero`
   with goal reward (the AssistanceZero baseline) and a noop assistant.
2. Re-enable the goal predictor head with `goal_loss_coeff > 0` but
   `goal_reward_scale = 0` as a representation-only auxiliary loss.
3. Paper-sized run on a GPU box with the BC or piKL human model and
   `num_simulations=100`.

## Known limitations

- **One X_h forward pass per expanded node.** With `num_simulations=100` and an
  11x10x10 world this dominates planning time; keep the estimator small or batch
  node evaluations (not done).
- **Stale value targets in the replay buffer.** Rewards are computed with the
  estimator at sampling time; older sequences in the replay buffer carry U_r
  values from an older X_h. The paper's Phase 2 has the same non-stationarity.
- **No goal predictor.** The AssistanceZero goal-prediction head is off because
  the assistant no longer needs goal-dependent rewards. Its representation benefit
  is lost; re-enabling it as a pure auxiliary loss is a possible ablation.
- **Sequence replay buffer needs a recurrent model.** With `interleave_lstm_every=-1`
  the model's `max_seq_len` is 0 and the repo's sequence replay buffer divides by
  it; the CPU smoke config therefore sets `use_replay_buffer=False`. Pre-existing.
- Same deviations as the PPO branch: mean-variant X_h, eps_X shift, human prior
  from data or heuristics.
