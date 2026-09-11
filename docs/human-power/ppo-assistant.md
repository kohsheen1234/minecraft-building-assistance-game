# `human-power/ppo-assistant`: Phase 2 in MBAG with PPO

This branch trains an MBAG assistant whose **only** reward is the intrinsic
human-power reward U_r(s') of Heitzig & Potham (2025), arXiv:2508.00159v2. The
assistant never receives goal-progress reward and never infers the goal. It builds
on `human-power/base` (see `docs/human-power/README.md` for the equations and the
paper-to-MBAG mapping).

## What is implemented

`mbag/rllib/human_power.py`:

| Paper | Code |
|---|---|
| V^e_h(s, g), eq. (6), TD target (13): v^e <- U_h(s',g) + gamma_h V^e_h(s',g) | `PowerEstimator.v_e_model`, trained in `PowerEstimator.update` with MSE against `u + gamma_h (1-u)(1-done) V^e_target(s')`. `(1-u)` is the goal-absorbing continuation, `V^e_target` a periodically synced copy (paper Appendix F uses target networks) |
| X_h(s), eq. (7), target (14): x <- \|G_h\| V^e_h(s,g)^zeta | `PowerEstimator.x_model`, MSE against `V^e(s,g)^zeta` (sampled-goal **mean** variant; the `\|G_h\|` factor is a constant rescaling the paper's design makes irrelevant to pi_r) |
| U_r(s) = -(sum_h X_h^-xi)^eta, eq. (8), one human | `PowerEstimator.rewards`: `robot_reward(X_h(s') + eps_X, xi, eta)` with eps_X from Appendix H.1 |
| U_h(s', g) = 1[s' in g] | `info["goal_completed"]` lifted into the batch column `power_goal_completed` by `MbagHumanPowerPPOTorchPolicy.postprocess_trajectory` |
| robot reward is U_r(s') of the next state (Appendix F.3) | `MbagHumanPowerPPO._update_power_and_replace_rewards` overwrites `SampleBatch.REWARDS` for the assistant with U_r(next_obs) and recomputes GAE |
| pi_r: entropy-regularised actor-critic (Section 3.1, AC variant) | PPO with the repo's entropy schedule; `gamma = gamma_r = 0.99` |
| Phase 1 human prior pi_h(s, g) | frozen goal-conditioned human: `lowest_block` heuristic (CPU smoke) or the sample BC checkpoint (11x10x10) |

Everything else is unchanged `MbagPPO`. The assistant's env reward is forced to
exactly zero by `own_reward_prop=1`, `per_player_goal_reward_scale=[1, 0]` and no
noop/action rewards for player 1, so the power reward is the only signal.

## Config knobs

| Sacred param | Default | Meaning |
|---|---|---|
| `power_zeta` | 2.0 | eq. (7) exponent, reliability preference |
| `power_xi` | 1.0 | eq. (8) inter-human inequality aversion (one human here) |
| `power_eta` | 1.1 | eq. (8) intertemporal inequality aversion |
| `power_gamma_h` | 0.99 | discount inside the V^e TD target |
| `power_x_epsilon` | 0.05 | eps_X added to X_h before the power; bounds U_r below by -(eps_X)^(-xi eta) |
| `power_lr`, `power_num_sgd_iter`, `power_minibatch_size` | 1e-3, 4, 256 | estimator optimisation per training step |
| `power_target_update_freq` | 1 | training steps between target-network syncs |
| `power_hidden_size`, `power_num_layers`, `power_filter_size` | 32, 2, 3 | estimator backbone (`mbag_convolutional_model`) |
| `power_assistant_policy_id` | `assistant` | which policy's rewards to replace |

Logged per training iteration under `info/learner/assistant/learner_stats`:
`power/human_power_bits_mean|min|max` (W_h of visited next states),
`power/robot_reward_mean` (U_r), `power/v_e_loss`, `power/x_loss`, `power/v_e_mean`,
`power/x_mean`, `power/goal_completion_rate`. Episode-level `goal_percentage` and
`human_actions_to_completion` come from the existing callbacks.

## Running

CPU smoke run (6x6x6 random goals, `lowest_block` heuristic human, tiny nets):

    TMPDIR=/tmp .venv/bin/python -m mbag.scripts.train with \
        iccea_power_assistant iccea_power_cpu_smoke num_training_iters=20

Paper-sized world with the sample BC human model, CPU-sized batches:

    TMPDIR=/tmp .venv/bin/python -m mbag.scripts.train with iccea_power_assistant \
        checkpoint_to_load_policies=data/logs/BC/sample_human_models/inf_blocks_True_teleportation_False/2024-04-10_18-51-43/1/checkpoint_000100 \
        checkpoint_name=sample_bc_human \
        num_workers=2 num_envs_per_worker=1 train_batch_size=3000 sgd_minibatch_size=300 \
        rollout_fragment_length=300 max_seq_len=300 horizon=300 randomize_first_episode_length=False \
        hidden_size=32 hidden_channels=32 num_layers=4 power_hidden_size=16 power_num_layers=2 \
        power_minibatch_size=300 entropy_coeff_horizon=30000 num_training_iters=10

Full paper-scale run (needs a GPU box; the config is the same, just drop the
overrides):

    python -m mbag.scripts.train with iccea_power_assistant \
        checkpoint_to_load_policies=path/to/human/model/checkpoint checkpoint_name=name

Tests:

    TMPDIR=/tmp .venv/bin/pytest tests/test_human_power.py -q --timeout=900

## Results log

### 2026-09-11, CPU smoke run (this Mac, 10 cores, no GPU)

Config: `iccea_power_assistant iccea_power_cpu_smoke train_batch_size=2400
num_training_iters=20`. 6x6x6 random goals, `lowest_block` heuristic human,
horizon 150, assistant net 2 conv layers x 32 channels, estimator nets 2 x 16.
20 iterations x 2400 env steps in 16 minutes. Full log:
`docs/human-power/results/2026-09-11-cpu-smoke-progress.csv`.

| iter | episode len | goal % | human actions to completion | W_h (bits) | U_r | V^e mean | assistant place / break / noop |
|-----:|------:|------:|------:|------:|------:|------:|:--|
| 1 | 105.1 | 0.970 | 107.5 | -1.91 | -4.29 | 0.491 | 57.0 / 5.8 / 42.3 |
| 5 | 100.9 | 1.000 | 100.0 | -1.92 | -4.34 | 0.477 | 41.8 / 15.8 / 43.3 |
| 11 | 77.5 | 1.000 | 77.3 | -2.12 | -5.05 | 0.447 | 27.8 / 7.1 / 42.6 |
| 15 | 88.5 | 0.950 | 84.9 | -2.02 | -4.67 | 0.448 | 38.2 / 5.3 / 45.0 |
| 20 | 93.0 | 0.843 | 87.2 | -2.08 | -4.96 | 0.439 | 43.7 / 4.4 / 45.0 |

Reference: the `lowest_block` human alone (noop assistant) finishes these goals in
28 to 53 steps.

What this shows:

- The pipeline is correct end to end. `assistant/goal_dependent_reward_mean` is
  exactly 0 at every iteration, the assistant's only reward is U_r(s') and it is
  negative everywhere as eq. (8) requires, and the two estimator nets train.
- Twenty iterations is far too short for V^e to calibrate: completions are 0.6% of
  transitions and V^e sits near its sigmoid initialisation (0.44 to 0.49), so W_h
  is essentially flat around -2 bits and the reward carries little state
  information yet. Nothing here should be read as a result about the objective.
- Behaviourally the assistant first reduced its interference (human actions to
  completion 107 -> 77 by iteration 11, mostly by placing fewer blocks), then
  drifted back to placing more blocks and completion rates fell. With an
  uninformative reward this is entropy-schedule drift, not learning.

### 2026-09-11, BC human model at 11x10x10, CPU (stopped)

Config: the second command above (3000 env steps per iteration, horizon 300).
Throughput on this Mac: **12.3 minutes per iteration**. In the first iteration the
BC human reached 4.8% goal completion on average and completed **zero** houses
within 300 steps (the paper's horizon is 1500), so every V^e target was 0 and the
estimator had no positive signal. Stopped after one iteration. Conclusion: the
paper-scale experiment needs the full 1500-step horizon and tens of millions of
env steps, i.e. the GPU box configuration (`num_workers=8`, `train_batch_size=32704`),
and is not a CPU job.

### 2026-09-11, CPU smoke config for 200 iterations

Config: `iccea_power_assistant iccea_power_cpu_smoke num_training_iters=200
power_num_sgd_iter=16` (same 6x6x6 world, `lowest_block` human, horizon 150).
200 iterations x 2400 env steps in 3.9 hours. Full log and checkpoints at 50, 125,
200 iterations: `docs/human-power/results/2026-09-11-cpu-smoke-200iter-progress.csv`.
Reference: the human alone finishes in 28 to 53 steps.

| iter | episode len | goal % | human actions to completion | W_h mean (bits) | W_h min / max | U_r | V^e mean | assistant place / break / noop | policy entropy |
|-----:|------:|------:|------:|------:|:--|------:|------:|:--|------:|
| 1 | 104.4 | 0.969 | 106.5 | -1.77 | -1.79 / -1.74 | -3.85 | 0.489 | 56.4 / 5.9 / 42.1 | 4.50 |
| 25 | 64.1 | 0.980 | 62.3 | -2.30 | -3.55 / -0.35 | -5.90 | 0.400 | 15.8 / 2.5 / 45.9 | 2.21 |
| 50 | 75.1 | 0.959 | 72.8 | -2.78 | -3.31 / -0.38 | -8.60 | 0.314 | 27.1 / 4.0 / 44.0 | 3.19 |
| 100 | 52.6 | 0.988 | 51.0 | -3.26 | -4.05 / -0.83 | -12.4 | 0.227 | 6.7 / 0.7 / 45.2 | 1.28 |
| 150 | 45.0 | 1.000 | 45.0 | -3.33 | -4.18 / -0.15 | -13.5 | 0.208 | 0.2 / 0.0 / 44.8 | 0.17 |
| 175 | 45.9 | 1.000 | 45.9 | -3.41 | -4.05 / -0.11 | -14.0 | 0.226 | 1.1 / 0.2 / 44.6 | 0.04 |
| 200 | 67.6 | 0.978 | 66.2 | -3.39 | -3.91 / -0.24 | -14.1 | 0.215 | 21.0 / 2.5 / 44.1 | 2.85 |

What this shows:

- **The estimator now separates states.** The spread of W_h within a batch grew
  from 0.04 bits at iteration 1 to about 4 bits by iteration 100 (min -4.05, max
  -0.83), so U_r carries real state information. Mean V^e fell from its sigmoid
  initialisation (0.49) to about 0.21 as discounting was learned; the mean W_h
  therefore drifts down over training, which is calibration of the estimate, not a
  loss of human power.
- **The assistant converged to non-interference.** By iteration 150 it placed 0.2
  blocks and broke 0.03 per episode, and the human finished in 45 actions, the same
  as building alone. In this smoke setting the assistant has no way to raise the
  human's power: the human has infinite blocks, can teleport, and knows the goal, so
  the power-maximising action is to stay out of the way. That is the paper's
  prediction for this situation, and it is what the goal-inference assistant
  notoriously fails to do early in training.
- **Late drift.** At iteration 200 placements rose back to 21 and entropy to 2.85.
  The PPO entropy coefficient schedule ends at iteration ~8 (20,000 env steps) and
  the reward scale is non-stationary because the estimator keeps changing; both are
  candidates. Longer runs should use a fixed small entropy coefficient and EMPO's
  z-space transform to stabilise reward scale.
- Whether the assistant can *increase* power, not merely avoid decreasing it, needs
  a setting where it can: finite blocks (it can gather and hand over resources),
  no teleportation (it can clear paths), or partial-house goals. Those are the
  paper-scale configs.

### Next experiments

1. Rerun the 200-iteration config with `entropy_coeff_end` held small and a longer
   `entropy_coeff_horizon`, to check whether the late drift disappears.
2. Ablate `power_x_epsilon` (0.05 vs 1.0, paper Appendix H.1 discusses eps_X = 1
   with a larger xi) since it sets the reward scale PPO sees.
3. Compare against `ppo_assistant` (goal-reward baseline) and a noop assistant on
   `human_actions_to_completion` with the same seeds.

## Known limitations and deviations

- **X_h is a mean over sampled goals**, not the paper's sum, and `eps_X` is added
  before the power. Both are constant rescalings/shifts that leave the argmax of
  pi_r unchanged in the paper's analysis, but they do change reward magnitudes.
- **Human prior from data or heuristics**, not from eqs. (1) to (3). The BC model
  was trained with fixed goals, so `goal_change_prob` stays 0.
- **`layer_builder` heuristic does not work with PPO** in this repo (its agent
  state confuses RLlib's collector); use `lowest_block`. Pre-existing.
- **`hidden_channels` must be set together with `hidden_size`** in overrides; the
  train script defaults them separately.
- No MCTS variant yet; that is `human-power/mcts-assistant`.
- No offline `evaluate.py power_checkpoint` option yet; the power estimator is
  saved inside the algorithm checkpoint (`MbagHumanPowerPPO.__getstate__`).
