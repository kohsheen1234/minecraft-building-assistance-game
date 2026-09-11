# Human-Power MCTS Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the AlphaZero-style assistant plan with the human-power reward: every MCTS node's reward becomes U_r(s') from the learned X_h network, and the value targets use the same reward, so search maximises long-term human power instead of goal progress.

**Architecture:** The planning env model gains an optional `power_reward_fn` hook that adds U_r(next_obs) to the node reward and records it in the node info so re-evaluations keep it. `MbagHumanPowerAlphaZeroPolicy` owns a `PowerEstimator`, installs the hook on every planning env it creates, replaces batch rewards with U_r(s') before value targets are computed, and ships the estimator inside its weights so rollout workers plan with the latest X_h. `MbagHumanPowerAlphaZero` trains the estimator on each fresh sample batch on the driver before the normal AlphaZero update.

**Tech Stack:** Ray RLlib 2.7.1, torch, existing `mbag.rllib.human_power.PowerEstimator`.

**Spec:** `docs/superpowers/specs/2026-09-11-human-power-design.md` (Section 6, `human-power/mcts-assistant`)

## Global Constraints

- Same as the ppo-assistant plan. Commits authored as `kohsheen1234`, no co-author trailer.
- The assistant's env reward is exactly zero (`goal_reward_scale=0`, `own_reward_prop=1`), `use_goal_predictor=False`, `goal_loss_coeff=0`. Without the hook every MCTS node reward would be 0.
- Config keys reuse the `power_*` names from the PPO branch.

---

## File map

| Path | Responsibility |
|---|---|
| `mbag/rllib/alpha_zero/planning.py` | `MbagEnvModel.power_reward_fn` hook; `power_reward` in info; applied in `step` and `get_reward_with_other_agent_actions` |
| `mbag/rllib/human_power.py` | `PowerEstimator.reward_for_obs(obs_tuple)` and `state_numpy()/load_state_numpy()` helpers |
| `mbag/rllib/human_power_alpha_zero.py` | `MbagHumanPowerAlphaZeroPolicy`, `MbagHumanPowerAlphaZeroConfig`, `MbagHumanPowerAlphaZero`, registration |
| `mbag/rllib/__init__.py`, `mbag/scripts/train.py` | import; policy class branch; `config.training(power_*)` |
| `mbag/scripts/train_configs.py` | `iccea_power_alphazero_assistant`, `iccea_power_alphazero_cpu_smoke` |
| `tests/test_human_power_alpha_zero.py` | env-model hook test; smoke training test |
| `docs/human-power/mcts-assistant.md` | branch README |

---

### Task 1: Power reward hook in the planning env model

**Files:** `mbag/rllib/alpha_zero/planning.py`, `mbag/rllib/human_power.py`, `tests/test_human_power_alpha_zero.py`

**Interfaces:**
- `MbagEnvModel.power_reward_fn: Optional[Callable[[MbagObs], float]]`, default `None`. In `step`, after the reward is computed: `power = float(self.power_reward_fn(obs)) if set else 0.0`; `info["power_reward"] = power`; `reward += power`. In `get_reward_with_other_agent_actions`: `reward += info.get("power_reward", 0.0)` at the end.
- `MbagEnvModelInfoDict.power_reward: float` (add to the TypedDict; `_get_player_info` does not produce it, so read with `.get`).
- `PowerEstimator.reward_for_obs(obs: MbagObs) -> float`: flattens the structured obs with the estimator's preprocessor and returns `rewards(flat[None])[0]`.
- `PowerEstimator.state_numpy() -> Dict[str, np.ndarray]` and `load_state_numpy(state)`.

- [ ] Test (env model): build `create_mbag_env_model` on a 5x5x5 basic goal with 2 players, set `env.power_reward_fn = lambda obs: -0.5`, step; assert `info["power_reward"] == -0.5` and `reward == env_reward_without_hook - 0.5`; assert `get_reward_with_other_agent_actions(...)` includes it when goal_logits are zeros.
- [ ] Implement, run, commit: `Add power_reward_fn hook to the MCTS env model`.

### Task 2: Policy, algorithm, wiring, smoke test

**Files:** `mbag/rllib/human_power_alpha_zero.py`, `mbag/rllib/__init__.py`, `mbag/scripts/train.py`, `tests/test_human_power_alpha_zero.py`

**Interfaces:**
- `MbagHumanPowerAlphaZeroPolicy(MbagAlphaZeroPolicy)`:
  - `__init__`: after super, build `self.power_estimator = PowerEstimator(observation_space, action_space, config["env_config"], **power kwargs from config)`; wrap `self.env_creator` so the returned env model has `power_reward_fn = self.power_estimator.reward_for_obs`; also set it on any env already in `self.envs`.
  - `get_weights()`: super weights plus `"power_estimator": self.power_estimator.state_numpy()`. `set_weights(weights)`: pop and load, then super.
  - `postprocess_trajectory`: add `POWER_GOAL_COMPLETED`, `POWER_NEXT_OBS`; set `sample_batch[REWARDS] = power_estimator.rewards(NEXT_OBS)`; then super (which computes `VALUE_TARGETS` from `REWARDS`).
  - `loss`: touch the two columns, then super.
- `MbagHumanPowerAlphaZeroConfig(MbagAlphaZeroConfig)`: same `power_*` attributes and `training()` override as the PPO config.
- `MbagHumanPowerAlphaZero(MbagAlphaZero)`: `_sample_and_add_to_replay_buffer` calls super, then trains `local_worker.get_policy(assistant).power_estimator` on the assistant batch (`OBS`, `POWER_NEXT_OBS`, `POWER_GOAL_COMPLETED`, `TERMINATEDS`) with `power_num_sgd_iter`/`power_minibatch_size`, and stores stats in `self._power_stats`. `training_step` calls super then merges `{"power/…"}` into `train_results[assistant]["custom_metrics"]`. Weights (including the estimator) sync at the end of super's `training_step`.
- Registration `register_trainable("MbagHumanPowerAlphaZero", …)`; import in `mbag/rllib/__init__.py`; in `train.py`: `if "HumanPowerAlphaZero" in run: policy_class = MbagHumanPowerAlphaZeroPolicy` before the `"AlphaZero" in run` branch, and `if isinstance(config, MbagHumanPowerAlphaZeroConfig): config.training(power_*)` after the AlphaZero block.

- [ ] Smoke test: `ex.run` with the `default_config` + `default_alpha_zero_config` values from `tests/test_train.py` inlined, `run="MbagHumanPowerAlphaZero"`, `num_players=2`, `heuristic="lowest_block"`, `mask_goal=True`, `use_goal_predictor=False`, `goal_loss_coeff=0`, `own_reward_prop=1`, `per_player_goal_reward_scale=[1, 0]`, `model="convolutional_alpha_zero"`, `num_simulations=5`, tiny nets. Assert `result["custom_metrics"]["assistant/goal_dependent_reward_mean"] == 0`, `result["assistant"]["custom_metrics"]["power/robot_reward_mean"] < 0` (or wherever RLlib surfaces it; check `result["info"]["learner"]` too), and `assistant/expected_reward_mean < 0` (MCTS saw negative power rewards).
- [ ] Implement, run tests, lint, commit: `Add MbagHumanPowerAlphaZero: plan with U_r(s') inside MCTS`.

### Task 3: Named configs and docs

- [ ] `iccea_power_alphazero_assistant`: copy of `assistancezero_assistant` with `run="MbagHumanPowerAlphaZero"`, `use_goal_predictor=False`, `goal_loss_coeff=0`, `prev_goal_kl_coeff=0`, `own_reward_prop=1`, `per_player_goal_reward_scale=[1, 0]`, `gamma=0.99`, `num_gpus=0`, `power_*` paper values, `experiment_tag=f"iccea_power_alphazero/{checkpoint_name}"`.
- [ ] `iccea_power_alphazero_cpu_smoke`: 6x6x6 random goals with the relaxed filters, `heuristic="lowest_block"`, `horizon=150`, `num_simulations=10`, `num_workers=2`, `sample_batch_size=1200`, `train_batch_size=32` sequences, `hidden_channels=32`, `num_layers=2`, `filter_size=3`, `interleave_lstm_every=-1`, small estimator, `num_training_iters=10`.
- [ ] Verify with `print_config`; run the smoke config for a few iterations; write `docs/human-power/mcts-assistant.md` (what, mapping, knobs, commands, results log, limitations: value targets in the replay buffer are computed with the estimator at sampling time; one X forward pass per expanded node).
- [ ] Lint, commit: `Add iccea_power_alphazero configs and document the MCTS human-power assistant`.
