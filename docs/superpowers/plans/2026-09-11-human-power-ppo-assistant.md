# Human-Power PPO Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an MBAG assistant with PPO whose only reward is the paper's intrinsic human-power reward U_r(s'), estimated online by two learned networks (V^e and X_h) exactly as in Phase 2 of Heitzig & Potham (2025), Section 3.1 and Appendix F.3.

**Architecture:** A `PowerEstimator` holds two small convolutional value networks built with the repo's `mbag_convolutional_model` (one sees the goal, one does not), trains them from the assistant's sample batch with targets (13) and (14), and produces U_r(s') per transition. `MbagHumanPowerPPO` subclasses `MbagPPO`, runs the estimator update and overwrites the assistant's rewards before the PPO update, following the GAIL reward-replacement pattern. A thin policy subclass lifts `goal_completed` and the next observation into batch columns.

**Tech Stack:** Ray RLlib 2.7.1, torch 2.8, sacred, existing `mbag.power.metrics`.

**Spec:** `docs/superpowers/specs/2026-09-11-human-power-design.md` (Sections 3, 4.5, 4.6, 4.7)

## Global Constraints

- Everything in the base plan's Global Constraints applies.
- Ray-based tests and scripts run with `TMPDIR=/tmp` on macOS.
- The assistant policy must receive zero env reward: `own_reward_prop=1`, `goal_reward_scale=0` for player 1, no noop/action rewards for player 1. Its only reward is U_r(s').
- Config key names: `power_zeta=2.0`, `power_xi=1.0`, `power_eta=1.1`, `power_gamma_h=0.99`, `power_x_epsilon=0.05`, `power_lr=1e-3`, `power_num_sgd_iter=4`, `power_minibatch_size=256`, `power_target_update_freq=1`, `power_assistant_policy_id="assistant"`, `power_hidden_size=32`, `power_num_layers=2`, `power_filter_size=3`.
- X_h is the sampled-goal mean estimator: target `V^e(s,g)^zeta`, no `|G_h|` factor (spec Section 3, rescaling invariance). U_r uses `X_h + power_x_epsilon` before the power (paper Appendix H.1, ε_X).
- The paper's per-transition robot reward is U_r(s'), the next state (Appendix F.3).

---

## File map

| Path | Responsibility |
|---|---|
| `mbag/rllib/human_power.py` | `PowerEstimator`, `MbagHumanPowerPPOTorchPolicy`, `MbagHumanPowerPPOConfig`, `MbagHumanPowerPPO`, registration |
| `mbag/rllib/__init__.py`, `mbag/scripts/train.py` | import for registration; policy class branch; sacred params; `config.training(...)` block |
| `mbag/scripts/train_configs.py` | `iccea_power_assistant`, `iccea_power_cpu_smoke` |
| `tests/test_human_power.py` | estimator unit tests, smoke training test |
| `docs/human-power/ppo-assistant.md` | branch README with commands and results log |

---

### Task 1: `PowerEstimator`

**Files:**
- Create: `mbag/rllib/human_power.py`
- Test: `tests/test_human_power.py`

**Interfaces:**
- Produces:
  - `POWER_GOAL_COMPLETED = "power_goal_completed"`, `POWER_NEXT_OBS = "power_next_obs"` batch column names.
  - `class PowerEstimator(nn.Module)`: `__init__(self, obs_space, action_space, env_config, *, zeta, xi, eta, gamma_h, x_epsilon, lr, hidden_size, num_layers, filter_size, target_update_freq, device="cpu")`.
    - `v_e(obs: TensorType) -> torch.Tensor` shape `(B,)` in (0,1); `x(obs) -> torch.Tensor` shape `(B,)` positive.
    - `update(obs, next_obs, goal_completed, terminated, *, num_sgd_iter, minibatch_size) -> Dict[str, float]` returns `{"v_e_loss", "x_loss", "v_e_mean", "x_mean", "goal_completion_rate"}`.
    - `rewards(next_obs) -> np.ndarray` shape `(B,)` = `robot_reward(x(next_obs) + x_epsilon, xi, eta)`; `power_bits(next_obs) -> np.ndarray` = `log2(x + x_epsilon)`.
    - `state_dict()`/`load_state_dict()` from `nn.Module` cover both nets and the target net; optimizers are re-created on load.
  - TD target (13): `target = U + gamma_h * (1 - U) * (1 - terminated) * V^e_target(next_obs)`, where `U = goal_completed`. The `(1 - U)` factor is the goal-absorbing continuation.
  - X target (14, mean variant): `V^e(obs).detach() ** zeta`.
  - Both trained with MSE via Adam.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_human_power.py
import numpy as np
import pytest
import torch

from mbag.environment.config import DEFAULT_CONFIG
from mbag.environment.mbag_env import MbagEnv


def _env_and_obs(num_steps=8):
    """A 2-player env; returns (env, flat assistant obs batch, next obs batch)."""
    import copy

    from ray.rllib.models.preprocessors import get_preprocessor

    from mbag.rllib.rllib_env import MbagMultiAgentEnv

    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "world_size": (5, 5, 5),
            "num_players": 2,
            "players": [{}, {}],
            "horizon": 20,
            "goal_generator": "basic",
            "abilities": {"teleportation": True, "flying": True, "inf_blocks": True},
        }
    )
    env = MbagEnv(config)
    ma_env = MbagMultiAgentEnv(config)
    prep = get_preprocessor(ma_env.observation_space)(ma_env.observation_space)
    obs_list, _ = env.reset()
    flat_obs, flat_next = [], []
    for _ in range(num_steps):
        prev = obs_list[1]
        obs_list, _, _, _ = env.step([(0, 0, 0), (0, 0, 0)])
        flat_obs.append(prep.transform(prev))
        flat_next.append(prep.transform(obs_list[1]))
    return ma_env, np.stack(flat_obs), np.stack(flat_next)


@pytest.mark.uses_rllib
def test_power_estimator_shapes_and_ranges():
    from mbag.rllib.human_power import PowerEstimator

    ma_env, obs, next_obs = _env_and_obs()
    est = PowerEstimator(
        ma_env.observation_space,
        ma_env.action_space,
        ma_env.config,
        zeta=2.0, xi=1.0, eta=1.1, gamma_h=0.99, x_epsilon=0.05, lr=1e-3,
        hidden_size=8, num_layers=1, filter_size=3, target_update_freq=1,
    )
    v = est.v_e(torch.as_tensor(obs))
    x = est.x(torch.as_tensor(obs))
    assert v.shape == (len(obs),) and torch.all((v > 0) & (v < 1))
    assert x.shape == (len(obs),) and torch.all(x > 0)
    r = est.rewards(next_obs)
    assert r.shape == (len(obs),) and np.all(r < 0)
    # More power -> higher reward, by construction of eq. (8).
    bits = est.power_bits(next_obs)
    order = np.argsort(bits)
    assert np.all(np.diff(r[order]) >= -1e-6)


@pytest.mark.uses_rllib
def test_power_estimator_td_update_learns_indicator():
    from mbag.rllib.human_power import PowerEstimator

    ma_env, obs, next_obs = _env_and_obs(num_steps=16)
    est = PowerEstimator(
        ma_env.observation_space, ma_env.action_space, ma_env.config,
        zeta=2.0, xi=1.0, eta=1.1, gamma_h=0.0, x_epsilon=0.05, lr=1e-2,
        hidden_size=8, num_layers=1, filter_size=3, target_update_freq=1,
    )
    # With gamma_h = 0 the TD target is exactly the indicator U.
    goal_completed = np.zeros(len(obs), dtype=np.float32)
    goal_completed[-1] = 1.0
    terminated = np.zeros(len(obs), dtype=np.float32)
    terminated[-1] = 1.0
    first = est.update(obs, next_obs, goal_completed, terminated, num_sgd_iter=1, minibatch_size=16)
    for _ in range(30):
        last = est.update(obs, next_obs, goal_completed, terminated, num_sgd_iter=1, minibatch_size=16)
    assert last["v_e_loss"] < first["v_e_loss"]
    assert last["goal_completion_rate"] == pytest.approx(1 / 16)
    v = est.v_e(torch.as_tensor(obs)).detach().numpy()
    assert v[-1] > v[:-1].mean()
    # X tracks V^e ** zeta.
    x = est.x(torch.as_tensor(obs)).detach().numpy()
    assert last["x_loss"] < first["x_loss"] or np.allclose(x, v**2, atol=0.2)


@pytest.mark.uses_rllib
def test_power_estimator_state_roundtrip():
    from mbag.rllib.human_power import PowerEstimator

    ma_env, obs, _ = _env_and_obs(num_steps=2)
    kwargs = dict(zeta=2.0, xi=1.0, eta=1.1, gamma_h=0.99, x_epsilon=0.05, lr=1e-3,
                  hidden_size=8, num_layers=1, filter_size=3, target_update_freq=1)
    a = PowerEstimator(ma_env.observation_space, ma_env.action_space, ma_env.config, **kwargs)
    b = PowerEstimator(ma_env.observation_space, ma_env.action_space, ma_env.config, **kwargs)
    b.load_state_dict(a.state_dict())
    t = torch.as_tensor(obs)
    assert torch.allclose(a.x(t), b.x(t)) and torch.allclose(a.v_e(t), b.v_e(t))
```

- [ ] **Step 2: Run to verify failure**

Run: `TMPDIR=/tmp .venv/bin/pytest tests/test_human_power.py -q`
Expected: `ModuleNotFoundError: No module named 'mbag.rllib.human_power'`

- [ ] **Step 3: Implement `PowerEstimator`** (first part of `mbag/rllib/human_power.py`)

```python
"""
Phase 2 of Heitzig & Potham (2025) inside MBAG: learn V^e_h(s, g) and X_h(s) online
from the assistant's rollouts and use U_r(s') as the assistant's only reward.
"""

import copy
import logging
from typing import Dict, Optional, cast

import numpy as np
import torch
from gymnasium import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from torch import nn

from mbag.power.metrics import robot_reward

logger = logging.getLogger(__name__)

POWER_GOAL_COMPLETED = "power_goal_completed"
POWER_NEXT_OBS = "power_next_obs"


class PowerEstimator(nn.Module):
    """
    Two goal-aware / goal-blind value networks on the repo's convolutional backbone.

    v_e_model(s, g) -> V^e in (0, 1)   (sigmoid on the value head; sees goal channels)
    x_model(s)      -> X_h  > 0        (softplus on the value head; goal masked)
    """

    def __init__(self, obs_space, action_space, env_config, *, zeta, xi, eta, gamma_h,
                 x_epsilon, lr, hidden_size, num_layers, filter_size,
                 target_update_freq, device="cpu"):
        super().__init__()
        self.zeta, self.xi, self.eta = zeta, xi, eta
        self.gamma_h, self.x_epsilon = gamma_h, x_epsilon
        self.lr, self.target_update_freq = lr, target_update_freq
        self.device = torch.device(device)
        num_outputs = int(np.prod(action_space.shape)) if action_space.shape else action_space.n

        def make(mask_goal: bool, name: str) -> TorchModelV2:
            return ModelCatalog.get_model_v2(
                obs_space, action_space, num_outputs,
                {
                    "custom_model": "mbag_convolutional_model",
                    "vf_share_layers": True,
                    "max_seq_len": 1,
                    "custom_model_config": {
                        "env_config": env_config,
                        "mask_goal": mask_goal,
                        "hidden_size": hidden_size,
                        "hidden_channels": hidden_size,
                        "num_layers": num_layers,
                        "filter_size": filter_size,
                        "num_value_layers": 2,
                        "use_fc_after_embedding": True,
                        "mask_action_distribution": False,
                        "interleave_lstm_every": -1,
                        "scale_obs": True,
                    },
                },
                framework="torch", name=name,
            )

        self.v_e_model = cast(nn.Module, make(False, "power_v_e"))
        self.v_e_target = copy.deepcopy(self.v_e_model)
        for p in self.v_e_target.parameters():
            p.requires_grad_(False)
        self.x_model = cast(nn.Module, make(True, "power_x"))
        self.to(self.device)
        self._make_optimizers()
        self.num_updates = 0

    def _make_optimizers(self):
        self.v_e_opt = torch.optim.Adam(self.v_e_model.parameters(), lr=self.lr)
        self.x_opt = torch.optim.Adam(self.x_model.parameters(), lr=self.lr)

    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict)
        self._make_optimizers()
        return result

    def _value(self, model: nn.Module, obs) -> torch.Tensor:
        obs_t = torch.as_tensor(obs, device=self.device)
        cast(TorchModelV2, model)(SampleBatch({SampleBatch.OBS: obs_t}))
        return cast(TorchModelV2, model).value_function()

    def v_e(self, obs, target: bool = False) -> torch.Tensor:
        return torch.sigmoid(self._value(self.v_e_target if target else self.v_e_model, obs))

    def x(self, obs) -> torch.Tensor:
        return nn.functional.softplus(self._value(self.x_model, obs))

    @torch.no_grad()
    def power_bits(self, next_obs) -> np.ndarray:
        x = self.x(next_obs).cpu().numpy() + self.x_epsilon
        return np.log2(x)

    @torch.no_grad()
    def rewards(self, next_obs) -> np.ndarray:
        """U_r(s') of eq. (8) for every next observation, with the eps_X shift."""
        x = self.x(next_obs).cpu().numpy() + self.x_epsilon
        return np.asarray(robot_reward(x, self.xi, self.eta), dtype=np.float32)

    def update(self, obs, next_obs, goal_completed, terminated, *, num_sgd_iter,
               minibatch_size) -> Dict[str, float]:
        n = len(obs)
        u = torch.as_tensor(goal_completed, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(terminated, dtype=torch.float32, device=self.device)
        v_e_losses, x_losses = [], []
        for _ in range(num_sgd_iter):
            perm = torch.randperm(n)
            for start in range(0, n, minibatch_size):
                idx = perm[start : start + minibatch_size].numpy()
                with torch.no_grad():
                    v_next = self.v_e(next_obs[idx], target=True)
                    # eq. (13) with goal-absorbing continuation.
                    target = u[idx] + self.gamma_h * (1 - u[idx]) * (1 - done[idx]) * v_next
                v_pred = self.v_e(obs[idx])
                v_e_loss = nn.functional.mse_loss(v_pred, target)
                self.v_e_opt.zero_grad(); v_e_loss.backward(); self.v_e_opt.step()
                # eq. (14), sampled-goal mean variant.
                x_target = v_pred.detach() ** self.zeta
                x_loss = nn.functional.mse_loss(self.x(obs[idx]), x_target)
                self.x_opt.zero_grad(); x_loss.backward(); self.x_opt.step()
                v_e_losses.append(float(v_e_loss)); x_losses.append(float(x_loss))
        self.num_updates += 1
        if self.num_updates % self.target_update_freq == 0:
            self.v_e_target.load_state_dict(self.v_e_model.state_dict())
        with torch.no_grad():
            v_all = self.v_e(obs); x_all = self.x(obs)
        return {
            "v_e_loss": float(np.mean(v_e_losses)),
            "x_loss": float(np.mean(x_losses)),
            "v_e_mean": float(v_all.mean()),
            "x_mean": float(x_all.mean()),
            "goal_completion_rate": float(u.mean()),
        }
```

If `ModelCatalog.get_model_v2` complains that `obs_space` lacks `original_space`, wrap it: RLlib policies receive a flattened `Box` whose `.original_space` is the `Tuple`. In the test, `ma_env.observation_space` is the raw `Tuple`; build the flat space with `get_preprocessor(space)(space).observation_space` and set `flat_space.original_space = space` inside `PowerEstimator.__init__` when the attribute is missing.

- [ ] **Step 4: Run tests**

Run: `TMPDIR=/tmp .venv/bin/pytest tests/test_human_power.py -q --timeout=300`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add mbag/rllib/human_power.py tests/test_human_power.py
git commit -m "Add PowerEstimator: online V^e and X_h networks with targets (13), (14)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Policy, algorithm, registration, and train-script wiring

**Files:**
- Modify: `mbag/rllib/human_power.py` (append)
- Modify: `mbag/rllib/__init__.py` (import), `mbag/scripts/train.py`
- Test: `tests/test_human_power.py` (append)

**Interfaces:**
- `MbagHumanPowerPPOTorchPolicy(MbagPPOTorchPolicy)`: `postprocess_trajectory` adds `POWER_GOAL_COMPLETED` (float32, shifted so index i is U_h(s_{i+1})) and `POWER_NEXT_OBS` (copy of `SampleBatch.NEXT_OBS`); `loss` touches both keys so RLlib keeps them.
- `MbagHumanPowerPPOConfig(MbagPPOConfig)` with the `power_*` attributes and a `training(...)` override.
- `MbagHumanPowerPPO(MbagPPO)`: lazily builds `self.power_estimator` from the assistant policy's spaces on first `training_step`; `training_step` = `MbagPPO.training_step` with `_update_power_and_replace_rewards(train_batch)` inserted before `standardize_fields`; adds `power/*` stats to the assistant's learner stats; `__getstate__`/`__setstate__` include `power_estimator` state.
- Registration: `register_trainable("MbagHumanPowerPPO", MbagHumanPowerPPO)`; `from . import human_power` in `mbag/rllib/__init__.py`; `from mbag.rllib.human_power import MbagHumanPowerPPOConfig, MbagHumanPowerPPOTorchPolicy` in `train.py`.
- `train.py`: policy class branch `if "HumanPowerPPO" in run: policy_class = MbagHumanPowerPPOTorchPolicy` **before** the `"PPO" in run` branch; sacred params `power_zeta ... power_filter_size` with the defaults from Global Constraints; `if isinstance(config, MbagHumanPowerPPOConfig): config.training(power_zeta=power_zeta, ...)` after the `MbagPPOConfig` block.

- [ ] **Step 1: Write the failing smoke test** (append to `tests/test_human_power.py`)

```python
@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_human_power_ppo_smoke():
    import tempfile

    from mbag.scripts.train import ex

    result = ex.run(
        config_updates={
            "run": "MbagHumanPowerPPO",
            "log_dir": tempfile.mkdtemp(),
            "width": 6, "height": 6, "depth": 6,
            "horizon": 20,
            "goal_generator": "random",
            "num_players": 2,
            "heuristic": "layer_builder",
            "policies_to_train": ["assistant"],
            "mask_goal": True,
            "goal_loss_coeff": 0,
            "use_extra_features": False,
            "own_reward_prop": 1,
            "per_player_goal_reward_scale": [1, 0],
            "gamma": 0.99,
            "num_workers": 0,
            "num_training_iters": 2,
            "train_batch_size": 200,
            "sgd_minibatch_size": 50,
            "rollout_fragment_length": 20,
            "hidden_size": 16, "num_layers": 1, "filter_size": 3,
            "power_hidden_size": 8, "power_num_layers": 1,
            "power_minibatch_size": 50,
            "evaluation_interval": None,
        }
    ).result
    assert result is not None
    stats = result["info"]["learner"]["assistant"]["learner_stats"]
    assert stats["power/robot_reward_mean"] < 0
    assert "power/human_power_bits_mean" in stats
    assert stats["power/v_e_loss"] >= 0
    # The assistant's env reward is exactly zero; goal-dependent reward is off.
    assert result["custom_metrics"]["assistant/goal_dependent_reward_mean"] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `TMPDIR=/tmp .venv/bin/pytest tests/test_human_power.py::test_human_power_ppo_smoke -q --timeout=600`
Expected: FAIL, `ValueError`/`KeyError` from `get_trainable_cls("MbagHumanPowerPPO")` or unknown sacred config key `power_hidden_size`.

- [ ] **Step 3: Implement** (append to `mbag/rllib/human_power.py`)

```python
from ray.rllib.algorithms.algorithm_config import NotProvided  # add to imports
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.execution.rollout_ops import standardize_fields, synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.metrics import NUM_AGENT_STEPS_SAMPLED, NUM_ENV_STEPS_SAMPLED, SAMPLE_TIMER, SYNCH_WORKER_WEIGHTS_TIMER
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import ResultDict
from ray.tune.registry import register_trainable
from ray.util.debug import log_once

from mbag.power.metrics import power_bits as _power_bits  # noqa
from .ppo import MbagPPO, MbagPPOConfig, MbagPPOTorchPolicy


class MbagHumanPowerPPOTorchPolicy(MbagPPOTorchPolicy):
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        n = len(sample_batch)
        completed = np.zeros(n, dtype=np.float32)
        infos = sample_batch.get(SampleBatch.INFOS)
        if infos is not None and n > 0 and isinstance(infos[0], dict) and "goal_completed" in infos[0]:
            # INFOS is shifted like OBS: INFOS[i + 1] is produced by ACTIONS[i].
            shifted = list(infos[1:])
            if episode is not None:
                agent_index = int(sample_batch[SampleBatch.AGENT_INDEX][0])
                agent_id = episode.get_agents()[agent_index]
                shifted.append(episode.last_info_for(agent_id))
            for i, info in enumerate(shifted[:n]):
                completed[i] = float(bool(info.get("goal_completed", False)))
        sample_batch[POWER_GOAL_COMPLETED] = completed
        sample_batch[POWER_NEXT_OBS] = np.array(sample_batch[SampleBatch.NEXT_OBS])
        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)

    def loss(self, model, dist_class, train_batch):
        # Touch the power columns so RLlib keeps them in training batches.
        train_batch[POWER_GOAL_COMPLETED]
        train_batch[POWER_NEXT_OBS]
        return super().loss(model, dist_class, train_batch)


class MbagHumanPowerPPOConfig(MbagPPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class)
        self.power_zeta = 2.0
        self.power_xi = 1.0
        self.power_eta = 1.1
        self.power_gamma_h = 0.99
        self.power_x_epsilon = 0.05
        self.power_lr = 1e-3
        self.power_num_sgd_iter = 4
        self.power_minibatch_size = 256
        self.power_target_update_freq = 1
        self.power_assistant_policy_id = "assistant"
        self.power_hidden_size = 32
        self.power_num_layers = 2
        self.power_filter_size = 3

    def training(self, *args, power_zeta=NotProvided, power_xi=NotProvided, power_eta=NotProvided,
                 power_gamma_h=NotProvided, power_x_epsilon=NotProvided, power_lr=NotProvided,
                 power_num_sgd_iter=NotProvided, power_minibatch_size=NotProvided,
                 power_target_update_freq=NotProvided, power_assistant_policy_id=NotProvided,
                 power_hidden_size=NotProvided, power_num_layers=NotProvided,
                 power_filter_size=NotProvided, **kwargs):
        super().training(*args, **kwargs)
        for name, value in locals().items():
            if name.startswith("power_") and value is not NotProvided:
                setattr(self, name, value)
        return self


class MbagHumanPowerPPO(MbagPPO):
    config: MbagHumanPowerPPOConfig  # type: ignore[assignment]

    @classmethod
    def get_default_config(cls):
        return MbagHumanPowerPPOConfig()

    @classmethod
    def get_default_policy_class(cls, config):
        return MbagHumanPowerPPOTorchPolicy

    power_estimator: Optional[PowerEstimator] = None
    _pending_power_state: Optional[dict] = None

    def _get_power_estimator(self) -> PowerEstimator:
        if self.power_estimator is None:
            policy = self.get_policy(self.config.power_assistant_policy_id)
            env_config = policy.config["model"]["custom_model_config"]["env_config"]
            self.power_estimator = PowerEstimator(
                policy.observation_space, policy.action_space, env_config,
                zeta=self.config.power_zeta, xi=self.config.power_xi, eta=self.config.power_eta,
                gamma_h=self.config.power_gamma_h, x_epsilon=self.config.power_x_epsilon,
                lr=self.config.power_lr, hidden_size=self.config.power_hidden_size,
                num_layers=self.config.power_num_layers, filter_size=self.config.power_filter_size,
                target_update_freq=self.config.power_target_update_freq,
            )
            if self._pending_power_state is not None:
                self.power_estimator.load_state_dict(self._pending_power_state)
                self._pending_power_state = None
        return self.power_estimator

    def _update_power_and_replace_rewards(self, train_batch: MultiAgentBatch) -> Dict[str, float]:
        pid = self.config.power_assistant_policy_id
        batch = train_batch.policy_batches[pid]
        batch.set_get_interceptor(None)
        batch.decompress_if_needed()
        est = self._get_power_estimator()
        obs = batch[SampleBatch.OBS]
        next_obs = batch[POWER_NEXT_OBS]
        completed = batch[POWER_GOAL_COMPLETED]
        terminated = batch[SampleBatch.TERMINATEDS].astype(np.float32)
        stats = est.update(obs, next_obs, completed, terminated,
                           num_sgd_iter=self.config.power_num_sgd_iter,
                           minibatch_size=self.config.power_minibatch_size)
        rewards = est.rewards(next_obs)
        bits = est.power_bits(next_obs)
        batch[SampleBatch.REWARDS] = rewards
        policy = self.workers.local_worker().get_policy(pid)
        episode_batches = []
        for episode_batch in batch.split_by_episode():
            episode_batches.append(compute_advantages(
                rollout=episode_batch,
                last_r=float(episode_batch[SampleBatch.VALUES_BOOTSTRAPPED][-1]),
                gamma=policy.config["gamma"], lambda_=policy.config["lambda"],
                use_gae=policy.config["use_gae"], use_critic=policy.config.get("use_critic", True),
                vf_preds=episode_batch[SampleBatch.VF_PREDS],
                rewards=episode_batch[SampleBatch.REWARDS]))
        train_batch.policy_batches[pid] = concat_samples(episode_batches)
        stats.update({"robot_reward_mean": float(rewards.mean()),
                      "human_power_bits_mean": float(bits.mean()),
                      "human_power_bits_min": float(bits.min()),
                      "human_power_bits_max": float(bits.max())})
        return stats

    def training_step(self) -> ResultDict:
        # Copy of MbagPPO.training_step with the power update spliced in.
        ...  # see ppo.py:338-442; insert after _add_anchor_policy_action_dist_inputs_to_sample_batch:
        #     power_stats = self._update_power_and_replace_rewards(train_batch)
        # and after train_one_step:
        #     if pid in train_results: train_results[pid][LEARNER_STATS_KEY].update({f"power/{k}": v ...})

    def __getstate__(self):
        state = super().__getstate__()
        if self.power_estimator is not None:
            state["power_estimator"] = self.power_estimator.state_dict()
        return state

    def __setstate__(self, state):
        power_state = state.pop("power_estimator", None)
        super().__setstate__(state)
        if power_state is not None:
            if self.power_estimator is None:
                self._pending_power_state = power_state
            else:
                self.power_estimator.load_state_dict(power_state)


register_trainable("MbagHumanPowerPPO", MbagHumanPowerPPO)
```

The `training_step` body must be the literal copy of `mbag/rllib/ppo.py:338-442` with the two insertions; the ellipsis above is only to avoid repeating 100 lines in the plan. The implementer copies it verbatim.

`mbag/rllib/__init__.py`: add `from . import human_power  # noqa: F401` after the `bc` import.

`mbag/scripts/train.py`:
- import `MbagHumanPowerPPOConfig, MbagHumanPowerPPOTorchPolicy` from `mbag.rllib.human_power`.
- sacred params (near the other PPO params):
  ```python
  power_zeta = 2.0
  power_xi = 1.0
  power_eta = 1.1
  power_gamma_h = 0.99
  power_x_epsilon = 0.05
  power_lr = 1e-3
  power_num_sgd_iter = 4
  power_minibatch_size = 256
  power_target_update_freq = 1
  power_assistant_policy_id = "assistant"
  power_hidden_size = 32
  power_num_layers = 2
  power_filter_size = 3
  ```
- policy class: `if "HumanPowerPPO" in run: policy_class = MbagHumanPowerPPOTorchPolicy` before `elif "PPO" in run`.
- after the `if isinstance(config, MbagPPOConfig): config.training(...)` block:
  ```python
  if isinstance(config, MbagHumanPowerPPOConfig):
      config.training(power_zeta=power_zeta, power_xi=power_xi, power_eta=power_eta,
                      power_gamma_h=power_gamma_h, power_x_epsilon=power_x_epsilon,
                      power_lr=power_lr, power_num_sgd_iter=power_num_sgd_iter,
                      power_minibatch_size=power_minibatch_size,
                      power_target_update_freq=power_target_update_freq,
                      power_assistant_policy_id=power_assistant_policy_id,
                      power_hidden_size=power_hidden_size, power_num_layers=power_num_layers,
                      power_filter_size=power_filter_size)
  ```

- [ ] **Step 4: Run tests**

Run: `TMPDIR=/tmp .venv/bin/pytest tests/test_human_power.py -q --timeout=600`
Expected: `4 passed`. Also `TMPDIR=/tmp .venv/bin/pytest tests/test_train.py -q -k "cross_play or kl_regularized" --timeout=600` still passes (MbagPPO untouched).

- [ ] **Step 5: Commit**

```bash
git add mbag/rllib/human_power.py mbag/rllib/__init__.py mbag/scripts/train.py tests/test_human_power.py
git commit -m "Add MbagHumanPowerPPO: PPO assistant rewarded only by U_r(s')

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Named configs

**Files:**
- Modify: `mbag/scripts/train_configs.py` (after `ppo_assistant`)

- [ ] **Step 1: Add the configs**

```python
    @ex.named_config
    def iccea_power_assistant():
        """
        Phase 2 of Heitzig & Potham (2025) in MBAG: PPO assistant whose only reward is
        U_r(s'). Same env and model as ppo_assistant, goal inference off, goal reward
        off for the assistant, paper hyperparameters. Needs checkpoint_to_load_policies
        pointing at a human model with policy id "human".
        """
        run = "MbagHumanPowerPPO"
        goal_generator = "craftassist"
        width = 11
        height = 10
        depth = 10
        num_players = 2
        randomize_first_episode_length = True
        random_start_locations = True
        num_training_iters = 100
        horizon = 1500
        noop_reward = 0.0
        get_resources_reward = 0.0
        teleportation = False
        inf_blocks = True
        entropy_coeff_start = 1
        entropy_coeff_end = 0.01
        entropy_coeff_horizon = 2_000_000
        own_reward_prop = 1
        per_player_goal_reward_scale = [1, 0]
        per_player_action_reward = [-0.2, 0]
        goal_loss_coeff = 0
        gamma = 0.99
        train_batch_size = 32704
        num_workers = 8
        num_envs_per_worker = 8
        num_gpus = 0
        lr = 0.0003
        kl_target = 0.01
        num_sgd_iter = 3
        rollout_fragment_length = 511
        batch_mode = "truncate_episodes"
        model = "convolutional"
        filter_size = 5
        hidden_size = 64
        max_seq_len = 511
        sgd_minibatch_size = 512
        num_layers = 8
        scale_obs = True
        vf_share_layers = True
        vf_loss_coeff = 0.01
        place_block_loss_coeff_schedule = [[0, 0], [1, 0]]
        evaluation_num_workers = 0
        evaluation_interval = None
        clip_param = 0.2
        gae_lambda = 0.95
        custom_action_dist = "mbag_bilevel_categorical"
        mask_goal = True
        interleave_lstm_every = -1
        policies_to_train = ["assistant"]
        checkpoint_to_load_policies = None
        checkpoint_name = ""
        load_policies_mapping = {"human": "human"}
        power_zeta = 2.0
        power_xi = 1.0
        power_eta = 1.1
        power_gamma_h = 0.99
        power_x_epsilon = 0.05
        power_hidden_size = 32
        power_num_layers = 4
        power_filter_size = 3
        experiment_tag = f"iccea_power/{checkpoint_name}"

    @ex.named_config
    def iccea_power_cpu_smoke():
        """
        CPU-sized variant: small world, heuristic goal-following human, tiny nets.
        Use with: python -m mbag.scripts.train with iccea_power_assistant iccea_power_cpu_smoke
        """
        goal_generator = "random"
        width = 6
        height = 6
        depth = 6
        horizon = 50
        teleportation = True
        heuristic = "layer_builder"
        checkpoint_to_load_policies = None
        load_policies_mapping = {}
        per_player_action_reward = [0, 0]
        num_training_iters = 10
        train_batch_size = 2000
        sgd_minibatch_size = 200
        rollout_fragment_length = 50
        max_seq_len = 50
        num_workers = 2
        num_envs_per_worker = 2
        hidden_size = 32
        num_layers = 2
        filter_size = 3
        power_hidden_size = 16
        power_num_layers = 2
        power_minibatch_size = 200
        entropy_coeff_horizon = 20_000
        experiment_tag = "iccea_power/cpu_smoke"
```

Check that `experiment_tag`, `heuristic`, `num_envs_per_worker`, `num_gpus`, `scale_obs`, `custom_action_dist` are existing sacred params (grep `train.py`). Drop any that are not.

- [ ] **Step 2: Verify the configs load**

Run: `TMPDIR=/tmp .venv/bin/python -m mbag.scripts.train print_config with iccea_power_assistant iccea_power_cpu_smoke 2>&1 | grep -E "run|power_|heuristic|goal_reward_scale"`
Expected: values as above, no sacred error.

- [ ] **Step 3: Commit**

```bash
git add mbag/scripts/train_configs.py
git commit -m "Add iccea_power_assistant and iccea_power_cpu_smoke named configs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Run the CPU experiments and document

**Files:**
- Create: `docs/human-power/ppo-assistant.md`

- [ ] **Step 1: CPU smoke run with the heuristic human**

```bash
TMPDIR=/tmp .venv/bin/python -m mbag.scripts.train with iccea_power_assistant iccea_power_cpu_smoke \
    num_training_iters=20 2>&1 | tee /tmp/iccea_smoke.log
```
Record per-iteration `power/human_power_bits_mean`, `power/robot_reward_mean`, `power/v_e_loss`, `goal_percentage_mean`, `human_actions_to_completion_mean` from the log or `progress.csv` under the log dir.

- [ ] **Step 2: BC human model run at the paper's world size**

```bash
TMPDIR=/tmp .venv/bin/python -m mbag.scripts.train with iccea_power_assistant \
    checkpoint_to_load_policies=data/logs/BC/sample_human_models/inf_blocks_True_teleportation_False/2024-04-10_18-51-43/1/checkpoint_000100 \
    checkpoint_name=sample_bc_human \
    num_workers=2 num_envs_per_worker=1 train_batch_size=3000 sgd_minibatch_size=300 \
    rollout_fragment_length=300 max_seq_len=300 horizon=300 \
    hidden_size=32 num_layers=4 power_hidden_size=16 power_num_layers=2 \
    num_training_iters=10 2>&1 | tee /tmp/iccea_bc.log
```

- [ ] **Step 3: Write `docs/human-power/ppo-assistant.md`**

Sections: what this branch adds; how the paper's Phase 2 maps to `PowerEstimator.update` (targets 13 and 14) and `_update_power_and_replace_rewards` (U_r(s') as reward); config knobs table; the two commands above; a results-log table with the recorded numbers and the log-dir paths; known limitations (mean-variant X, ε_X, BC human trained with fixed goals, no MCTS yet).

- [ ] **Step 4: Lint and commit**

```bash
source .venv/bin/activate && ./lint.sh
git add docs/human-power/ppo-assistant.md
git commit -m "Document the PPO human-power assistant and first CPU results

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review against the spec

- Spec 4.5 `AttainmentValueModel`/`PowerModel` are realised as the two nets inside `PowerEstimator`; training targets (13), (14) and the reward overwrite are in Task 1 and Task 2. Spec 4.6 train config in Task 3. Spec 4.7 power metrics are logged as learner stats (`power/human_power_bits_mean`, `power/robot_reward_mean`) rather than env infos, because U_r is computed from batches, not inside the env; `goal_percentage` and `human_actions_to_completion` already flow through the callbacks. The `evaluate.py power_checkpoint` option from Spec 4.7 is deferred to a follow-up and noted in the branch README.
- Names are consistent: `PowerEstimator`, `POWER_GOAL_COMPLETED`, `POWER_NEXT_OBS`, `MbagHumanPowerPPOTorchPolicy`, `MbagHumanPowerPPOConfig`, `MbagHumanPowerPPO`, `power_*` keys, `iccea_power_assistant`, `iccea_power_cpu_smoke`.
