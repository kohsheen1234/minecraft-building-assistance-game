"""
Phase 2 of Heitzig & Potham (2025), "Model-Based Soft Maximization of Suitable
Metrics of Long-Term Human Power" (arXiv:2508.00159v2), inside MBAG.

The assistant is trained with PPO, but its reward is replaced by the paper's
intrinsic reward U_r(s') computed from two networks that are learned online from the
assistant's own rollouts:

* V^e_h(s, g): probability that the human completes goal g from state s under the
  current policies, trained with the TD target of eq. (13),
  v^e <- U_h(s', g) + gamma_h * V^e_h(s', g), where U_h = 1[goal completed].
* X_h(s): the effective number of attainable goals of eq. (7), trained with the
  sampled-goal target of eq. (14), x <- V^e_h(s, g)^zeta (mean over goals variant).

U_r(s') = -(X_h(s') + eps_X)^(-xi * eta) then follows from eq. (8) with one human.
"""

import copy
import logging
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
from gymnasium import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from torch import nn

from mbag.power.metrics import robot_reward

logger = logging.getLogger(__name__)

POWER_GOAL_COMPLETED = "power_goal_completed"
POWER_NEXT_OBS = "power_next_obs"


def _flat_obs_space_with_original(obs_space: spaces.Space) -> spaces.Space:
    """
    RLlib models expect the flattened observation Box with ``original_space`` set to
    the structured space. Policies already hand us such a space; raw env spaces are
    wrapped here.
    """
    if hasattr(obs_space, "original_space"):
        return obs_space
    preprocessor = get_preprocessor(obs_space)(obs_space)
    flat_space = preprocessor.observation_space
    cast(Any, flat_space).original_space = obs_space
    return flat_space


class PowerEstimator(nn.Module):
    """
    Two value networks on the repo's convolutional backbone:

    * ``v_e_model(s, g)`` sees the goal channels; sigmoid on the value head gives
      V^e in (0, 1). A frozen copy ``v_e_target`` provides TD targets.
    * ``x_model(s)`` has the goal masked; softplus on the value head gives X_h > 0.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        env_config,
        *,
        zeta: float,
        xi: float,
        eta: float,
        gamma_h: float,
        x_epsilon: float,
        lr: float,
        hidden_size: int,
        num_layers: int,
        filter_size: int,
        target_update_freq: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.zeta, self.xi, self.eta = zeta, xi, eta
        self.gamma_h, self.x_epsilon = gamma_h, x_epsilon
        self.lr, self.target_update_freq = lr, target_update_freq
        self.device = torch.device(device)

        obs_space = _flat_obs_space_with_original(obs_space)
        if isinstance(action_space, spaces.Discrete):
            num_outputs = int(action_space.n)
        else:
            num_outputs = int(np.prod(action_space.shape or (1,)))

        def make(mask_goal: bool, name: str) -> nn.Module:
            model = ModelCatalog.get_model_v2(
                obs_space,
                action_space,
                num_outputs,
                {
                    "custom_model": "mbag_convolutional_model",
                    "vf_share_layers": True,
                    "max_seq_len": 1,
                    "custom_model_config": {
                        "env_config": env_config,
                        "mask_goal": mask_goal,
                        "mask_other_players": False,
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
                framework="torch",
                name=name,
            )
            return cast(nn.Module, model)

        self.v_e_model = make(False, "power_v_e")
        self.v_e_target = copy.deepcopy(self.v_e_model)
        for parameter in self.v_e_target.parameters():
            parameter.requires_grad_(False)
        self.x_model = make(True, "power_x")
        self.to(self.device)
        self._make_optimizers()
        self.num_updates = 0

    def _make_optimizers(self) -> None:
        self.v_e_opt = torch.optim.Adam(self.v_e_model.parameters(), lr=self.lr)
        self.x_opt = torch.optim.Adam(self.x_model.parameters(), lr=self.lr)

    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict)
        self._make_optimizers()
        return result

    def _value(self, model: nn.Module, obs) -> torch.Tensor:
        obs_t = torch.as_tensor(np.asarray(obs), device=self.device)
        model_v2 = cast(TorchModelV2, model)
        model_v2(SampleBatch({SampleBatch.OBS: obs_t}))
        return model_v2.value_function()

    def v_e(self, obs, target: bool = False) -> torch.Tensor:
        """V^e_h(s, g) in (0, 1); ``target`` selects the frozen TD-target copy."""
        model = self.v_e_target if target else self.v_e_model
        return torch.sigmoid(self._value(model, obs))

    def x(self, obs) -> torch.Tensor:
        """X_h(s) > 0 (eq. 7, mean-over-goals variant)."""
        return nn.functional.softplus(self._value(self.x_model, obs))

    @torch.no_grad()
    def power_bits(self, obs) -> np.ndarray:
        """W_h(s) = log2 (X_h(s) + eps_X), eq. (10)."""
        x = self.x(obs).cpu().numpy() + self.x_epsilon
        return np.asarray(np.log2(x), dtype=np.float32)

    @torch.no_grad()
    def rewards(self, next_obs) -> np.ndarray:
        """U_r(s') of eq. (8) for every next observation, with the eps_X shift."""
        x = self.x(next_obs).cpu().numpy() + self.x_epsilon
        return np.asarray(robot_reward(x, self.xi, self.eta), dtype=np.float32)

    def update(
        self,
        obs,
        next_obs,
        goal_completed,
        terminated,
        *,
        num_sgd_iter: int,
        minibatch_size: int,
    ) -> Dict[str, float]:
        """One round of TD updates for V^e (eq. 13) and regression for X_h (eq. 14)."""
        obs = np.asarray(obs)
        next_obs = np.asarray(next_obs)
        n = len(obs)
        u = torch.as_tensor(
            np.asarray(goal_completed), dtype=torch.float32, device=self.device
        )
        done = torch.as_tensor(
            np.asarray(terminated), dtype=torch.float32, device=self.device
        )
        v_e_losses, x_losses = [], []
        for _ in range(num_sgd_iter):
            perm = torch.randperm(n).numpy()
            for start in range(0, n, minibatch_size):
                idx = perm[start : start + minibatch_size]
                with torch.no_grad():
                    v_next = self.v_e(next_obs[idx], target=True)
                    # eq. (13): entering a goal state pays 1 and ends that goal's
                    # continuation (goal-absorbing); terminal states end it too.
                    target = (
                        u[idx] + self.gamma_h * (1 - u[idx]) * (1 - done[idx]) * v_next
                    )
                v_pred = self.v_e(obs[idx])
                v_e_loss = nn.functional.mse_loss(v_pred, target)
                self.v_e_opt.zero_grad()
                v_e_loss.backward()
                self.v_e_opt.step()

                # eq. (14), sampled-goal mean variant: x <- V^e(s, g)^zeta.
                x_target = v_pred.detach() ** self.zeta
                x_loss = nn.functional.mse_loss(self.x(obs[idx]), x_target)
                self.x_opt.zero_grad()
                x_loss.backward()
                self.x_opt.step()

                v_e_losses.append(float(v_e_loss))
                x_losses.append(float(x_loss))

        self.num_updates += 1
        if self.num_updates % self.target_update_freq == 0:
            self.v_e_target.load_state_dict(self.v_e_model.state_dict())

        with torch.no_grad():
            v_all = self.v_e(obs)
            x_all = self.x(obs)
        return {
            "v_e_loss": float(np.mean(v_e_losses)),
            "x_loss": float(np.mean(x_losses)),
            "v_e_mean": float(v_all.mean()),
            "x_mean": float(x_all.mean()),
            "goal_completion_rate": float(u.mean()),
        }
