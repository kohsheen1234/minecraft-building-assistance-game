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
    flat_space: spaces.Space = preprocessor.observation_space
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
        original_space = cast(Any, obs_space).original_space
        self.preprocessor = get_preprocessor(original_space)(original_space)
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

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._make_optimizers()
        return result

    def _value(self, model: nn.Module, obs) -> torch.Tensor:
        obs_t = torch.as_tensor(np.asarray(obs), device=self.device)
        model_v2 = cast(TorchModelV2, model)
        model_v2(SampleBatch({SampleBatch.OBS: obs_t}))
        return cast(torch.Tensor, model_v2.value_function())

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

    def reward_for_obs(self, obs) -> float:
        """U_r for a single structured MBAG observation (world, inventory, timestep)."""
        flat = self.preprocessor.transform(obs)
        return float(self.rewards(flat[None])[0])

    def state_numpy(self) -> Dict[str, np.ndarray]:
        """State dict as numpy arrays, for shipping inside RLlib policy weights."""
        return {key: value.cpu().numpy() for key, value in self.state_dict().items()}

    def load_state_numpy(self, state: Dict[str, np.ndarray]) -> None:
        self.load_state_dict(
            {key: torch.as_tensor(value) for key, value in state.items()}
        )

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


# --------------------------------------------------------------------------------
# Policy, config, and algorithm
# --------------------------------------------------------------------------------

from ray.rllib.algorithms.algorithm_config import NotProvided  # noqa: E402
from ray.rllib.evaluation.postprocessing import compute_advantages  # noqa: E402
from ray.rllib.execution.rollout_ops import (  # noqa: E402
    standardize_fields,
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import (  # noqa: E402
    multi_gpu_train_one_step,
    train_one_step,
)
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples  # noqa: E402
from ray.rllib.utils.metrics import (  # noqa: E402
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY  # noqa: E402
from ray.rllib.utils.typing import ResultDict  # noqa: E402
from ray.tune.registry import register_trainable  # noqa: E402
from ray.util.debug import log_once  # noqa: E402

from .ppo import MbagPPO, MbagPPOConfig, MbagPPOTorchPolicy  # noqa: E402


class MbagHumanPowerPPOTorchPolicy(MbagPPOTorchPolicy):
    """
    PPO policy that lifts two extra columns into every sample batch so the algorithm
    can compute the human-power reward in ``training_step``:

    * ``POWER_GOAL_COMPLETED[i]`` = U_h(s_{i+1}) = 1[goal completed by ACTIONS[i]]
    * ``POWER_NEXT_OBS[i]`` = s_{i+1} (a copy of NEXT_OBS that survives into training)
    """

    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        n = len(sample_batch)
        completed = np.zeros(n, dtype=np.float32)
        infos = sample_batch.get(SampleBatch.INFOS)
        if (
            infos is not None
            and n > 0
            and isinstance(infos[0], dict)
            and "goal_completed" in infos[0]
        ):
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
        return super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode
        )

    def loss(self, model, dist_class, train_batch):
        # Touch the power columns so RLlib keeps them in training batches (it drops
        # columns that are only read during postprocessing).
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

    def training(
        self,
        *args,
        power_zeta=NotProvided,
        power_xi=NotProvided,
        power_eta=NotProvided,
        power_gamma_h=NotProvided,
        power_x_epsilon=NotProvided,
        power_lr=NotProvided,
        power_num_sgd_iter=NotProvided,
        power_minibatch_size=NotProvided,
        power_target_update_freq=NotProvided,
        power_assistant_policy_id=NotProvided,
        power_hidden_size=NotProvided,
        power_num_layers=NotProvided,
        power_filter_size=NotProvided,
        **kwargs,
    ):
        super().training(*args, **kwargs)
        provided = {
            "power_zeta": power_zeta,
            "power_xi": power_xi,
            "power_eta": power_eta,
            "power_gamma_h": power_gamma_h,
            "power_x_epsilon": power_x_epsilon,
            "power_lr": power_lr,
            "power_num_sgd_iter": power_num_sgd_iter,
            "power_minibatch_size": power_minibatch_size,
            "power_target_update_freq": power_target_update_freq,
            "power_assistant_policy_id": power_assistant_policy_id,
            "power_hidden_size": power_hidden_size,
            "power_num_layers": power_num_layers,
            "power_filter_size": power_filter_size,
        }
        for name, value in provided.items():
            if value is not NotProvided:
                setattr(self, name, value)
        return self


class MbagHumanPowerPPO(MbagPPO):
    """
    MbagPPO whose assistant is rewarded only by U_r(s') from a PowerEstimator that is
    trained online on the same batches (paper Section 3.1, Phase 2).
    """

    config: MbagHumanPowerPPOConfig

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
                policy.observation_space,
                policy.action_space,
                env_config,
                zeta=self.config.power_zeta,
                xi=self.config.power_xi,
                eta=self.config.power_eta,
                gamma_h=self.config.power_gamma_h,
                x_epsilon=self.config.power_x_epsilon,
                lr=self.config.power_lr,
                hidden_size=self.config.power_hidden_size,
                num_layers=self.config.power_num_layers,
                filter_size=self.config.power_filter_size,
                target_update_freq=self.config.power_target_update_freq,
            )
            if self._pending_power_state is not None:
                self.power_estimator.load_state_dict(self._pending_power_state)
                self._pending_power_state = None
        return self.power_estimator

    def _update_power_and_replace_rewards(
        self, train_batch: MultiAgentBatch
    ) -> Dict[str, float]:
        """
        Train V^e and X_h on the assistant's batch (targets 13 and 14), then replace
        the assistant's rewards with U_r(s') and recompute advantages.
        """
        assert self.workers is not None
        pid = self.config.power_assistant_policy_id
        batch = train_batch.policy_batches[pid]
        batch.set_get_interceptor(None)
        batch.decompress_if_needed()
        estimator = self._get_power_estimator()

        obs = batch[SampleBatch.OBS]
        next_obs = batch[POWER_NEXT_OBS]
        completed = batch[POWER_GOAL_COMPLETED]
        terminated = np.asarray(batch[SampleBatch.TERMINATEDS], dtype=np.float32)
        stats = estimator.update(
            obs,
            next_obs,
            completed,
            terminated,
            num_sgd_iter=self.config.power_num_sgd_iter,
            minibatch_size=self.config.power_minibatch_size,
        )

        rewards = estimator.rewards(next_obs)
        bits = estimator.power_bits(next_obs)
        batch[SampleBatch.REWARDS] = rewards

        policy = self.workers.local_worker().get_policy(pid)
        assert policy is not None
        episode_batches = []
        for episode_batch in batch.split_by_episode():
            episode_batches.append(
                compute_advantages(
                    rollout=episode_batch,
                    last_r=float(episode_batch[SampleBatch.VALUES_BOOTSTRAPPED][-1]),
                    gamma=policy.config["gamma"],
                    lambda_=policy.config["lambda"],
                    use_gae=policy.config["use_gae"],
                    use_critic=policy.config.get("use_critic", True),
                    vf_preds=episode_batch[SampleBatch.VF_PREDS],
                    rewards=episode_batch[SampleBatch.REWARDS],
                )
            )
        train_batch.policy_batches[pid] = concat_samples(episode_batches)

        stats.update(
            {
                "robot_reward_mean": float(rewards.mean()),
                "human_power_bits_mean": float(bits.mean()),
                "human_power_bits_min": float(bits.min()),
                "human_power_bits_max": float(bits.max()),
            }
        )
        return stats

    def training_step(self) -> ResultDict:
        # Copy of MbagPPO.training_step with the power update spliced in before
        # advantages are standardized.
        assert self.workers is not None

        with self._timers[SAMPLE_TIMER]:
            if self.config.count_steps_by == "agent_steps":
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_agent_steps=self.config.train_batch_size,
                )
            else:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers, max_env_steps=self.config.train_batch_size
                )

        assert not isinstance(train_batch, list)
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        train_batch = self._add_anchor_policy_action_dist_inputs_to_sample_batch(
            train_batch
        )

        with self._timers["power_update"]:
            power_stats = self._update_power_and_replace_rewards(train_batch)

        train_batch = standardize_fields(train_batch, ["advantages"])
        if self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        assistant_pid = self.config.power_assistant_policy_id
        if assistant_pid in train_results:
            train_results[assistant_pid][LEARNER_STATS_KEY].update(
                {f"power/{key}": value for key, value in power_stats.items()}
            )

        policies_to_update = list(train_results.keys())

        policy_map = self.workers.local_worker().policy_map
        assert policy_map is not None
        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: policy_map[pid].num_grad_updates for pid in policies_to_update
            },
        }

        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            if self.workers.num_remote_workers() > 0:
                from_worker_or_learner_group = None
                if self.config._enable_learner_api:
                    from_worker_or_learner_group = self.learner_group
                self.workers.sync_weights(
                    from_worker_or_learner_group=from_worker_or_learner_group,
                    policies=policies_to_update,
                    global_vars=global_vars,
                )

        for policy_id, policy_info in train_results.items():
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            policy = self.get_policy(policy_id)
            assert isinstance(policy, MbagPPOTorchPolicy)
            policy.update_kl(kl_divergence)

            scaled_vf_loss = (
                self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            assert isinstance(train_batch, MultiAgentBatch)
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        self.workers.local_worker().set_global_vars(global_vars)

        return train_results

    def __getstate__(self):
        state = super().__getstate__()
        if self.power_estimator is not None:
            state["power_estimator"] = {
                key: value.cpu()
                for key, value in self.power_estimator.state_dict().items()
            }
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
