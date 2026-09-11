"""
Human-power objective (Heitzig & Potham 2025, arXiv:2508.00159v2) inside the
AlphaZero-style MBAG assistant.

Every MCTS node's reward is U_r(s') from the learned X_h network (via the planning
env model's ``power_reward_fn`` hook), the value targets use the same reward, and the
PowerEstimator is trained on the driver from each fresh sample batch and shipped to
the rollout workers inside the policy weights.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.typing import ModelWeights, ResultDict
from ray.tune.registry import register_trainable

from .alpha_zero import MbagAlphaZero, MbagAlphaZeroConfig
from .alpha_zero.alpha_zero_policy import MbagAlphaZeroPolicy
from .human_power import POWER_GOAL_COMPLETED, POWER_NEXT_OBS, PowerEstimator

logger = logging.getLogger(__name__)

POWER_ESTIMATOR_WEIGHTS_KEY = "__power_estimator__"


def _power_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        zeta=config.get("power_zeta", 2.0),
        xi=config.get("power_xi", 1.0),
        eta=config.get("power_eta", 1.1),
        gamma_h=config.get("power_gamma_h", 0.99),
        x_epsilon=config.get("power_x_epsilon", 0.05),
        lr=config.get("power_lr", 1e-3),
        hidden_size=config.get("power_hidden_size", 32),
        num_layers=config.get("power_num_layers", 2),
        filter_size=config.get("power_filter_size", 3),
        target_update_freq=config.get("power_target_update_freq", 1),
    )


class MbagHumanPowerAlphaZeroPolicy(MbagAlphaZeroPolicy):
    """
    MbagAlphaZeroPolicy that owns a PowerEstimator, plans with U_r(s') at every MCTS
    node, and uses U_r(s') as the reward for value targets.
    """

    power_estimator: PowerEstimator

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.power_estimator = PowerEstimator(
            observation_space,
            action_space,
            config["env_config"],
            **_power_kwargs(config),
        )

        base_env_creator = self.env_creator

        def env_creator():
            env_model = base_env_creator()
            env_model.power_reward_fn = self.power_estimator.reward_for_obs
            return env_model

        self.env_creator = env_creator
        for env_model in self.envs:
            env_model.power_reward_fn = self.power_estimator.reward_for_obs

    def get_weights(self) -> ModelWeights:
        weights = dict(super().get_weights())
        weights[POWER_ESTIMATOR_WEIGHTS_KEY] = self.power_estimator.state_numpy()
        return weights

    def set_weights(self, weights: ModelWeights) -> None:
        weights = dict(weights)
        power_state = weights.pop(POWER_ESTIMATOR_WEIGHTS_KEY, None)
        super().set_weights(weights)
        if power_state is not None:
            self.power_estimator.load_state_numpy(power_state)

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
            shifted = list(infos[1:])
            if episode is not None:
                agent_index = int(sample_batch[SampleBatch.AGENT_INDEX][0])
                agent_id = episode.get_agents()[agent_index]
                shifted.append(episode.last_info_for(agent_id))
            for i, info in enumerate(shifted[:n]):
                completed[i] = float(bool(info.get("goal_completed", False)))
        sample_batch[POWER_GOAL_COMPLETED] = completed
        next_obs = np.array(sample_batch[SampleBatch.NEXT_OBS])
        sample_batch[POWER_NEXT_OBS] = next_obs
        # U_r(s') replaces the (zero) env reward before value targets are computed.
        sample_batch[SampleBatch.REWARDS] = self.power_estimator.rewards(next_obs)
        return super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode
        )

    def loss(self, model, dist_class, train_batch):
        train_batch[POWER_GOAL_COMPLETED]
        train_batch[POWER_NEXT_OBS]
        return super().loss(model, dist_class, train_batch)


class MbagHumanPowerAlphaZeroConfig(MbagAlphaZeroConfig):
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


class MbagHumanPowerAlphaZero(MbagAlphaZero):
    config: MbagHumanPowerAlphaZeroConfig  # type: ignore[assignment]

    @classmethod
    def get_default_config(cls):
        return MbagHumanPowerAlphaZeroConfig()

    @classmethod
    def get_default_policy_class(cls, config):
        return MbagHumanPowerAlphaZeroPolicy

    _power_stats: Optional[Dict[str, float]] = None

    def _sample_and_add_to_replay_buffer(self):
        new_sample_batch, prediction_metrics = (
            super()._sample_and_add_to_replay_buffer()
        )
        self._power_stats = self._update_power_estimator(new_sample_batch)
        return new_sample_batch, prediction_metrics

    def _update_power_estimator(
        self, sample_batch: MultiAgentBatch
    ) -> Optional[Dict[str, float]]:
        assert self.workers is not None
        pid = self.config.power_assistant_policy_id
        if pid not in sample_batch.policy_batches:
            return None
        policy = self.workers.local_worker().get_policy(pid)
        assert isinstance(policy, MbagHumanPowerAlphaZeroPolicy)
        batch = sample_batch.policy_batches[pid]
        batch.set_get_interceptor(None)
        batch.decompress_if_needed()
        stats = policy.power_estimator.update(
            batch[SampleBatch.OBS],
            batch[POWER_NEXT_OBS],
            batch[POWER_GOAL_COMPLETED],
            np.asarray(batch[SampleBatch.TERMINATEDS], dtype=np.float32),
            num_sgd_iter=self.config.power_num_sgd_iter,
            minibatch_size=self.config.power_minibatch_size,
        )
        rewards = policy.power_estimator.rewards(batch[POWER_NEXT_OBS])
        bits = policy.power_estimator.power_bits(batch[POWER_NEXT_OBS])
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
        self._power_stats = None
        train_results = super().training_step()
        if self._power_stats is not None:
            pid = self.config.power_assistant_policy_id
            train_results.setdefault(pid, {}).setdefault("custom_metrics", {}).update(
                {f"power/{key}": value for key, value in self._power_stats.items()}
            )
        return train_results


register_trainable("MbagHumanPowerAlphaZero", MbagHumanPowerAlphaZero)
