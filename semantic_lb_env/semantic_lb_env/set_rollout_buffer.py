from typing import Generator, NamedTuple, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.buffers import (
    MaskableDictRolloutBuffer,
    MaskableDictRolloutBufferSamples,
)
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize

class SetDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    action_masks: th.Tensor
    feasibility_labels: th.Tensor

class SetDictRolloutBuffer(MaskableDictRolloutBuffer):

    feasibility_labels: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, gae_lambda, gamma, n_envs)

    def reset(self) -> None:
        super().reset()

        self.feasibility_labels = np.zeros(
            (self.buffer_size, self.n_envs, self.mask_dims), dtype=np.float32
        )

    def add(
        self,
        *args,
        action_masks: Optional[np.ndarray] = None,
        feasibility_labels: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if feasibility_labels is not None:
            self.feasibility_labels[self.pos] = feasibility_labels.reshape((self.n_envs, self.mask_dims))

        super().add(*args, action_masks=action_masks, **kwargs)

    def get(
        self, batch_size: Optional[int] = None
    ) -> Generator[SetDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:

            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "action_masks",
                "feasibility_labels",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None
    ) -> SetDictRolloutBufferSamples:
        return SetDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
            feasibility_labels=self.to_torch(
                self.feasibility_labels[batch_inds].reshape(-1, self.mask_dims)
            ),
        )
