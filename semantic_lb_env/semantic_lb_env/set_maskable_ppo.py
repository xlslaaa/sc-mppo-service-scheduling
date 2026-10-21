from typing import Any, Dict, Optional, Type, Union
import torch as th
import torch.nn.functional as F
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor

from .set_rollout_buffer import SetDictRolloutBuffer

class SetMaskablePPO(MaskablePPO):

    def __init__(
        self,
        policy: Union[str, Type[MaskableActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rollout_buffer_class: Optional[Type[SetDictRolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        feas_coef: float = 0.1,
    ):

        if rollout_buffer_class is None:
            rollout_buffer_class = SetDictRolloutBuffer

        self.feas_coef = feas_coef

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation"
        assert isinstance(rollout_buffer, SetDictRolloutBuffer), \
            f"rollout_buffer 必须是 SetDictRolloutBuffer,当前类型 {type(rollout_buffer)}"

        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():

                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                if use_masking:

                    action_masks = np.array([env.get_attr("action_masks", indices=i)[0]()
                                            for i in range(self.n_envs)])

                    action_masks_tensor = th.as_tensor(action_masks, device=self.device)
                else:
                    action_masks_tensor = None

                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks_tensor)

            actions = actions.cpu().numpy()

            new_obs, rewards, dones, infos = env.step(actions)

            feasibility_labels = np.array([
                info.get("true_mask", np.ones(self.action_space.n, dtype=np.float32))
                for info in infos
            ], dtype=np.float32)

            self.num_timesteps += self.n_envs

            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, th.nn.modules.container.Sequential):

                actions = actions.reshape((self.n_envs, self.action_space.shape[0]))

            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None:

                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks if use_masking else None,
                feasibility_labels=feasibility_labels,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():

            obs_tensor = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:

        self.policy.set_training_mode(True)

        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        feasibility_losses = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, th.nn.modules.container.Sequential):

                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if clip_range_vf is None:
                    values_pred = values
                else:

                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)

                feasibility_logits = getattr(self.policy, "_last_feasibility_logits", None)
                if feasibility_logits is not None and self.feas_coef > 0:

                    feasibility_loss = F.binary_cross_entropy_with_logits(
                        feasibility_logits,
                        rollout_data.feasibility_labels,
                        reduction="mean",
                    )
                    feasibility_losses.append(feasibility_loss.item())
                else:
                    feasibility_loss = th.tensor(0.0, device=self.device)
                    feasibility_losses.append(0.0)

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.feas_coef * feasibility_loss
                )

                pg_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                approx_kl_divs.append(th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy())

            self._n_updates += 1
            if self.target_kl is not None and np.mean(approx_kl_divs) > 1.5 * self.target_kl:
                print(f"Early stopping at step {epoch} due to reaching max kl: {np.mean(approx_kl_divs):.2f}")
                continue_training = False
                break

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/feasibility_loss", np.mean(feasibility_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        return continue_training
