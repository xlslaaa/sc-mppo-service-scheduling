from __future__ import annotations

from typing import Any, Dict, Generator, List, NamedTuple, Optional, Sequence

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn
from torch.nn import functional as F

from semantic_lb_env.cmdp import CONSTRAINT_NAMES, active_limits, constraint_names as _constraint_names

class P3OBufferSamples(NamedTuple):

    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    cost_advantages: th.Tensor
    cost_returns: th.Tensor

class CostRolloutBuffer(RolloutBuffer):

    def __init__(self, buffer_size: int, observation_space: spaces.Space,
                 action_space: spaces.Space, device="auto", gae_lambda: float = 1.0,
                 gamma: float = 0.99, n_envs: int = 1,
                 n_constraints: int = len(CONSTRAINT_NAMES),
                 cost_gamma: Optional[float] = None,
                 cost_gae_lambda: Optional[float] = None) -> None:
        self.n_constraints = int(n_constraints)
        self.cost_gamma = float(gamma if cost_gamma is None else cost_gamma)
        self.cost_gae_lambda = float(gae_lambda if cost_gae_lambda is None else cost_gae_lambda)
        super().__init__(buffer_size, observation_space, action_space, device=device,
                         gae_lambda=gae_lambda, gamma=gamma, n_envs=n_envs)

    def reset(self) -> None:
        shape = (self.buffer_size, self.n_envs, self.n_constraints)
        self.costs = np.zeros(shape, dtype=np.float32)
        self.cost_values = np.zeros(shape, dtype=np.float32)
        self.cost_advantages = np.zeros(shape, dtype=np.float32)
        self.cost_returns = np.zeros(shape, dtype=np.float32)
        super().reset()

    def add_costs(self, costs: np.ndarray, cost_values: np.ndarray) -> None:
        pos = self.pos
        self.costs[pos] = np.asarray(costs, dtype=np.float32).reshape(self.n_envs, self.n_constraints)
        self.cost_values[pos] = np.asarray(cost_values, dtype=np.float32).reshape(
            self.n_envs, self.n_constraints)

    def compute_cost_returns_and_advantage(self, last_cost_values: th.Tensor,
                                           dones: np.ndarray) -> None:
        last = last_cost_values.detach().cpu().numpy().reshape(self.n_envs, self.n_constraints)
        last_gae = np.zeros((self.n_envs, self.n_constraints), dtype=np.float32)
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = (1.0 - np.asarray(dones, dtype=np.float32)).reshape(self.n_envs, 1)
                next_values = last
            else:
                next_non_terminal = (1.0 - self.episode_starts[step + 1]).reshape(self.n_envs, 1)
                next_values = self.cost_values[step + 1]
            delta = (self.costs[step]
                     + self.cost_gamma * next_values * next_non_terminal
                     - self.cost_values[step])
            last_gae = delta + self.cost_gamma * self.cost_gae_lambda * next_non_terminal * last_gae
            self.cost_advantages[step] = last_gae
        self.cost_returns = self.cost_advantages + self.cost_values

    def get(self, batch_size: Optional[int] = None) -> Generator[P3OBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for tensor in ["observations", "actions", "values", "log_probs", "advantages",
                           "returns", "costs", "cost_values", "cost_advantages", "cost_returns"]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> P3OBufferSamples:
        n = len(batch_inds)
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.cost_advantages[batch_inds].reshape(n, self.n_constraints),
            self.cost_returns[batch_inds].reshape(n, self.n_constraints),
        )
        return P3OBufferSamples(*tuple(map(self.to_torch, data)))

class P3O(PPO):

    def __init__(self, policy, env, *, cost_limits: Dict[str, float],
                 penalty_coef=5.0, cost_vf_coef: float = 0.5,
                 cost_gamma: Optional[float] = None, cost_gae_lambda: Optional[float] = None,
                 cost_episode_len: int = 120, cost_net_arch: Sequence[int] = (64, 64), constraint_names: Sequence[str] = CONSTRAINT_NAMES,
                 **kwargs: Any) -> None:
        self.constraint_names = tuple(str(n) for n in constraint_names); missing = [n for n in self.constraint_names if n not in cost_limits]
        if missing:
            raise ValueError("cost_limits 缺少约束 %s；必须与本次运行的 constraint_names %s 一一对应" % (missing, list(self.constraint_names)))
        self.n_constraints = len(self.constraint_names)
        self.cost_limits_per_step = np.array([float(cost_limits[n]) for n in self.constraint_names],
                                            dtype=np.float64)
        if isinstance(penalty_coef, dict):
            missing = [n for n in self.constraint_names if n not in penalty_coef]
            if missing:
                raise ValueError("penalty_coef 作为字典时必须覆盖 %s，缺 %s"
                                 % (list(self.constraint_names), missing))
            coefs = [float(penalty_coef[n]) for n in self.constraint_names]
        else:
            coefs = [float(penalty_coef)] * self.n_constraints
        if any(c < 0.0 for c in coefs):
            raise ValueError("κ 必须非负，收到 %r" % (coefs,))
        self.penalty_coef_vec = np.array(coefs, dtype=np.float64)
        self.cost_vf_coef = float(cost_vf_coef)
        self.cost_episode_len = int(cost_episode_len)
        self.cost_net_arch = tuple(int(w) for w in cost_net_arch)

        gamma = float(kwargs.get("gamma", 0.99))
        gae_lambda = float(kwargs.get("gae_lambda", 0.95))
        self.cost_gamma = gamma if cost_gamma is None else float(cost_gamma)
        self.cost_gae_lambda = gae_lambda if cost_gae_lambda is None else float(cost_gae_lambda)

        if self.cost_gamma >= 1.0:
            self.discount_horizon = float(self.cost_episode_len)
        else:
            self.discount_horizon = float(
                (1.0 - self.cost_gamma ** self.cost_episode_len) / (1.0 - self.cost_gamma))
        self.cost_limits_return = self.cost_limits_per_step * self.discount_horizon

        kwargs.setdefault("rollout_buffer_class", CostRolloutBuffer)
        buffer_kwargs = dict(kwargs.get("rollout_buffer_kwargs") or {})
        buffer_kwargs.setdefault("n_constraints", self.n_constraints)
        buffer_kwargs.setdefault("cost_gamma", self.cost_gamma)
        buffer_kwargs.setdefault("cost_gae_lambda", self.cost_gae_lambda)
        kwargs["rollout_buffer_kwargs"] = buffer_kwargs

        self.diagnostics: List[Dict[str, float]] = []
        super().__init__(policy, env, **kwargs)
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("本实现只覆盖 Discrete 动作空间（本 env 是 15 个离散动作），"
                             "收到 %r" % (self.action_space,))

    def _setup_model(self) -> None:
        super()._setup_model()

        rng_state = th.get_rng_state()
        obs_dim = int(np.prod(self.observation_space.shape))
        layers: List[nn.Module] = []
        last = obs_dim
        for width in self.cost_net_arch:
            layers += [nn.Linear(last, width), nn.Tanh()]
            last = width
        layers += [nn.Linear(last, self.n_constraints)]

        self.cost_critic = nn.Sequential(*layers).to(self.device)

        self.cost_optimizer = self.policy.optimizer_class(
            self.cost_critic.parameters(), lr=self.lr_schedule(1.0),
            **self.policy.optimizer_kwargs)
        th.set_rng_state(rng_state)
        self._penalty_coefs_t = th.as_tensor(self.penalty_coef_vec, dtype=th.float32,
                                            device=self.device)
        self._cost_limits_return_t = th.as_tensor(self.cost_limits_return, dtype=th.float32,
                                                 device=self.device)

    def _excluded_save_params(self) -> List[str]:

        return super()._excluded_save_params() + ["cost_critic", "cost_optimizer",
                                                 "_penalty_coefs_t", "_cost_limits_return_t"]

    def _costs_from_infos(self, infos: Sequence[Dict[str, Any]]) -> np.ndarray:
        out = np.zeros((len(infos), len(self.constraint_names)), dtype=np.float32)
        for i, info in enumerate(infos):
            costs = info.get("cmdp_costs")
            if costs is None:
                raise ValueError("info 里没有 cmdp_costs：P3O 必须跑在 CMDP env 上"
                                 "（make_env(..., constraint_config=...)）。")
            for j, name in enumerate(self.constraint_names):
                out[i, j] = float(costs[name])
        return out

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback,
                         rollout_buffer: RolloutBuffer, n_rollout_steps: int) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
                cost_values = self.cost_critic(obs_tensor)
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            costs = self._costs_from_infos(infos)
            actions = actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if (done and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                        terminal_cost_value = self.cost_critic(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value
                    costs[idx] += self.cost_gamma * terminal_cost_value.cpu().numpy()
            rollout_buffer.add_costs(costs, cost_values.cpu().numpy())
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts,
                               values, log_probs)
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            last_obs_tensor = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(last_obs_tensor)
            last_cost_values = self.cost_critic(last_obs_tensor)
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        rollout_buffer.compute_cost_returns_and_advantage(last_cost_values=last_cost_values,
                                                          dones=dones)
        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.policy.optimizer, self.cost_optimizer])
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        buf = self.rollout_buffer

        jc_return = buf.cost_returns.reshape(-1, self.n_constraints).mean(axis=0)
        gap = (th.as_tensor(jc_return, dtype=th.float32, device=self.device)
               - self._cost_limits_return_t)

        entropy_losses: List[float] = []
        pg_losses: List[float] = []
        value_losses: List[float] = []
        cost_value_losses: List[float] = []
        penalty_values: List[float] = []
        clip_fractions: List[float] = []
        penalty_active = np.zeros(self.n_constraints, dtype=np.float64)
        n_batches = 0
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in buf.get(self.batch_size):

                actions = rollout_data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                r = ratio.unsqueeze(1)
                ca = rollout_data.cost_advantages
                cost_surrogate = th.max(
                    ca * r, ca * th.clamp(r, 1 - clip_range, 1 + clip_range)).mean(dim=0)

                violation = F.relu(cost_surrogate + gap)
                penalty = (self._penalty_coefs_t * violation).sum()

                cost_values_pred = self.cost_critic(rollout_data.observations)
                cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values_pred)
                cost_value_losses.append(cost_value_loss.item())
                penalty_values.append(penalty.item())
                penalty_active += (violation.detach().cpu().numpy() > 0.0).astype(np.float64)
                n_batches += 1

                loss = (policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                        + self.cost_vf_coef * cost_value_loss + penalty)
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print("Early stopping at step %d due to reaching max kl: %.2f"
                              % (epoch, approx_kl_div))
                    break

                self.policy.optimizer.zero_grad()
                self.cost_optimizer.zero_grad()
                loss.backward()

                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                th.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                self.cost_optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(buf.values.flatten(), buf.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self._record_cost_diagnostics(jc_return, penalty_active, n_batches,
                                      penalty_values, cost_value_losses, pg_losses, value_losses)

    def _record_cost_diagnostics(self, jc_return: np.ndarray, penalty_active: np.ndarray,
                                 n_batches: int, penalty_values: Sequence[float],
                                 cost_value_losses: Sequence[float],
                                 policy_losses: Sequence[float],
                                 value_losses: Sequence[float]) -> None:
        active = penalty_active / max(n_batches, 1)
        row: Dict[str, float] = {
            "timesteps": float(self.num_timesteps),
            "n_updates": float(self._n_updates),
            "penalty_mean": float(np.mean(penalty_values)) if penalty_values else 0.0,
            "cost_value_loss": float(np.mean(cost_value_losses)) if cost_value_losses else 0.0,
            "policy_loss_mean": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss_mean": float(np.mean(value_losses)) if value_losses else 0.0,
        }
        for j, name in enumerate(self.constraint_names):
            per_step = float(jc_return[j] / self.discount_horizon)
            row["jc_return_%s" % name] = float(jc_return[j])
            row["jc_per_step_%s" % name] = per_step
            row["limit_per_step_%s" % name] = float(self.cost_limits_per_step[j])
            row["limit_return_%s" % name] = float(self.cost_limits_return[j])
            row["gap_return_%s" % name] = float(jc_return[j] - self.cost_limits_return[j])
            row["penalty_active_frac_%s" % name] = float(active[j])
            row["kappa_%s" % name] = float(self.penalty_coef_vec[j])
            self.logger.record("p3o/jc_per_step_%s" % name, per_step)
            self.logger.record("p3o/penalty_active_frac_%s" % name, float(active[j]))
        self.logger.record("p3o/penalty", row["penalty_mean"])
        self.logger.record("p3o/cost_value_loss", row["cost_value_loss"])
        self.diagnostics.append(row)

    def provenance(self) -> Dict[str, Any]:
        return {
            "algo": "p3o",
            "citation": "P3O = Zhang et al., IJCAI 2022, doi:10.24963/ijcai.2022/520",
            "penalty_coef": {n: float(self.penalty_coef_vec[j])
                             for j, n in enumerate(self.constraint_names)},
            "cost_limits_per_step": {n: float(self.cost_limits_per_step[j])
                                     for j, n in enumerate(self.constraint_names)},
            "cost_limits_return": {n: float(self.cost_limits_return[j])
                                   for j, n in enumerate(self.constraint_names)},
            "cost_gamma": self.cost_gamma, "constraint_names": list(self.constraint_names),
            "cost_gae_lambda": self.cost_gae_lambda,
            "cost_vf_coef": self.cost_vf_coef,
            "cost_episode_len": self.cost_episode_len,
            "discount_horizon": self.discount_horizon,
            "cost_net_arch": list(self.cost_net_arch),
            "normalize_cost_advantage": False,
            "gamma": float(self.gamma),
            "gae_lambda": float(self.gae_lambda),
            "n_steps": int(self.n_steps),
            "batch_size": int(self.batch_size),
            "n_epochs": int(self.n_epochs),
            "learning_rate_initial": float(self.lr_schedule(1.0)),
            "n_train_calls": len(self.diagnostics),
        }

P3O_ARM_ENV_KWARGS: Dict[str, Any] = dict(mask=False, semantic_state=True, semantic_reward=True)

P3O_ARM_WORKLOAD_SEED_OFFSET = 6

P3O_ARM_FIXED_HYPERPARAMS: Dict[str, Any] = dict(gamma=0.97, gae_lambda=0.92, verbose=0,
                                                 device="cpu")

def build_p3o_arm(env: Any, *, constraint_config: Any, penalty_coef: Any,
                  cost_vf_coef: float, learning_rate: float, n_steps: int,
                  batch_size: int, seed: int) -> "P3O":
    return P3O(
        "MlpPolicy",
        env,
        cost_limits=active_limits(constraint_config), constraint_names=_constraint_names(constraint_config),
        cost_episode_len=env.unwrapped.config.episode_len,
        penalty_coef=penalty_coef,
        cost_vf_coef=cost_vf_coef,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
        **P3O_ARM_FIXED_HYPERPARAMS,
    )
