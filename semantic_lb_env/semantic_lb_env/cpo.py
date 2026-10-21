from __future__ import annotations

import copy
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import TRPO
from sb3_contrib.common.utils import conjugate_gradient_solver
from scipy.optimize import minimize
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from semantic_lb_env.cmdp import CONSTRAINT_NAMES, active_limits, constraint_names as _constraint_names
from semantic_lb_env.p3o import (P3O, P3O_ARM_ENV_KWARGS, P3O_ARM_FIXED_HYPERPARAMS,
                                 P3O_ARM_WORKLOAD_SEED_OFFSET, CostRolloutBuffer)

STEP_CASES = ("trpo", "constrained", "recovery")

def solve_linear_feasibility(c: np.ndarray, S: np.ndarray, ridge: float = 1e-12
                             ) -> Tuple[np.ndarray, float]:
    c = np.asarray(c, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64) + ridge * np.eye(len(c))
    if np.all(c <= 0.0):
        return np.zeros_like(c), 0.0

    def fun(nu: np.ndarray) -> Tuple[float, np.ndarray]:
        Snu = S @ nu
        return -(nu @ c - 0.5 * nu @ Snu), -(c - Snu)

    res = minimize(fun, np.zeros_like(c), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, None)] * len(c),
                   options=dict(maxiter=2000, ftol=1e-15, gtol=1e-12))
    nu = np.maximum(res.x, 0.0)
    return nu, float(0.5 * nu @ S @ nu)

def solve_cpo_dual(q: float, r: np.ndarray, S: np.ndarray, c: np.ndarray, delta: float,
                   x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    r = np.asarray(r, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    eps = 1e-16

    def Q(nu: np.ndarray) -> float:
        return float(q - 2.0 * nu @ r + nu @ S @ nu)

    def fun(nu: np.ndarray) -> Tuple[float, np.ndarray]:
        qv = max(Q(nu), eps)
        root = np.sqrt(2.0 * delta * qv)
        grad = np.sqrt(2.0 * delta) * (S @ nu - r) / np.sqrt(qv) - c
        return root - nu @ c, grad

    starts = [np.zeros_like(c)] + ([np.maximum(np.asarray(x0, dtype=np.float64), 0.0)]
                                   if x0 is not None else [])
    best = None
    for start in starts:
        res = minimize(fun, start, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * len(c),
                       options=dict(maxiter=2000, ftol=1e-15, gtol=1e-12))
        if best is None or res.fun < best.fun:
            best = res
    nu = np.maximum(best.x, 0.0)
    lam = float(np.sqrt(max(Q(nu), eps) / (2.0 * delta)))
    return nu, lam

def plan_cpo_step(*, q: float, r: np.ndarray, S: np.ndarray, c: np.ndarray,
                  b_dot_s: np.ndarray, delta: float) -> Dict[str, Any]:
    c = np.asarray(c, dtype=np.float64)
    m = len(c)
    trpo_scale = float(np.sqrt(2.0 * delta / q)) if q > 0 else 0.0
    predicted = c + trpo_scale * np.asarray(b_dot_s, dtype=np.float64)
    if np.all(predicted <= 0.0):
        return {"case": "trpo", "nu": np.zeros(m), "lam": (1.0 / trpo_scale if trpo_scale > 0
                                                           else float("inf")),
                "feasibility_half_norm_sq": 0.0, "s_coef": None, "p_coefs": None}
    nu_f, half_norm_sq = solve_linear_feasibility(c, S)
    if half_norm_sq > delta:
        denom = float(nu_f @ np.asarray(S, dtype=np.float64) @ nu_f)
        scale = float(np.sqrt(2.0 * delta / denom)) if denom > 1e-300 else 0.0
        return {"case": "recovery", "nu": nu_f, "lam": 0.0,
                "feasibility_half_norm_sq": half_norm_sq,
                "s_coef": 0.0, "p_coefs": -scale * nu_f}
    nu, lam = solve_cpo_dual(q, r, S, c, delta, x0=nu_f)
    return {"case": "constrained", "nu": nu, "lam": lam,
            "feasibility_half_norm_sq": half_norm_sq,
            "s_coef": 1.0 / lam, "p_coefs": -nu / lam}

def accept_step(*, kl_ok: bool, reward_gain_ok: bool, require_reward_gain: bool,
                cost_change: np.ndarray, allowed_cost_increase: np.ndarray) -> bool:
    cost_ok = bool(np.all(np.asarray(cost_change) <= np.asarray(allowed_cost_increase)))
    return bool(kl_ok) and (bool(reward_gain_ok) or not require_reward_gain) and cost_ok

class CPO(TRPO):

    collect_rollouts = P3O.collect_rollouts
    _costs_from_infos = P3O._costs_from_infos

    def __init__(self, policy, env, *, cost_limits: Dict[str, float],
                 cost_gamma: Optional[float] = None, cost_gae_lambda: Optional[float] = None,
                 cost_episode_len: int = 120, cost_net_arch: Sequence[int] = (64, 64),
                 constraint_names: Sequence[str] = CONSTRAINT_NAMES, **kwargs: Any) -> None:
        self.constraint_names = tuple(str(n) for n in constraint_names)
        missing = [n for n in self.constraint_names if n not in cost_limits]
        if missing:
            raise ValueError("cost_limits 缺少约束 %s；必须与本次运行的 constraint_names %s 一一对应"
                             % (missing, list(self.constraint_names)))
        self.n_constraints = len(self.constraint_names)
        self.cost_limits_per_step = np.array([float(cost_limits[n]) for n in self.constraint_names],
                                            dtype=np.float64)
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
        self.diagnostics: List[Dict[str, Any]] = []
        super().__init__(policy, env, **kwargs)
        if not isinstance(self.action_space, spaces.Discrete):
            raise ValueError("本实现只覆盖 Discrete 动作空间（本 env 是离散动作），收到 %r"
                             % (self.action_space,))
        if self.sub_sampling_factor != 1:
            raise ValueError("sub_sampling_factor 必须为 1：子采样会让代价通道与 reward 通道"
                             "看到不同的样本（TRPO 的子采样分支不带代价字段）")

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

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["cost_critic", "cost_optimizer"]

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.policy.optimizer, self.cost_optimizer])

        policy_objective_values = []
        kl_divergences = []
        line_search_results = []
        value_losses = []
        cost_value_losses = []

        buf = self.rollout_buffer

        jc_return = buf.cost_returns.reshape(-1, self.n_constraints).mean(axis=0).astype(np.float64)
        cost_value_mean = buf.cost_values.reshape(-1, self.n_constraints).mean(axis=0).astype(np.float64)
        c_gap = jc_return - self.cost_limits_return
        diag: Dict[str, Any] = {}

        for rollout_data in buf.get(batch_size=None):
            actions = rollout_data.actions.long().flatten()

            with th.no_grad():
                old_distribution = copy.copy(self.policy.get_distribution(rollout_data.observations))

            distribution = self.policy.get_distribution(rollout_data.observations)
            log_prob = distribution.log_prob(actions)

            advantages = rollout_data.advantages
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (rollout_data.advantages.std() + 1e-8)

            ratio = th.exp(log_prob - rollout_data.old_log_prob)
            policy_objective = (advantages * ratio).mean()
            kl_div = kl_divergence(distribution, old_distribution).mean()

            self.policy.optimizer.zero_grad()
            actor_params, policy_objective_gradients, grad_kl, grad_shape = self._compute_actor_grad(
                kl_div, policy_objective)
            hessian_vector_product_fn = partial(self.hessian_vector_product, actor_params, grad_kl)
            search_direction = conjugate_gradient_solver(
                hessian_vector_product_fn, policy_objective_gradients, max_iter=self.cg_max_steps)

            cost_adv = rollout_data.cost_advantages
            cost_adv_mean = cost_adv.mean(dim=0).double().cpu().numpy()
            cost_surrogate_old = (cost_adv * ratio.detach().unsqueeze(1)).mean(dim=0)
            cost_grads = []
            for j in range(self.n_constraints):
                cost_objective = (cost_adv[:, j] * ratio).mean()
                grads = th.autograd.grad(cost_objective, actor_params, retain_graph=True)
                cost_grads.append(th.cat([g.reshape(-1) for g in grads]))

            rng_state = th.get_rng_state()
            cost_directions = [conjugate_gradient_solver(hessian_vector_product_fn, b,
                                                         max_iter=self.cg_max_steps)
                               for b in cost_grads]
            th.set_rng_state(rng_state)

            s_h_s = th.matmul(search_direction, hessian_vector_product_fn(search_direction, retain_graph=False))
            line_search_max_step_size = 2 * self.target_kl
            line_search_max_step_size /= s_h_s
            line_search_max_step_size = th.sqrt(line_search_max_step_size)

            with th.no_grad():
                q = float(s_h_s)
                r = np.array([float(policy_objective_gradients @ p) for p in cost_directions])
                S = np.array([[float(bi @ pj) for pj in cost_directions] for bi in cost_grads])
                S = 0.5 * (S + S.T)
                b_dot_s = np.array([float(b @ search_direction) for b in cost_grads])
            plan = plan_cpo_step(q=q, r=r, S=S, c=c_gap, b_dot_s=b_dot_s, delta=self.target_kl)
            if plan["case"] != "trpo":
                with th.no_grad():
                    full_step = float(plan["s_coef"]) * search_direction
                    for coef, p in zip(plan["p_coefs"], cost_directions):
                        full_step = full_step + float(coef) * p
                search_direction = full_step
                line_search_max_step_size = th.ones((), dtype=search_direction.dtype)
            allowed_cost_increase = np.maximum(-c_gap, 0.0)
            require_reward_gain = plan["case"] != "recovery"

            line_search_backtrack_coeff = 1.0
            original_actor_params = [param.detach().clone() for param in actor_params]

            is_line_search_success = False
            accepted_cost_change = np.zeros(self.n_constraints)
            accepted_coeff = 0.0
            with th.no_grad():
                for _ in range(self.line_search_max_iter):
                    start_idx = 0
                    for param, original_param, shape in zip(actor_params, original_actor_params, grad_shape):
                        n_params = param.numel()
                        param.data = (
                            original_param.data
                            + line_search_backtrack_coeff
                            * line_search_max_step_size
                            * search_direction[start_idx: (start_idx + n_params)].view(shape)
                        )
                        start_idx += n_params

                    distribution = self.policy.get_distribution(rollout_data.observations)
                    log_prob = distribution.log_prob(actions)
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    new_policy_objective = (advantages * ratio).mean()
                    kl_div = kl_divergence(distribution, old_distribution).mean()

                    cost_change = ((cost_adv * ratio.unsqueeze(1)).mean(dim=0)
                                   - cost_surrogate_old).double().cpu().numpy()
                    if accept_step(kl_ok=bool(kl_div < self.target_kl),
                                   reward_gain_ok=bool(new_policy_objective > policy_objective),
                                   require_reward_gain=require_reward_gain,
                                   cost_change=cost_change,
                                   allowed_cost_increase=allowed_cost_increase):
                        is_line_search_success = True
                        accepted_cost_change = cost_change
                        accepted_coeff = float(line_search_backtrack_coeff)
                        break

                    line_search_backtrack_coeff *= self.line_search_shrinking_factor

                line_search_results.append(is_line_search_success)

                if not is_line_search_success:
                    for param, original_param in zip(actor_params, original_actor_params):
                        param.data = original_param.data.clone()
                    policy_objective_values.append(policy_objective.item())
                    kl_divergences.append(0.0)
                else:
                    policy_objective_values.append(new_policy_objective.item())
                    kl_divergences.append(kl_div.item())
            diag = dict(plan=plan, success=is_line_search_success, coeff=accepted_coeff,
                        cost_change=accepted_cost_change, q=q,
                        kl=kl_divergences[-1], cost_adv_mean=cost_adv_mean,
                        cost_value_mean=cost_value_mean)

        for _ in range(self.n_critic_updates):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values_pred = self.policy.predict_values(rollout_data.observations)
                value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
                value_losses.append(value_loss.item())

                self.policy.optimizer.zero_grad()
                value_loss.backward()
                for param in actor_params:
                    param.grad = None
                self.policy.optimizer.step()

                cost_values_pred = self.cost_critic(rollout_data.observations)
                cost_value_loss = F.mse_loss(rollout_data.cost_returns, cost_values_pred)
                cost_value_losses.append(cost_value_loss.item())
                self.cost_optimizer.zero_grad()
                cost_value_loss.backward()
                self.cost_optimizer.step()

        self._n_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(),
                                           self.rollout_buffer.returns.flatten())

        self.logger.record("train/policy_objective", np.mean(policy_objective_values))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/kl_divergence_loss", np.mean(kl_divergences))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/is_line_search_success", np.mean(line_search_results))
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self._record_cost_diagnostics(jc_return, c_gap, diag, cost_value_losses)

    def _record_cost_diagnostics(self, jc_return: np.ndarray, c_gap: np.ndarray,
                                 diag: Dict[str, Any], cost_value_losses: Sequence[float]) -> None:
        plan = diag["plan"]
        row: Dict[str, Any] = {
            "timesteps": float(self.num_timesteps),
            "n_updates": float(self._n_updates),
            "case": plan["case"],
            "line_search_success": bool(diag["success"]),
            "line_search_coeff": float(diag["coeff"]),
            "kl": float(diag["kl"]),
            "lambda": float(plan["lam"]),
            "q": float(diag["q"]),
            "feasibility_half_norm_sq": float(plan["feasibility_half_norm_sq"]),
            "cost_value_loss": float(np.mean(cost_value_losses)) if cost_value_losses else 0.0,
        }
        for j, name in enumerate(self.constraint_names):
            row["jc_return_%s" % name] = float(jc_return[j])
            row["jc_per_step_%s" % name] = float(jc_return[j] / self.discount_horizon)
            row["limit_per_step_%s" % name] = float(self.cost_limits_per_step[j])
            row["limit_return_%s" % name] = float(self.cost_limits_return[j])
            row["gap_return_%s" % name] = float(c_gap[j])
            row["nu_%s" % name] = float(plan["nu"][j])
            row["surrogate_cost_change_%s" % name] = float(diag["cost_change"][j])

            row["cost_adv_mean_%s" % name] = float(diag["cost_adv_mean"][j])
            row["cost_value_mean_%s" % name] = float(diag["cost_value_mean"][j])
            self.logger.record("cpo/jc_per_step_%s" % name, row["jc_per_step_%s" % name])
            self.logger.record("cpo/nu_%s" % name, row["nu_%s" % name])
        for case in STEP_CASES:
            self.logger.record("cpo/case_%s" % case, float(plan["case"] == case))
        self.logger.record("cpo/cost_value_loss", row["cost_value_loss"])
        self.diagnostics.append(row)

    def provenance(self) -> Dict[str, Any]:
        cases = [r["case"] for r in self.diagnostics]
        return {
            "algo": "cpo",
            "citation": "CPO = Achiam, Held, Tamar, Abbeel, ICML 2017（**原文待核**）",
            "constraint_handling": "multi_constraint_dual_over_nu_lambda_closed_form",
            "recovery_step": "min_norm_feasible_direction_scaled_to_trust_region",
            "cost_limits_per_step": {n: float(self.cost_limits_per_step[j])
                                     for j, n in enumerate(self.constraint_names)},
            "cost_limits_return": {n: float(self.cost_limits_return[j])
                                   for j, n in enumerate(self.constraint_names)},
            "constraint_names": list(self.constraint_names),
            "cost_gamma": self.cost_gamma,
            "cost_gae_lambda": self.cost_gae_lambda,
            "cost_episode_len": self.cost_episode_len,
            "discount_horizon": self.discount_horizon,
            "cost_net_arch": list(self.cost_net_arch),
            "normalize_cost_advantage": False,
            "normalize_advantage": bool(self.normalize_advantage),
            "target_kl": float(self.target_kl),
            "cg_max_steps": int(self.cg_max_steps),
            "cg_damping": float(self.cg_damping),
            "line_search_shrinking_factor": float(self.line_search_shrinking_factor),
            "line_search_max_iter": int(self.line_search_max_iter),
            "n_critic_updates": int(self.n_critic_updates),
            "gamma": float(self.gamma),
            "gae_lambda": float(self.gae_lambda),
            "n_steps": int(self.n_steps),
            "batch_size": int(self.batch_size),
            "learning_rate_initial": float(self.lr_schedule(1.0)),
            "n_train_calls": len(self.diagnostics),
            "case_counts": {case: cases.count(case) for case in STEP_CASES},
            "line_search_success_rate": (float(np.mean([r["line_search_success"]
                                                        for r in self.diagnostics]))
                                         if self.diagnostics else None),
        }

CPO_ARM_ENV_KWARGS: Dict[str, Any] = P3O_ARM_ENV_KWARGS
CPO_ARM_WORKLOAD_SEED_OFFSET = P3O_ARM_WORKLOAD_SEED_OFFSET

CPO_ARM_FIXED_HYPERPARAMS: Dict[str, Any] = dict(
    P3O_ARM_FIXED_HYPERPARAMS,

    target_kl=0.01,
    cg_max_steps=15,
    cg_damping=0.1,
    line_search_shrinking_factor=0.8,
    line_search_max_iter=10,
    n_critic_updates=10,
)

def build_cpo_arm(env: Any, *, constraint_config: Any, learning_rate: float, n_steps: int,
                  batch_size: int, seed: int) -> CPO:
    return CPO(
        "MlpPolicy",
        env,
        cost_limits=active_limits(constraint_config),
        constraint_names=_constraint_names(constraint_config),
        cost_episode_len=env.unwrapped.config.episode_len,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        seed=seed,
        **CPO_ARM_FIXED_HYPERPARAMS,
    )
