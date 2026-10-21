from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

CONSTRAINT_NAMES = ("sla", "cost", "failure"); TIERED_CONSTRAINT_NAMES = ("sla_high", "sla_bulk", "cost", "failure"); CONSTRAINT_MODES = ("aggregate", "tiered")

MULTIPLIER_MODES = ("integral", "pid")

@dataclass(frozen=True)
class ConstraintConfig:

    sla_limit: float = 0.02
    cost_limit: float = 0.45
    failure_limit: float = 0.037
    multiplier_lr: float = 0.05
    multiplier_max: float = 20.0
    initial_multiplier: float = 0.0

    multiplier_mode: str = "integral"
    multiplier_kp: float = 0.0
    multiplier_kd: float = 0.0; constraint_mode: str = "aggregate"; sla_high_limit: Optional[float] = None; sla_bulk_limit: Optional[float] = None

    @property
    def limits(self) -> Dict[str, float]:
        return {
            "sla": self.sla_limit,
            "cost": self.cost_limit,
            "failure": self.failure_limit,
        }

class LagrangianRewardWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, config: Optional[ConstraintConfig] = None) -> None:
        super().__init__(env)
        self.constraint_config = config or ConstraintConfig()
        if self.constraint_config.multiplier_mode not in MULTIPLIER_MODES:
            raise ValueError(
                "multiplier_mode 必须是 %s 之一，收到 %r"
                % (list(MULTIPLIER_MODES), self.constraint_config.multiplier_mode)
            )
        self.constraint_names = constraint_names(self.constraint_config); self.multipliers = {name: float(self.constraint_config.initial_multiplier) for name in self.constraint_names}

        self._integral = {name: float(self.constraint_config.initial_multiplier) for name in self.constraint_names}
        self._prev_costs: Dict[str, Optional[float]] = {name: None for name in self.constraint_names}
        self.episode_cost_sums = {name: 0.0 for name in self.constraint_names}
        self.episode_steps = 0

    def reset(self, **kwargs):
        self.episode_cost_sums = {name: 0.0 for name in self.constraint_names}
        self.episode_steps = 0
        return self.env.reset(**kwargs)

    def action_masks(self) -> np.ndarray:
        return self.env.action_masks()

    def step(self, action):
        obs, objective_reward, terminated, truncated, info = self.env.step(action)
        costs = info["cmdp_costs"]
        penalty = sum(self.multipliers[name] * float(costs[name]) for name in self.constraint_names)
        reward = float(objective_reward - penalty)

        self.episode_steps += 1
        for name in self.constraint_names:
            self.episode_cost_sums[name] += float(costs[name])

        info["objective_reward"] = float(objective_reward)
        info["lagrangian_penalty"] = float(penalty)
        info["lagrangian_reward"] = reward
        info["lagrange_multipliers"] = dict(self.multipliers)

        if terminated or truncated:
            divisor = max(self.episode_steps, 1); limits = active_limits(self.constraint_config)
            mean_costs = {name: self.episode_cost_sums[name] / divisor for name in self.constraint_names}
            info["cmdp_episode"] = {
                "mean_costs": mean_costs,
                "limits": limits,
                "violations": {
                    name: max(0.0, mean_costs[name] - limits[name])
                    for name in self.constraint_names
                },
            }

        return obs, reward, terminated, truncated, info

    def update_multipliers(self, mean_costs: Mapping[str, float]) -> Dict[str, float]:
        limits = active_limits(self.constraint_config)
        config = self.constraint_config
        upper = config.multiplier_max
        for name in self.constraint_names:
            cost = float(mean_costs[name])
            if config.multiplier_mode == "integral":
                updated = self.multipliers[name] + config.multiplier_lr * (cost - limits[name])
                self.multipliers[name] = float(np.clip(updated, 0.0, upper))
            else:
                delta = cost - limits[name]
                integral = float(np.clip(self._integral[name] + config.multiplier_lr * delta, 0.0, upper))
                self._integral[name] = integral
                previous = self._prev_costs[name]
                derivative = 0.0 if previous is None else max(0.0, cost - previous)
                self.multipliers[name] = float(
                    np.clip(config.multiplier_kp * delta + integral + config.multiplier_kd * derivative, 0.0, upper)
                )
            self._prev_costs[name] = cost
        return dict(self.multipliers)

class LagrangianCallback(BaseCallback):

    def __init__(self) -> None:
        super().__init__()
        self.rows: List[Dict[str, float]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("cmdp_episode")
            if episode is None:
                continue
            mean_costs = episode["mean_costs"]
            wrapper = self.training_env.envs[0]
            if hasattr(wrapper, "env"):
                env = wrapper.env
            else:
                env = wrapper
            while env is not None and not isinstance(env, LagrangianRewardWrapper):
                env = getattr(env, "env", None)
            if env is None:
                continue
            multipliers = env.update_multipliers(mean_costs)
            row: Dict[str, float] = {"timesteps": float(self.num_timesteps)}
            for name in env.constraint_names:
                row[f"mean_{name}_cost"] = float(mean_costs[name])
                row[f"{name}_limit"] = float(episode["limits"][name])
                row[f"{name}_violation"] = float(episode["violations"][name])
                row[f"lambda_{name}"] = float(multipliers[name])
            self.rows.append(row)
        return True

def constraint_names(config: ConstraintConfig) -> Tuple[str, ...]:
    mode = str(config.constraint_mode)
    if mode not in CONSTRAINT_MODES:
        raise ValueError(
            "constraint_mode 必须是 %s 之一，收到 %r" % (list(CONSTRAINT_MODES), mode)
        )
    if mode == "aggregate":
        return CONSTRAINT_NAMES
    missing = [
        field
        for field, value in (("sla_high_limit", config.sla_high_limit),
                             ("sla_bulk_limit", config.sla_bulk_limit))
        if value is None
    ]
    if missing:
        raise ValueError(
            "constraint_mode='tiered' 要求显式给出 %s；这两条上限必须由定标探针先于"
            "数据标定，不得沿用 sla_limit（聚合上限与摊薄后的分档上限量纲不同）。" % missing
        )
    return TIERED_CONSTRAINT_NAMES

def active_limits(config: ConstraintConfig) -> Dict[str, float]:
    names = constraint_names(config)
    if names is CONSTRAINT_NAMES:
        return config.limits
    return {
        "sla_high": float(config.sla_high_limit),
        "sla_bulk": float(config.sla_bulk_limit),
        "cost": config.cost_limit,
        "failure": config.failure_limit,
    }
