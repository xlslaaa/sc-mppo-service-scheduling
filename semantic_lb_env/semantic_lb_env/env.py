from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces; from semantic_lb_env.tiers import is_high_priority, sla_cost_weight; from semantic_lb_env.trace_replay import level_for_episode, plan_from_env_config; from semantic_lb_env.env_config import TelemetryMaskConfig

@dataclass(frozen=True)
class EnvConfig:
    num_domains: int = 3
    service_types: int = 6
    services_per_domain: int = 5
    episode_len: int = 120
    seed: int = 7
    scenario_seed: Optional[int] = None
    workload_seed: Optional[int] = None

    constrained_objective: bool = False

    arrival_scale: float = 1.0
    service_rate_scale: float = 1.0
    queue_decay: float = 0.90
    load_noise: float = 0.04

    cross_domain_candidates: bool = True

    domain_skew: float = 0.18

    cross_demand_frac: float = 0.0

    sla_aware_candidates: bool = False

    resource_aware_candidates: bool = False

    sla_projection_slack: float = 1.0

    sla_scale: float = 1.0

    failure_frac: float = 0.0

    failure_duration: int = 12

    failure_visible: bool = True

    cross_domain_rtt: float = 0.0

    cross_domain_cost_mult: float = 1.0

    semantic_qos_coupling: bool = True

    trace_profile_path: Optional[str] = None

    semantic_reward: float = 2.0
    mismatch_penalty: float = 6.0
    latency_weight: float = 1.0
    sla_penalty: float = 2.5
    balance_weight: float = 0.8
    reliability_weight: float = 0.8
    cost_weight: float = 0.25

    balance_metric: str = "variance"

    cost_model: str = "synthetic_uniform"
    cost_profile_path: Optional[str] = None
    cost_iso_scale: bool = True; mask_control: str = "enforced"; mask_control_seed: int = 20260826; industrial_pattern: str = "none"; industrial_pattern_path: Optional[str] = None; topology_variant: int = 0; sla_cost_priority_weighted: bool = False; trace_replay: bool = False; trace_replay_block: int = -1; trace_replay_block_len: int = -1; trace_replay_episodes: int = -1; trace_replay_source: Optional[str] = None; telemetry_mask_config: Optional[TelemetryMaskConfig] = None

class SemanticLoadBalancingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        use_semantic_mask: bool = True,
        include_semantic_state: bool = True,
        reward_semantic: bool = True,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig()
        self.use_semantic_mask = use_semantic_mask
        self.include_semantic_state = include_semantic_state
        self.reward_semantic = reward_semantic

        c = self.config
        self.num_services = c.num_domains * c.services_per_domain
        scenario_seed = c.seed if c.scenario_seed is None else c.scenario_seed
        workload_seed = c.seed + 1 if c.workload_seed is None else c.workload_seed
        self.scenario_rng = np.random.default_rng(scenario_seed)
        self.workload_rng = np.random.default_rng(workload_seed)

        self._failure_seed = workload_seed + 9973
        self.failure_rng = np.random.default_rng(self._failure_seed); self._storm_seed = workload_seed + 7919; self.storm_rng = np.random.default_rng(self._storm_seed); self._telemetry_config = c.telemetry_mask_config; self._risk_predictor = None; self._beta_scheduler = None; self._constraint_budgets = {'sla': 1.0, 'cost': 1.0, 'failure': 1.0}; self._telemetry_spent = {}; self._telemetry_budget_limits = None; _tc = c.telemetry_mask_config; self._init_telemetry_masking() if (_tc is not None and _tc.use_telemetry_mask) else None

        self.service_domain = np.repeat(np.arange(c.num_domains), c.services_per_domain)
        self.service_type = np.array(
            [(d * 2 + i) % c.service_types for d in range(c.num_domains) for i in range(c.services_per_domain)],
            dtype=np.int64,
        )
        self.semantic_matrix = self._build_semantic_matrix()

        self.base_capacity = self.scenario_rng.uniform(0.75, 1.25, size=self.num_services).astype(np.float32)
        self.cost = self.scenario_rng.uniform(0.25, 1.0, size=self.num_services).astype(np.float32)
        self.base_reliability = self.scenario_rng.uniform(0.88, 0.995, size=self.num_services).astype(np.float32)

        self._trace_latency_quantiles = self._load_trace_latency(c.trace_profile_path)
        if self._trace_latency_quantiles is not None:
            self.base_latency = self._trace_latency_quantiles.mean(axis=1).astype(np.float32)
        else:
            self.base_latency = self.scenario_rng.uniform(8.0, 28.0, size=self.num_services).astype(np.float32)
        self._apply_cost_model(); self._industrial = self._load_industrial_pattern(c.industrial_pattern, c.industrial_pattern_path); self._replay = plan_from_env_config(c); self._replay_ep = -1; self._replay_mult = None

        self._trace_arrival_quantiles = self._load_trace_arrival(c.trace_profile_path)

        self._latency_semantic_mult = np.ones(self.num_services, dtype=np.float32)
        if c.semantic_qos_coupling:
            for service_id in range(self.num_services):
                domain = self.service_domain[service_id]
                stype = self.service_type[service_id]
                if self.semantic_matrix[domain, stype] == 1:
                    self._latency_semantic_mult[service_id] = 0.75
                    self.base_reliability[service_id] = min(0.998, self.base_reliability[service_id] + 0.025)
                    self.base_capacity[service_id] *= 1.12
                else:
                    self._latency_semantic_mult[service_id] = 1.35
                    self.base_capacity[service_id] *= 0.82
        self.base_latency = (self.base_latency * self._latency_semantic_mult).astype(np.float32)

        semantic_dim = c.num_domains + c.service_types if include_semantic_state else 0
        obs_dim = self.num_services * 2 + semantic_dim + 4
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_services)

        self.t = 0; self._storm_until = -1; self._storm_events = 0; self._storm_steps = 0
        self.queues = np.zeros(self.num_services, dtype=np.float32)
        self.utilization = np.zeros(self.num_services, dtype=np.float32)
        self.current_request: Dict[str, float] = {}

        self.instance_down = np.zeros(self.num_services, dtype=bool)
        self.down_until = np.zeros(self.num_services, dtype=np.int64)

        f = float(c.failure_frac)
        d = max(int(c.failure_duration), 1)
        self._failure_hazard = 0.0 if f <= 0.0 else min(f, 0.95) / (d * (1.0 - min(f, 0.95)))

    def _load_trace_latency(self, profile_path: Optional[str]) -> Optional[np.ndarray]:
        if not profile_path:
            return None
        path = Path(profile_path)
        if not path.is_absolute():
            path = (Path(__file__).resolve().parent.parent / path)
        if not path.exists():
            return None

        profile = json.loads(path.read_text())
        latency = profile.get("latency")
        if not latency or not latency.get("services"):
            return None

        svc = sorted(
            latency["services"].values(),
            key=lambda s: s.get("call_frequency", 0.0),
            reverse=True,
        )
        quant = np.array([s["latency_ms_quantiles"] for s in svc], dtype=np.float64)

        median_col = latency["quantile_levels"].index(0.50)
        pooled_median = float(np.median(quant[:, median_col]))
        target_median = 18.0
        scale = target_median / pooled_median if pooled_median > 0 else 1.0
        quant *= scale

        self._trace_quantile_levels = np.asarray(latency["quantile_levels"], dtype=np.float64)

        idx = (self.scenario_rng.permutation(self.num_services) % quant.shape[0])
        return quant[idx]

    def _sample_service_latency(self, action: int) -> float:
        if self._trace_latency_quantiles is None:
            return float(self.base_latency[action])
        levels = self._trace_quantile_levels
        curve = self._trace_latency_quantiles[action]
        u = float(self.workload_rng.random())
        sampled = float(np.interp(u, levels, curve))
        return sampled * float(self._latency_semantic_mult[action])

    def _load_trace_arrival(self, profile_path: Optional[str]) -> Optional[np.ndarray]:
        if not profile_path:
            return None
        path = Path(profile_path)
        if not path.is_absolute():
            path = (Path(__file__).resolve().parent.parent / path)
        if not path.exists():
            return None
        profile = json.loads(path.read_text())
        arrival = profile.get("arrival")
        if not arrival or not arrival.get("call_rate_quantiles"):
            return None

        curve = np.asarray(arrival["call_rate_quantiles"], dtype=np.float64)
        levels = np.asarray(arrival["quantile_levels"], dtype=np.float64)
        median_col = int(np.argmin(np.abs(levels - 0.50)))
        cr_median = float(curve[median_col])
        if cr_median <= 0:
            return None

        cap = 0.6
        target_median = 0.14
        cr_ref = cr_median * (cap / target_median - 1.0)
        mapped = cap * curve / (curve + cr_ref)
        self._trace_arrival_levels = levels
        return mapped

    def _sample_demand(self) -> float:
        c = self.config
        if self._trace_arrival_quantiles is None:
            return float(self.workload_rng.gamma(shape=2.0, scale=0.07) * c.arrival_scale)
        u = float(self.workload_rng.random())
        sampled = float(np.interp(u, self._trace_arrival_levels, self._trace_arrival_quantiles))

        return sampled * c.arrival_scale

    def _build_semantic_matrix(self) -> np.ndarray:
        c = self.config
        matrix = np.zeros((c.num_domains, c.service_types), dtype=np.int8)
        for domain in range(c.num_domains):
            primary = [(domain * 2) % c.service_types, (domain * 2 + 1) % c.service_types]
            secondary = [(domain * 2 + 2) % c.service_types]
            matrix[domain, primary] = 1
            matrix[domain, secondary] = 1
        return matrix if not c.topology_variant else self._topology_variant_matrix(matrix)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.workload_rng = np.random.default_rng(seed)

            self.failure_rng = np.random.default_rng(seed + 9973); self.storm_rng = np.random.default_rng(seed + 7919)
        self.t = 0; self._storm_until = -1; self._storm_events = 0; self._storm_steps = 0; self._replay_mult = self._replay_advance(); self._telemetry_reset_budgets()
        self.queues = self.workload_rng.uniform(0.05, 0.20, size=self.num_services).astype(np.float32)
        self.utilization = self.workload_rng.uniform(0.05, 0.25, size=self.num_services).astype(np.float32)

        self.instance_down = np.zeros(self.num_services, dtype=bool)
        self.down_until = np.zeros(self.num_services, dtype=np.int64)
        if self.config.failure_frac > 0.0:
            self._init_failures()
        self.current_request = self._sample_request()
        return self._get_obs(), self._info()

    def _init_failures(self) -> None:
        c = self.config
        d = max(int(c.failure_duration), 1)
        frac = min(float(c.failure_frac), 0.95)
        down = self.failure_rng.random(self.num_services) < frac
        remaining = self.failure_rng.integers(1, d + 1, size=self.num_services)
        self.instance_down = down
        self.down_until = np.where(down, remaining, 0).astype(np.int64)

    def _advance_failures(self) -> None:
        if self.config.failure_frac <= 0.0:
            return
        d = max(int(self.config.failure_duration), 1)

        self.instance_down = self.instance_down & (self.t < self.down_until)

        draws = self.failure_rng.random(self.num_services)
        newly_down = (~self.instance_down) & (draws < self._failure_hazard)
        self.instance_down = self.instance_down | newly_down
        self.down_until = np.where(newly_down, self.t + d, self.down_until)

    def _true_action_mask(self) -> np.ndarray:
        req_domain = int(self.current_request["domain"])
        req_type = int(self.current_request["type"])
        compatible_domain = self.semantic_matrix[self.service_domain, req_type] == 1
        compatible_type = self.service_type == req_type
        same_domain = self.service_domain == req_domain

        local_arm = compatible_domain & same_domain

        if self.config.cross_domain_candidates:
            cross_arm = compatible_type
        else:
            cross_arm = compatible_type & same_domain
        mask = cross_arm | local_arm

        if not mask.any():
            mask = (compatible_type | (compatible_domain & same_domain))

        if self.config.failure_frac > 0.0 and self.config.failure_visible:
            healthy = mask & ~self.instance_down
            if healthy.any():
                mask = healthy

        if self.config.sla_aware_candidates:
            mask = self._prune_by_sla(mask, req_type)
        if self.config.resource_aware_candidates:
            mask = self._prune_by_resource(mask)
        return mask.astype(bool) if (self._telemetry_config is None or not self._telemetry_config.use_telemetry_mask) else self._dynamic_mask_telemetry(mask, self.current_request).astype(bool)

    def _projected_latency(self) -> np.ndarray:
        c = self.config
        effective_capacity = self.base_capacity * c.service_rate_scale
        queue_delay = 80.0 * self.queues / (effective_capacity + 0.1)
        utilization_delay = 40.0 * np.maximum(0.0, self.utilization - 0.65)
        projected = self.base_latency + queue_delay + utilization_delay

        if c.cross_domain_rtt != 0.0:
            req_domain = int(self.current_request["domain"])
            projected = projected + c.cross_domain_rtt * (self.service_domain != req_domain)
        return projected

    def _prune_by_sla(self, mask: np.ndarray, req_type: int) -> np.ndarray:
        sla = float(self.current_request["sla"])
        feasible = self._projected_latency() <= sla * float(self.config.sla_projection_slack)
        pruned = mask & feasible
        return pruned if pruned.any() else mask

    def _prune_by_resource(self, mask: np.ndarray) -> np.ndarray:
        c = self.config
        priority = float(self.current_request["priority"])
        demand = float(self.current_request["demand"])

        rel_floor = 0.90 * priority
        reliable_enough = self.base_reliability >= rel_floor

        effective_capacity = self.base_capacity * c.service_rate_scale
        headroom = (self.queues * c.queue_decay + demand) <= (0.85 * effective_capacity)
        pruned = mask & reliable_enough & headroom
        return pruned if pruned.any() else mask

    def action_masks(self) -> np.ndarray:
        if not self.use_semantic_mask:
            return np.ones(self.num_services, dtype=bool)
        return self._true_action_mask() if self.config.mask_control == "enforced" else self._mask_control_override()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        c = self.config
        action = int(action)
        true_mask = self._true_action_mask(); _recall_keys = self._candidate_recall_keys(true_mask)
        valid = bool(true_mask[action])

        req_domain = int(self.current_request["domain"])
        req_type = int(self.current_request["type"])

        served_cross_domain = float(int(self.service_domain[action]) != req_domain)
        local_compatible = (self.service_domain == req_domain) & (
            self.semantic_matrix[self.service_domain, req_type] == 1
        )
        local_available = local_compatible & (self.utilization < 0.85)
        cross_domain_needed = float(not bool(local_available.any()))
        sla = float(self.current_request["sla"])
        priority = float(self.current_request["priority"])
        demand = float(self.current_request["demand"])

        semantic_match = self._semantic_score(action, req_domain, req_type)
        semantic_penalty_latency = 1.0 + (1.0 - semantic_match) * 1.8
        effective_capacity = self.base_capacity[action] * c.service_rate_scale
        queue_delay = 80.0 * self.queues[action] / (effective_capacity + 0.1)
        utilization_delay = 40.0 * max(0.0, self.utilization[action] - 0.65)
        base_service_latency = self._sample_service_latency(action)

        cross_domain_rtt = float(c.cross_domain_rtt) * served_cross_domain
        latency = float(
            (base_service_latency + queue_delay + utilization_delay) * semantic_penalty_latency
            + cross_domain_rtt
        )
        latency_norm = min(latency / 220.0, 1.5)
        sla_violation = max(0.0, (latency - sla) / max(sla, 1.0))
        reliability = float(np.clip(self.base_reliability[action] - 0.10 * self.utilization[action], 0.0, 1.0))
        cost = float(self.cost[action])

        if c.cross_domain_cost_mult != 1.0 and served_cross_domain > 0.0:
            cost *= float(c.cross_domain_cost_mult)

        normalized_cost = cost / float(np.max(self.cost))

        healthy_candidate_exists = bool((true_mask & ~self.instance_down).any())
        routed_to_down = bool(self.instance_down[action])

        instance_down_frac = float(np.mean(self.instance_down))

        success = bool(valid and semantic_match > 0.0 and self.workload_rng.random() < reliability)

        if routed_to_down:
            success = False
        avoidable_routed_to_down = routed_to_down and healthy_candidate_exists
        unavoidable_down_failure = routed_to_down and not healthy_candidate_exists
        background_failure = (not success) and not routed_to_down

        reward = 0.0
        if self.reward_semantic:
            reward += c.semantic_reward * semantic_match
        if not valid:
            reward -= c.mismatch_penalty
        reward -= c.latency_weight * latency_norm
        reward += c.reliability_weight * reliability

        if not c.constrained_objective:
            reward -= c.sla_penalty * sla_violation * (0.5 + priority)
            if not success:
                reward -= c.reliability_weight
            reward -= c.cost_weight * cost

        self._advance_load(action, demand)
        load_balance = float(np.var(self.utilization))
        jain_index = self._jain_index(self.utilization)

        if c.balance_metric == "jain":
            reward += c.balance_weight * (jain_index - 1.0)
        else:
            reward -= c.balance_weight * load_balance
        self.t += 1
        terminated = False
        truncated = self.t >= c.episode_len
        self.current_request = self._sample_request()

        self._advance_failures()
        info = self._info(); _sla_w = sla_cost_weight(priority) if c.sla_cost_priority_weighted else 1.0
        info.update(
            {
                "valid_action": valid,
                "semantic_match": semantic_match,
                "latency": latency,
                "sla": sla, "priority": priority,
                "sla_violation": float(sla_violation > 0.0),
                "throughput": float(success),
                "request_success": float(success),
                "request_failure": float(not success),
                "reliability": reliability,
                "load_variance": load_balance,
                "jain_index": jain_index,
                "served_cross_domain": served_cross_domain,
                "cross_domain_needed": cross_domain_needed,

                "cross_domain_success": served_cross_domain * float(success),

                "unmet_when_cross_needed": cross_domain_needed * float(not success),
                "cost": cost,
                "normalized_cost": normalized_cost,

                "instance_down_frac": instance_down_frac,
                "healthy_candidate_exists": float(healthy_candidate_exists),
                "infeasible_request": float(not healthy_candidate_exists),
                "routed_to_down": float(routed_to_down),
                "avoidable_routed_to_down": float(avoidable_routed_to_down),
                "unavoidable_down_failure": float(unavoidable_down_failure),
                "background_failure": float(background_failure),

                "cross_domain_rtt_ms": cross_domain_rtt,
                "cmdp_costs": {
                    "sla": float(sla_violation > 0.0) * _sla_w, "sla_high": float(sla_violation > 0.0) * _sla_w * float(is_high_priority(priority)), "sla_bulk": float(sla_violation > 0.0) * _sla_w * float(not is_high_priority(priority)),
                    "cost": normalized_cost,
                    "failure": float(not success),
                },
                "true_mask_size": int(true_mask.sum()), "action": int(action), **_recall_keys, "true_mask": tuple(bool(b) for b in true_mask),
            }
        ); self._telemetry_budget_step(info["cmdp_costs"])
        return self._get_obs(), float(reward), terminated, truncated, info

    def _sample_request(self) -> Dict[str, float]:
        c = self.config
        phase = (np.sin(2.0 * np.pi * self.t / max(c.episode_len, 1)) + 1.0) / 2.0

        skew = float(c.domain_skew)
        domain_probs = np.array([0.34 + skew * phase, 0.33, 0.33 - skew * phase], dtype=np.float64)
        domain_probs = np.clip(domain_probs, 0.05, 0.90)
        domain_probs /= domain_probs.sum()
        domain = int(self.workload_rng.choice(c.num_domains, p=domain_probs))
        valid_types = np.where(self.semantic_matrix[domain] == 1)[0]
        req_type = int(self.workload_rng.choice(valid_types))

        if c.cross_demand_frac > 0.0 and float(self.workload_rng.random()) < c.cross_demand_frac:
            incompatible = np.where(self.semantic_matrix[domain] == 0)[0]
            serviceable = incompatible[(self.semantic_matrix[:, incompatible] == 1).any(axis=0)]
            if serviceable.size > 0:
                req_type = int(self.workload_rng.choice(serviceable))
        priority = float(self.workload_rng.choice([0.4, 0.7, 1.0], p=[0.45, 0.35, 0.20]))

        sla = float(self.workload_rng.choice([55.0, 80.0, 120.0], p=[0.25, 0.50, 0.25]) / priority) * c.sla_scale
        demand = self._sample_demand(); demand = demand if self._replay_mult is None else demand * self._replay_mult
        return self._industrial_pattern_apply({"domain": domain, "type": req_type, "sla": sla, "priority": priority, "demand": demand})

    def _advance_load(self, action: int, demand: float) -> None:
        c = self.config
        self.queues *= c.queue_decay
        self.queues[action] += demand
        noise = self.workload_rng.normal(0.0, c.load_noise, size=self.num_services).astype(np.float32)
        self.utilization = np.clip(0.78 * self.utilization + 0.22 * self.queues + noise, 0.0, 1.0)
        self.queues = np.clip(self.queues, 0.0, 1.0)

    @staticmethod
    def _jain_index(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        denom = float(x.size) * float(np.sum(x * x))
        if denom <= 0.0:
            return 1.0
        return float(np.sum(x) ** 2 / denom)

    def _semantic_score(self, action: int, req_domain: int, req_type: int) -> float:
        score = 0.0
        if self.service_type[action] == req_type:
            score += 0.7
        if self.service_domain[action] == req_domain:
            score += 0.3
        elif self.semantic_matrix[self.service_domain[action], req_type] == 1:
            score += 0.15
        return float(min(score, 1.0))

    def _get_obs(self) -> np.ndarray:
        c = self.config
        parts = [self.queues, self.utilization]
        if self.include_semantic_state:
            domain_oh = np.zeros(c.num_domains, dtype=np.float32)
            type_oh = np.zeros(c.service_types, dtype=np.float32)
            domain_oh[int(self.current_request["domain"])] = 1.0
            type_oh[int(self.current_request["type"])] = 1.0
            parts.extend([domain_oh, type_oh])
        misc = np.array(
            [
                min(float(self.current_request["sla"]) / 150.0, 1.0),
                float(self.current_request["priority"]),
                min(float(self.current_request["demand"]) / 0.5, 1.0),
                self.t / max(c.episode_len, 1),
            ],
            dtype=np.float32,
        )
        parts.append(misc)
        return np.concatenate(parts).astype(np.float32)

    def _info(self) -> Dict:
        return {
            "t": self.t,
            "request_domain": int(self.current_request["domain"]),
            "request_type": int(self.current_request["type"]),
            "mask_size": int(self.action_masks().sum()),
            "true_mask_size": int(self._true_action_mask().sum()),
        }

    def _apply_cost_model(self) -> None:
        c = self.config

        self._precoupling_cost_keys = (
            np.asarray(self.base_capacity, dtype=np.float64).copy(),
            np.asarray(self.base_latency, dtype=np.float64).copy(),
        )
        model = str(c.cost_model)
        if model == "synthetic_uniform":
            return
        if model != "power_based":
            raise ValueError(
                "cost_model 只支持 'synthetic_uniform' 与 'power_based'，收到 %r" % (c.cost_model,)
            )

        profile = self._load_cost_profile(c.cost_profile_path)
        synthetic = self.cost.copy()

        phys = self._derive_physical_cost(profile)
        tier_index = phys["tier_index"]
        energy_wh = phys["energy_wh"]
        cny = phys["cny"]
        pue = phys["pue"]
        tariff = phys["tariff"]
        n = self.num_services

        self.cost_energy_wh = energy_wh.astype(np.float64)
        self.cost_cny_per_request = cny.astype(np.float64)

        if c.cost_iso_scale:

            lo, hi = float(synthetic.min()), float(synthetic.max())
            rlo, rhi = float(cny.min()), float(cny.max())
            if rhi - rlo <= 0.0:
                scaled = np.full(n, 0.5 * (lo + hi), dtype=np.float64)
            else:
                scaled = lo + (cny - rlo) * ((hi - lo) / (rhi - rlo))
        else:

            scaled = cny * (float(synthetic.mean()) / float(cny.mean()))
        self.cost = scaled.astype(np.float32)

        self.cost_synthetic = synthetic
        self.cost_model_meta = {
            "cost_model": model,
            "profile_name": profile.get("name"),
            "iso_scale": bool(c.cost_iso_scale),
            "assignment": profile.get("assignment", "capacity_rank"),
            "tier_index": tier_index.tolist(),
            "tier_power_w": [float(t["power_active_w"]) for t in profile["tiers"]],
            "pue": pue,
            "tariff_cny_per_kwh": tariff,
            "energy_wh_per_request_mean": float(energy_wh.mean()),
            "cny_per_request_mean": float(cny.mean()),
            "cost_max": float(np.max(self.cost)),
        }

    def _load_cost_profile(self, profile_path: Optional[str]) -> Dict:
        if not profile_path:
            raise ValueError("cost_model='power_based' 必须同时给出 cost_profile_path")
        path = Path(profile_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent / path
        if not path.exists():
            raise FileNotFoundError("找不到成本 profile：%s" % (path,))
        profile = json.loads(path.read_text())
        schema = profile.get("schema")
        if schema != "power_cost_profile/v1":
            raise ValueError("成本 profile schema 不是 power_cost_profile/v1，收到 %r" % (schema,))
        tiers = profile.get("tiers") or []
        if not tiers:
            raise ValueError("成本 profile 里没有 tiers")
        for tier in tiers:
            if float(tier.get("power_active_w", 0.0)) <= 0.0:
                raise ValueError("tier %r 的 power_active_w 必须为正" % (tier.get("name"),))
        for key in ("pue", "tariff_cny_per_kwh"):
            if float((profile.get(key) or {}).get("value", 0.0)) <= 0.0:
                raise ValueError("成本 profile 缺少正的 %s.value" % (key,))
        return profile

    def _derive_physical_cost(self, profile: Dict) -> Dict:
        tiers = profile["tiers"]
        n = self.num_services
        assignment = str(profile.get("assignment", "capacity_rank"))

        cap_key, lat_key = getattr(
            self,
            "_precoupling_cost_keys",
            (
                np.asarray(self.base_capacity, dtype=np.float64),
                np.asarray(self.base_latency, dtype=np.float64),
            ),
        )
        if assignment == "capacity_rank":
            key = cap_key
        elif assignment == "latency_rank":
            key = lat_key
        else:
            raise ValueError(
                "成本 profile 的 assignment 只支持 'capacity_rank' / 'latency_rank'，收到 %r"
                % (assignment,)
            )
        order = np.argsort(key, kind="stable")
        shares = np.asarray([float(t["share"]) for t in tiers], dtype=np.float64)
        shares = shares / shares.sum()
        bounds = np.rint(np.cumsum(shares) * n).astype(np.int64)
        bounds[-1] = n
        power_w = np.zeros(n, dtype=np.float64)
        tier_index = np.zeros(n, dtype=np.int64)
        start = 0
        for k, tier in enumerate(tiers):
            stop = int(max(bounds[k], start))
            slots = order[start:stop]
            power_w[slots] = float(tier["power_active_w"])
            tier_index[slots] = k
            start = stop
        pue = float(profile["pue"]["value"])
        tariff = float(profile["tariff_cny_per_kwh"]["value"])
        service_time_s = lat_key / 1000.0
        energy_wh = power_w * service_time_s / 3600.0 * pue
        cny = energy_wh / 1000.0 * tariff
        return {
            "power_w": power_w,
            "tier_index": tier_index,
            "energy_wh": energy_wh,
            "cny": cny,
            "pue": pue,
            "tariff": tariff,
            "assignment": assignment,
        }

    def physical_cost_vectors(self, profile_path: Optional[str] = None) -> Dict:

        path = profile_path or self.config.cost_profile_path or "configs/power_cost_profile.json"
        return self._derive_physical_cost(self._load_cost_profile(path))

    def _deadline_feasible_mask(self) -> np.ndarray:
        req_domain = int(self.current_request["domain"])
        req_type = int(self.current_request["type"])
        sla = float(self.current_request["sla"])

        same_domain = self.service_domain == req_domain
        compatible_domain = self.semantic_matrix[self.service_domain, req_type] == 1
        compatible_type = self.service_type == req_type

        reference = self.reference_action_mask()
        score = 0.7 * compatible_type.astype(np.float64)
        score = np.minimum(score + np.where(same_domain, 0.3, 0.15 * compatible_domain), 1.0)
        effective_capacity = self.base_capacity * self.config.service_rate_scale
        queue_delay = 80.0 * self.queues / (effective_capacity + 0.1)
        utilization_delay = 40.0 * np.maximum(0.0, self.utilization - 0.65)
        projected = (
            np.asarray(self.base_latency, dtype=np.float64) + queue_delay + utilization_delay
        ) * (1.0 + (1.0 - score) * 1.8)
        if self.config.cross_domain_rtt != 0.0:
            projected = projected + float(self.config.cross_domain_rtt) * (~same_domain)
        return reference & ~self.instance_down & (projected <= sla)

    def _candidate_recall_keys(self, candidate_mask: np.ndarray) -> Dict[str, int]:
        feasible = self._deadline_feasible_mask()
        return {
            "feasible_size": int(feasible.sum()),
            "retained_feasible_size": int((np.asarray(candidate_mask, dtype=bool) & feasible).sum()),
        }

    def _mask_control_entropy(self) -> Tuple[int, ...]:
        req = self.current_request
        vals = (
            int(self.config.mask_control_seed),
            int(self.t),
            int(req["domain"]),
            int(req["type"]),
            int(round(float(req["sla"]) * 1e6)),
            int(round(float(req["priority"]) * 1e6)),
            int(round(float(req["demand"]) * 1e6)),
        )
        return tuple(v & 0xFFFFFFFF for v in vals)

    def _mask_control_override(self) -> np.ndarray:
        mode = str(self.config.mask_control)
        if mode == "advisory":
            return np.ones(self.num_services, dtype=bool)
        if mode != "size_matched":
            raise ValueError(
                "未知的 mask_control=%r；合法值只有 'enforced' / 'size_matched' / 'advisory'" % (mode,)
            )
        k = int(self._true_action_mask().sum())
        rng = np.random.default_rng(self._mask_control_entropy())
        out = np.zeros(self.num_services, dtype=bool)
        out[rng.choice(self.num_services, size=k, replace=False)] = True
        return out

    def _load_industrial_pattern(self, name: str, path: Optional[str]) -> Optional[Dict[str, float]]:
        if str(name) == "none":
            return None
        if not path:
            raise ValueError("industrial_pattern=%r 必须同时给出 industrial_pattern_path"
                             "（缺文件是硬错误，不静默回退到 none）" % (name,))
        import json as _json

        _p = Path(path)
        if not _p.is_absolute():
            _p = Path(__file__).resolve().parent.parent / _p
        if not _p.exists():
            raise FileNotFoundError("找不到工业负载模式 profile：%s" % (_p,))
        blob = _json.loads(_p.read_text(encoding="utf-8"))
        if blob.get("schema") != "industrial_workload_profile/v1":
            raise ValueError("%s 不是 industrial_workload_profile/v1：schema=%r"
                             % (path, blob.get("schema")))
        table = blob.get("patterns", {})
        if name not in table:
            raise ValueError("未知的 industrial_pattern=%r；档位表里只有 %s"
                             % (name, sorted(table)))
        raw = table[name]
        L = int(max(self.config.episode_len, 1))
        P = int(raw.get("takt_period_steps", 0))
        A = float(raw.get("takt_amplitude", 0.0))
        n = float(raw.get("storm_expected_count", 0.0))
        D = int(raw.get("storm_duration_steps", 0))
        sm = float(raw.get("storm_demand_mult", 1.0))
        sp = float(raw.get("storm_priority", 0.0))
        sf = float(raw.get("shift_at_frac", 0.0))
        shm = float(raw.get("shift_demand_mult", 1.0))
        if not (0.0 <= A < 1.0):
            raise ValueError("takt_amplitude 必须在 [0, 1)：%r" % (A,))
        if A > 0.0:
            if P < 2 or P % 2 != 0:
                raise ValueError("takt_period_steps 必须是 >=2 的偶数（否则两相步数不等、"
                                 "均值守恒不再是精确事实）：%r" % (P,))
            if L % P != 0:
                raise ValueError("takt_period_steps=%d 必须整除 episode_len=%d，"
                                 "否则 episode 上的乘子均值不等于 1.0" % (P, L))
        if not (0.0 <= sf < 1.0):
            raise ValueError("shift_at_frac 必须在 [0, 1)：%r" % (sf,))
        if shm <= 0.0 or sm <= 0.0:
            raise ValueError("demand 乘子必须为正：storm=%r shift=%r" % (sm, shm))
        if not (0.0 <= sp <= 1.0):
            raise ValueError("storm_priority 必须在 [0, 1]（0 = 不改写优先级）：%r" % (sp,))
        hazard = self._industrial_storm_hazard(n, D, L) if n > 0.0 else 0.0
        mean_mult = 1.0
        if hazard > 0.0:
            mean_mult *= 1.0 + (float(D) * n / float(L)) * (sm - 1.0)
        if sf > 0.0:
            mean_mult *= (1.0 - sf) + sf * shm
        return {
            "pattern": name, "takt_period_steps": P, "takt_amplitude": A,
            "storm_hazard": hazard, "storm_expected_count": n, "storm_duration_steps": D,
            "storm_demand_mult": sm, "storm_priority": sp,
            "shift_at_frac": sf, "shift_step_index": int(sf * L), "shift_demand_mult": shm,
            "mean_demand_multiplier_analytic": mean_mult, "episode_len": L,
            "profile_path": str(path),
        }

    @staticmethod
    def _industrial_storm_hazard(expected_count: float, duration: int, episode_len: int) -> float:
        n = float(expected_count)
        idle = float(episode_len) - float(duration) * n
        if n <= 0.0:
            return 0.0
        if idle <= 1.0:
            raise ValueError("storm_expected_count=%r × duration=%r 几乎占满 episode_len=%r，"
                             "hazard 无解" % (expected_count, duration, episode_len))
        h = n / idle
        if not (0.0 < h <= 1.0):
            raise ValueError("推导出的 hazard 不在 (0, 1]：%r" % (h,))
        return h

    def _industrial_pattern_apply(self, req: Dict[str, float]) -> Dict[str, float]:
        p = self._industrial
        if p is None:
            return req
        t = int(self.t)
        m = 1.0
        P = int(p["takt_period_steps"])
        A = float(p["takt_amplitude"])
        if A > 0.0 and P >= 2:
            m *= (1.0 + A) if (t % P) < (P // 2) else (1.0 - A)
        if float(p["shift_at_frac"]) > 0.0 and t >= int(p["shift_step_index"]):
            m *= float(p["shift_demand_mult"])
        if float(p["storm_hazard"]) > 0.0:
            if t >= self._storm_until:
                if float(self.storm_rng.random()) < float(p["storm_hazard"]):
                    self._storm_until = t + int(p["storm_duration_steps"])
                    self._storm_events += 1
            if t < self._storm_until:
                self._storm_steps += 1
                m *= float(p["storm_demand_mult"])
                sp = float(p["storm_priority"])
                if sp > 0.0:
                    old = float(req["priority"])
                    req["sla"] = float(req["sla"]) * (old / sp)
                    req["priority"] = sp
        if m != 1.0:
            req["demand"] = float(req["demand"]) * m
        return req

    def _topology_variant_matrix(self, historical: np.ndarray) -> np.ndarray:
        from . import topology as _tp

        c = self.config
        matrix = _tp.build_matrix(c.topology_variant, c.num_domains, c.service_types); _tp.assert_scale_compatible(c.topology_variant, c.services_per_domain)
        if matrix.shape != historical.shape or matrix.dtype != historical.dtype:
            raise ValueError(
                f"topology_variant={c.topology_variant} 产出的矩阵形状/dtype 与历史路径不符："
                f"{matrix.shape}/{matrix.dtype} vs {historical.shape}/{historical.dtype}"
            )

        if not (matrix.sum(axis=1) == _tp.ROW_DEGREE).all():
            raise ValueError(
                f"topology_variant={c.topology_variant} 的行度不是 {_tp.ROW_DEGREE}："
                f"{matrix.sum(axis=1).tolist()} ⇒ workload_rng 流会错位，拒绝构造 env"
            )
        return matrix

    def reference_action_mask(self) -> np.ndarray:
        req_domain = int(self.current_request["domain"])
        req_type = int(self.current_request["type"])
        same_domain = self.service_domain == req_domain
        compatible_domain = self.semantic_matrix[self.service_domain, req_type] == 1
        compatible_type = self.service_type == req_type
        return compatible_type | (compatible_domain & same_domain)

    def deadline_feasible_mask(self) -> np.ndarray:
        return self._deadline_feasible_mask()

    def trace_profile_status(self) -> Dict[str, object]:
        return {
            "requested": self.config.trace_profile_path,
            "latency_loaded": self._trace_latency_quantiles is not None,
            "arrival_loaded": self._trace_arrival_quantiles is not None,
        }

    def _replay_advance(self) -> Optional[float]:
        p = self._replay
        if p is None:
            return None
        self._replay_ep += 1
        return level_for_episode(p, self._replay_ep)

    def _telemetry_reset_budgets(self) -> None:
        cfg = self._telemetry_config
        if cfg is None:
            return
        if self._telemetry_budget_limits is None:
            if cfg.budget_limits is not None:
                if len(cfg.budget_limits) != len(cfg.budget_keys):
                    raise ValueError(
                        f"budget_limits 长度 ({len(cfg.budget_limits)}) 与 budget_keys 长度 "
                        f"({len(cfg.budget_keys)}) 不一致")
                self._telemetry_budget_limits = {
                    k: float(v) for k, v in zip(cfg.budget_keys, cfg.budget_limits)}
            else:
                from semantic_lb_env.cmdp import ConstraintConfig, active_limits

                _defaults = active_limits(ConstraintConfig())

                _missing = [k for k in cfg.budget_keys if k not in _defaults]
                if _missing:
                    raise KeyError(
                        f"budget_keys 含 ConstraintConfig 未定义上限的键 {_missing}；"
                        f"可用: {sorted(_defaults)}。请显式给 budget_limits。")
                self._telemetry_budget_limits = {k: float(_defaults[k]) for k in cfg.budget_keys}
        self._telemetry_spent = {k: 0.0 for k in cfg.budget_keys}
        self.update_constraint_budgets({k: 1.0 for k in cfg.budget_keys})

    def _telemetry_budget_step(self, step_costs: Dict[str, float]) -> None:
        cfg = self._telemetry_config
        if cfg is None:
            return
        horizon = float(self.config.episode_len)
        _new = {}
        for k in cfg.budget_keys:
            if k not in step_costs:
                raise KeyError(
                    f"cmdp_costs 缺少预算键 {k!r}；实际可用键: {sorted(step_costs)}")
            self._telemetry_spent[k] += float(step_costs[k])
            allowance = float(self._telemetry_budget_limits[k]) * horizon
            _new[k] = 0.0 if allowance <= 0.0 else float(
                np.clip(1.0 - self._telemetry_spent[k] / allowance, 0.0, 1.0))
        self.update_constraint_budgets(_new)

    def _init_telemetry_masking(self):
        from semantic_lb_env.predictors import RiskPredictor
        from semantic_lb_env.beta_scheduler import BetaScheduler

        cfg = self._telemetry_config

        model_path = cfg.risk_predictor_path
        if model_path is None:

            from semantic_lb_env.env_config import DEFAULT_RISK_PREDICTOR_PATH; model_path = DEFAULT_RISK_PREDICTOR_PATH

        if Path(model_path).exists():
            self._risk_predictor = RiskPredictor.load(str(model_path), device=cfg.predictor_device)
        else:

            self._risk_predictor = RiskPredictor(input_dim=13)
            self._risk_predictor.to(cfg.predictor_device)

        self._beta_scheduler = BetaScheduler(
            strategy=cfg.beta_strategy,
            beta_min=cfg.beta_min,
            beta_max=cfg.beta_max,
            epsilon=cfg.beta_epsilon, beta_fixed=cfg.beta_fixed
        )

    def _extract_risk_features(self, mask: np.ndarray, request: Dict[str, float]) -> np.ndarray:
        c = self.config
        candidate_indices = np.where(mask)[0]
        n_candidates = len(candidate_indices)

        if n_candidates == 0:
            return np.empty((0, 13), dtype=np.float32)

        features = np.zeros((n_candidates, 13), dtype=np.float32)

        req_domain = int(request["domain"])
        req_sla = float(request["sla"])
        req_priority = float(request["priority"])
        req_demand = float(request["demand"])

        max_queue = 2.0
        max_latency = 100.0
        max_demand = 1.5
        max_sla = 300.0
        max_rtt = 100.0; min_rel = 0.88; rel_span = 0.12

        for i, service_id in enumerate(candidate_indices):

            features[i, 0] = min(self.queues[service_id] / max_queue, 1.0)

            features[i, 1] = self.utilization[service_id]

            features[i, 2] = min(self.base_latency[service_id] / max_latency, 1.0)

            features[i, 3] = min(req_demand / max_demand, 1.0)

            features[i, 4] = min(req_sla / max_sla, 1.0)

            features[i, 5] = req_priority

            features[i, 6] = float(self.service_domain[service_id] != req_domain)

            if c.cross_domain_rtt != 0.0 and self.service_domain[service_id] != req_domain:
                features[i, 7] = min(c.cross_domain_rtt / max_rtt, 1.0)
            else:
                features[i, 7] = 0.0

            features[i, 8] = self._constraint_budgets.get('sla', 1.0)
            features[i, 9] = self._constraint_budgets.get('cost', 1.0)
            features[i, 10] = self._constraint_budgets.get('failure', 1.0)

            features[i, 11] = float(self.t) / float(c.episode_len); features[i, 12] = min(max((float(self.base_reliability[service_id]) - min_rel) / rel_span, 0.0), 1.0)

        return features

    def _compute_budget_remaining(self) -> float:
        cfg = self._telemetry_config

        budgets = []
        weights = []

        for key, weight in zip(cfg.budget_keys, cfg.budget_weights):
            budget = self._constraint_budgets.get(key, 1.0)
            budgets.append(budget)
            weights.append(weight)

        weighted_budget = sum(b * w for b, w in zip(budgets, weights))

        return float(np.clip(weighted_budget, 0.0, 1.0))

    def _dynamic_mask_telemetry(self, base_mask: np.ndarray, request: Dict[str, float]) -> np.ndarray:
        cfg = self._telemetry_config

        if not base_mask.any():
            return base_mask

        features = self._extract_risk_features(base_mask, request)

        if features.shape[0] == 0:
            return base_mask

        mu, sigma = self._risk_predictor.predict(features, device=cfg.predictor_device)

        budget = self._compute_budget_remaining()
        beta = self._beta_scheduler(budget)

        candidate_indices = np.where(base_mask)[0]
        ucb_mask = np.copy(base_mask)

        if cfg.ucb_threshold_sla is not None:
            threshold_sla = cfg.ucb_threshold_sla
        else:

            threshold_sla = min(float(request["sla"]) / 300.0, 1.0)

        threshold_failure = cfg.ucb_threshold_failure

        for i, service_id in enumerate(candidate_indices):
            ucb_sla = mu[i, 0] + beta * sigma[i, 0]
            ucb_failure = mu[i, 1] + beta * sigma[i, 1]

            if ucb_sla > threshold_sla or ucb_failure > threshold_failure:
                ucb_mask[service_id] = False

        if cfg.fallback_to_base_mask and not ucb_mask.any():
            return base_mask

        return ucb_mask

    def update_constraint_budgets(self, budgets: Dict[str, float]):
        self._constraint_budgets.update(budgets)
