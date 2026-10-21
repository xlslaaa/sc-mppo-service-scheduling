from __future__ import annotations

from typing import Dict, List, Mapping, Optional

import numpy as np; from semantic_lb_env.tiers import tier_masks as priority_tier_masks

DEFAULT_CONSTRAINT_LIMITS: Dict[str, float] = {
    "sla": 0.02,
    "cost": 0.45,
    "failure": 0.037,
}

def evaluate_policy(
    env,
    policy,
    episodes: int = 20,
    use_masks: bool = False,
    seed: int = 1000,
    deterministic: bool = True,
    constraint_limits: Optional[Mapping[str, float]] = None,
    own_limits: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    limits = dict(constraint_limits) if constraint_limits is not None else dict(DEFAULT_CONSTRAINT_LIMITS)

    import time

    rows: List[Dict[str, float]] = []

    ep_cmdp_costs: List[Dict[str, float]] = []

    decision_us: List[float] = []
    mask_build_us: List[float] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        ep_cost_sums = {"sla": 0.0, "cost": 0.0, "failure": 0.0}
        ep_steps = 0
        while not done:
            _t_mask = time.perf_counter()
            action_masks = env.action_masks() if use_masks else None
            _t_pred = time.perf_counter()
            if use_masks:
                action, _ = policy.predict(obs, deterministic=deterministic, action_masks=action_masks)
            else:
                action, _ = policy.predict(obs, deterministic=deterministic)
            _t_done = time.perf_counter()
            decision_us.append((_t_done - _t_mask) * 1e6)
            mask_build_us.append((_t_pred - _t_mask) * 1e6 if use_masks else 0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)
            rows.append({"episode": ep, "reward": reward, **info})
            costs = info.get("cmdp_costs")
            if costs is not None:
                for name in costs:
                    ep_cost_sums[name] = ep_cost_sums.get(name, 0.0) + float(costs[name])
                ep_steps += 1
        rows.append({"episode": ep, "episode_reward": ep_reward})
        divisor = max(ep_steps, 1)
        ep_cmdp_costs.append({name: ep_cost_sums[name] / divisor for name in ep_cost_sums})

    step_rows = [r for r in rows if "latency" in r]
    rewards = [r["reward"] for r in step_rows]
    latencies = [r["latency"] for r in step_rows]
    violations = [r["sla_violation"] for r in step_rows]
    throughputs = [r["throughput"] for r in step_rows]
    failures = [r.get("request_failure", 1.0 - r["throughput"]) for r in step_rows]
    normalized_costs = [r.get("normalized_cost", r["cost"]) for r in step_rows]
    valid_actions = [r["valid_action"] for r in step_rows]
    semantic = [r["semantic_match"] for r in step_rows]
    load_var = [r["load_variance"] for r in step_rows]

    jain = [r.get("jain_index", 0.0) for r in step_rows]

    served_cross = [r.get("served_cross_domain", 0.0) for r in step_rows]
    cross_needed = [r.get("cross_domain_needed", 0.0) for r in step_rows]
    cross_success = [r.get("cross_domain_success", 0.0) for r in step_rows]
    unmet_cross = [r.get("unmet_when_cross_needed", 0.0) for r in step_rows]

    cand_sizes = [r["true_mask_size"] for r in step_rows if "true_mask_size" in r]

    down_frac = [r.get("instance_down_frac", 0.0) for r in step_rows]
    routed_down = [r.get("routed_to_down", 0.0) for r in step_rows]
    avoidable_down = [r.get("avoidable_routed_to_down", 0.0) for r in step_rows]
    infeasible = [r.get("infeasible_request", 0.0) for r in step_rows]
    unavoidable_down = [r.get("unavoidable_down_failure", 0.0) for r in step_rows]
    background_fail = [r.get("background_failure", 0.0) for r in step_rows]
    rtt_paid = [r.get("cross_domain_rtt_ms", 0.0) for r in step_rows]
    ep_rewards = [r["episode_reward"] for r in rows if "episode_reward" in r]

    prio_arr = np.asarray([r.get("priority", float("nan")) for r in step_rows], dtype=float)
    viol_arr = np.asarray(violations, dtype=float)
    tier_masks = priority_tier_masks(prio_arr)

    def _tier_miss(tier: str) -> float:
        mask = tier_masks[tier]
        return float(np.mean(viol_arr[mask])) if bool(mask.any()) else 0.0

    actions = [r["action"] for r in step_rows if "action" in r]
    energy_wh_per_req = 0.0
    cny_per_req = 0.0
    if step_rows and len(actions) == len(step_rows):

        base_env = env if hasattr(env, "physical_cost_vectors") else getattr(env, "unwrapped", None)
        try:
            phys = base_env.physical_cost_vectors() if base_env is not None else None
        except (FileNotFoundError, ValueError, AttributeError):
            phys = None
        if phys is not None:
            idx = np.asarray(actions, dtype=np.int64)
            energy_wh_per_req = float(np.mean(np.asarray(phys["energy_wh"])[idx]))
            cny_per_req = float(np.mean(np.asarray(phys["cny"])[idx]))

    feas_arr = np.asarray([r.get("feasible_size", float("nan")) for r in step_rows], dtype=float)
    kept_arr = np.asarray([r.get("retained_feasible_size", float("nan")) for r in step_rows], dtype=float)
    cand_arr = np.asarray([r.get("true_mask_size", float("nan")) for r in step_rows], dtype=float)
    _has_recall = np.isfinite(feas_arr) & np.isfinite(kept_arr)
    _defined = _has_recall & (feas_arr > 0.0)
    _cand_defined = _defined & np.isfinite(cand_arr)
    recall_micro = float(kept_arr[_defined].sum() / feas_arr[_defined].sum()) if _defined.any() else 0.0
    recall_macro = float(np.mean(kept_arr[_defined] / feas_arr[_defined])) if _defined.any() else 0.0
    _cand_total = float(cand_arr[_cand_defined].sum()) if _cand_defined.any() else 0.0
    precision_micro = float(kept_arr[_cand_defined].sum() / _cand_total) if _cand_total > 0.0 else 0.0
    infeasible_rate = float(np.mean(feas_arr[_has_recall] == 0.0)) if _has_recall.any() else 0.0
    mean_feasible = float(np.mean(feas_arr[_has_recall])) if _has_recall.any() else 0.0

    result = {
        "mean_reward": float(np.mean(rewards)),
        "mean_episode_reward": float(np.mean(ep_rewards)),
        "mean_latency": float(np.mean(latencies)),
        "p95_latency": float(np.percentile(latencies, 95)),

        "p99_latency": float(np.percentile(latencies, 99)),
        "latency_std": float(np.std(latencies)),
        "latency_iqr": float(np.percentile(latencies, 75) - np.percentile(latencies, 25)),
        "sla_violation_rate": float(np.mean(violations)),
        "throughput_rate": float(np.mean(throughputs)),
        "failure_rate": float(np.mean(failures)),
        "mean_normalized_cost": float(np.mean(normalized_costs)),
        "valid_action_rate": float(np.mean(valid_actions)),
        "semantic_match": float(np.mean(semantic)),
        "load_variance": float(np.mean(load_var)),
        "jain_index": float(np.mean(jain)),

        "cross_domain_route_rate": float(np.mean(served_cross)),
        "cross_domain_need_rate": float(np.mean(cross_needed)),
        "cross_domain_success_rate": (
            float(np.sum(cross_success) / np.sum(served_cross))
            if np.sum(served_cross) > 0 else 0.0
        ),
        "unmet_demand_rate": float(np.mean(unmet_cross)),

        "mean_candidate_set_size": float(np.mean(cand_sizes)) if cand_sizes else 0.0,

        "instance_down_frac": float(np.mean(down_frac)),
        "routed_to_down_rate": float(np.mean(routed_down)),
        "avoidable_routed_to_down_rate": float(np.mean(avoidable_down)),
        "infeasible_request_rate": float(np.mean(infeasible)),
        "unavoidable_down_failure_rate": float(np.mean(unavoidable_down)),
        "background_failure_rate": float(np.mean(background_fail)),

        "mean_cross_domain_rtt": float(np.mean(rtt_paid)),
        "cost_per_successful_request": (
            float(np.sum(normalized_costs) / np.sum(throughputs))
            if np.sum(throughputs) > 0

            else float(np.sum(normalized_costs))
        ),

        "deadline_miss_rate_prio_high": _tier_miss("high"),
        "deadline_miss_rate_prio_mid": _tier_miss("mid"),
        "deadline_miss_rate_prio_low": _tier_miss("low"),

        "request_share_prio_high": float(np.mean(tier_masks["high"])),
        "request_share_prio_mid": float(np.mean(tier_masks["mid"])),
        "request_share_prio_low": float(np.mean(tier_masks["low"])),

        "decision_latency_us_mean": float(np.mean(decision_us)) if decision_us else 0.0,
        "decision_latency_us_p99": float(np.percentile(decision_us, 99)) if decision_us else 0.0,
        "mask_build_latency_us_mean": float(np.mean(mask_build_us)) if mask_build_us else 0.0,

        "energy_wh_per_1k_requests": 1000.0 * energy_wh_per_req,
        "cost_cny_per_1k_requests": 1000.0 * cny_per_req,

        "candidate_recall_micro": recall_micro,
        "candidate_recall_macro": recall_macro,
        "candidate_precision_micro": precision_micro,
        "deadline_infeasible_step_rate": infeasible_rate,
        "mean_feasible_set_size": mean_feasible,
    }

    def _score(bounds: Mapping[str, float], prefix: str = "") -> None:
        all_ok = np.ones(len(ep_cmdp_costs), dtype=bool)
        for name, limit in bounds.items():
            ep_means = np.array([c[name] for c in ep_cmdp_costs], dtype=float)
            satisfied = ep_means <= limit
            result[f"{prefix}{name}_satisfaction_rate"] = float(np.mean(satisfied))
            result[f"{prefix}{name}_violation"] = float(np.mean(np.maximum(0.0, ep_means - limit)))
            all_ok &= satisfied

        result[f"{prefix}all_constraints_satisfaction_rate"] = (
            float(np.mean(all_ok)) if len(ep_cmdp_costs) else 0.0
        )

    _score(limits)
    if own_limits is not None:
        _score(own_limits, prefix="own_")

    return result
