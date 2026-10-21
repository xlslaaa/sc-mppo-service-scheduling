from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from semantic_lb_env.baselines import LeastLoadedPolicy, RandomPolicy, RoundRobinPolicy
from semantic_lb_env.cmdp import (
    MULTIPLIER_MODES, CONSTRAINT_MODES, active_limits,
    ConstraintConfig,
    LagrangianCallback,
    LagrangianRewardWrapper,
)
from semantic_lb_env.env import EnvConfig, SemanticLoadBalancingEnv
from semantic_lb_env.evaluate import evaluate_policy; from semantic_lb_env.p3o import P3O_ARM_ENV_KWARGS, P3O_ARM_WORKLOAD_SEED_OFFSET, build_p3o_arm

_TRACE_PROFILE_PATH: Optional[str] = None

_ARRIVAL_SCALE: float = 1.0
_SERVICE_RATE_SCALE: float = 1.0

_CROSS_DOMAIN: bool = True
_DOMAIN_SKEW: float = 0.18
_CROSS_DEMAND_FRAC: float = 0.0

_SLA_AWARE: bool = False
_RESOURCE_AWARE: bool = False
_SLA_SCALE: float = 1.0

_BALANCE_METRIC: str = "variance"
_BALANCE_WEIGHT: float = 0.8

_FAILURE_FRAC: float = 0.0
_FAILURE_DURATION: int = 12
_FAILURE_VISIBLE: bool = True
_CROSS_DOMAIN_RTT: float = 0.0
_CROSS_DOMAIN_COST_MULT: float = 1.0

_SEMANTIC_QOS_COUPLING: bool = True

_COST_MODEL: str = "synthetic_uniform"
_COST_PROFILE_PATH: Optional[str] = None
_COST_ISO_SCALE: bool = True

_MASK_CONTROL: str = "enforced"
_MASK_CONTROL_SEED: int = 20260826; _INDUSTRIAL_PATTERN: str = "none"; _INDUSTRIAL_PATTERN_PATH = None; _SLA_COST_PRIORITY_WEIGHTED: bool = False; _TRACE_REPLAY: bool = False; _TRACE_REPLAY_BLOCK: int = -1; _TRACE_REPLAY_BLOCK_LEN: int = -1; _TRACE_REPLAY_EPISODES: int = -1; _TRACE_REPLAY_SOURCE = None; _TRACE_REPLAY_PROV: dict = {}; _SERVICES_PER_DOMAIN: int = EnvConfig().services_per_domain; _SCALE_TIER_PROV: dict = {}; _TELEMETRY_MASK_CONFIG = None; _TELEMETRY_PROV: dict = {}; from semantic_lb_env.beta_scheduler import BetaScheduler as _BetaScheduler

class RewardLogger(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.rows = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info:
                self.rows.append(
                    {
                        "timesteps": self.num_timesteps,
                        "episode_reward": ep_info["r"],
                        "episode_length": ep_info["l"],
                    }
                )
        return True

def make_env(
    scenario_seed: int,
    workload_seed: int,
    mask: bool = True,
    semantic_state: bool = True,
    semantic_reward: bool = True,
    constraint_config: Optional[ConstraintConfig] = None,
):
    config = EnvConfig(
        seed=scenario_seed,
        scenario_seed=scenario_seed,
        workload_seed=workload_seed,
        constrained_objective=constraint_config is not None,
        trace_profile_path=_TRACE_PROFILE_PATH,
        arrival_scale=_ARRIVAL_SCALE,
        service_rate_scale=_SERVICE_RATE_SCALE,
        cross_domain_candidates=_CROSS_DOMAIN,
        domain_skew=_DOMAIN_SKEW,
        cross_demand_frac=_CROSS_DEMAND_FRAC,
        sla_aware_candidates=_SLA_AWARE,
        resource_aware_candidates=_RESOURCE_AWARE,
        sla_scale=_SLA_SCALE,
        balance_metric=_BALANCE_METRIC,
        balance_weight=_BALANCE_WEIGHT,
        failure_frac=_FAILURE_FRAC,
        failure_duration=_FAILURE_DURATION,
        failure_visible=_FAILURE_VISIBLE,
        cross_domain_rtt=_CROSS_DOMAIN_RTT,
        cross_domain_cost_mult=_CROSS_DOMAIN_COST_MULT,
        semantic_qos_coupling=_SEMANTIC_QOS_COUPLING,
        cost_model=_COST_MODEL,
        cost_profile_path=_COST_PROFILE_PATH,
        cost_iso_scale=_COST_ISO_SCALE,
        mask_control=_MASK_CONTROL,
        mask_control_seed=_MASK_CONTROL_SEED, industrial_pattern=_INDUSTRIAL_PATTERN, industrial_pattern_path=_INDUSTRIAL_PATTERN_PATH, sla_cost_priority_weighted=_SLA_COST_PRIORITY_WEIGHTED, trace_replay=_TRACE_REPLAY, trace_replay_block=_TRACE_REPLAY_BLOCK, trace_replay_block_len=_TRACE_REPLAY_BLOCK_LEN, trace_replay_episodes=_TRACE_REPLAY_EPISODES, trace_replay_source=_TRACE_REPLAY_SOURCE, services_per_domain=_SERVICES_PER_DOMAIN, telemetry_mask_config=_TELEMETRY_MASK_CONFIG,
    )
    env = SemanticLoadBalancingEnv(
        config=config,
        use_semantic_mask=mask,
        include_semantic_state=semantic_state,
        reward_semantic=semantic_reward,
    )
    if constraint_config is not None:
        env = LagrangianRewardWrapper(env, constraint_config)
    return Monitor(env)

def make_constraint_config(args: argparse.Namespace) -> ConstraintConfig:
    return ConstraintConfig(
        sla_limit=args.sla_limit,
        cost_limit=args.cost_limit,
        failure_limit=args.failure_limit,
        multiplier_lr=args.multiplier_lr,
        multiplier_max=args.multiplier_max,

        multiplier_mode=getattr(args, "multiplier_mode", "integral"),
        multiplier_kp=getattr(args, "multiplier_kp", 0.0),
        multiplier_kd=getattr(args, "multiplier_kd", 0.0), constraint_mode=getattr(args, "constraint_mode", "aggregate"), sla_high_limit=getattr(args, "sla_high_limit", None), sla_bulk_limit=getattr(args, "sla_bulk_limit", None),
    )

def train_and_eval(args: argparse.Namespace) -> None:

    global _TRACE_PROFILE_PATH, _ARRIVAL_SCALE, _SERVICE_RATE_SCALE
    global _CROSS_DOMAIN, _DOMAIN_SKEW, _CROSS_DEMAND_FRAC
    global _SLA_AWARE, _RESOURCE_AWARE, _SLA_SCALE
    global _BALANCE_METRIC, _BALANCE_WEIGHT
    global _FAILURE_FRAC, _FAILURE_DURATION, _FAILURE_VISIBLE
    global _CROSS_DOMAIN_RTT, _CROSS_DOMAIN_COST_MULT
    global _SEMANTIC_QOS_COUPLING
    global _COST_MODEL, _COST_PROFILE_PATH, _COST_ISO_SCALE
    global _MASK_CONTROL, _MASK_CONTROL_SEED; global _INDUSTRIAL_PATTERN, _INDUSTRIAL_PATTERN_PATH; global _SLA_COST_PRIORITY_WEIGHTED; global _TRACE_REPLAY, _TRACE_REPLAY_BLOCK, _TRACE_REPLAY_BLOCK_LEN, _TRACE_REPLAY_EPISODES, _TRACE_REPLAY_SOURCE, _TRACE_REPLAY_PROV; global _SERVICES_PER_DOMAIN, _SCALE_TIER_PROV; global _TELEMETRY_MASK_CONFIG, _TELEMETRY_PROV
    _TRACE_PROFILE_PATH = getattr(args, "trace_profile", None)
    _ARRIVAL_SCALE = getattr(args, "arrival_scale", 1.0)
    _SERVICE_RATE_SCALE = getattr(args, "service_rate_scale", 1.0)
    _CROSS_DOMAIN = not getattr(args, "no_cross_domain", False)
    _DOMAIN_SKEW = getattr(args, "domain_skew", 0.18)
    _CROSS_DEMAND_FRAC = getattr(args, "cross_demand_frac", 0.0)
    _SLA_AWARE = getattr(args, "sla_aware_candidates", False)
    _RESOURCE_AWARE = getattr(args, "resource_aware_candidates", False)
    _SLA_SCALE = getattr(args, "sla_scale", 1.0)
    _BALANCE_METRIC = getattr(args, "balance_metric", "variance")
    _BALANCE_WEIGHT = getattr(args, "balance_weight", 0.8)
    _FAILURE_FRAC = getattr(args, "failure_frac", 0.0)
    _FAILURE_DURATION = getattr(args, "failure_duration", 12)

    _FAILURE_VISIBLE = not getattr(args, "failure_invisible", False)
    _CROSS_DOMAIN_RTT = getattr(args, "cross_domain_rtt", 0.0)
    _CROSS_DOMAIN_COST_MULT = getattr(args, "cross_domain_cost_mult", 1.0)

    _SEMANTIC_QOS_COUPLING = not getattr(args, "no_semantic_qos_coupling", False)
    _COST_MODEL = getattr(args, "cost_model", "synthetic_uniform")
    _COST_PROFILE_PATH = getattr(args, "cost_profile", None)

    _COST_ISO_SCALE = not getattr(args, "no_cost_iso_scale", False)
    _MASK_CONTROL = getattr(args, "mask_control", "enforced")
    _MASK_CONTROL_SEED = getattr(args, "mask_control_seed", 20260826); _INDUSTRIAL_PATTERN = getattr(args, "industrial_pattern", "none"); _INDUSTRIAL_PATTERN_PATH = getattr(args, "industrial_pattern_path", None); _SLA_COST_PRIORITY_WEIGHTED = bool(getattr(args, "sla_cost_priority_weighted", False)); reject_p3o_under_tiered(args); (_TRACE_REPLAY, _TRACE_REPLAY_BLOCK, _TRACE_REPLAY_BLOCK_LEN, _TRACE_REPLAY_EPISODES, _TRACE_REPLAY_SOURCE) = _trace_replay_args(args); (_SERVICES_PER_DOMAIN, _SCALE_TIER_PROV) = resolve_scale_tier(args); import sys; sys.path.insert(0, str(Path(__file__).parent.parent.parent)); from scripts._telemetry_helpers import make_telemetry_config, reject_missing_predictor, telemetry_provenance; _TELEMETRY_MASK_CONFIG = make_telemetry_config(args); reject_missing_predictor(_TELEMETRY_MASK_CONFIG); _TELEMETRY_PROV = telemetry_provenance(_TELEMETRY_MASK_CONFIG)
    if _COST_MODEL != "synthetic_uniform" and not _COST_PROFILE_PATH:

        raise SystemExit("--cost-model %s 必须同时给出 --cost-profile" % (_COST_MODEL,))

    out_dir = Path(args.output_dir); _TRACE_REPLAY_PROV = trace_replay_provenance()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_seed = args.seed
    scenario_seed = args.scenario_seed if args.scenario_seed is not None else args.seed
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed + 10_000
    eval_episodes = args.eval_episodes
    total_timesteps = args.timesteps

    metrics = []
    curves = []
    cmdp_curves = []
    constraint_config = make_constraint_config(args); _OWN_LIMITS = active_limits(constraint_config) if constraint_config.constraint_mode == "tiered" else None; __import__('scripts._telemetry_helpers', fromlist=['x']).reject_budget_limit_mismatch(_TELEMETRY_MASK_CONFIG, constraint_config)

    base_env = make_env(scenario_seed, train_seed, mask=True)
    num_actions = base_env.unwrapped.num_services
    for name, policy, use_masks in [
        ("random_semantic", RandomPolicy(num_actions, respect_mask=True, seed=train_seed), True),
        ("round_robin_semantic", RoundRobinPolicy(num_actions, respect_mask=True), True),
        ("least_loaded_semantic", LeastLoadedPolicy(num_actions, num_actions, respect_mask=True), True),
        ("random_no_semantic", RandomPolicy(num_actions, respect_mask=False, seed=train_seed), False),
    ]:
        env = make_env(scenario_seed, eval_seed, mask=use_masks)
        row = evaluate_policy(env.unwrapped, policy, episodes=eval_episodes, use_masks=use_masks, seed=eval_seed, own_limits=_OWN_LIMITS, deterministic=args.eval_deterministic)
        row["method"] = name
        metrics.append(row)

    ppo_env = make_env(scenario_seed, train_seed + 1, mask=False, semantic_state=True, semantic_reward=True)
    ppo_logger = RewardLogger()
    ppo = PPO(
        "MlpPolicy",
        ppo_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.97,
        gae_lambda=0.92,
        verbose=0,
        seed=train_seed,
        device="cpu",
    )
    ppo.learn(total_timesteps=total_timesteps, callback=ppo_logger)
    ppo.save(out_dir / "ppo_no_mask")
    for row in ppo_logger.rows:
        row["method"] = "ppo_no_mask"
        curves.append(row)
    env = make_env(scenario_seed, eval_seed, mask=False)
    row = evaluate_policy(env.unwrapped, ppo, episodes=eval_episodes, use_masks=False, seed=eval_seed, own_limits=_OWN_LIMITS, deterministic=args.eval_deterministic)
    row["method"] = "ppo_no_mask"
    metrics.append(row)

    lag_ppo_env = make_env(
        scenario_seed,
        train_seed + 6,
        mask=False,
        semantic_state=True,
        semantic_reward=True,
        constraint_config=constraint_config,
    )
    lag_ppo_logger = RewardLogger()
    lag_ppo_cmdp = LagrangianCallback()
    lag_ppo = PPO(
        "MlpPolicy",
        lag_ppo_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.97,
        gae_lambda=0.92,
        verbose=0,
        seed=train_seed,
        device="cpu",
    )
    lag_ppo.learn(total_timesteps=total_timesteps, callback=[lag_ppo_logger, lag_ppo_cmdp])
    lag_ppo.save(out_dir / "ppo_lagrangian")
    for row in lag_ppo_logger.rows:
        row["method"] = "ppo_lagrangian"
        curves.append(row)
    for row in lag_ppo_cmdp.rows:
        row["method"] = "ppo_lagrangian"
        cmdp_curves.append(row)
    env = make_env(scenario_seed, eval_seed, mask=False)
    row = evaluate_policy(env.unwrapped, lag_ppo, episodes=eval_episodes, use_masks=False, seed=eval_seed, own_limits=_OWN_LIMITS, deterministic=args.eval_deterministic)
    row["method"] = "ppo_lagrangian"
    metrics.append(row)

    mask_env = make_env(scenario_seed, train_seed + 2, mask=True, semantic_state=True, semantic_reward=True)
    mask_logger = RewardLogger()
    mask_eval_env = make_env(scenario_seed, eval_seed + 1, mask=True, semantic_state=True, semantic_reward=True)
    eval_callback = MaskableEvalCallback(
        mask_eval_env,
        best_model_save_path=str(out_dir / "best_maskable"),
        log_path=str(out_dir / "eval_logs"),
        eval_freq=max(args.n_steps * 4, 512),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=0,
    )
    maskable = MaskablePPO(
        "MlpPolicy",
        mask_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.97,
        gae_lambda=0.92,
        verbose=0,
        seed=train_seed,
        device="cpu",
    )
    maskable.learn(total_timesteps=total_timesteps, callback=[mask_logger, eval_callback])
    maskable.save(out_dir / "maskable_ppo_semantic")
    for row in mask_logger.rows:
        row["method"] = "maskable_ppo_semantic"
        curves.append(row)
    env = make_env(scenario_seed, eval_seed, mask=True)
    row = evaluate_policy(env.unwrapped, maskable, episodes=eval_episodes, use_masks=True, seed=eval_seed, own_limits=_OWN_LIMITS, deterministic=args.eval_deterministic)
    row["method"] = "maskable_ppo_semantic"
    metrics.append(row)

    sc_env = make_env(
        scenario_seed,
        train_seed + 7,
        mask=True,
        semantic_state=True,
        semantic_reward=True,
        constraint_config=constraint_config,
    )
    sc_logger = RewardLogger()
    sc_cmdp = LagrangianCallback()
    sc_mppo = MaskablePPO(
        "MlpPolicy",
        sc_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.97,
        gae_lambda=0.92,
        verbose=0,
        seed=train_seed,
        device="cpu",
    )
    sc_mppo.learn(total_timesteps=total_timesteps, callback=[sc_logger, sc_cmdp])
    sc_mppo.save(out_dir / "sc_mppo")
    for row in sc_logger.rows:
        row["method"] = "sc_mppo"
        curves.append(row)
    for row in sc_cmdp.rows:
        row["method"] = "sc_mppo"
        cmdp_curves.append(row)
    env = make_env(scenario_seed, eval_seed, mask=True)
    row = evaluate_policy(env.unwrapped, sc_mppo, episodes=eval_episodes, use_masks=True, seed=eval_seed, own_limits=_OWN_LIMITS, deterministic=args.eval_deterministic)
    row["method"] = "sc_mppo"
    metrics.append(row)

    ab_env = make_env(scenario_seed, train_seed + 4, mask=True, semantic_state=True, semantic_reward=False)
    ab_logger = RewardLogger()
    ab_model = MaskablePPO(
        "MlpPolicy",
        ab_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=0.97,
        gae_lambda=0.92,
        verbose=0,
        seed=train_seed,
        device="cpu",
    )
    ab_model.learn(total_timesteps=total_timesteps, callback=ab_logger)
    ab_model.save(out_dir / "maskable_ppo_no_semantic_reward")
    for row in ab_logger.rows:
        row["method"] = "maskable_ppo_no_semantic_reward"
        curves.append(row)
    env = make_env(scenario_seed, eval_seed, mask=True, semantic_reward=False)
    row = evaluate_policy(env.unwrapped, ab_model, episodes=eval_episodes, use_masks=True, seed=eval_seed, own_limits=_OWN_LIMITS, deterministic=args.eval_deterministic)
    row["method"] = "maskable_ppo_no_semantic_reward"
    metrics.append(row)

    if getattr(args, "extra_ablation", False):

        sr_env = make_env(
            scenario_seed,
            train_seed + 8,
            mask=True,
            semantic_state=True,
            semantic_reward=False,
            constraint_config=constraint_config,
        )
        sr_logger = RewardLogger()
        sr_cmdp = LagrangianCallback()
        sr_model = MaskablePPO(
            "MlpPolicy",
            sr_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=0.97,
            gae_lambda=0.92,
            verbose=0,
            seed=train_seed,
            device="cpu",
        )
        sr_model.learn(total_timesteps=total_timesteps, callback=[sr_logger, sr_cmdp])
        sr_model.save(out_dir / "sc_mppo_no_semantic_reward")
        for row in sr_logger.rows:
            row["method"] = "sc_mppo_no_semantic_reward"
            curves.append(row)
        for row in sr_cmdp.rows:
            row["method"] = "sc_mppo_no_semantic_reward"
            cmdp_curves.append(row)
        env = make_env(scenario_seed, eval_seed, mask=True, semantic_reward=False)
        row = evaluate_policy(env.unwrapped, sr_model, episodes=eval_episodes, use_masks=True, seed=eval_seed, own_limits=_OWN_LIMITS, deterministic=args.eval_deterministic)
        row["method"] = "sc_mppo_no_semantic_reward"
        metrics.append(row)

    if not args.skip_dqn:

        dqn_env = make_env(scenario_seed, train_seed + 5, mask=False, semantic_state=True, semantic_reward=True)
        dqn_logger = RewardLogger()
        dqn = DQN(
            "MlpPolicy",
            dqn_env,
            learning_rate=args.learning_rate,
            buffer_size=10_000,
            learning_starts=256,
            batch_size=args.batch_size,
            gamma=0.97,
            train_freq=16,
            gradient_steps=1,
            target_update_interval=512,
            verbose=0,
            seed=train_seed,
            device="cpu",
        )
        dqn.learn(total_timesteps=min(total_timesteps, args.dqn_timesteps), callback=dqn_logger); dqn_provenance = {"nominal_dqn_timesteps": int(args.dqn_timesteps), "requested_timesteps": int(min(total_timesteps, args.dqn_timesteps)), "total_timesteps_effective": int(dqn.num_timesteps), "nominal_workload_seed": int(train_seed) + 5, "effective_sb3_seed": int(train_seed)}
        dqn.save(out_dir / "dqn_no_mask")
        for row in dqn_logger.rows:
            row["method"] = "dqn_no_mask"
            curves.append(row)
        env = make_env(scenario_seed, eval_seed, mask=False)
        row = evaluate_policy(env.unwrapped, dqn, episodes=eval_episodes, use_masks=False, seed=eval_seed, own_limits=_OWN_LIMITS, deterministic=args.eval_deterministic)
        row["method"] = "dqn_no_mask"
        metrics.append(row)

    p3o_provenance = None; _cpo_summary = _maybe_train_cpo_arm(args, scenario_seed=scenario_seed, train_seed=train_seed, eval_seed=eval_seed, total_timesteps=total_timesteps, eval_episodes=eval_episodes, constraint_config=constraint_config, out_dir=out_dir, metrics=metrics, curves=curves, _OWN_LIMITS=_OWN_LIMITS); _deepset_summary = _maybe_train_deepset_arm(args, scenario_seed=scenario_seed, train_seed=train_seed, eval_seed=eval_seed, total_timesteps=total_timesteps, eval_episodes=eval_episodes, out_dir=out_dir, metrics=metrics, curves=curves, _OWN_LIMITS=_OWN_LIMITS)
    if args.include_p3o:

        p3o_env = make_env(
            scenario_seed,
            train_seed + P3O_ARM_WORKLOAD_SEED_OFFSET,
            constraint_config=constraint_config,
            **P3O_ARM_ENV_KWARGS
        )
        p3o_logger = RewardLogger()
        p3o_model = build_p3o_arm(
            p3o_env,

            constraint_config=constraint_config,
            penalty_coef=args.p3o_penalty_coef,
            cost_vf_coef=args.p3o_cost_vf_coef,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            seed=train_seed,
        )

        p3o_model.learn(total_timesteps=total_timesteps, callback=p3o_logger)
        p3o_model.save(out_dir / "p3o")
        for row in p3o_logger.rows:
            row["method"] = "p3o"
            curves.append(row)

        p3o_cost_curve = pd.DataFrame(p3o_model.diagnostics)
        p3o_cost_curve.to_csv(out_dir / "p3o_cost_curve.csv", index=False)

        from p3o_checks import failure_reasons, format_verdict, onebite_verdict
        p3o_onebite = onebite_verdict(pd.DataFrame(curves), p3o_cost_curve)
        print("[p3o] " + format_verdict(p3o_onebite))
        if not p3o_onebite["ok"]:
            raise RuntimeError("P3O 跑中牙未过 ⇒ 本 run 的 P3O 产物作废："
                               + "；".join(failure_reasons(p3o_onebite)))
        p3o_provenance = p3o_model.provenance()
        p3o_provenance["onebite"] = p3o_onebite

        p3o_provenance["total_timesteps_effective"] = int(p3o_model.num_timesteps)
        p3o_provenance["nominal_workload_seed"] = int(train_seed) + 6
        p3o_provenance["effective_sb3_seed"] = int(train_seed)
        env = make_env(scenario_seed, eval_seed, mask=False)
        row = evaluate_policy(env.unwrapped, p3o_model, episodes=eval_episodes, use_masks=False, seed=eval_seed, own_limits=_OWN_LIMITS, deterministic=args.eval_deterministic)
        row["method"] = "p3o"
        metrics.append(row)

    metrics_df = pd.DataFrame(metrics)
    curves_df = pd.DataFrame(curves)
    cmdp_curves_df = pd.DataFrame(cmdp_curves)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    curves_df.to_csv(out_dir / "training_curve.csv", index=False)
    cmdp_curves_df.to_csv(out_dir / "cmdp_curve.csv", index=False)

    summary = {
        "best_by_reward": metrics_df.sort_values("mean_episode_reward", ascending=False).iloc[0].to_dict(),
        "best_by_sla": metrics_df.sort_values("sla_violation_rate", ascending=True).iloc[0].to_dict(),
        "timesteps": total_timesteps,
        "eval_episodes": eval_episodes,
        "train_seed": train_seed,
        "scenario_seed": scenario_seed,
        "eval_seed": eval_seed, **_TRACE_REPLAY_PROV,
        "constraint_limits": active_limits(constraint_config),

        "env_config": {
            "trace_profile_path": _TRACE_PROFILE_PATH,
            "arrival_scale": _ARRIVAL_SCALE,
            "service_rate_scale": _SERVICE_RATE_SCALE,
            "cross_domain_candidates": _CROSS_DOMAIN,
            "domain_skew": _DOMAIN_SKEW,
            "cross_demand_frac": _CROSS_DEMAND_FRAC,
            "sla_aware_candidates": _SLA_AWARE,
            "resource_aware_candidates": _RESOURCE_AWARE,
            "sla_scale": _SLA_SCALE,
            "balance_metric": _BALANCE_METRIC,
            "balance_weight": _BALANCE_WEIGHT,
            "failure_frac": _FAILURE_FRAC,
            "failure_duration": _FAILURE_DURATION,
            "failure_visible": _FAILURE_VISIBLE,
            "cross_domain_rtt": _CROSS_DOMAIN_RTT,
            "cross_domain_cost_mult": _CROSS_DOMAIN_COST_MULT,
            "semantic_qos_coupling": _SEMANTIC_QOS_COUPLING,
            "cost_model": _COST_MODEL,
            "cost_profile_path": _COST_PROFILE_PATH,
            "cost_iso_scale": _COST_ISO_SCALE,
            "mask_control": _MASK_CONTROL,
            "mask_control_seed": _MASK_CONTROL_SEED, "industrial_pattern": _INDUSTRIAL_PATTERN, "industrial_pattern_path": _INDUSTRIAL_PATTERN_PATH, "sla_cost_priority_weighted": _SLA_COST_PRIORITY_WEIGHTED, "trace_replay": _TRACE_REPLAY, "trace_replay_block": _TRACE_REPLAY_BLOCK, "trace_replay_block_len": _TRACE_REPLAY_BLOCK_LEN, "trace_replay_episodes": _TRACE_REPLAY_EPISODES, "trace_replay_source": _TRACE_REPLAY_SOURCE, "scale_tier": _SCALE_TIER_PROV, "telemetry_mask_enabled": _TELEMETRY_MASK_CONFIG is not None, **_TELEMETRY_PROV,
        },

        "algo_config": {
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "multiplier_lr": args.multiplier_lr,
            "multiplier_max": args.multiplier_max,
            "multiplier_mode": getattr(args, "multiplier_mode", "integral"),
            "multiplier_kp": getattr(args, "multiplier_kp", 0.0),
            "multiplier_kd": getattr(args, "multiplier_kd", 0.0), "constraint_mode": getattr(args, "constraint_mode", "aggregate"), "sla_high_limit": getattr(args, "sla_high_limit", None), "sla_bulk_limit": getattr(args, "sla_bulk_limit", None),
            "dqn_timesteps": args.dqn_timesteps,
            "skip_dqn": args.skip_dqn,
            "extra_ablation": getattr(args, "extra_ablation", None),
        },

        "seed_provenance": {
            "sb3_model_seed_all_methods": train_seed,
            "nominal_workload_seed_offsets": {
                "heuristic_baselines": 0, "ppo_no_mask": 1, "maskable_ppo_semantic": 2,
                "maskable_ppo_no_semantic_reward": 4, "dqn_no_mask": 5,
                "ppo_lagrangian": 6, "sc_mppo": 7, "sc_mppo_no_semantic_reward": 8,
            },
            "eval_workload_seed_in_run": eval_seed,
            "note": ("nominal_workload_seed_offsets 只在 env 构造时生效；SB3 学习方法"
                     "在 learn() 处按 sb3_model_seed_all_methods 重播种 ⇒ 训练期各"
                     "学习方法共用同一条工作负载流。启发式基线不过 SB3，偏移生效。"
                     "重打分口径另见 reeval_protocol.py 的 EVAL_SEED / --eval-seed。"),
        },
    }; summary.update(_cpo_summary); summary.update(_deepset_summary)

    if not args.skip_dqn: summary["dqn_config"] = dqn_provenance
    if p3o_provenance is not None:
        summary["p3o_config"] = p3o_provenance
    summary.update({"trace_profile_status": base_env.unwrapped.trace_profile_status()} if _TRACE_PROFILE_PATH else {}); (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(metrics_df.sort_values("mean_episode_reward", ascending=False).to_string(index=False))
    print(f"\nSaved results to {out_dir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/latest")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario-seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None); parser.add_argument("--eval-deterministic", type=lambda x: x.lower() in ("true", "1", "yes"), default=True, help="Whether to use deterministic (argmax) policy during evaluation. Default True for reproducibility; set to False to evaluate stochastic policy sampling.")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-dqn", action="store_true")
    parser.add_argument(
        "--extra-ablation",
        action="store_true",
        help="Also train sc_mppo_no_semantic_reward (mask + CMDP, semantic reward off), "
             "the single-variable ablation for the semantic-reward innovation point.",
    )
    parser.add_argument("--dqn-timesteps", type=int, default=5000); parser.add_argument("--include-cpo", action="store_true", help="Also train the CPO baseline (Achiam et al., ICML 2017): TRPO + one cost critic per constraint, with the linearised constraints solved inside the trust-region subproblem (no penalty, no multiplier shaping). Default off is a bit-for-bit no-op: no import, no RNG draw, summary.json gains no key."); parser.add_argument("--include-deepset", action="store_true", help="Also train a DRL-FORCH-style Deep Set baseline (permutation-equivariant actor + invariant critic, masked PPO, no CMDP). Default off is a bit-for-bit no-op.")
    parser.add_argument(
        "--include-p3o",
        action="store_true",
        help="Also train the P3O baseline (Zhang et al., IJCAI 2022): PPO + an "
        "independent cost critic + a clipped exact penalty inside the policy loss, "
        "no dual variable. Default off is a bit-for-bit no-op: the block is skipped, "
        "no RNG is drawn and summary.json gains no key.",
    )
    parser.add_argument(
        "--p3o-penalty-coef",
        type=float,
        default=5.0,
        help="Fixed penalty coefficient kappa for --include-p3o, shared by the three "
        "constraints. P3O has no dual ascent, so this value is the whole "
        "constraint-enforcement knob; it must be chosen by the pre-data calibration "
        "probe, not tuned on the family's endpoints.",
    )
    parser.add_argument(
        "--p3o-cost-vf-coef",
        type=float,
        default=0.5,
        help="Weight of the cost-critic regression loss for --include-p3o "
        "(mirrors PPO's vf_coef default of 0.5).",
    )
    parser.add_argument("--sla-limit", type=float, default=0.02)
    parser.add_argument("--cost-limit", type=float, default=0.45)
    parser.add_argument("--failure-limit", type=float, default=0.037)
    parser.add_argument("--multiplier-lr", type=float, default=0.05)
    parser.add_argument("--multiplier-max", type=float, default=20.0)
    parser.add_argument(
        "--multiplier-mode",
        choices=list(MULTIPLIER_MODES),
        default="integral",
        help="Lagrange-multiplier controller. 'integral' (default) keeps the "
        "existing rule lambda <- clip(lambda + K_I(J_c - d)); 'pid' adds the "
        "proportional and derivative terms of Stooke et al. (2020). With "
        "--multiplier-kp 0 --multiplier-kd 0 the two modes are bit-identical.",
    )
    parser.add_argument(
        "--multiplier-kp",
        type=float,
        default=0.0,
        help="K_P for --multiplier-mode pid. 0.0 (default) reproduces the "
        "integral-only rule bit-for-bit.",
    )
    parser.add_argument(
        "--multiplier-kd",
        type=float,
        default=0.0,
        help="K_D for --multiplier-mode pid, applied to max(0, J_c - J_c_prev) "
        "so it only damps rising cost. 0.0 (default) reproduces the "
        "integral-only rule bit-for-bit.",
    )

    parser.add_argument(
        "--constraint-mode",
        choices=list(CONSTRAINT_MODES),
        default="aggregate",
        help="CMDP constraint set. 'aggregate' (default) keeps the three "
        "channels (sla, cost, failure) and is bit-identical to every run made "
        "before this flag existed. 'tiered' splits the SLA channel into "
        "sla_high / sla_bulk by request urgency (semantic_lb_env/tiers.py) and "
        "gives each its own multiplier; the aggregate sla channel then leaves "
        "the constraint set, because PER STEP sla_high + sla_bulk == sla holds "
        "bit-exactly (smeared accounting) so keeping all three would double-count. "
        "At the EPISODE-MEAN level the same sum is only equal in exact arithmetic: "
        "it is three separate divisions and about 25.5%% of n=120 episodes differ by "
        "1 ulp, so never use the mean-level version as a bit-exact assertion.",
    )
    parser.add_argument(
        "--sla-high-limit",
        type=float,
        default=None,
        help="Per-step limit on the high-urgency SLA channel. Only read under "
        "--constraint-mode tiered, where it is REQUIRED: the default None makes "
        "constraint_names() raise instead of silently reusing --sla-limit, whose "
        "scale differs (the tiered costs are smeared, i.e. zero on steps of the "
        "other tier). Calibrate it with the step-3 probe before touching data.",
    )
    parser.add_argument(
        "--sla-bulk-limit",
        type=float,
        default=None,
        help="Per-step limit on the non-high-urgency SLA channel. Same contract "
        "as --sla-high-limit: required under tiered mode, never defaulted.",
    )

    parser.add_argument(
        "--sla-cost-priority-weighted",
        action="store_true",
        help="A2 control arm: keep ONE aggregate sla constraint but make its cost "
        "priority-weighted, i.e. 1(miss) * (0.5 + priority) / (0.5 + E[priority]) "
        "with E[priority] = 0.625 derived from the env's own priority sampling "
        "(semantic_lb_env/tiers.py is the single definition point). Zero new "
        "hyperparameters, zero eval-schema change, no kappa recalibration, and the "
        "constraint set stays the three aggregate channels -- that is the whole "
        "point: E5's tiered arm has to beat THIS, not just beat the unweighted "
        "aggregate arm. Off by default and bit-identical to every earlier run (the "
        "weight is the literal 1.0 then). NOTE the normalisation only makes the "
        "WEIGHT have unit mean under the priority marginal; misses concentrate on "
        "the tight-deadline tier, so the weighted cost mean sits ABOVE the "
        "unweighted one and reusing --sla-limit is therefore slightly STRICTER, not "
        "equal-strength. Do not combine with --constraint-mode tiered: the "
        "pre-registration declares them as two mutually exclusive routes.",
    )
    parser.add_argument(
        "--trace-profile",
        default=None,
        help="Path to a fitted trace_profile.json for trace-calibrated dynamics. "
        "When omitted, the environment uses its original synthetic sampling.",
    )
    parser.add_argument(
        "--arrival-scale",
        type=float,
        default=1.0,
        help="Load-stress multiplier on arrival rate (robustness scenarios). "
        "1.0 = nominal; >1.0 raises demand. Applied to training and eval alike.",
    )
    parser.add_argument(
        "--service-rate-scale",
        type=float,
        default=1.0,
        help="Multiplier on service capacity (robustness scenarios). "
        "1.0 = nominal; <1.0 shrinks capacity. Applied to training and eval alike.",
    )
    parser.add_argument(
        "--no-cross-domain",
        action="store_true",
        help="Ablate the cross-business-domain candidate arm (innovation point 1): "
        "restrict the mask/candidate set to the request's own domain. Default off "
        "keeps the full cross-domain candidate generation. Applied to training and "
        "eval alike, so a model trained with it must be re-evaluated with it too.",
    )
    parser.add_argument(
        "--domain-skew",
        type=float,
        default=0.18,
        help="Amplitude of the request-domain imbalance (see EnvConfig.domain_skew). "
        "0.18 (default) reproduces the original workload bit-for-bit; larger values "
        "(e.g. 0.55) concentrate load on one domain to create the cross-domain "
        "pressure scenario. Applied to training and eval alike.",
    )
    parser.add_argument(
        "--cross-demand-frac",
        type=float,
        default=0.0,
        help="Fraction of requests whose function type the origin domain cannot "
        "serve (semantic_matrix[domain,type]==0), forcing a cross-domain route "
        "(see EnvConfig.cross_demand_frac). 0.0 (default) reproduces the original "
        "workload bit-for-bit; positive values inject genuine cross-domain need, "
        "the pressure Innovation Point 1 exists for. Applied to training and eval "
        "alike, so a model trained with it must be re-evaluated with it too.",
    )
    parser.add_argument(
        "--sla-aware-candidates",
        action="store_true",
        help="Innovation-Point-1 completion: prune the candidate set by the "
        "request's SLA budget (drop instances whose projected latency would miss "
        "the deadline). Default off. Applied to training and eval alike.",
    )
    parser.add_argument(
        "--resource-aware-candidates",
        action="store_true",
        help="Innovation-Point-1 completion: prune the candidate set by the "
        "request's resource preference derived from priority (reliability floor) "
        "and demand (capacity headroom). Default off. Applied to training and "
        "eval alike.",
    )
    parser.add_argument(
        "--sla-scale",
        type=float,
        default=1.0,
        help="Multiplier on the sampled SLA budget (see EnvConfig.sla_scale). "
        "1.0 (default) reproduces the original workload bit-for-bit; values <1 "
        "(e.g. 0.15) tighten deadlines so the SLA candidate arm actually bites. "
        "Applied to training and eval alike.",
    )
    parser.add_argument(
        "--failure-frac",
        type=float,
        default=0.0,
        help="Realism perturbation axis A: steady-state fraction of service "
        "instances that are down at any moment (see EnvConfig.failure_frac). "
        "0.0 (default) short-circuits the whole failure path and reproduces every "
        "existing arm bit-for-bit. Design levels: 0.10 / 0.20 / 0.30. Failures are "
        "drawn from a dedicated failure_rng, so the request/latency/reliability "
        "streams are bit-identical to the failure_frac=0 anchor (strictly paired). "
        "Applied to training and eval alike.",
    )
    parser.add_argument(
        "--failure-duration",
        type=int,
        default=12,
        help="Steps a failed instance stays down before recovering (see "
        "EnvConfig.failure_duration). 12 = 10%% of episode_len=120. Together with "
        "--failure-frac it fixes the per-step hazard h = f / (D*(1-f)), so the "
        "steady-state down fraction equals --failure-frac.",
    )
    parser.add_argument(
        "--failure-invisible",
        action="store_true",
        help="Realism perturbation axis A, second arm: keep failed instances in the "
        "candidate set (they still fail on arrival) instead of filtering them out. "
        "Default off = an oracle health checker removes down instances from the "
        "action mask. NOTE the visible arm therefore measures the mechanism value "
        "of a perfect health check injected through the mask, NOT the agent's own "
        "failure awareness (health state is not in the 43-dim observation). The "
        "invisible arm is a sensitivity experiment w.r.t. exogenous failure noise. "
        "Applied to training and eval alike.",
    )
    parser.add_argument(
        "--cross-domain-rtt",
        type=float,
        default=0.0,
        help="Realism perturbation axis B: extra network RTT (ms) added when a "
        "request is served by an out-of-domain instance (see "
        "EnvConfig.cross_domain_rtt). 0.0 (default) is IEEE754-exact +0.0 and "
        "reproduces every existing arm bit-for-bit. Design levels: 20 / 50 / 100. "
        "Added outside the semantic-mismatch latency multiplier (link property, not "
        "a semantic penalty) and mirrored into _projected_latency so the SLA-pruning "
        "arm does not filter as if cross-domain were free. Applied to training and "
        "eval alike -- except for the deliberately zero-shot transfer arms, which "
        "re-evaluate nominal models under RTT>0 and must be reported as a separate "
        "protocol.",
    )
    parser.add_argument(
        "--cross-domain-cost-mult",
        type=float,
        default=1.0,
        help="Sensitivity arm for axis B: multiplier on the cost of cross-domain "
        "traffic (see EnvConfig.cross_domain_cost_mult). 1.0 (default) is exact and "
        "changes nothing. The normalized_cost denominator stays np.max(self.cost), "
        "so this arm can push normalized_cost above 1.0 -- that is the intended "
        "statement 'cross-domain traffic is dearer against a fixed budget', and it "
        "keeps the shared 0.457 cost yardstick comparable across all arms.",
    )
    parser.add_argument(
        "--no-semantic-qos-coupling",
        action="store_true",
        help="Decoupling ablation: drop the built-in 'semantically-native => better "
        "QoS' correlation (latency x0.75, reliability +0.025, capacity x1.12 for "
        "native instances; latency x1.35, capacity x0.82 otherwise). Semantic masking, "
        "candidate generation and the reward-side semantic bonus are all untouched. "
        "Default (flag absent) keeps the coupling and reproduces every existing arm "
        "bit-for-bit. Purpose: the coupling is hard-coded causality, so it inflates "
        "any semantic-aware vs semantic-agnostic comparison; this arm asks how much of "
        "the semantic mechanism's value survives decoupling. MUST match between "
        "training and evaluation. See EnvConfig.semantic_qos_coupling.",
    )
    parser.add_argument(
        "--cost-model",
        choices=["synthetic_uniform", "power_based"],
        default="synthetic_uniform",
        help="Cost yardstick. 'synthetic_uniform' (default) keeps cost ~ U(0.25, 1.0), "
        "provenance grade C and statistically independent of capacity/latency; it "
        "reproduces every existing arm bit-for-bit. 'power_based' derives the per-request "
        "unit cost from published device power specs x published PUE x published "
        "electricity tariff (requires --cost-profile). Cost is this project's only win "
        "that survives the whole grid, so it must be shown to survive a citable "
        "yardstick. MUST match between training and evaluation. See EnvConfig.cost_model.",
    )
    parser.add_argument(
        "--cost-profile",
        default=None,
        help="Path to a power_cost_profile/v1 JSON (e.g. configs/power_cost_profile.json). "
        "Required by --cost-model power_based; a missing file is a hard error rather than "
        "a silent fallback, so no run can be labelled power_based while carrying "
        "synthetic costs.",
    )
    parser.add_argument(
        "--no-cost-iso-scale",
        action="store_true",
        help="Only meaningful with --cost-model power_based. By default the derived "
        "physical costs are affinely mapped back onto the same seed's synthetic "
        "[min, max], which keeps np.max(self.cost) (the normalized_cost denominator), "
        "the shared 0.457 budget and the cost_weight magnitude intact so that the single "
        "variable is the cost vector's shape/ordering. This flag switches to mean-matching "
        "only, preserving the physical spread; np.max(self.cost) then changes, so such an "
        "arm needs its own --own-cost-limit and must not share a cost-satisfaction table "
        "with existing arms. Note that PUE and tariff cancel exactly under the default "
        "iso-scale mapping (any common positive factor does).",
    )
    parser.add_argument(
        "--mask-control",
        choices=["enforced", "size_matched", "advisory"],
        default="enforced",
        help="Candidate-vs-mask decoupling control. H1 bundles two effects: the semantic "
        "candidate set makes the action space SMALLER and makes it RIGHT. All 23 existing "
        "arms only have the diagonal (both or neither), so 'masking helps' cannot be "
        "attributed. 'enforced' (default) = the semantic set both proposes and enforces, "
        "reproducing every existing arm bit-for-bit. 'size_matched' = enforce a random "
        "subset of EXACTLY the semantic mask's per-step cardinality, drawn uniformly over "
        "all services and independent of semantics -> keeps 'smaller', drops 'right'. "
        "'advisory' = still compute the semantic set (valid/success/reward/the five recall "
        "columns are all unchanged) but return an all-ones mask -> keeps 'right', drops "
        "enforcement; NOT the same as ppo_no_mask, which changes the algorithm class too. "
        "MUST match between training and evaluation unless the arm is deliberately labelled "
        "zero-shot. Neither non-default value touches valid/success/reward or any shared RNG "
        "stream. See EnvConfig.mask_control and env.py's _mask_control_override docstring.",
    )
    parser.add_argument(
        "--mask-control-seed",
        type=int,
        default=20260826,
        help="Only read under --mask-control size_matched. Feeds a one-shot SeedSequence "
        "derived from (this seed, t, the current request); no shared RNG stream is touched, "
        "so workload_rng/failure_rng/scenario_rng stay bit-identical. Because the request "
        "sequence is exogenous, every method in an arm sees the same control masks at each "
        "step for a given seed -- the same pairing property failure_rng was built for.",
    )
    parser.add_argument(
        "--industrial-pattern",
        choices=["none", "takt", "alarm_storm", "shift", "composite"],
        default="none",
        help="Industrial workload TIME STRUCTURE (line takt / alarm storm / shift step). "
        "none (default) reproduces every existing artifact bit-for-bit: the hook returns the "
        "very same request dict, does no arithmetic and draws no random numbers. The other "
        "levels only reshape demand OVER TIME (plus the request priority during a storm); "
        "topology, cost and capacity are untouched. Level parameters are frozen in "
        "configs/industrial_workload_profile.json and are NOT tunable on the CLI. Note that "
        "alarm_storm/shift/composite are NOT mean-preserving, so each needs a companion "
        "--arrival-scale <analytic mean> arm; otherwise the readout is just 'more load'. "
        "MUST match between training and evaluation unless the arm is labelled zero-shot.",
    )
    parser.add_argument(
        "--industrial-pattern-path",
        default=None,
        help="Path to an industrial_workload_profile/v1 JSON. Required whenever "
        "--industrial-pattern is not none; a missing file or a schema mismatch is a hard "
        "error rather than a silent fallback.",
    ); add_trace_replay_args(parser); add_scale_tier_args(parser); add_telemetry_args(parser)
    args = parser.parse_args()

    if args.sla_cost_priority_weighted and args.constraint_mode == "tiered":
        parser.error("--sla-cost-priority-weighted 与 --constraint-mode tiered 互斥："
                     "A2 是「一条聚合约束 + 优先级加权 cost」，A3 是「两条分档约束 + 裸指示 "
                     "cost」；同时开会把 tiered 那两条在未加权 cost 上定标的上限静默改掉。")
    reject_p3o_under_tiered(args, parser=parser)
    return args

def reject_p3o_under_tiered(args: argparse.Namespace,
                            parser: Optional[argparse.ArgumentParser] = None) -> None:
    if not (getattr(args, "include_p3o", False)
            and str(getattr(args, "constraint_mode", "aggregate")) == "tiered"):
        return
    msg = ("--include-p3o 与 --constraint-mode tiered 互斥：κ* = 0.003 是在**三**约束的 V(κ) 上"
           "标定的（scripts/probe_strong_baselines.py 的 select_kappa），四约束下既没有标定值也"
           "没有标定规则，而 P3O 无对偶上升 ⇒ κ 就是全部的约束执行力度。要在 tiered 下跑 P3O："
           "先按四约束重标 κ。")
    if parser is not None:
        parser.error(msg)
    raise SystemExit("参数组合被拒：" + msg)

def add_trace_replay_args(parser: argparse.ArgumentParser) -> None:
    from semantic_lb_env.trace_replay import CONFIG_FIELDS, DEFAULT_SOURCE, POLICY, UNSET
    flag = ["--" + k.replace("_", "-") for k in CONFIG_FIELDS]
    parser.add_argument(
        flag[0], action="store_true",
        help="Replay the 721-point real temporal curve as a SLOW layer: one load level per "
             "episode, held for episodes_per_point consecutive episodes (policy %r), applied to "
             "request DEMAND on top of the existing fast layer. Off (default) reproduces every "
             "existing artifact bit-for-bit: not one arithmetic op, not one RNG draw, and not one "
             "of the four provenance keys is written. Turning it on REQUIRES %s (there is no "
             "default block: a default would turn 'forgot to declare' into 'silently replayed "
             "block 0' with nothing visible in the artifact)."
             % (POLICY, ", ".join(flag[1:4])),
    )
    parser.add_argument(
        flag[1], type=int, default=UNSET,
        help="Which equal-length block of the 721-point series this run consumes. Blocks are the "
             "smallest addressable unit (no free-floating start offset); only COMPLETE blocks are "
             "addressable, so the tail points are unreachable on purpose. %d = not declared."
             % UNSET,
    )
    parser.add_argument(
        flag[2], type=int, default=UNSET,
        help="Block length in sampling points. Together with %s it pins start_offset = "
             "block * block_len, which the artifact re-derives as a cross-key identity. "
             "%d = not declared." % (flag[1], UNSET),
    )
    parser.add_argument(
        flag[3], type=int, default=UNSET,
        help="How many episodes this run DECLARES it will replay. ceil(episodes / "
             "episodes_per_point) must fit inside the block: running past the end raises rather "
             "than wrapping around, because a silent wrap turns '12 h of real curve' into 'one "
             "segment played on repeat' with nothing visible in the artifact. %d = not declared."
             % UNSET,
    )

    parser.add_argument(
        flag[4], default=None,
        help=f"Replay source JSON (schema trace_temporal_profile/v1). Default {DEFAULT_SOURCE}. "
             f"The PUBLISHED traces/trace_profile.json is refused by schema+point-count, not by "
             f"filename: it has 30 timestamps over a 29-minute window and is the "
             f"*RPC_RT-contaminated (40.8%%) version.",
    )

def _trace_replay_args(args: argparse.Namespace) -> tuple:
    from semantic_lb_env.trace_replay import CONFIG_FIELDS, UNSET
    on = bool(getattr(args, CONFIG_FIELDS[0], False))
    ints = tuple(int(getattr(args, f, UNSET)) for f in CONFIG_FIELDS[1:4])
    return (on,) + ints + (getattr(args, CONFIG_FIELDS[4], None),)

def trace_replay_provenance() -> dict:
    from semantic_lb_env.trace_replay import plan_from_env_config
    p = plan_from_env_config(make_env(0, 0).unwrapped.config)
    return {} if p is None else p.provenance()

def add_scale_tier_args(parser: argparse.ArgumentParser) -> None:
    from semantic_lb_env import scale_tiers as _st
    parser.add_argument(
        "--services-per-domain", type=int, default=None,
        help="Instances per business domain (N = num_domains x this). Default None = the PUBLISHED "
             "tier (%d, N=%d), which reproduces every existing artifact bit-for-bit. The frozen "
             "ladder is %r (N %r): NOT three hand-picked numbers but the COMPLETE solution set of "
             "'semantic-coupling coverage == published' over spd <= %d (%s). Off-solution tiers are "
             "REFUSED here, before any directory is made or any step is trained, because a knob that "
             "looks like 'scale only' silently changes the coverage of the very mechanism under test "
             "(failure mode (104)). Any non-published tier changes obs/action dims, so no published "
             "weight zip loads and EVERY arm must be retrained in-tier, with the three CMDP limits "
             "recalibrated per tier by the same calibrate_limits recipe; cross-tier runs are NOT "
             "pairable (RNG consumption scales with N) and must go to a separate --out-root."
             % (_st.published_spd(), _st.published()["n_services"], list(_st.LADDER),
                [d["n_services"] for d in _st.ladder_shapes()], _st.SPD_SEARCH_MAX,
                _st.LADDER_DERIVATION),
    )

def resolve_scale_tier(args: argparse.Namespace) -> tuple:
    from semantic_lb_env import scale_tiers as _st
    spd, info, line = _st.resolve(getattr(args, "services_per_domain", None),
                                  context="run_experiment")
    if line:
        print(line, flush=True)
    return spd, _st.provenance(info)

def add_telemetry_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--telemetry-mask-enabled", action="store_true", help="启用 telemetry-aware 动态掩码（Week 1-2 新增）。默认关闭，逐位复现既有行为。")
    parser.add_argument("--telemetry-predictor-path", default=None, help="风险预测器模型路径。**默认值是 None 而不是一份路径副本**：不传时走 dataclass 的 None，由 env 解析到 env_config.DEFAULT_RISK_PREDICTOR_PATH（唯一定义点）。此前这里硬抄了一份**相对**路径，从别的 cwd 起跑就指不到文件，而 env 在文件缺失时会静默造未训练的网")
    parser.add_argument("--telemetry-beta-strategy", choices=list(_BetaScheduler.STRATEGIES), default="linear", help="β 调度策略（合法取值从 BetaScheduler.STRATEGIES 现取，本文件不手抄清单）")
    parser.add_argument("--telemetry-beta-min", type=float, default=0.5, help="预算充足时的 β 最小值")
    parser.add_argument("--telemetry-beta-max", type=float, default=2.0, help="预算耗尽时的 β 最大值")
    parser.add_argument("--telemetry-beta-fixed", type=float, default=None, help="--telemetry-beta-strategy constant 下的固定 β（A1 消融臂）。**故意没有默认值**：这个数决定消融臂的平均保守程度，正式跑里取探针实测的自适应臂平均 β")

def _maybe_train_cpo_arm(args: argparse.Namespace, *, scenario_seed: int, train_seed: int,
                         eval_seed: int, total_timesteps: int, eval_episodes: int,
                         constraint_config: ConstraintConfig, out_dir: Path, metrics: list,
                         curves: list, _OWN_LIMITS) -> dict:
    if not getattr(args, "include_cpo", False):
        return {}
    from semantic_lb_env.cpo import CPO_ARM_ENV_KWARGS, CPO_ARM_WORKLOAD_SEED_OFFSET, build_cpo_arm
    cpo_env = make_env(
        scenario_seed,
        train_seed + CPO_ARM_WORKLOAD_SEED_OFFSET,
        constraint_config=constraint_config,
        **CPO_ARM_ENV_KWARGS
    )
    cpo_logger = RewardLogger()
    cpo_model = build_cpo_arm(
        cpo_env,
        constraint_config=constraint_config,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seed=train_seed,
    )
    cpo_model.learn(total_timesteps=total_timesteps, callback=cpo_logger)
    cpo_model.save(out_dir / "cpo")
    for row in cpo_logger.rows:
        row["method"] = "cpo"
        curves.append(row)

    pd.DataFrame(cpo_model.diagnostics).to_csv(out_dir / "cpo_cost_curve.csv", index=False)
    provenance = cpo_model.provenance()

    provenance["total_timesteps_effective"] = int(cpo_model.num_timesteps)
    provenance["nominal_workload_seed"] = int(train_seed) + CPO_ARM_WORKLOAD_SEED_OFFSET
    provenance["effective_sb3_seed"] = int(train_seed)

    import os as _os
    import torch as _th
    provenance["torch_threads"] = {
        "intra_op": int(_th.get_num_threads()), "interop": int(_th.get_num_interop_threads()),
        **{k: _os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    env = make_env(scenario_seed, eval_seed, mask=False)

    row = evaluate_policy(env.unwrapped, cpo_model, episodes=eval_episodes, use_masks=False, seed=eval_seed, own_limits=_OWN_LIMITS, deterministic=args.eval_deterministic)
    row["method"] = "cpo"
    metrics.append(row)
    return {"cpo_config": provenance}

def _maybe_train_deepset_arm(args: argparse.Namespace, *, scenario_seed: int, train_seed: int,
                             eval_seed: int, total_timesteps: int, eval_episodes: int,
                             out_dir: Path, metrics: list, curves: list, _OWN_LIMITS) -> dict:
    if not getattr(args, "include_deepset", False):
        return {}
    import os as _os
    import torch as _th
    from semantic_lb_env.deepset_policy import (DEEPSET_ARM_ENV_KWARGS, DEEPSET_ARM_WORKLOAD_SEED_OFFSET,
                                                DEEPSET_INFO_LEVEL, DEEPSET_POLICY_KWARGS,
                                                build_deepset_arm, wrap_for_deepset)
    ds_env = wrap_for_deepset(make_env(scenario_seed, train_seed + DEEPSET_ARM_WORKLOAD_SEED_OFFSET,
                                       **DEEPSET_ARM_ENV_KWARGS))
    ds_logger = RewardLogger()
    ds_model = build_deepset_arm(ds_env, learning_rate=args.learning_rate, n_steps=args.n_steps,
                                 batch_size=args.batch_size, seed=train_seed)
    ds_model.learn(total_timesteps=total_timesteps, callback=ds_logger)
    ds_model.save(out_dir / "deepset_ppo")
    for row in ds_logger.rows:
        row["method"] = "deepset_ppo"
        curves.append(row)
    provenance = {
        "algo": "deepset_ppo", "info_level": DEEPSET_INFO_LEVEL,
        "architecture": {k: (v.__name__ if isinstance(v, type) else v) for k, v in DEEPSET_POLICY_KWARGS.items()},
        "n_parameters": int(sum(p.numel() for p in ds_model.policy.parameters())), "source_sha256": __import__("semantic_lb_env.deepset_policy", fromlist=["x"]).deepset_source_sha256(),
        "total_timesteps_effective": int(ds_model.num_timesteps),

        "nominal_workload_seed": int(ds_env.unwrapped.config.workload_seed),
        "effective_sb3_seed": int(ds_model.seed),
        "torch_threads": {"intra_op": int(_th.get_num_threads()), "interop": int(_th.get_num_interop_threads()),
                          **{k: _os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
    }

    env = wrap_for_deepset(make_env(scenario_seed, eval_seed, mask=True).unwrapped)
    row = evaluate_policy(env, ds_model, episodes=eval_episodes, use_masks=True, seed=eval_seed, own_limits=_OWN_LIMITS, deterministic=args.eval_deterministic)
    row["method"] = "deepset_ppo"
    metrics.append(row)
    return {"deepset_config": provenance}

if __name__ == "__main__":
    train_and_eval(parse_args())
