from __future__ import annotations

import argparse
from pathlib import Path

from run_experiment import parse_args as base_parse_args
from run_experiment import train_and_eval

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the experiment across multiple training seeds with a fixed "
        "scenario topology and a fixed evaluation workload, so the per-seed "
        "results form paired samples suitable for Friedman/Wilcoxon tests."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--scenario-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=10042)
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output-dir", default="results/multiseed_30k")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--skip-dqn", action="store_true", default=True)
    parser.add_argument(
        "--include-dqn",
        action="store_true",
        help="Train and evaluate the DQN baseline as well. This additive flag "
        "exists because --skip-dqn above is store_true with default=True, i.e. "
        "passing it can only re-assert the default and there is no way to unset "
        "it from this runner. Pair with --dqn-timesteps to give DQN the same "
        "budget as the PPO family (its default 5000 is a stub, not a fair arm).",
    )
    parser.add_argument(
        "--extra-ablation",
        action="store_true",
        help="Forwarded to run_experiment: also train sc_mppo_no_semantic_reward "
             "(mask + CMDP, semantic reward off).",
    )
    parser.add_argument("--dqn-timesteps", type=int, default=5000)
    parser.add_argument("--sla-limit", type=float, default=0.02)
    parser.add_argument("--cost-limit", type=float, default=0.45)
    parser.add_argument("--failure-limit", type=float, default=0.037)
    parser.add_argument("--multiplier-lr", type=float, default=0.05)
    parser.add_argument("--multiplier-max", type=float, default=20.0)
    parser.add_argument(
        "--multiplier-mode",
        choices=["integral", "pid"],
        default="integral",
        help="Forwarded to run_experiment: Lagrange-multiplier controller. "
        "'integral' (default) is the existing rule; 'pid' adds the P and D terms "
        "of Stooke et al. (2020) and is bit-identical to it at kp=kd=0.",
    )
    parser.add_argument("--multiplier-kp", type=float, default=0.0); parser.add_argument("--constraint-mode", choices=["aggregate", "tiered"], default="aggregate", help="转发给 run_experiment：CMDP 约束集合。aggregate（默认）保持 sla/cost/failure 三条、逐位复现本旗标之前的全部运行；tiered 把 sla 按请求紧迫度拆成 sla_high/sla_bulk 两条各带独立乘子（分档线在 semantic_lb_env/tiers.py），聚合 sla 随之退出约束集合。tiered 下 --sla-high-limit 与 --sla-bulk-limit 必须显式给出。")
    parser.add_argument("--multiplier-kd", type=float, default=0.0); parser.add_argument("--sla-high-limit", type=float, default=None, help="转发给 run_experiment：高紧迫度档的每步上限。仅在 --constraint-mode tiered 下被读，且**必须**显式给出 —— 默认 None 会让 constraint_names() 当场抛错，而不是静默沿用 --sla-limit（摊薄计法下两者量纲不同）。由 ③ 定标探针先于数据标定。"); parser.add_argument("--sla-bulk-limit", type=float, default=None, help="转发给 run_experiment：非高紧迫度档的每步上限。契约同 --sla-high-limit。")
    parser.add_argument(
        "--trace-profile",
        default=None,
        help="Path to a fitted trace_profile.json for trace-calibrated dynamics "
        "(propagated to every per-seed run). Omit for synthetic dynamics.",
    )
    parser.add_argument(
        "--arrival-scale",
        type=float,
        default=1.0,
        help="Load-stress multiplier on arrival rate (propagated to every "
        "per-seed run). 1.0 = nominal load.",
    )
    parser.add_argument(
        "--service-rate-scale",
        type=float,
        default=1.0,
        help="Multiplier on service capacity (propagated to every per-seed run). "
        "1.0 = nominal capacity.",
    )
    parser.add_argument(
        "--no-cross-domain",
        action="store_true",
        help="Forwarded to run_experiment: ablate the cross-domain candidate arm "
        "(candidates restricted to the request's own business domain).",
    )
    parser.add_argument(
        "--domain-skew",
        type=float,
        default=0.18,
        help="Forwarded to run_experiment: request-domain imbalance amplitude "
        "(0.18 = default; higher concentrates load on the manufacturing domain).",
    )
    parser.add_argument(
        "--cross-demand-frac",
        type=float,
        default=0.0,
        help="Forwarded to run_experiment: fraction of requests whose function "
        "type the origin domain cannot serve (structural cross-domain need). "
        "0.0 = default; positive (e.g. 0.35) injects genuine cross-domain pressure.",
    )
    parser.add_argument(
        "--sla-aware-candidates",
        action="store_true",
        help="Forwarded to run_experiment: prune candidates whose projected "
        "latency exceeds the request's SLA budget (SLA feature enters candidate "
        "selection, not just reward).",
    )
    parser.add_argument(
        "--resource-aware-candidates",
        action="store_true",
        help="Forwarded to run_experiment: prune candidates that do not fit the "
        "request's resource preference (priority->reliability floor, demand->"
        "capacity headroom).",
    )
    parser.add_argument(
        "--sla-scale",
        type=float,
        default=1.0,
        help="Forwarded to run_experiment: multiplier on the sampled SLA budget "
        "(1.0 = default, bit-for-bit unchanged; <1 tightens deadlines so the SLA "
        "candidate arm actually bites, e.g. 0.15).",
    )
    parser.add_argument(
        "--balance-metric",
        choices=["variance", "jain"],
        default="variance",
        help="转发给 run_experiment：负载均衡奖励项形式。"
        "variance（默认）逐位复现主实验；jain 改为 Jain 公平性奖励。",
    )
    parser.add_argument(
        "--balance-weight",
        type=float,
        default=0.8,
        help="转发给 run_experiment：负载均衡项权重（见 EnvConfig.balance_weight）。"
        "0.8（默认）逐位复现主实验；jain 模式下常用不可忽略权重（如 12）。",
    )
    parser.add_argument(
        "--failure-frac",
        type=float,
        default=0.0,
        help="转发给 run_experiment：真实性扰动轴 A，任一时刻处于故障的服务实例稳态"
        "比例。0.0（默认）整段短路，逐位复现既有实验臂；设计档位 0.10/0.20/0.30。",
    )
    parser.add_argument(
        "--failure-duration",
        type=int,
        default=12,
        help="转发给 run_experiment：单次故障持续步数（12 = episode_len=120 的 10%%）。",
    )
    parser.add_argument(
        "--failure-invisible",
        action="store_true",
        help="转发给 run_experiment：故障实例留在候选集中（仍必然失败），而不是被 "
        "oracle 健康检查从动作掩码里剔除。默认关闭 = 可见臂。",
    )
    parser.add_argument(
        "--cross-domain-rtt",
        type=float,
        default=0.0,
        help="转发给 run_experiment：真实性扰动轴 B，跨业务域承接叠加的网络 RTT（ms）。"
        "0.0（默认）逐位复现既有实验臂；设计档位 20/50/100。",
    )
    parser.add_argument(
        "--cross-domain-cost-mult",
        type=float,
        default=1.0,
        help="转发给 run_experiment：跨域流量的成本倍率（轴 B 的独立敏感性臂）。"
        "1.0（默认）精确无变化。",
    )
    parser.add_argument(
        "--no-semantic-qos-coupling",
        action="store_true",
        help="转发给 run_experiment：解耦消融，拆掉 env 内生的\"语义主营 ⇒ QoS 更好\"耦合"
        "（主营时延 ×0.75/可靠性 +0.025/容量 ×1.12，非主营时延 ×1.35/容量 ×0.82）。"
        "语义掩码、候选生成、reward 侧语义加分一律不动。默认关闭 = 保留耦合、逐位复现"
        "既有全部实验臂。训练与评估两端必须一致。",
    )
    parser.add_argument(
        "--cost-model",
        choices=["synthetic_uniform", "power_based"],
        default="synthetic_uniform",
        help="转发给 run_experiment：成本标尺。synthetic_uniform（默认）= cost ~ U(0.25,1.0)，"
        "溯源 C 级、与容量/时延独立，逐位复现既有全部实验臂；power_based 用公开功耗规格 × "
        "公开 PUE/电价推导单位成本（需 --cost-profile）。训练与评估两端必须一致。",
    )
    parser.add_argument(
        "--cost-profile",
        default=None,
        help="转发给 run_experiment：power_cost_profile/v1 JSON 路径，power_based 必需。",
    )
    parser.add_argument(
        "--no-cost-iso-scale",
        action="store_true",
        help="转发给 run_experiment：power_based 下改为只对齐均值、保留物理离散度"
        "（默认是仿射映射回同 seed 合成向量的 [min,max]，保住成本预算口径）。",
    )
    parser.add_argument(
        "--mask-control",
        choices=["enforced", "size_matched", "advisory"],
        default="enforced",
        help="转发给 run_experiment：候选–mask 耦合对照。enforced（默认）= 语义候选集既提议"
        "又强制，逐位复现既有全部实验臂；size_matched = 强制一个与语义掩码逐步等势、但内容"
        "与语义无关的随机子集（只留「动作空间变小」）；advisory = 语义集照算但返回全 1 掩码"
        "（只留「建议」、去掉强制）。训练与评估两端必须一致，除非该臂被显式标为零样本。",
    )
    parser.add_argument(
        "--mask-control-seed",
        type=int,
        default=20260826,
        help="转发给 run_experiment：只在 --mask-control size_matched 下被读；不消耗任何"
        "共享 RNG 流。",
    )
    parser.add_argument(
        "--industrial-pattern",
        choices=["none", "takt", "alarm_storm", "shift", "composite"],
        default="none",
        help="转发给 run_experiment：工业负载时间结构档位（节拍/告警风暴/班次阶跃）。none（默认）= 逐位复现既有全部产物（钩子是恒等映射、不抽任何随机数）；其余档位只改 demand 的**时间分布**与风暴期请求优先级，拓扑/成本/容量一字不动。档位参数在 configs/industrial_workload_profile.json 里冻结，不在 CLI 上调。注意 alarm_storm/shift/composite **不**均值守恒 ⇒ 必须同时跑一条--arrival-scale <解析均值> 的同均值对照臂，否则测到的只是「负载变大」。",
    )
    parser.add_argument(
        "--industrial-pattern-path",
        default=None,
        help="转发给 run_experiment：industrial_workload_profile/v1 JSON 路径。--industrial-pattern 非 none 时必需；文件缺失/schema 不符是硬错误而非静默回退。",
    )

    parser.add_argument(
        "--include-p3o",
        action="store_true",
        help="转发给 run_experiment：额外训练 P3O 文献基线（--include-p3o）。默认关闭 ⇒ "
        "逐位复现既有全部产物（旗标关时 run_experiment 那段整体不执行、不抽 RNG、"
        "summary.json 不多键）。与 --include-dqn 同为加法式旗标。",
    )
    parser.add_argument(
        "--p3o-penalty-coef",
        type=float,
        default=5.0,
        help="转发给 run_experiment：P3O 的固定罚系数 κ（三条约束共用）。P3O 没有对偶上升，"
        "这个值就是全部的约束强度旋钮 ⇒ 必须由数据前的定标探针选，不许拿家族终点调。",
    )
    parser.add_argument(
        "--p3o-cost-vf-coef",
        type=float,
        default=0.5,
        help="转发给 run_experiment：P3O 代价 critic 回归损失的权重（对齐 PPO vf_coef 的 0.5）。",
    ); _add_scale_tier_cli(parser); _add_cpo_cli(parser); _add_deepset_cli(parser)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        run_args = base_parse_args_defaults()
        run_args.seed = seed
        run_args.scenario_seed = args.scenario_seed
        run_args.eval_seed = args.eval_seed
        run_args.timesteps = args.timesteps
        run_args.eval_episodes = args.eval_episodes
        run_args.learning_rate = args.learning_rate
        run_args.n_steps = args.n_steps
        run_args.batch_size = args.batch_size
        run_args.skip_dqn = args.skip_dqn and not args.include_dqn
        run_args.extra_ablation = args.extra_ablation
        run_args.dqn_timesteps = args.dqn_timesteps
        run_args.sla_limit = args.sla_limit
        run_args.cost_limit = args.cost_limit
        run_args.failure_limit = args.failure_limit
        run_args.multiplier_lr = args.multiplier_lr
        run_args.multiplier_max = args.multiplier_max
        run_args.multiplier_mode = args.multiplier_mode
        run_args.multiplier_kp = args.multiplier_kp
        run_args.multiplier_kd = args.multiplier_kd; run_args.constraint_mode = args.constraint_mode; run_args.sla_high_limit = args.sla_high_limit; run_args.sla_bulk_limit = args.sla_bulk_limit
        run_args.trace_profile = args.trace_profile
        run_args.arrival_scale = args.arrival_scale
        run_args.service_rate_scale = args.service_rate_scale
        run_args.no_cross_domain = args.no_cross_domain
        run_args.domain_skew = args.domain_skew
        run_args.cross_demand_frac = args.cross_demand_frac
        run_args.sla_aware_candidates = args.sla_aware_candidates
        run_args.resource_aware_candidates = args.resource_aware_candidates
        run_args.sla_scale = args.sla_scale
        run_args.balance_metric = args.balance_metric
        run_args.balance_weight = args.balance_weight
        run_args.failure_frac = args.failure_frac
        run_args.failure_duration = args.failure_duration
        run_args.failure_invisible = args.failure_invisible
        run_args.cross_domain_rtt = args.cross_domain_rtt
        run_args.cross_domain_cost_mult = args.cross_domain_cost_mult
        run_args.no_semantic_qos_coupling = args.no_semantic_qos_coupling
        run_args.cost_model = args.cost_model
        run_args.cost_profile = args.cost_profile
        run_args.no_cost_iso_scale = args.no_cost_iso_scale
        run_args.mask_control = args.mask_control
        run_args.mask_control_seed = args.mask_control_seed
        run_args.industrial_pattern = args.industrial_pattern
        run_args.industrial_pattern_path = args.industrial_pattern_path
        run_args.include_p3o = args.include_p3o; run_args.include_cpo = args.include_cpo; run_args.include_deepset = args.include_deepset
        run_args.p3o_penalty_coef = args.p3o_penalty_coef
        run_args.p3o_cost_vf_coef = args.p3o_cost_vf_coef; run_args.services_per_domain = args.services_per_domain
        run_args.output_dir = str(root / f"seed_{seed}")

        print(f"\n===== training seed {seed} -> {run_args.output_dir} =====")
        train_and_eval(run_args)

    print(f"\nAll seeds finished. Aggregate with:\n  python scripts/aggregate_stats.py --root {root}")

def base_parse_args_defaults() -> argparse.Namespace:

    import sys

    argv_backup = sys.argv
    sys.argv = ["run_experiment.py"]
    try:
        return base_parse_args()
    finally:
        sys.argv = argv_backup

def _add_scale_tier_cli(parser: argparse.ArgumentParser) -> None:
    from run_experiment import add_scale_tier_args
    add_scale_tier_args(parser)

def _add_cpo_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--include-cpo",
        action="store_true",
        help="转发给 run_experiment：额外训练 CPO 文献基线（Achiam et al., ICML 2017）。默认关闭 ⇒ "
        "逐位复现既有全部产物（旗标关时 run_experiment 那段第一行就返回、不抽 RNG、"
        "summary.json 不多键）。",
    )

def _add_deepset_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--include-deepset",
        action="store_true",
        help="转发给 run_experiment：额外训练 DRL-FORCH 风格的 DeepSet 基线（等变 actor + 不变 critic，"
        "掩码 PPO，无 CMDP）。默认关闭 ⇒ 逐位复现既有全部产物。",
    )

if __name__ == "__main__":
    main()
