from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sb3_contrib import MaskablePPO, TRPO
from stable_baselines3 import DQN, PPO

from semantic_lb_env.baselines import LeastLoadedPolicy, RandomPolicy, RoundRobinPolicy
from semantic_lb_env.env import EnvConfig, SemanticLoadBalancingEnv
from semantic_lb_env.evaluate import DEFAULT_CONSTRAINT_LIMITS, evaluate_policy

SCENARIO_SEED = 42
EVAL_SEED = 10042

_EVAL_SEED = EVAL_SEED

_SCENARIO_SEED = SCENARIO_SEED

_TRACE_PROFILE_PATH = None

_ARRIVAL_SCALE = 1.0
_SERVICE_RATE_SCALE = 1.0

_CROSS_DOMAIN = True
_DOMAIN_SKEW = 0.18
_CROSS_DEMAND_FRAC = 0.0

_SLA_AWARE = False
_RESOURCE_AWARE = False
_SLA_SCALE = 1.0

_BALANCE_METRIC = "variance"
_BALANCE_WEIGHT = 0.8

_FAILURE_FRAC = 0.0
_FAILURE_DURATION = 12
_FAILURE_VISIBLE = True
_CROSS_DOMAIN_RTT = 0.0
_CROSS_DOMAIN_COST_MULT = 1.0

_SEMANTIC_QOS_COUPLING = True

_COST_MODEL = "synthetic_uniform"
_COST_PROFILE_PATH = None
_COST_ISO_SCALE = True
_MASK_CONTROL = "enforced"
_MASK_CONTROL_SEED = 20260826
_INDUSTRIAL_PATTERN: str = "none"
_INDUSTRIAL_PATTERN_PATH = None; _TRACE_REPLAY: bool = False; _TRACE_REPLAY_BLOCK: int = -1; _TRACE_REPLAY_BLOCK_LEN: int = -1; _TRACE_REPLAY_EPISODES: int = -1; _TRACE_REPLAY_SOURCE = None; _TRACE_REPLAY_PROV: dict = {}

_TOPOLOGY_VARIANT = 0; _SERVICES_PER_DOMAIN: int = EnvConfig().services_per_domain; _TELEMETRY_MASK_CONFIG = None; _TELEMETRY_PROV: dict = {}

_OWN_LIMITS = None

_SHARED_LIMITS = dict(DEFAULT_CONSTRAINT_LIMITS)

def make_env(mask: bool = True, semantic_reward: bool = True) -> SemanticLoadBalancingEnv:
    return SemanticLoadBalancingEnv(
        config=EnvConfig(
            seed=_SCENARIO_SEED,
            scenario_seed=_SCENARIO_SEED,
            workload_seed=_EVAL_SEED,
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
            mask_control_seed=_MASK_CONTROL_SEED,
            industrial_pattern=_INDUSTRIAL_PATTERN,
            industrial_pattern_path=_INDUSTRIAL_PATTERN_PATH,
            topology_variant=_TOPOLOGY_VARIANT, trace_replay=_TRACE_REPLAY, trace_replay_block=_TRACE_REPLAY_BLOCK, trace_replay_block_len=_TRACE_REPLAY_BLOCK_LEN, trace_replay_episodes=_TRACE_REPLAY_EPISODES, trace_replay_source=_TRACE_REPLAY_SOURCE, services_per_domain=_SERVICES_PER_DOMAIN, telemetry_mask_config=_TELEMETRY_MASK_CONFIG,
        ),
        use_semantic_mask=mask,
        include_semantic_state=True,
        reward_semantic=semantic_reward,
    )

LEARNING_MODELS = [
    ("ppo_no_mask", PPO, False, True),
    ("ppo_lagrangian", PPO, False, True),
    ("maskable_ppo_semantic", MaskablePPO, True, True),
    ("sc_mppo", MaskablePPO, True, True),
    ("maskable_ppo_no_semantic_reward", MaskablePPO, True, False),

    ("sc_mppo_no_semantic_reward", MaskablePPO, True, False),

    ("dqn_no_mask", DQN, False, True), ("p3o", PPO, False, True), ("cpo", TRPO, False, True), ("deepset_ppo", MaskablePPO, True, True),
]

def eval_seed_dir(seed_dir: Path, episodes: int, deterministic: bool) -> pd.DataFrame:
    rows = []

    probe = make_env()
    num_actions = probe.num_services
    baselines = [
        ("random_semantic", RandomPolicy(num_actions, respect_mask=True, seed=_EVAL_SEED), True, True),
        ("round_robin_semantic", RoundRobinPolicy(num_actions, respect_mask=True), True, True),
        ("least_loaded_semantic", LeastLoadedPolicy(num_actions, num_actions, respect_mask=True), True, True),
        ("random_no_semantic", RandomPolicy(num_actions, respect_mask=False, seed=_EVAL_SEED), False, True),
    ]
    for name, policy, use_masks, sem in baselines:
        env = make_env(mask=use_masks, semantic_reward=sem)
        row = evaluate_policy(
            env, policy, episodes=episodes, use_masks=use_masks, seed=_EVAL_SEED,
            deterministic=deterministic, constraint_limits=_SHARED_LIMITS,
            own_limits=_OWN_LIMITS,
        )
        row["method"] = name
        rows.append(row)

    for stem, loader, use_masks, sem in LEARNING_MODELS:
        path = seed_dir / f"{stem}.zip"
        if not path.exists():
            continue
        model = loader.load(path, device="cpu")
        env = make_env(mask=use_masks, semantic_reward=sem); env = (__import__('semantic_lb_env.deepset_policy', fromlist=['x']).wrap_for_deepset(env) if stem == "deepset_ppo" else env)
        row = evaluate_policy(
            env, model, episodes=episodes, use_masks=use_masks, seed=_EVAL_SEED,
            deterministic=deterministic, constraint_limits=_SHARED_LIMITS,
            own_limits=_OWN_LIMITS,
        )
        row["method"] = stem
        rows.append(row)

    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved multi-seed models under deterministic and "
        "stochastic protocols with constraint-satisfaction metrics. No retraining."
    )
    parser.add_argument("--source-root", required=True,
                        help="Multi-seed dir containing seed_XX/*.zip (e.g. results/multiseed_30k).")
    parser.add_argument("--out-root", required=True,
                        help="Where to write reeval seed_XX/metrics.csv (per protocol subdir).")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--trace-profile", default=None,
                        help="Path to the SAME trace_profile.json the models were "
                        "trained with; omit for synthetic-trained models.")
    parser.add_argument("--arrival-scale", type=float, default=1.0,
                        help="MUST match the arrival_scale the models were trained "
                        "with (robustness scenarios). 1.0 = nominal load.")
    parser.add_argument("--service-rate-scale", type=float, default=1.0,
                        help="MUST match the service_rate_scale the models were "
                        "trained with. 1.0 = nominal capacity.")
    parser.add_argument("--own-cost-limit", type=float, default=None,
                        help="The --cost-limit this run was TRAINED with. Given "
                        "together with --own-sla-limit/--own-failure-limit, adds "
                        "own_*_satisfaction_rate / own_*_violation columns scored "
                        "against that budget. The shared-bar columns "
                        "(cost_satisfaction_rate etc.) stay on the fixed "
                        "DEFAULT_CONSTRAINT_LIMITS so methods remain comparable.")
    parser.add_argument("--own-sla-limit", type=float, default=None,
                        help="The --sla-limit this run was trained with.")
    parser.add_argument("--own-failure-limit", type=float, default=None,
                        help="The --failure-limit this run was trained with.")
    parser.add_argument("--failure-floor", type=float, default=None,
                        help="Override ONLY the failure entry of the shared bar "
                        "(sla/cost stay on DEFAULT_CONSTRAINT_LIMITS, keeping "
                        "those satisfaction columns' published meaning). Use a "
                        "value above the ~0.049 structural failure-rate floor "
                        "(e.g. 0.07) to turn the otherwise-dead failure_"
                        "satisfaction_rate into a live completeness check. Does "
                        "NOT mutate DEFAULT_CONSTRAINT_LIMITS; same additive "
                        "philosophy as --own-*-limit.")
    parser.add_argument("--no-cross-domain", action="store_true",
                        help="Ablate the cross-business-domain candidate arm "
                        "(innovation point 1): candidates are restricted to the "
                        "request's own domain. MUST match how the models were "
                        "TRAINED; use only to re-evaluate models trained with the "
                        "same setting. Default keeps the cross-domain arm on.")
    parser.add_argument("--domain-skew", type=float, default=0.18,
                        help="Request-domain imbalance amplitude the models were "
                        "TRAINED with. 0.18 = original default; larger (e.g. 0.55) "
                        "concentrates load on one domain to stress cross-domain "
                        "routing. Must match training.")
    parser.add_argument("--cross-demand-frac", type=float, default=0.0,
                        help="Fraction of requests whose function TYPE the origin "
                        "domain cannot serve (structural cross-domain need) the "
                        "models were TRAINED with. 0.0 = original default (no forced "
                        "cross-domain demand); positive (e.g. 0.35) injects genuine "
                        "cross-domain pressure. Must match training.")
    parser.add_argument("--sla-aware-candidates", action="store_true",
                        help="Enable the SLA-budget candidate-pruning arm the models "
                        "were TRAINED with (innovation point 1: SLA feature shapes the "
                        "candidate set). Must match training.")
    parser.add_argument("--resource-aware-candidates", action="store_true",
                        help="Enable the resource-preference candidate-pruning arm the "
                        "models were TRAINED with (innovation point 1: priority/demand "
                        "shape the candidate set). Must match training.")
    parser.add_argument("--sla-scale", type=float, default=1.0,
                        help="Multiplier on the sampled SLA budget the models were "
                        "TRAINED with. 1.0 = original default; <1 (e.g. 0.15) tightens "
                        "deadlines so the SLA arm actually bites. Must match training.")
    parser.add_argument("--balance-metric", choices=["variance", "jain"], default="variance",
                        help="负载均衡奖励度量，必须与训练时一致。'variance'（默认）逐位复现"
                        "主实验；'jain' 用 balance_weight*(Jain-1) 公平性奖励。")
    parser.add_argument("--balance-weight", type=float, default=0.8,
                        help="负载均衡奖励权重，必须与训练时一致（默认 0.8）。jain 模式下"
                        "把公平性信号从 ~0.02 量级放大到该权重量级。")
    parser.add_argument("--failure-frac", type=float, default=0.0,
                        help="真实性扰动轴 A：任一时刻处于故障的实例稳态比例。MUST match "
                        "training（把 failure_frac=0 训出的模型放到故障 env 上重评估会重新"
                        "引入 train/eval 不匹配）。0.0 = 默认，逐位复现历史评估。")
    parser.add_argument("--failure-duration", type=int, default=12,
                        help="单次故障持续步数。MUST match training。默认 12。")
    parser.add_argument("--failure-invisible", action="store_true",
                        help="故障实例留在候选集中而不被 oracle 健康检查剔除。MUST match "
                        "training。默认关闭 = 可见臂。")
    parser.add_argument("--cross-domain-rtt", type=float, default=0.0,
                        help="真实性扰动轴 B：跨业务域承接叠加的网络 RTT（ms）。一般 MUST "
                        "match training；**唯一例外**是零样本迁移臂——刻意用它在 rtt>0 下"
                        "重评估 --source-root results/multiseed_trace_recal 里 rtt=0 训出的"
                        "标称模型，以度量未见扰动下的退化。该用法必须作为独立协议单列汇报，"
                        "不得与配平重训臂的数字混报。0.0 = 默认。")
    parser.add_argument("--cross-domain-cost-mult", type=float, default=1.0,
                        help="跨域流量成本倍率（轴 B 敏感性臂）。MUST match training。"
                        "1.0 = 默认，精确无变化。")
    parser.add_argument("--no-semantic-qos-coupling", action="store_true",
                        help="解耦消融：拆掉 env 内生的\"语义主营 ⇒ QoS 更好\"耦合"
                        "（主营时延 ×0.75/可靠性 +0.025/容量 ×1.12，非主营时延 ×1.35/"
                        "容量 ×0.82）；语义掩码、候选生成、reward 侧语义加分一律不动。"
                        "MUST match training——解耦臂的模型在解耦环境里训出，用带耦合的"
                        "环境评估会把分布错配读成性能下降。默认关闭 = 保留耦合。")
    parser.add_argument("--cost-model", choices=["synthetic_uniform", "power_based"],
                        default="synthetic_uniform",
                        help="成本标尺。synthetic_uniform = 默认，cost ~ U(0.25,1.0)（溯源 C 级），"
                        "逐位复现全部历史评估；power_based 用公开功耗规格 × 公开 PUE/电价推导"
                        "（需 --cost-profile）。MUST match training。见 EnvConfig.cost_model。")
    parser.add_argument("--cost-profile", default=None,
                        help="power_cost_profile/v1 JSON 路径（如 configs/power_cost_profile.json）。"
                        "--cost-model power_based 必需；文件缺失是硬错误而非静默回退。")
    parser.add_argument("--no-cost-iso-scale", action="store_true",
                        help="只在 power_based 下有意义。默认把物理成本仿射映射回同 seed 合成向量的"
                        "[min,max]（保住 max(self.cost)、共享预算 0.457 与 cost_weight 量级，"
                        "唯一变量是成本向量的形状）；本旗标改为只对齐均值、保留物理离散度，"
                        "此时该臂必须自带 --own-cost-limit。MUST match training。")
    parser.add_argument("--mask-control", choices=["enforced", "size_matched", "advisory"],
                        default="enforced",
                        help="候选–mask 耦合对照。enforced（默认）= 语义候选集既提议又强制，"
                        "逐位复现全部历史评估；size_matched = 强制一个与语义掩码逐步等势、内容"
                        "与语义无关的均匀随机子集（只留「动作空间变小」）；advisory = 语义集照算"
                        "（valid/success/reward/召回率五列全不变）但 action_masks() 返回全 1"
                        "（只留「建议」）。三者都不碰 valid/reward 与任何共享 RNG 流。"
                        "**训练与评估两端一致时**是配平臂；此处单独改 = 零样本换掩码臂，"
                        "必须按 zs_rtt_* 的先例显式标注、不得与配平臂混报。"
                        "见 EnvConfig.mask_control 与 env.py 的 _mask_control_override docstring。")
    parser.add_argument("--mask-control-seed", type=int, default=20260826,
                        help="只在 --mask-control size_matched 下被读；进的是一条由"
                        "(该种子, t, 当前请求) 派生的一次性 SeedSequence，"
                        "workload_rng/failure_rng/scenario_rng 逐位不动。")
    parser.add_argument("--industrial-pattern",
                        choices=["none", "takt", "alarm_storm", "shift", "composite"],
                        default="none",
                        help="工业负载时间结构档位（节拍/告警风暴/班次阶跃）。none（默认）= 逐位复现既有全部产物（钩子是恒等映射、不抽任何随机数）；其余档位只改 demand 的**时间分布**与风暴期请求优先级，拓扑/成本/容量一字不动。档位参数在 configs/industrial_workload_profile.json 里冻结，不在 CLI 上调。注意 alarm_storm/shift/composite **不**均值守恒 ⇒ 必须同时跑一条--arrival-scale <解析均值> 的同均值对照臂，否则测到的只是「负载变大」。"
                        "在重打分侧单独改 = 零样本工业模式臂（先例 zs_rtt_*），"
                        "必须显式标注、不得与配平重训臂混报。")
    parser.add_argument("--industrial-pattern-path", default=None,
                        help="industrial_workload_profile/v1 JSON 路径。--industrial-pattern 非 none 时必需；文件缺失/schema 不符是硬错误而非静默回退。")
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED,
                        help="评估工作负载种子（每个 episode 用 eval_seed + ep 重建 "
                        "workload_rng/failure_rng，见 env.py:442-448）。默认 10042 = "
                        "全部历史重打分所用的那**一个**实现 ⇒ 逐位复现、空操作。"
                        "换一个值 = 在**同一批已训模型**上复制评估负载方差（不重训）；"
                        "这是既有 8 种子谱**结构性缺失**的方差成分。**不改** "
                        "SCENARIO_SEED（拓扑必须与训练一致）⇒ 换的是测试轨迹，"
                        "不是被测系统。口径要求：换过 eval seed 的产物必须写进独立 "
                        "out-root，且**不得**与 10042 的数字混入同一个配对家族"
                        "（它们不是重复实验，而是同一模型的两次不同评估）。")
    parser.add_argument("--scenario-seed", type=int, default=SCENARIO_SEED,
                        help="场景实现种子（scenario_rng：base_capacity / cost / "
                        "base_reliability + trace 时延曲线到槽位的 permutation）。"
                        "默认 42 = 全部历史臂训练与重打分所用的那**一个**实现 ⇒ 逐位"
                        "复现、空操作。换一个值 = 固定规模（3 域 × 5 服务 = 15 实例、"
                        "obs_dim 43 不变）下把已训模型放到**另一个场景实现**上零样本"
                        "重打分。**不换**语义结构（service_domain / service_type / "
                        "semantic_matrix 决定性生成）⇒ 换的是被测场景，不是规模也不是"
                        "语义拓扑。口径要求：独立 out-root；**不得**与 42 的数字混入同"
                        "一个配对家族；`own_*` 满足率的跨场景解读必须同报'上限是在 42 "
                        "上标定的'这一条。")
    parser.add_argument("--topology-variant", type=int, default=0,
                        help="固定规模**多拓扑**变体 id。0（默认）= 历史语义矩阵 ⇒ 逐位"
                        "复现、空操作。1..216 = semantic_lb_env/topology.py 规范枚举里"
                        "的第 id 个变体：换的是 semantic_matrix（哪个业务域主营哪些功能"
                        "类型），保行度 3 / 保 QoS 加成实例数 9 / 保每域候选集规模 "
                        "(6,7,7)；规模（15 实例 / obs 43 / action 15）与实例目录 "
                        "service_domain / service_type 不变 ⇒ 已发表模型零样本可载。"
                        "与 --scenario-seed 是**两条不同的轴**：那条换实例属性实现、"
                        "语义结构不动（故禁令 B7 说它不是多拓扑），本条恰好相反 —— "
                        "scenario_rng 的三次抽样与 trace permutation 一字不动。"
                        "id 28 = 历史组合本身（走变体代码路径），必须与 0 逐位相同。"
                        "口径要求：独立 out-root；不得与 0 档或 --scenario-seed 档混入"
                        "同一个配对家族；零样本迁移，不得与重训数字混报。"); _add_trace_replay_cli(parser); _add_scale_tier_cli(parser); _add_telemetry_cli(parser)
    args = parser.parse_args()

    global _TRACE_PROFILE_PATH, _ARRIVAL_SCALE, _SERVICE_RATE_SCALE, _OWN_LIMITS
    global _SHARED_LIMITS, _CROSS_DOMAIN, _DOMAIN_SKEW, _CROSS_DEMAND_FRAC
    global _SLA_AWARE, _RESOURCE_AWARE, _SLA_SCALE, _BALANCE_METRIC, _BALANCE_WEIGHT
    global _FAILURE_FRAC, _FAILURE_DURATION, _FAILURE_VISIBLE
    global _CROSS_DOMAIN_RTT, _CROSS_DOMAIN_COST_MULT
    global _SEMANTIC_QOS_COUPLING
    global _COST_MODEL, _COST_PROFILE_PATH, _COST_ISO_SCALE
    global _MASK_CONTROL, _MASK_CONTROL_SEED, _EVAL_SEED, _INDUSTRIAL_PATTERN, _INDUSTRIAL_PATTERN_PATH
    global _SCENARIO_SEED
    global _TOPOLOGY_VARIANT; global _TRACE_REPLAY, _TRACE_REPLAY_BLOCK, _TRACE_REPLAY_BLOCK_LEN, _TRACE_REPLAY_EPISODES, _TRACE_REPLAY_SOURCE, _TRACE_REPLAY_PROV; global _SERVICES_PER_DOMAIN; global _TELEMETRY_MASK_CONFIG, _TELEMETRY_PROV
    _TRACE_PROFILE_PATH = args.trace_profile
    _ARRIVAL_SCALE = args.arrival_scale
    _SERVICE_RATE_SCALE = args.service_rate_scale
    _CROSS_DOMAIN = not args.no_cross_domain
    _DOMAIN_SKEW = args.domain_skew
    _CROSS_DEMAND_FRAC = args.cross_demand_frac
    _SLA_AWARE = args.sla_aware_candidates
    _RESOURCE_AWARE = args.resource_aware_candidates
    _SLA_SCALE = args.sla_scale
    _BALANCE_METRIC = args.balance_metric
    _BALANCE_WEIGHT = args.balance_weight
    _FAILURE_FRAC = args.failure_frac
    _FAILURE_DURATION = args.failure_duration
    _FAILURE_VISIBLE = not args.failure_invisible
    _CROSS_DOMAIN_RTT = args.cross_domain_rtt
    _CROSS_DOMAIN_COST_MULT = args.cross_domain_cost_mult
    _SEMANTIC_QOS_COUPLING = not args.no_semantic_qos_coupling
    _COST_MODEL = args.cost_model
    _COST_PROFILE_PATH = args.cost_profile
    _COST_ISO_SCALE = not args.no_cost_iso_scale
    _MASK_CONTROL = args.mask_control
    _MASK_CONTROL_SEED = args.mask_control_seed
    _EVAL_SEED = args.eval_seed
    _SCENARIO_SEED = args.scenario_seed
    _INDUSTRIAL_PATTERN = args.industrial_pattern
    _INDUSTRIAL_PATTERN_PATH = args.industrial_pattern_path
    _TOPOLOGY_VARIANT = args.topology_variant; (_TRACE_REPLAY, _TRACE_REPLAY_BLOCK, _TRACE_REPLAY_BLOCK_LEN, _TRACE_REPLAY_EPISODES, _TRACE_REPLAY_SOURCE) = _trace_replay_args_of(args); _SERVICES_PER_DOMAIN = _scale_tier_of(args)
    if _TOPOLOGY_VARIANT:

        from semantic_lb_env import topology as _tp
        inv = _tp.invariants(_TOPOLOGY_VARIANT)
        print(f"[topology-variant] 语义矩阵变体 {_TOPOLOGY_VARIANT}/{_tp.POOL_SIZE}："
              f"主营集 {inv['primaries']}，与历史差 {inv['hamming_from_historical']}/18 格、"
              f"{inv['domains_changed']}/3 个域换了主营集；行度 {inv['row_degree']}、"
              f"QoS 加成 {inv['qos_bonus_count']}/15、每域候选集规模 "
              f"{inv['candidate_size_multiset']}、列度 {inv['column_degree']}"
              f"（保列度={inv['column_degree_preserved']}）。"
              f"scenario_seed 仍为 {_SCENARIO_SEED}、eval_seed 仍为 {_EVAL_SEED} ⇒ "
              f"换的是**语义拓扑**，不是实例属性实现、不是测试轨迹、不是规模。"
              f"本批是**零样本迁移**产物（模型训练于历史拓扑），不得与 variant 0 档或 "
              f"--scenario-seed 档混入同一个配对家族；CMDP 上限是在历史拓扑上标定的，"
              f"own_* 满足率须带此披露。")
    if _INDUSTRIAL_PATTERN != "none" and not _INDUSTRIAL_PATTERN_PATH:
        raise SystemExit("--industrial-pattern %s 必须同时给出 --industrial-pattern-path"
                         % (_INDUSTRIAL_PATTERN,))
    if _EVAL_SEED != EVAL_SEED:
        print(f"[eval-seed] 评估工作负载种子 {EVAL_SEED} -> {_EVAL_SEED}；"
              f"SCENARIO_SEED 仍为 {SCENARIO_SEED}（拓扑不变）。"
              f"这一批产物**不得**与 {EVAL_SEED} 的产物混入同一个配对家族。")
    if _SCENARIO_SEED != SCENARIO_SEED:
        print(f"[scenario-seed] 场景实现种子 {SCENARIO_SEED} -> {_SCENARIO_SEED}；"
              f"规模与语义结构不变（15 实例 / obs_dim 43），换的是实例属性实现 ⇒ "
              f"本批是**零样本迁移**产物（模型仍训练于 {SCENARIO_SEED}）。"
              f"这一批产物**不得**与 {SCENARIO_SEED} 的产物混入同一个配对家族；"
              f"CMDP 上限是在 {SCENARIO_SEED} 上标定的，own_* 满足率须带此披露。")
    if _COST_MODEL != "synthetic_uniform" and not _COST_PROFILE_PATH:
        raise SystemExit("--cost-model %s 必须同时给出 --cost-profile" % (_COST_MODEL,))

    if args.failure_floor is not None:
        _SHARED_LIMITS = dict(DEFAULT_CONSTRAINT_LIMITS)
        _SHARED_LIMITS["failure"] = args.failure_floor
        print(f"shared bar failure floor overridden: {DEFAULT_CONSTRAINT_LIMITS['failure']} "
              f"-> {args.failure_floor} (sla/cost stay on DEFAULT)")

    own = {"sla": args.own_sla_limit, "cost": args.own_cost_limit,
           "failure": args.own_failure_limit}; _TRACE_REPLAY_PROV = _trace_replay_gate(args); (_TELEMETRY_MASK_CONFIG, _TELEMETRY_PROV) = _resolve_telemetry(args)
    given = [k for k, v in own.items() if v is not None]
    if given:
        missing = [k for k, v in own.items() if v is None]
        if missing:
            raise SystemExit(
                f"--own-*-limit must be given for all three constraints; missing: {missing}"
            )
        _OWN_LIMITS = own
        print(f"own limits (per-run budget): {_OWN_LIMITS}")

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    seed_dirs = sorted(source_root.glob("seed_*"))
    if not seed_dirs:
        raise SystemExit(f"No seed_*/ under {source_root}")

    for protocol, deterministic in [("deterministic", True), ("stochastic", False)]:
        proto_root = out_root / protocol
        for seed_dir in seed_dirs:
            df = eval_seed_dir(seed_dir, args.eval_episodes, deterministic)
            dest = proto_root / seed_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            df.to_csv(dest / "metrics.csv", index=False)
            print(f"[{protocol}] {seed_dir.name}: {len(df)} methods -> {dest/'metrics.csv'}")
        print(f"--- {protocol} done: {proto_root} "
              f"(run aggregate_stats.py --root {proto_root})")

    import json
    import sys

    from semantic_lb_env import topology as _tp_prov
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "reeval_provenance.json", "w", encoding="utf-8") as f:
        json.dump({
            "eval_seed_effective": _EVAL_SEED,
            "eval_seed_published_default": EVAL_SEED,
            "is_published_eval_workload": _EVAL_SEED == EVAL_SEED,
            "scenario_seed": _SCENARIO_SEED,
            "scenario_seed_published_default": SCENARIO_SEED,
            "is_published_scenario": _SCENARIO_SEED == SCENARIO_SEED,
            "industrial_pattern": _INDUSTRIAL_PATTERN,
            "industrial_pattern_path": _INDUSTRIAL_PATTERN_PATH,
            "topology_variant": _TOPOLOGY_VARIANT,
            "is_published_topology": _TOPOLOGY_VARIANT == 0, "scale_tier": _scale_tier_prov(),

            "trace_replay_declared": {
                "trace_replay": _TRACE_REPLAY,
                "trace_replay_block": _TRACE_REPLAY_BLOCK,
                "trace_replay_block_len": _TRACE_REPLAY_BLOCK_LEN,
                "trace_replay_episodes": _TRACE_REPLAY_EPISODES,
                "trace_replay_source": _TRACE_REPLAY_SOURCE,
            },
            **_TRACE_REPLAY_PROV,
            **_TELEMETRY_PROV,
            "eval_episodes": args.eval_episodes,
            "source_root": str(source_root),
            "n_seed_dirs": len(seed_dirs),
            "argv": sys.argv[1:],
            "note": ("eval_seed_effective != eval_seed_published_default 时，本批产物是"
                     "同一批模型在**另一个评估工作负载实现**上的重打分，不是重复实验；"
                     "不得与默认批混入同一个配对家族。is_published_scenario 为 false 时"
                     "同理，且更强：本批是**零样本迁移**（模型训练于 scenario 42），"
                     "own_* 满足率还须同报'CMDP 上限在 42 上标定'这一混淆项。"
                     "is_published_topology 为 false 时同理（模型训练于历史语义矩阵），"
                     "且它与 scenario_seed 是**两条不同的轴**：前者换语义拓扑、后者换"
                     "实例属性实现，三者两两之间都不得混进同一个配对家族。"),
            "topology_primaries": (None if _TOPOLOGY_VARIANT == 0 else
                                   [list(x) for x in
                                    _tp_prov.primaries_of(_TOPOLOGY_VARIANT)]),
            "topology_pool_sha256": _tp_prov.POOL_SHA256,
        }, f, indent=2, ensure_ascii=False)

def _add_trace_replay_cli(parser) -> None:
    from run_experiment import add_trace_replay_args
    add_trace_replay_args(parser)

def _trace_replay_args_of(args) -> tuple:
    from run_experiment import _trace_replay_args
    return _trace_replay_args(args)

def trace_replay_provenance() -> dict:
    from semantic_lb_env.trace_replay import plan_from_env_config
    p = plan_from_env_config(make_env().unwrapped.config)
    return {} if p is None else p.provenance()

def _trace_replay_gate(args) -> dict:
    if not _TRACE_REPLAY:

        from semantic_lb_env.trace_replay import UNSET
        if (_TRACE_REPLAY_BLOCK, _TRACE_REPLAY_BLOCK_LEN, _TRACE_REPLAY_EPISODES,
                _TRACE_REPLAY_SOURCE) != (UNSET, UNSET, UNSET, None):
            raise SystemExit(
                "给了 --trace-replay-* 却没给 --trace-replay：五个旗标是一整块，"
                "半开半关的批次会让边车里「声明了什么」与「真回放了什么」不是一回事；"
                "要么全给、要么一个都别给。")
        return {}
    if _INDUSTRIAL_PATTERN != "none":

        raise SystemExit(
            "--trace-replay 与 --industrial-pattern %s 同时开：两者都在改 demand 的时间结构，"
            "叠加后没有任何同均值对照臂能把两份效应拆开（轴 6 的禁令 B1 与本轴的同一条）。"
            "要测复合结构必须另冻一族并单独定标。" % (_INDUSTRIAL_PATTERN,))

    if _TRACE_REPLAY_EPISODES < args.eval_episodes:
        raise SystemExit(
            "--trace-replay-episodes %d < --eval-episodes %d：本入口每个策略各造一个评估 "
            "env、各自从块头起消费 %d 个 episode ⇒ 会在最后几个 episode 上越出声明窗口"
            "当场抛（回放越窗不取模、不截断）。把声明量提到 ≥ %d。"
            % (_TRACE_REPLAY_EPISODES, args.eval_episodes, args.eval_episodes,
               args.eval_episodes))
    prov = trace_replay_provenance()
    tr = prov["trace_replay"]
    print(f"[trace-replay] E4 慢层回放：块 {prov['trace_block']}/{tr['n_blocks']}"
          f"（块长 {tr['block_len']} 点、起点偏移 {prov['trace_start_offset']}、尾部 "
          f"{tr['unused_tail_points']} 点不可寻址），声明 {tr['n_episodes']} 个 episode ⇒ "
          f"消费 {tr['n_points_consumed']} 个采样点，策略 {tr['policy']!r}；"
          f"源 {tr['source']} sha256 {tr['source_sha256'][:16]}、"
          f"索引指纹 {prov['trace_index_fingerprint'][:16]}。"
          f"模型训练时**没有**慢层 ⇒ 本批是**零样本**产物，不得与任何配平重训臂、也不得与"
          f"训练侧回放产物混入同一个配对家族。慢层只改 demand 的**时间结构**：恒定水平的"
          f"慢层与 --arrival-scale 在端点上逐位相同（§7 2.3 A）⇒ 「回放 vs 标称」的差里"
          f"混着均值变化，**只能**与 --arrival-scale <本窗口倍率均值> 的同均值对照臂比。")
    return prov

def _add_scale_tier_cli(parser) -> None:
    import run_experiment
    run_experiment.add_scale_tier_args(parser)

def _scale_tier_of(args) -> int:
    from semantic_lb_env import scale_tiers as _st
    from semantic_lb_env import topology as _tp
    spd, info, line = _st.resolve(getattr(args, "services_per_domain", None), context="reeval")
    _tp.assert_scale_compatible(int(getattr(args, "topology_variant", 0) or 0), spd)
    if line:
        print(line, flush=True)
        print("[scale-tier] 本入口的额外一条：非已发表档位上**已发表权重装不进**"
              "（obs/action 维都变了，探针 E 节可装载 0/14）⇒ `--source-root` 必须指向"
              "**本档自己重训**出来的 seed_*/，拿已发表 multiseed 目录来重打分会在第一个"
              "`load()` 上炸，而不是静默按已发表档位跑。", flush=True)
    return spd

def _scale_tier_prov() -> dict:
    from semantic_lb_env import scale_tiers as _st
    spd = int(make_env().unwrapped.config.services_per_domain)
    return _st.provenance(_st.describe(spd))

def _add_telemetry_cli(parser) -> None:
    import run_experiment
    run_experiment.add_telemetry_args(parser)

def _resolve_telemetry(args) -> tuple:
    from scripts._telemetry_helpers import (make_telemetry_config,
                                            reject_missing_predictor,
                                            telemetry_provenance)
    cfg = make_telemetry_config(args)
    reject_missing_predictor(cfg)
    return cfg, telemetry_provenance(cfg)

if __name__ == "__main__":
    main()
