from typing import Optional
import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_MISSING = object()

_ARG_TO_FIELD = {
    "telemetry_predictor_path": "risk_predictor_path",
    "telemetry_beta_strategy": "beta_strategy",
    "telemetry_beta_min": "beta_min",
    "telemetry_beta_max": "beta_max",
    "telemetry_beta_fixed": "beta_fixed",
}

def make_telemetry_config(args: argparse.Namespace) -> Optional:
    if not getattr(args, "telemetry_mask_enabled", False):
        return None

    from semantic_lb_env.env_config import TelemetryMaskConfig
    kwargs = {}
    for arg_name, field_name in _ARG_TO_FIELD.items():
        value = getattr(args, arg_name, _MISSING)
        if value is not _MISSING:
            kwargs[field_name] = value
    return TelemetryMaskConfig(use_telemetry_mask=True, **kwargs)

def resolve_predictor_path(cfg) -> "Path":
    from semantic_lb_env.env_config import DEFAULT_RISK_PREDICTOR_PATH
    declared = getattr(cfg, "risk_predictor_path", None)
    return Path(DEFAULT_RISK_PREDICTOR_PATH) if declared is None else Path(declared)

PROVENANCE_KEYS = (
    "telemetry_predictor_path", "telemetry_predictor_sha256",
    "telemetry_beta_strategy", "telemetry_beta_fixed",
    "telemetry_beta_min", "telemetry_beta_max", "telemetry_beta_epsilon",
    "telemetry_ucb_threshold_sla", "telemetry_ucb_threshold_failure",
    "telemetry_fallback_to_base_mask",
    "telemetry_budget_keys", "telemetry_budget_weights",
    "telemetry_budget_limits", "telemetry_budget_limits_source",
)

def telemetry_provenance(cfg) -> dict:
    if cfg is None:
        return {}

    import hashlib

    path = resolve_predictor_path(cfg)
    sha = None
    if path.is_file():
        sha = hashlib.sha256(path.read_bytes()).hexdigest()

    declared_limits = getattr(cfg, "budget_limits", None)

    limits_source = ("explicit" if declared_limits is not None else
                     "cmdp.ConstraintConfig() 字段默认值经 active_limits() 现取"
                     "（**不随本跑的 --sla-limit/--cost-limit/--failure-limit 覆盖而变**）")

    return {
        "telemetry_predictor_path": str(path),
        "telemetry_predictor_sha256": sha,
        "telemetry_beta_strategy": cfg.beta_strategy,
        "telemetry_beta_fixed": cfg.beta_fixed,
        "telemetry_beta_min": cfg.beta_min,
        "telemetry_beta_max": cfg.beta_max,
        "telemetry_beta_epsilon": cfg.beta_epsilon,
        "telemetry_ucb_threshold_sla": cfg.ucb_threshold_sla,
        "telemetry_ucb_threshold_failure": cfg.ucb_threshold_failure,
        "telemetry_fallback_to_base_mask": cfg.fallback_to_base_mask,
        "telemetry_budget_keys": list(cfg.budget_keys),
        "telemetry_budget_weights": list(cfg.budget_weights),
        "telemetry_budget_limits": None if declared_limits is None else list(declared_limits),
        "telemetry_budget_limits_source": limits_source,
    }

def reject_missing_predictor(cfg) -> None:
    if cfg is None:
        return
    path = resolve_predictor_path(cfg)
    if not path.is_file():
        raise SystemExit(
            f"遥测掩码已启用，但风险预测器文件不存在：{path}\n"
            f"（声明值 risk_predictor_path={getattr(cfg, 'risk_predictor_path', None)!r}）\n"
            f"env 在这种情况下会**静默造一个未训练的网**去筛候选，产物里看不出来 —— "
            f"所以入口这里硬拒。请检查 --telemetry-predictor-path，或先训一个："
            f"scripts/train_risk_predictor.py")

def reject_budget_limit_mismatch(cfg, constraint_config) -> None:
    if cfg is None or getattr(cfg, "budget_limits", None) is not None:
        return
    from semantic_lb_env.cmdp import ConstraintConfig, active_limits

    pristine = active_limits(ConstraintConfig())
    effective = active_limits(constraint_config)
    diffs = {}
    for k in cfg.budget_keys:
        a, b = pristine.get(k, _MISSING), effective.get(k, _MISSING)
        if a is not b and a != b:
            diffs[k] = (None if a is _MISSING else a, None if b is _MISSING else b)
    if diffs:
        raise SystemExit(
            f"遥测预算上限与本跑的 CMDP 生效上限不一致：{diffs}（左=遥测实际用的 "
            f"ConstraintConfig() 默认值，右=本跑生效值，None=该键在那一侧不存在）。\n"
            f"后果：预算 b_k 按一套上限耗尽、约束按另一套执行 ⇒ β 的保守程度与真实"
            f"约束余量脱钩，是个不会报错也读不出的混杂。\n"
            f"请显式给 TelemetryMaskConfig.budget_limits（按 budget_keys "
            f"{list(cfg.budget_keys)} 的顺序），把'用哪套上限'变成登记值。")

