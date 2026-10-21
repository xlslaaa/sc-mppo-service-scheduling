from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

CURVE_VALUE_COLUMNS = ("timesteps", "episode_reward", "episode_length")

def _series(curves: pd.DataFrame, method: str) -> pd.DataFrame:
    if "method" not in curves.columns:
        raise KeyError("训练曲线里没有 method 列 ⇒ 判据无法按臂取行（不得静默跳过）")
    sub = curves[curves["method"] == method]
    if sub.empty:
        raise KeyError("训练曲线里没有 method=%r 的行 ⇒ 该臂没跑或臂名写错" % method)
    missing = [c for c in CURVE_VALUE_COLUMNS if c not in sub.columns]
    if missing:
        raise KeyError("训练曲线缺列 %s ⇒ 判据不能退化成只比行数" % missing)
    return sub[list(CURVE_VALUE_COLUMNS)].reset_index(drop=True)

def onebite_verdict(curves: pd.DataFrame, cost_curve: pd.DataFrame,
                    method: str = "p3o", reference: str = "ppo_lagrangian") -> Dict[str, Any]:
    a, b = _series(curves, method), _series(curves, reference)
    same_shape = a.shape == b.shape
    curves_differ = (not same_shape) or (not a.equals(b))
    if cost_curve is None or len(cost_curve) == 0:
        raise KeyError("代价侧读数为空 ⇒ P3O 的 train() 一次都没被调用过")
    if "penalty_mean" not in cost_curve.columns:
        raise KeyError("代价侧读数缺 penalty_mean 列 ⇒ 判据取不到键（不得静默通过）")
    penalty_max = float(cost_curve["penalty_mean"].max())
    return {
        "method": method,
        "reference": reference,
        "curve_rows": int(len(a)),
        "reference_curve_rows": int(len(b)),
        "curves_differ": bool(curves_differ),
        "penalty_mean_max": penalty_max,
        "penalty_ever_positive": bool(penalty_max > 0.0),
        "ok": bool(curves_differ and penalty_max > 0.0),
        "note": ("曲线不同是必要非充分条件：p3o 的 λ 恒为 0、ppo_lagrangian 的 λ 由 "
                 "callback 推离 0，故即使罚项失效两者也会不同。充分侧见 "
                 "tests/test_p3o_additivity.py 的 κ=0 / κ>0 逐位牙。"),
    }

def format_verdict(verdict: Dict[str, Any]) -> str:
    return ("1-bite[%s vs %s] 曲线逐位不同=%s（%d vs %d 行） penalty_mean 最大=%.4f 正值=%s"
            % (verdict["method"], verdict["reference"], verdict["curves_differ"],
               verdict["curve_rows"], verdict["reference_curve_rows"],
               verdict["penalty_mean_max"], verdict["penalty_ever_positive"]))

def failure_reasons(verdict: Dict[str, Any]) -> List[str]:
    out = []
    if not verdict["curves_differ"]:
        out.append("p3o 与 %s 的训练曲线逐位相同 ⇒ 旗标没生效或产物被覆盖"
                   % verdict["reference"])
    if not verdict["penalty_ever_positive"]:
        out.append("penalty_mean 全程为 0 ⇒ κ 或代价通道没生效")
    return out
