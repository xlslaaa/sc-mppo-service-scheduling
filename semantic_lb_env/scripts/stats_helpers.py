from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

__all__ = ["WilcoxonResult", "wilcoxon_exact", "holm_correct_family", "exact_two_sided_floor"]

RANK_RULE = "average"
TWO_SIDED_TAIL_RULE = "P(min(W+, W-) <= observed), equality included"
SUPPORTED_ZERO_METHODS = ("wilcox",)
DEFAULT_ENUMERATION_LIMIT = 20
DEFAULT_MC_RESAMPLES = 200_000
DEFAULT_MC_SEED = 20260824

_NULL_CACHE: dict = {}

@dataclass
class WilcoxonResult:

    p_raw: float
    statistic: float
    method_label: str

    n_total: int
    n_eff: int
    n_zero: int
    has_tie: bool
    max_abs_diff: float

    zero_method: str = "wilcox"
    zero_tolerance: float = 0.0
    alternative: str = "two-sided"
    continuity_correction: bool = False
    rank_rule: str = RANK_RULE
    two_sided_tail_rule: str = TWO_SIDED_TAIL_RULE
    mid_p: bool = False

    enumeration_count: Optional[int] = None
    mc_resamples: Optional[int] = None
    mc_seed: Optional[int] = None
    mc_standard_error: Optional[float] = None

    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

def _null_signs(n: int) -> np.ndarray:
    if n not in _NULL_CACHE:
        _NULL_CACHE[n] = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
    return _NULL_CACHE[n]

def exact_two_sided_floor(n_eff: int) -> float:
    if n_eff <= 0:
        return 1.0
    return min(1.0, 2.0 / float(2 ** n_eff))

def _p_from_null(null_wplus: np.ndarray, w_obs: float, total: float,
                 alternative: str) -> Tuple[float, np.ndarray]:
    if alternative == "two-sided":
        t_obs = min(w_obs, total - w_obs)
        hit = np.minimum(null_wplus, total - null_wplus) <= t_obs + 1e-12
    elif alternative == "greater":
        hit = null_wplus >= w_obs - 1e-12
    elif alternative == "less":
        hit = null_wplus <= w_obs + 1e-12
    else:
        raise ValueError("alternative 必须是 two-sided / greater / less，收到 %r" % (alternative,))
    return float(np.mean(hit)), hit

def _mc_p(ranks: np.ndarray, w_obs: float, total: float, alternative: str,
          resamples: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = ranks.size
    p_hits = 0
    block = max(1, min(resamples, 20_000))
    done = 0
    while done < resamples:
        b = min(block, resamples - done)
        signs = rng.integers(0, 2, size=(b, n)).astype(float)
        _, hit = _p_from_null(signs @ ranks, w_obs, total, alternative)
        p_hits += int(hit.sum())
        done += b
    p = (p_hits + 1.0) / (resamples + 1.0)
    se = float(np.sqrt(max(p * (1.0 - p), 0.0) / resamples))
    return p, se

def wilcoxon_exact(
    x: Sequence[float],
    y: Optional[Sequence[float]] = None,
    *,
    alternative: str = "two-sided",
    zero_method: str = "wilcox",
    zero_tolerance: float = 0.0,
    enumeration_limit: int = DEFAULT_ENUMERATION_LIMIT,
    mc_resamples: int = DEFAULT_MC_RESAMPLES,
    mc_seed: int = DEFAULT_MC_SEED,
) -> WilcoxonResult:
    if zero_method not in SUPPORTED_ZERO_METHODS:
        raise NotImplementedError(
            "zero_method=%r 未实现。本模块只实现 %r；pratt / zsplit 需要各自的秩与零分布，"
            "未逐一验证前不提供，也绝不静默退回其他口径。"
            % (zero_method, SUPPORTED_ZERO_METHODS)
        )
    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance 不能为负，收到 %r" % (zero_tolerance,))

    d = np.asarray(x, dtype=float)
    if y is not None:
        d = d - np.asarray(y, dtype=float)
    if d.ndim != 1:
        raise ValueError("只接受一维配对差值，收到 shape=%r" % (d.shape,))
    if np.isnan(d).any():
        raise ValueError("差向量含 NaN，调用方须先决定缺失值口径（本模块不猜）")

    n_total = int(d.size)
    max_abs = float(np.abs(d).max()) if n_total else 0.0
    is_zero = np.abs(d) <= zero_tolerance
    n_zero = int(is_zero.sum())
    dz = d[~is_zero]
    n_eff = int(dz.size)
    notes: List[str] = []

    base = dict(n_total=n_total, n_eff=n_eff, n_zero=n_zero, max_abs_diff=max_abs,
                zero_method=zero_method, zero_tolerance=zero_tolerance,
                alternative=alternative)

    if n_eff == 0:
        exact_zero = bool(n_total == 0 or max_abs == 0.0)
        label = "structural_zero" if exact_zero else "near_zero_short_circuit"
        if not exact_zero:
            notes.append(
                "全部差值落在 zero_tolerance=%g 内但并非逐位为零（max|d|=%.3e）；"
                "这是容差归零，不是结构性恒等，报告时须分开陈述。" % (zero_tolerance, max_abs))
        return WilcoxonResult(p_raw=1.0, statistic=0.0, method_label=label,
                              has_tie=False, warnings=notes, **base)

    abs_dz = np.abs(dz)
    ranks = rankdata(abs_dz)
    has_tie = bool(np.unique(abs_dz).size < n_eff)
    total = float(ranks.sum())
    w_obs = float(ranks[dz > 0.0].sum())

    if n_zero:
        notes.append(
            "%d/%d 个差值为零，按 zero_method='wilcox' 去零后 n_eff=%d；"
            "双侧精确 p 的结构下界为 %.4f。"
            % (n_zero, n_total, n_eff, exact_two_sided_floor(n_eff)))

    if n_eff <= enumeration_limit:
        null = _null_signs(n_eff) @ ranks
        p_raw, _ = _p_from_null(null, w_obs, total, alternative)
        label = "exact_tie_aware" if has_tie else "exact_no_tie"
        return WilcoxonResult(p_raw=p_raw, statistic=w_obs, method_label=label,
                              has_tie=has_tie, enumeration_count=int(2 ** n_eff),
                              warnings=notes, **base)

    p_raw, se = _mc_p(ranks, w_obs, total, alternative, mc_resamples, mc_seed)
    notes.append(
        "n_eff=%d 超过 enumeration_limit=%d，改用固定种子置换 Monte Carlo："
        "B=%d、seed=%d、p=(k+1)/(B+1)、MC 标准误 %.2e。"
        % (n_eff, enumeration_limit, mc_resamples, mc_seed, se))
    return WilcoxonResult(p_raw=p_raw, statistic=w_obs,
                          method_label="permutation_monte_carlo", has_tie=has_tie,
                          mc_resamples=mc_resamples, mc_seed=mc_seed,
                          mc_standard_error=se, warnings=notes, **base)

def holm_correct_family(p_values: Iterable[float], alpha: float = 0.05
                        ) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return p.copy(), np.zeros(0, dtype=bool)
    if np.isnan(p).any():
        raise ValueError("Holm 家族里有 NaN p 值；structural_zero 应显式记为 1.0")
    order = np.argsort(p, kind="stable")
    m = p.size
    adj_sorted = np.maximum.accumulate(p[order] * (m - np.arange(m)))
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    p_holm = np.empty_like(adj_sorted)
    p_holm[order] = adj_sorted
    return p_holm, p_holm <= alpha