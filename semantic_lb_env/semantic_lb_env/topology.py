from __future__ import annotations

import hashlib
import itertools
import json
from typing import Dict, List, Sequence, Tuple

import numpy as np

NUM_DOMAINS = 3
SERVICE_TYPES = 6
SERVICES_PER_DOMAIN = 5

HISTORICAL_PRIMARIES: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(sorted({(2 * d) % SERVICE_TYPES,
                  (2 * d + 1) % SERVICE_TYPES,
                  (2 * d + 2) % SERVICE_TYPES}))
    for d in range(NUM_DOMAINS)
)

CANDIDATE_SIZE_MULTISET: Tuple[int, ...] = (6, 7, 7)

QOS_BONUS_COUNT = 9

ROW_DEGREE = 3

def service_type_vector() -> np.ndarray:
    return np.array(
        [(d * 2 + i) % SERVICE_TYPES
         for d in range(NUM_DOMAINS) for i in range(SERVICES_PER_DOMAIN)],
        dtype=np.int64,
    )

def service_domain_vector() -> np.ndarray:
    return np.repeat(np.arange(NUM_DOMAINS), SERVICES_PER_DOMAIN)

def type_counts() -> np.ndarray:
    return np.bincount(service_type_vector(), minlength=SERVICE_TYPES)

def own_types(domain: int) -> Tuple[int, ...]:
    sd, st = service_domain_vector(), service_type_vector()
    return tuple(sorted(set(st[sd == domain].tolist())))

def invariants_of_primaries_ok(primaries: Sequence[Sequence[int]]) -> bool:
    sd, st = service_domain_vector(), service_type_vector()
    m = matrix_of(primaries)
    if not (m.sum(axis=1) == ROW_DEGREE).all():
        return False
    bonus = int(sum(1 for s in range(sd.size) if m[sd[s], st[s]] == 1))
    return bonus == QOS_BONUS_COUNT

def candidate_sizes(primaries: Sequence[int]) -> List[int]:
    counts = type_counts()
    return sorted(4 + int(counts[t]) for t in primaries)

def _build_pool() -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    per_domain = [list(itertools.combinations(own_types(d), ROW_DEGREE))
                  for d in range(NUM_DOMAINS)]
    pool = []
    for combo in itertools.product(*per_domain):
        if all(tuple(candidate_sizes(combo[d])) == CANDIDATE_SIZE_MULTISET
               for d in range(NUM_DOMAINS)):
            pool.append(tuple(tuple(x) for x in combo))
    return tuple(pool)

POOL: Tuple[Tuple[Tuple[int, ...], ...], ...] = _build_pool()
POOL_SIZE = len(POOL)
HISTORICAL_POOL_INDEX = POOL.index(HISTORICAL_PRIMARIES)
HISTORICAL_VARIANT_ID = HISTORICAL_POOL_INDEX + 1

POOL_SHA256 = hashlib.sha256(
    json.dumps([[list(x) for x in c] for c in POOL], separators=(",", ":")).encode("utf-8")
).hexdigest()

def matrix_of(primaries: Sequence[Sequence[int]]) -> np.ndarray:
    m = np.zeros((NUM_DOMAINS, SERVICE_TYPES), dtype=np.int8)
    for d, types in enumerate(primaries):
        for t in types:
            m[d, t] = 1
    return m

def column_degree(primaries: Sequence[Sequence[int]]) -> Tuple[int, ...]:
    return tuple(int(x) for x in matrix_of(primaries).sum(axis=0))

HISTORICAL_COLUMN_DEGREE = column_degree(HISTORICAL_PRIMARIES)

COLUMN_DEGREE_PRESERVING_IDS: Tuple[int, ...] = tuple(
    i + 1 for i, c in enumerate(POOL) if column_degree(c) == HISTORICAL_COLUMN_DEGREE
)

def hamming_from_historical(variant: int) -> int:
    return int((matrix_of(primaries_of(variant)) != matrix_of(HISTORICAL_PRIMARIES)).sum())

def primaries_of(variant: int) -> Tuple[Tuple[int, ...], ...]:
    if variant == 0:
        return HISTORICAL_PRIMARIES
    if not (1 <= variant <= POOL_SIZE):
        raise ValueError(
            f"topology_variant={variant} 越界：合法值是 0（历史）或 1..{POOL_SIZE}"
            f"（三不变量池，历史本身 = id {HISTORICAL_VARIANT_ID}）"
        )
    return POOL[variant - 1]

def build_matrix(variant: int, num_domains: int = NUM_DOMAINS,
                 service_types: int = SERVICE_TYPES, services_per_domain: int = SERVICES_PER_DOMAIN) -> np.ndarray:
    if variant != 0 and (num_domains, service_types) != (NUM_DOMAINS, SERVICE_TYPES):
        raise ValueError(
            f"topology_variant 只为主线规模 {NUM_DOMAINS}×{SERVICE_TYPES} 推导，"
            f"收到 {num_domains}×{service_types}；换规模必须重新推导枚举，不是重新跑。"
        )
    assert_scale_compatible(variant, services_per_domain); return matrix_of(primaries_of(variant))

def invariants(variant: int) -> Dict[str, object]:
    primaries = primaries_of(variant)
    m = matrix_of(primaries)
    sd, st = service_domain_vector(), service_type_vector()
    bonus = int(sum(1 for s in range(sd.size) if m[sd[s], st[s]] == 1))
    return {
        "variant": variant,
        "primaries": [list(x) for x in primaries],
        "row_degree": [int(x) for x in m.sum(axis=1)],
        "qos_bonus_count": bonus,
        "candidate_size_multiset": [candidate_sizes(primaries[d]) for d in range(NUM_DOMAINS)],
        "column_degree": list(column_degree(primaries)),
        "column_degree_preserved": column_degree(primaries) == HISTORICAL_COLUMN_DEGREE,
        "hamming_from_historical": hamming_from_historical(variant),
        "domains_changed": int(sum(1 for d in range(NUM_DOMAINS)
                                   if primaries[d] != HISTORICAL_PRIMARIES[d])),
        "primaries_within_own_types": all(set(primaries[d]) <= set(own_types(d))
                                          for d in range(NUM_DOMAINS)),
    }

def assert_scale_compatible(variant: int, services_per_domain: int) -> None:
    if variant != 0 and int(services_per_domain) != SERVICES_PER_DOMAIN:
        raise ValueError(
            f"topology_variant={variant} 与 services_per_domain={services_per_domain} 不可同开："
            f"本文件的 216 个变体池是按每域 {SERVICES_PER_DOMAIN} 个实例推导的 —— 唯一在筛的那条"
            f"不变量（每域候选集规模 {CANDIDATE_SIZE_MULTISET}）与它依赖的全局类型计数"
            f" {list(type_counts())} 都随 services_per_domain 变，spd≠{SERVICES_PER_DOMAIN} 时"
            f"筛选条件静默失效而 POOL_SHA256 一字不变（失效模式 (106)）。"
            f"要在别的规模上跑多拓扑：先按新规模**重新推导**池并重冻 ⑦B 家族。"
        )
