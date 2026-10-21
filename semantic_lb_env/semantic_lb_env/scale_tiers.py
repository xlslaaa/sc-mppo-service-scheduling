from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple

SPD_SEARCH_MAX = 60

LADDER: Tuple[int, ...] = (5, 10, 15)
LADDER_DERIVATION = "solution_spd(SPD_SEARCH_MAX)"

def base_shape() -> Dict[str, int]:
    from .env import EnvConfig
    c = EnvConfig()
    return {"num_domains": int(c.num_domains), "service_types": int(c.service_types),
            "services_per_domain": int(c.services_per_domain)}

def coupling_coverage(num_domains: int, service_types: int, services_per_domain: int) -> Dict:
    from .env import EnvConfig, SemanticLoadBalancingEnv
    env = SemanticLoadBalancingEnv(config=EnvConfig(
        num_domains=num_domains, service_types=service_types,
        services_per_domain=services_per_domain))
    n = int(env.num_services)
    hit = [int(env.semantic_matrix[env.service_domain[i], env.service_type[i]] == 1)
           for i in range(n)]
    k = int(sum(hit))
    rows = env.semantic_matrix.sum(axis=1).astype(int).tolist()
    uniq = sorted({tuple(int(v) for v in row) for row in env.semantic_matrix})
    return {"num_domains": num_domains, "service_types": service_types,
            "services_per_domain": services_per_domain, "n_services": n,
            "coupled_services": k, "coverage": k / n if n else float("nan"),
            "obs_dim": int(env.observation_space.shape[0]),
            "action_n": int(env.action_space.n),
            "semantic_dim": int(num_domains + service_types),
            "row_degrees": rows, "unique_semantic_rows": len(uniq)}

def same_coverage(a: Dict, b: Dict) -> bool:
    return a["coupled_services"] * b["n_services"] == b["coupled_services"] * a["n_services"]

@lru_cache(maxsize=None)
def published() -> Dict:
    return coupling_coverage(**base_shape())

def published_spd() -> int:
    return int(base_shape()["services_per_domain"])

@lru_cache(maxsize=None)
def solution_spd(max_spd: int = SPD_SEARCH_MAX) -> Tuple[int, ...]:
    base, pub = base_shape(), published()
    return tuple(spd for spd in range(1, int(max_spd) + 1)
                 if same_coverage(coupling_coverage(base["num_domains"],
                                                    base["service_types"], spd), pub))

def tier_index(services_per_domain: int) -> Optional[int]:
    spd = int(services_per_domain)
    return LADDER.index(spd) + 1 if spd in LADDER else None

def describe(services_per_domain: int) -> Dict:
    spd = int(services_per_domain)
    base, pub = base_shape(), published()
    info = coupling_coverage(base["num_domains"], base["service_types"], spd)
    info["coverage_preserved"] = same_coverage(info, pub)
    info["published_coverage"] = pub["coverage"]
    info["published_services_per_domain"] = int(pub["services_per_domain"])
    info["is_published_scale"] = spd == int(pub["services_per_domain"])
    info["in_ladder"] = spd in LADDER
    info["tier_index"] = tier_index(spd)
    info["ladder"] = list(LADDER)
    return info

def ladder_shapes() -> Tuple[Dict, ...]:
    return tuple(describe(spd) for spd in LADDER)

def check(services_per_domain: int, allow_coverage_drift: bool = False,
          context: str = "") -> Dict:
    spd = int(services_per_domain)
    where = (" [%s]" % context) if context else ""
    if spd < 1:
        raise ValueError("services_per_domain=%r 非法：至少 1%s" % (services_per_domain, where))
    info = describe(spd)
    if not info["coverage_preserved"] and not allow_coverage_drift:
        raise ValueError(
            "services_per_domain=%d 的语义耦合覆盖面 %d/%d = %.6f ≠ 已发表 %d/%d = %.6f%s ⇒ "
            "这个看着「只改规模」的旋钮同时改掉了被测机制的覆盖面（失效模式 (104)），"
            "跑出来的差里混着「机制盖住的系统变少了」，没有任何对照臂能把两份拆开。"
            "覆盖面守恒的完整解集（穷举 spd ≤ %d）= %r ⇒ 阶梯 %r。"
            "真要看非守恒档位的曲线形状，显式传 allow_coverage_drift=True，"
            "且那种档位**一步都不许训**（S2）。"
            % (spd, info["coupled_services"], info["n_services"], info["coverage"],
               published()["coupled_services"], published()["n_services"],
               published()["coverage"], where, SPD_SEARCH_MAX, list(solution_spd()),
               list(LADDER)))
    return info

def banner(info: Dict, context: str = "") -> str:
    tier = info["tier_index"]
    return (
        "[scale-tier]%s services_per_domain %d -> %d：N = %d、obs %d、action %d，"
        "语义耦合覆盖面 %d/%d = %.6f（已发表 %.6f，守恒=%s），"
        "阶梯 spd %r 第 %s 档。**这不是「只把系统放大」**：① 已发表 zip 在此档装不进"
        "（obs/action 都变了，探针 E 节可装载 0/14）⇒ 本档的每条臂都必须**在本档重训**；"
        "② 三个 CMDP 上限只在 N=%d 上标定过 ⇒ 必须按 `calibrate_limits` 的同一个配方"
        "**逐档重标**（S4），沿用已发表上限的跑不得进裁决；③ 解析成本下界跨档不可比"
        "（S3，档内抽签散布盖过档间差）；④ RNG 消耗量随 N 变 ⇒ 跨档请求流对不上，"
        "配对检验**只在档内**成立，跨档只能读趋势。产物必须独立 out-root，"
        "不得与其它档或已发表档混入同一个配对家族。"
        % ((" [%s]" % context) if context else "", info["published_services_per_domain"],
           info["services_per_domain"], info["n_services"], info["obs_dim"],
           info["action_n"], info["coupled_services"], info["n_services"],
           info["coverage"], info["published_coverage"], info["coverage_preserved"],
           list(LADDER), tier if tier is not None else "（不在阶梯里）",
           published()["n_services"]))

def resolve(services_per_domain: Optional[int], context: str = "") -> Tuple[int, Dict, str]:
    spd = published_spd() if services_per_domain is None else int(services_per_domain)
    info = check(spd, context=context)
    return spd, info, "" if info["is_published_scale"] else banner(info, context)

PROVENANCE_KEYS: Tuple[str, ...] = (
    "services_per_domain", "n_services", "obs_dim", "action_n", "coupled_services",
    "coverage", "coverage_preserved", "published_services_per_domain", "published_coverage",
    "is_published_scale", "in_ladder", "tier_index", "ladder",
)

def provenance(info: Dict) -> Dict:
    return {key: info[key] for key in PROVENANCE_KEYS}
