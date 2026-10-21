from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np

PRIORITY_TIER_HIGH_MIN = 0.85
PRIORITY_TIER_MID_MIN = 0.55

PRIORITY_TIERS = ("high", "mid", "low")

def is_high_priority(priority: Union[float, int]) -> bool:
    return bool(float(priority) >= PRIORITY_TIER_HIGH_MIN)

def tier_masks(priorities: Sequence[float]) -> Dict[str, np.ndarray]:
    prio = np.asarray(priorities, dtype=float)
    return {
        "high": prio >= PRIORITY_TIER_HIGH_MIN,
        "mid": (prio >= PRIORITY_TIER_MID_MIN) & (prio < PRIORITY_TIER_HIGH_MIN),
        "low": prio < PRIORITY_TIER_MID_MIN,
    }

E_PRIORITY = 0.625
SLA_COST_WEIGHT_OFFSET = 0.5
SLA_COST_WEIGHT_DENOM = SLA_COST_WEIGHT_OFFSET + E_PRIORITY

def sla_cost_weight(priority: Union[float, int]) -> float:
    return (SLA_COST_WEIGHT_OFFSET + float(priority)) / SLA_COST_WEIGHT_DENOM
