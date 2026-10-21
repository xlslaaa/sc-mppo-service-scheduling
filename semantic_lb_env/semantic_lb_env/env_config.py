from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_RISK_PREDICTOR_PATH = Path(__file__).resolve().parent.parent / 'predictors' / 'risk_mlp_v4.pth'

@dataclass
class TelemetryMaskConfig:

    use_telemetry_mask: bool = False

    risk_predictor_path: Optional[str] = None

    predictor_device: str = 'cpu'

    beta_strategy: str = 'linear'

    beta_fixed: Optional[float] = None

    beta_min: float = 0.5

    beta_max: float = 2.0

    beta_epsilon: float = 0.1

    ucb_threshold_sla: Optional[float] = None

    ucb_threshold_failure: float = 0.5

    fallback_to_base_mask: bool = True

    budget_keys: tuple = ('sla', 'cost', 'failure')

    budget_weights: tuple = (0.5, 0.3, 0.2)

    budget_limits: Optional[tuple] = None

    def __post_init__(self):
        if self.use_telemetry_mask:

            from semantic_lb_env.beta_scheduler import BetaScheduler
            if self.beta_strategy not in BetaScheduler.STRATEGIES:
                raise ValueError(
                    f"无效的 beta_strategy: {self.beta_strategy}，"
                    f"必须是 {BetaScheduler.STRATEGIES} 之一"
                )

            if self.beta_strategy == 'constant':
                if self.beta_fixed is None:
                    raise ValueError(
                        "beta_strategy='constant' 必须显式给 beta_fixed —— 它决定消融臂的"
                        "平均保守程度，不能走默认值"
                    )
            elif self.beta_fixed is not None:
                raise ValueError(
                    f"beta_fixed={self.beta_fixed} 只在 beta_strategy='constant' 下生效，"
                    f"当前策略是 '{self.beta_strategy}' ⇒ 该值会被静默忽略"
                )

            if self.beta_strategy != 'constant' and self.beta_min >= self.beta_max:
                raise ValueError(
                    f"beta_min ({self.beta_min}) 必须小于 beta_max ({self.beta_max})；"
                    f"要把 β 冻成常数请用 beta_strategy='constant' + beta_fixed"
                )

            if len(self.budget_keys) != len(self.budget_weights):
                raise ValueError(
                    f"budget_keys 长度 ({len(self.budget_keys)}) "
                    f"与 budget_weights 长度 ({len(self.budget_weights)}) 不一致"
                )

            weight_sum = sum(self.budget_weights)
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"budget_weights 之和 ({weight_sum:.6f}) 必须等于 1.0"
                )
