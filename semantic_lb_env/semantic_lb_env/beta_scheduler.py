import numpy as np
from typing import Optional

class BetaScheduler:

    STRATEGIES = ('linear', 'inverse', 'piecewise', 'constant')

    def __init__(
        self,
        strategy: str = 'linear',
        beta_min: float = 0.5,
        beta_max: float = 2.0,
        epsilon: float = 0.1,
        beta_fixed: Optional[float] = None
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"未知策略: {strategy}，必须是 {self.STRATEGIES} 之一")

        if strategy == 'constant':
            if beta_fixed is None:
                raise ValueError(
                    "strategy='constant' 必须显式给 beta_fixed —— 它决定消融臂的平均"
                    "保守程度，不能走默认值")
            if beta_fixed < 0:
                raise ValueError(f"beta_fixed ({beta_fixed}) 必须非负")
        elif beta_min >= beta_max:

            raise ValueError(
                f"beta_min ({beta_min}) 必须小于 beta_max ({beta_max})；"
                f"要把 β 冻成常数请用 strategy='constant' + beta_fixed")

        if beta_min < 0 or beta_max < 0:
            raise ValueError(f"beta_min 和 beta_max 必须非负")

        if epsilon <= 0:
            raise ValueError(f"epsilon ({epsilon}) 必须为正")

        self.strategy = strategy
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.epsilon = epsilon
        self.beta_fixed = beta_fixed

    def __call__(self, budget_remaining: float) -> float:

        b_t = np.clip(budget_remaining, 0.0, 1.0)

        if self.strategy == 'linear':
            return self._linear(b_t)
        elif self.strategy == 'inverse':
            return self._inverse(b_t)
        elif self.strategy == 'piecewise':
            return self._piecewise(b_t)
        elif self.strategy == 'constant':
            return float(self.beta_fixed)

    def _linear(self, b_t: float) -> float:
        return self.beta_max - (self.beta_max - self.beta_min) * b_t

    def _inverse(self, b_t: float) -> float:

        raw = 1.0 / (b_t + self.epsilon)

        raw_min = 1.0 / (1.0 + self.epsilon)
        raw_max = 1.0 / self.epsilon

        normalized = (raw - raw_min) / (raw_max - raw_min)
        beta = self.beta_min + normalized * (self.beta_max - self.beta_min)

        return beta

    def _piecewise(self, b_t: float) -> float:
        if b_t < 0.2:
            return 2.0
        elif b_t < 0.5:
            return 1.0
        else:
            return 0.3

    def get_config(self) -> dict:
        return {
            'strategy': self.strategy,
            'beta_min': self.beta_min,
            'beta_max': self.beta_max,
            'epsilon': self.epsilon,
            'beta_fixed': self.beta_fixed
        }

    @classmethod
    def from_config(cls, config: dict) -> 'BetaScheduler':
        return cls(**config)

def plot_beta_curves(
    strategies: list = ['linear', 'inverse', 'piecewise'],
    beta_min: float = 0.5,
    beta_max: float = 2.0,
    output_path: Optional[str] = None
):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib 未安装，跳过可视化")
        return

    budget_range = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 6))

    for strategy in strategies:
        scheduler = BetaScheduler(
            strategy=strategy,
            beta_min=beta_min,
            beta_max=beta_max
        )
        beta_values = [scheduler(b) for b in budget_range]

        plt.plot(budget_range, beta_values, label=f'{strategy}', linewidth=2)

    plt.xlabel('预算剩余 (budget_remaining)', fontsize=12)
    plt.ylabel('β 权重', fontsize=12)
    plt.title('β 调度策略对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.axhline(y=beta_min, color='gray', linestyle='--', alpha=0.5, label=f'β_min={beta_min}')
    plt.axhline(y=beta_max, color='gray', linestyle='--', alpha=0.5, label=f'β_max={beta_max}')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 图表已保存：{output_path}")
    else:
        plt.show()

if __name__ == '__main__':

    print("=" * 60)
    print("β 调度器演示")
    print("=" * 60)

    strategies = ['linear', 'inverse', 'piecewise']
    test_budgets = [0.0, 0.2, 0.5, 0.8, 1.0]

    for strategy in strategies:
        scheduler = BetaScheduler(strategy=strategy)
        print(f"\n策略: {strategy}")
        print(f"  配置: beta_min={scheduler.beta_min}, beta_max={scheduler.beta_max}")
        print(f"  budget → β:")

        for b in test_budgets:
            beta = scheduler(b)
            print(f"    {b:.1f} → {beta:.3f}")

    print("\n" + "=" * 60)
    print("生成可视化曲线...")
    plot_beta_curves(output_path='beta_scheduler_curves.png')
