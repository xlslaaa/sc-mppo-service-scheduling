from __future__ import annotations

import numpy as np

class RoundRobinPolicy:
    def __init__(self, num_actions: int, respect_mask: bool = True) -> None:
        self.num_actions = num_actions
        self.respect_mask = respect_mask
        self.idx = 0

    def predict(self, obs, deterministic=True, action_masks=None):
        mask = action_masks if action_masks is not None and self.respect_mask else np.ones(self.num_actions, dtype=bool)
        valid = np.where(mask)[0]
        action = int(valid[self.idx % len(valid)])
        self.idx += 1
        return action, None

class LeastLoadedPolicy:
    def __init__(self, num_actions: int, num_services: int, respect_mask: bool = True) -> None:
        self.num_actions = num_actions
        self.num_services = num_services
        self.respect_mask = respect_mask

    def predict(self, obs, deterministic=True, action_masks=None):
        mask = action_masks if action_masks is not None and self.respect_mask else np.ones(self.num_actions, dtype=bool)
        utilization = np.asarray(obs)[self.num_services : self.num_services * 2]
        scores = utilization.copy()
        scores[~mask] = np.inf
        return int(np.argmin(scores)), None

class RandomPolicy:
    def __init__(self, num_actions: int, respect_mask: bool = True, seed: int = 0) -> None:
        self.num_actions = num_actions
        self.respect_mask = respect_mask
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=True, action_masks=None):
        mask = action_masks if action_masks is not None and self.respect_mask else np.ones(self.num_actions, dtype=bool)
        valid = np.where(mask)[0]
        return int(self.rng.choice(valid)), None
