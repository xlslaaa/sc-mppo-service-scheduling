from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

INFO_LEVELS = ("iso_info", "full_info")

_SVC_COLS_ISO = ("queue", "utilization")
_SVC_COLS_EXTRA = ("base_capacity", "base_latency", "base_reliability", "cost")

def service_feature_names(num_domains: int, service_types: int, info_level: str) -> Tuple[str, ...]:
    if info_level not in INFO_LEVELS:
        raise ValueError("info_level 必须是 %s 之一，收到 %r" % (INFO_LEVELS, info_level))
    cols = list(_SVC_COLS_ISO)
    cols += ["domain_oh_%d" % i for i in range(num_domains)]
    cols += ["type_oh_%d" % i for i in range(service_types)]
    if info_level == "full_info":
        cols += list(_SVC_COLS_EXTRA)
    return tuple(cols)

def request_feature_names(num_domains: int, service_types: int) -> Tuple[str, ...]:
    cols = ["req_domain_oh_%d" % i for i in range(num_domains)]
    cols += ["req_type_oh_%d" % i for i in range(service_types)]
    cols += ["sla_norm", "priority", "demand_norm", "t_frac"]
    return tuple(cols)

class SetObsWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        n_max: Optional[int] = None,
        info_level: str = "iso_info",
    ) -> None:
        super().__init__(env)
        if info_level not in INFO_LEVELS:
            raise ValueError("info_level 必须是 %s 之一，收到 %r" % (INFO_LEVELS, info_level))
        self.info_level = info_level

        base = env.unwrapped
        self.n_services = int(base.num_services)
        self.n_max = int(self.n_services if n_max is None else n_max)
        if self.n_max < self.n_services:
            raise ValueError(
                "n_max=%d 小于环境的 num_services=%d ⇒ 会静默丢掉服务"
                % (self.n_max, self.n_services)
            )

        c = base.config
        self.num_domains = int(c.num_domains)
        self.service_types = int(c.service_types)

        self.svc_cols = service_feature_names(self.num_domains, self.service_types, info_level)
        self.req_cols = request_feature_names(self.num_domains, self.service_types)
        f_svc = len(self.svc_cols)
        f_req = len(self.req_cols)

        self._static = np.zeros((self.n_max, f_svc), dtype=np.float32)
        dom_off = len(_SVC_COLS_ISO)
        typ_off = dom_off + self.num_domains
        for i in range(self.n_services):
            self._static[i, dom_off + int(base.service_domain[i])] = 1.0
            self._static[i, typ_off + int(base.service_type[i])] = 1.0
        if info_level == "full_info":
            extra_off = typ_off + self.service_types
            n = self.n_services
            self._static[:n, extra_off + 0] = base.base_capacity

            self._static[:n, extra_off + 1] = np.clip(base.base_latency / 150.0, 0.0, 1.0)
            self._static[:n, extra_off + 2] = base.base_reliability
            self._static[:n, extra_off + 3] = base.cost

        self._slot_valid = np.zeros(self.n_max, dtype=np.float32)
        self._slot_valid[: self.n_services] = 1.0

        self.observation_space = spaces.Dict(
            {
                "services": spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_max, f_svc), dtype=np.float32),
                "request": spaces.Box(low=-np.inf, high=np.inf, shape=(f_req,), dtype=np.float32),
                "slot_valid": spaces.Box(low=0.0, high=1.0, shape=(self.n_max,), dtype=np.float32),
                "base_mask": spaces.Box(low=0.0, high=1.0, shape=(self.n_max,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(self.n_max)

    def _build_obs(self) -> Dict[str, np.ndarray]:
        base = self.env.unwrapped
        n = self.n_services

        svc = self._static.copy()
        svc[:n, 0] = base.queues
        svc[:n, 1] = base.utilization

        req = np.zeros(len(self.req_cols), dtype=np.float32)
        r = base.current_request
        req[int(r["domain"])] = 1.0
        req[self.num_domains + int(r["type"])] = 1.0
        off = self.num_domains + self.service_types

        req[off + 0] = min(float(r["sla"]) / 150.0, 1.0)
        req[off + 1] = float(r["priority"])
        req[off + 2] = min(float(r["demand"]) / 0.5, 1.0)
        req[off + 3] = base.t / max(base.config.episode_len, 1)

        return {
            "services": svc,
            "request": req,
            "slot_valid": self._slot_valid.copy(),
            "base_mask": self._padded_mask(),
        }

    def _padded_mask(self) -> np.ndarray:
        m = np.zeros(self.n_max, dtype=np.float32)
        m[: self.n_services] = self.env.unwrapped.action_masks().astype(np.float32)
        return m

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict]:
        _, info = self.env.reset(**kwargs)
        return self._build_obs(), info

    def step(self, action):
        a = int(action)
        if a >= self.n_services:

            raise IndexError(
                "动作 %d 落在 padding 槽位（N=%d, n_max=%d）⇒ action_masks 未被施加"
                % (a, self.n_services, self.n_max)
            )
        _, reward, terminated, truncated, info = self.env.step(a)
        return self._build_obs(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        m = np.zeros(self.n_max, dtype=bool)
        m[: self.n_services] = self.env.unwrapped.action_masks()
        return m

    def policy_kwargs(self, **overrides) -> Dict:
        kw = dict(svc_cols=self.svc_cols, req_cols=self.req_cols)
        kw.update(overrides)
        return kw
