from __future__ import annotations

from typing import Any, Dict, Optional

import torch as th
from sb3_contrib import MaskablePPO
from torch import nn

from semantic_lb_env.set_obs import SetObsWrapper
from semantic_lb_env.set_policy import PAD_LOGIT, SetPointerPolicy, _masked_mean, resolve_layout

_MAX_FILL = -1e9

class EquivariantLayer(nn.Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lam = nn.Linear(in_dim, out_dim, bias=False)
        self.gam = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: th.Tensor, slot_valid: th.Tensor) -> th.Tensor:
        filled = x.masked_fill(slot_valid.unsqueeze(-1) < 0.5, _MAX_FILL)
        pooled, _ = filled.max(dim=1, keepdim=True)
        return self.lam(x) - self.gam(pooled)

class DeepSetPolicy(SetPointerPolicy):

    def _build(self, lr_schedule) -> None:
        obs = self.observation_space
        f_svc, f_req = int(obs["services"].shape[1]), int(obs["request"].shape[0])

        _, _, self.num_domains, self.service_types = resolve_layout(self.svc_cols, self.req_cols)
        if f_svc != len(self.svc_cols) or f_req != len(self.req_cols):
            raise ValueError("观测列数 (%d, %d) 与 policy_kwargs 的列名数 (%d, %d) 不符 ⇒ 两者不是"
                             "同一个 SetObsWrapper 给的" % (f_svc, f_req, len(self.svc_cols),
                                                         len(self.req_cols)))
        H, f_in = self.hidden_dim, f_svc + f_req
        self.mlp_extractor = None
        self.feasibility_net = None
        self.actor_layers = nn.ModuleList(
            [EquivariantLayer(f_in if i == 0 else H, H) for i in range(self.n_layers)]
            + [EquivariantLayer(H, 1)])
        self.critic_layers = nn.ModuleList(
            [EquivariantLayer(f_in if i == 0 else H, H) for i in range(self.n_layers + 1)])
        self.critic_head = nn.Sequential(nn.Linear(H, H), self.activation_fn(), nn.Linear(H, 1))
        self.act = self.activation_fn()
        if self.ortho_init:
            raise ValueError("DeepSetPolicy 用 PyTorch 默认初始化（同原仓库）⇒ 必须传 ortho_init=False")
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def _stack(self, layers: nn.ModuleList, x: th.Tensor, slot_valid: th.Tensor,
               last_linear: bool) -> th.Tensor:
        for i, layer in enumerate(layers):
            x = layer(x, slot_valid)
            if not (last_linear and i == len(layers) - 1):
                x = self.act(x)
        return x

    def core_outputs(self, obs) -> Dict[str, th.Tensor]:
        services = obs["services"].float()
        request = obs["request"].float()
        slot_valid = obs["slot_valid"].float()
        x = th.cat([services, request.unsqueeze(1).expand(-1, services.shape[1], -1)], dim=-1)
        score = self._stack(self.actor_layers, x, slot_valid, last_linear=True).squeeze(-1)
        action_logits = th.where(slot_valid > 0.5, score, th.full_like(score, PAD_LOGIT))
        h = self._stack(self.critic_layers, x, slot_valid, last_linear=True)
        values = self.critic_head(_masked_mean(h, slot_valid))
        return {"score": score, "action_logits": action_logits, "values": values,
                "service_embeddings": h}

DEEPSET_INFO_LEVEL = "full_info"
DEEPSET_ARM_ENV_KWARGS: Dict[str, Any] = dict(mask=True, semantic_state=True, semantic_reward=True)
DEEPSET_ARM_WORKLOAD_SEED_OFFSET = 9
DEEPSET_ARM_FIXED_HYPERPARAMS: Dict[str, Any] = dict(gamma=0.97, gae_lambda=0.92, verbose=0,
                                                     device="cpu")
DEEPSET_POLICY_KWARGS: Dict[str, Any] = dict(hidden_dim=64, n_layers=2, activation_fn=nn.ELU, ortho_init=False)

def wrap_for_deepset(env) -> SetObsWrapper:
    return SetObsWrapper(env, n_max=None, info_level=DEEPSET_INFO_LEVEL)

def build_deepset_arm(env, *, learning_rate: float, n_steps: int, batch_size: int,
                      seed: int) -> MaskablePPO:
    if not isinstance(env, SetObsWrapper) or env.info_level != DEEPSET_INFO_LEVEL:
        raise TypeError("DeepSet 臂的 env 必须经 wrap_for_deepset 包装（info_level=%r）"
                        % DEEPSET_INFO_LEVEL)
    return MaskablePPO(DeepSetPolicy, env, learning_rate=learning_rate, n_steps=n_steps,
                       batch_size=batch_size, seed=seed,
                       policy_kwargs=env.policy_kwargs(**DEEPSET_POLICY_KWARGS),
                       **DEEPSET_ARM_FIXED_HYPERPARAMS)

def deepset_source_sha256() -> str:
    import hashlib
    import inspect
    parts = [inspect.getsource(obj) for obj in (EquivariantLayer, DeepSetPolicy, wrap_for_deepset,
                                                  build_deepset_arm)]
    parts += [repr(v) for v in (DEEPSET_INFO_LEVEL, DEEPSET_ARM_ENV_KWARGS, DEEPSET_ARM_WORKLOAD_SEED_OFFSET,
                                DEEPSET_ARM_FIXED_HYPERPARAMS, DEEPSET_POLICY_KWARGS, _MAX_FILL)]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
