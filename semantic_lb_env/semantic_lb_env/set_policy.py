import inspect
import re
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn

from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from semantic_lb_env.set_obs import (
    INFO_LEVELS,
    request_feature_names,
    service_feature_names,
)

PAD_LOGIT = -1e8

RELATIONS = ("same_domain", "same_type", "cross_domain")

def resolve_layout(
    svc_cols: Tuple[str, ...], req_cols: Tuple[str, ...]
) -> Tuple[slice, slice, int, int]:
    dom = [i for i, c in enumerate(svc_cols) if c.startswith("domain_oh_")]
    typ = [i for i, c in enumerate(svc_cols) if c.startswith("type_oh_")]
    if not dom or not typ:
        raise ValueError("svc_cols 里找不到 domain_oh_* / type_oh_* 列：%r" % (svc_cols,))
    if dom != list(range(dom[0], dom[-1] + 1)) or typ != list(range(typ[0], typ[-1] + 1)):
        raise ValueError("svc_cols 的 domain/type one-hot 列不连续，本模块的 slice 假设不成立")
    num_domains, service_types = len(dom), len(typ)

    req_dom = [i for i, c in enumerate(req_cols) if c.startswith("req_domain_oh_")]
    req_typ = [i for i, c in enumerate(req_cols) if c.startswith("req_type_oh_")]
    if req_dom != list(range(num_domains)):
        raise ValueError(
            "request 的域 one-hot 必须是最前面 %d 列，实测在 %r ⇒ SetEncoder 的派生特征会错位"
            % (num_domains, req_dom)
        )
    if req_typ != list(range(num_domains, num_domains + service_types)):
        raise ValueError(
            "request 的类型 one-hot 必须紧随其后占 %d 列，实测在 %r ⇒ SetEncoder 的派生特征会错位"
            % (service_types, req_typ)
        )
    return slice(dom[0], dom[-1] + 1), slice(typ[0], typ[-1] + 1), num_domains, service_types

def _masked_mean(x: th.Tensor, mask: th.Tensor) -> th.Tensor:
    m = mask.unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

class _DictPassThroughExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Space) -> None:
        super().__init__(observation_space, features_dim=1)

    def forward(self, observations):
        raise RuntimeError(
            "SetPointerPolicy 不走 features_extractor 这条路；特征提取在 _core/SetEncoder 里。"
            "看到这个异常说明有一条父类方法没被覆盖。"
        )

class SetEncoder(nn.Module):

    def __init__(
        self,
        f_svc: int,
        f_req: int,
        dom_slice: slice,
        typ_slice: slice,
        hidden_dim: int = 64,
        n_layers: int = 2,
        activation_fn: Type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.dom_slice = dom_slice
        self.typ_slice = typ_slice
        H = self.hidden_dim
        act = activation_fn

        self.svc_in = nn.Sequential(nn.Linear(f_svc + 2, H), act(), nn.Linear(H, H))
        self.req_in = nn.Sequential(nn.Linear(f_req, H), act(), nn.Linear(H, H))

        self.film = nn.Linear(H, 2 * H)

        self.attn_k = nn.Linear(H, H)
        self.attn_v = nn.Linear(H, H)

        self.rel_w = nn.ModuleList(
            [nn.ModuleList([nn.Linear(H, H, bias=False) for _ in RELATIONS]) for _ in range(self.n_layers)]
        )
        self.self_w = nn.ModuleList([nn.Linear(H, H) for _ in range(self.n_layers)])
        self.ctx_w = nn.ModuleList([nn.Linear(H, H, bias=False) for _ in range(self.n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.n_layers)])
        self.act = act()

    def _adjacency(self, services: th.Tensor, slot_valid: th.Tensor) -> List[th.Tensor]:
        dom = services[:, :, self.dom_slice]
        typ = services[:, :, self.typ_slice]
        same_dom = th.bmm(dom, dom.transpose(1, 2))
        same_typ = th.bmm(typ, typ.transpose(1, 2))

        n = services.shape[1]
        eye = th.eye(n, device=services.device, dtype=services.dtype).unsqueeze(0)
        pair_valid = slot_valid.unsqueeze(1) * slot_valid.unsqueeze(2) * (1.0 - eye)

        out = {
            "same_domain": same_dom * pair_valid,
            "same_type": same_typ * pair_valid,
            "cross_domain": (1.0 - same_dom) * pair_valid,
        }
        return [out[r] for r in RELATIONS]

    def forward(
        self, services: th.Tensor, request: th.Tensor, slot_valid: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        dom = services[:, :, self.dom_slice]
        typ = services[:, :, self.typ_slice]

        req_dom = request[:, : dom.shape[-1]].unsqueeze(1)
        req_typ = request[:, dom.shape[-1] : dom.shape[-1] + typ.shape[-1]].unsqueeze(1)
        same_dom_req = (dom * req_dom).sum(-1, keepdim=True)
        same_typ_req = (typ * req_typ).sum(-1, keepdim=True)

        x = th.cat([services, same_dom_req, same_typ_req], dim=-1)
        h = self.svc_in(x)
        q = self.req_in(request)

        gamma, beta = self.film(q).chunk(2, dim=-1)
        h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        h = h * slot_valid.unsqueeze(-1)

        scale = float(self.hidden_dim) ** 0.5
        logits = (self.attn_k(h) * q.unsqueeze(1)).sum(-1) / scale
        logits = logits.masked_fill(slot_valid < 0.5, PAD_LOGIT)
        w = th.softmax(logits, dim=1).unsqueeze(-1)
        ctx = (w * self.attn_v(h)).sum(dim=1)

        adj = self._adjacency(services, slot_valid)
        for layer in range(self.n_layers):
            msg = self.self_w[layer](h) + self.ctx_w[layer](ctx).unsqueeze(1)
            for r, a in enumerate(adj):
                agg = th.bmm(a, h) / a.sum(dim=-1, keepdim=True).clamp(min=1.0)
                msg = msg + self.rel_w[layer][r](agg)
            h = self.norms[layer](h + self.act(msg))
            h = h * slot_valid.unsqueeze(-1)

        return h, ctx, q

class SetPointerPolicy(MaskableActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        svc_cols: Optional[Tuple[str, ...]] = None,
        req_cols: Optional[Tuple[str, ...]] = None,
        hidden_dim: int = 64,
        n_layers: int = 2,
        use_feasibility_head: bool = False,
        beta_feasibility: float = 0.0,
        net_arch: Optional[Any] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        **kwargs: Any,
    ) -> None:
        if svc_cols is None or req_cols is None:
            raise ValueError(
                "svc_cols / req_cols 必须给，且必须来自 SetObsWrapper.policy_kwargs() —— "
                "它们是列序的唯一来源，手写就又多一个定义点"
            )
        if use_feasibility_head and beta_feasibility == 0.0:
            raise ValueError(
                "use_feasibility_head=True 但 beta_feasibility=0.0 ⇒ 可行性头拿不到任何梯度，"
                "产物里却会多出它的权重键，看起来像'联合训练过'。要关就 use_feasibility_head=False。"
            )
        if (not use_feasibility_head) and beta_feasibility != 0.0:
            raise ValueError(
                "beta_feasibility=%r 只在 use_feasibility_head=True 下生效，否则会被静默忽略"
                % (beta_feasibility,)
            )
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("SetPointerPolicy 只接受 SetObsWrapper 的 Dict 观测空间")

        self.svc_cols = tuple(svc_cols)
        self.req_cols = tuple(req_cols)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.use_feasibility_head = bool(use_feasibility_head)
        self.beta_feasibility = float(beta_feasibility)

        kwargs.setdefault("features_extractor_class", _DictPassThroughExtractor)
        kwargs.setdefault("normalize_images", False)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        raise RuntimeError(
            "SetPointerPolicy 不使用 MlpExtractor（它的输入维随观测展平而变，等于把 N 焊进参数）。"
            "网络在 _build 里直接构造。"
        )

    def _build(self, lr_schedule: Schedule) -> None:
        obs = self.observation_space
        f_svc = int(obs["services"].shape[1])
        f_req = int(obs["request"].shape[0])
        dom_slice, typ_slice, self.num_domains, self.service_types = resolve_layout(
            self.svc_cols, self.req_cols
        )

        if f_svc != len(self.svc_cols):
            raise ValueError(
                "观测 services 有 %d 列，svc_cols 声称 %d 列 ⇒ policy_kwargs 与 wrapper "
                "不是同一次构造出来的（多半是 info_level 改了一边）" % (f_svc, len(self.svc_cols))
            )
        if f_req != len(self.req_cols):
            raise ValueError(
                "观测 request 有 %d 列，req_cols 声称 %d 列 ⇒ policy_kwargs 与 wrapper 不一致"
                % (f_req, len(self.req_cols))
            )

        H = self.hidden_dim
        self.mlp_extractor = None
        self.encoder = SetEncoder(
            f_svc=f_svc,
            f_req=f_req,
            dom_slice=dom_slice,
            typ_slice=typ_slice,
            hidden_dim=H,
            n_layers=self.n_layers,
            activation_fn=self.activation_fn,
        )

        self.pointer_q = nn.Linear(H, H, bias=False)
        self.pointer_c = nn.Linear(H, H, bias=False)
        self.pointer_h = nn.Linear(H, H)
        self.pointer_out = nn.Linear(H, 1)

        self.value_net = nn.Sequential(nn.Linear(3 * H, H), self.activation_fn(), nn.Linear(H, 1))
        if self.use_feasibility_head:
            self.feasibility_net = nn.Sequential(
                nn.Linear(2 * H, H), self.activation_fn(), nn.Linear(H, 1)
            )
        else:
            self.feasibility_net = None

        if self.ortho_init:
            self.encoder.apply(lambda m: self.init_weights(m, np.sqrt(2)))
            for mod in (self.pointer_q, self.pointer_c, self.pointer_h):
                self.init_weights(mod, np.sqrt(2))
            self.init_weights(self.pointer_out, 0.01)
            self.value_net.apply(lambda m: self.init_weights(m, 1.0))
            if self.feasibility_net is not None:
                self.feasibility_net.apply(lambda m: self.init_weights(m, 0.01))

        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                svc_cols=self.svc_cols,
                req_cols=self.req_cols,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                use_feasibility_head=self.use_feasibility_head,
                beta_feasibility=self.beta_feasibility,
            )
        )
        return data

    def core_outputs(self, obs: PyTorchObs) -> Dict[str, th.Tensor]:
        services = obs["services"].float()
        request = obs["request"].float()
        slot_valid = obs["slot_valid"].float()

        h, ctx, q = self.encoder(services, request, slot_valid)

        z = self.pointer_h(h) + self.pointer_q(q).unsqueeze(1) + self.pointer_c(ctx).unsqueeze(1)
        score = self.pointer_out(th.tanh(z)).squeeze(-1)

        out: Dict[str, th.Tensor] = {"score": score}
        if self.feasibility_net is not None:
            feas = self.feasibility_net(
                th.cat([h, ctx.unsqueeze(1).expand_as(h)], dim=-1)
            ).squeeze(-1)
            out["feasibility_logits"] = feas

            score = score + self.beta_feasibility * nn.functional.logsigmoid(feas)

        action_logits = th.where(
            slot_valid > 0.5, score, th.full_like(score, PAD_LOGIT)
        )

        pooled = _masked_mean(h, slot_valid)
        values = self.value_net(th.cat([pooled, ctx, q], dim=-1))

        out["action_logits"] = action_logits
        out["values"] = values
        out["service_embeddings"] = h
        return out

    def _distribution(
        self, action_logits: th.Tensor, action_masks: Optional[np.ndarray]
    ) -> MaskableDistribution:
        dist = self.action_dist.proba_distribution(action_logits=action_logits)
        if action_masks is not None:
            dist.apply_masking(action_masks)
        return dist

    def forward(
        self,
        obs: PyTorchObs,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        out = self.core_outputs(obs)
        dist = self._distribution(out["action_logits"], action_masks)
        actions = dist.get_actions(deterministic=deterministic)
        return actions, out["values"], dist.log_prob(actions)

    def evaluate_actions(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        out = self.core_outputs(obs)

        self._last_feasibility_logits = out.get("feasibility_logits", None)
        dist = self._distribution(out["action_logits"], action_masks)
        return out["values"], dist.log_prob(actions), dist.entropy()

    def get_distribution(
        self, obs: PyTorchObs, action_masks: Optional[np.ndarray] = None
    ) -> MaskableDistribution:
        return self._distribution(self.core_outputs(obs)["action_logits"], action_masks)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        return self.core_outputs(obs)["values"]

    def extract_features(self, obs, features_extractor=None):
        raise RuntimeError(
            "SetPointerPolicy 不走 extract_features；所有前向路径经 core_outputs。"
        )

def sb3_huge_neg_from_source() -> float:
    from sb3_contrib.common.maskable.distributions import MaskableCategorical

    src = inspect.getsource(MaskableCategorical.apply_masking)
    m = re.search(r"HUGE_NEG\s*=\s*th\.tensor\(\s*(-?[\d.eE+-]+)", src)
    if not m:
        raise RuntimeError(
            "在 MaskableCategorical.apply_masking 里找不到 HUGE_NEG 的字面量 ⇒ "
            "sb3_contrib 的实现变了，PAD_LOGIT 这个副本的前提失效"
        )
    return float(m.group(1))
