from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

SCHEMA = "trace_temporal_profile/v1"

DEFAULT_SOURCE = "semantic_lb_env/traces/trace_temporal_profile.json"

POLICY = "hold"
FINGERPRINT_PAYLOAD_VERSION = "trace_replay_fingerprint/v1"
ROOT = Path(__file__).resolve().parents[2]

class ReplayContractError(ValueError):
    pass

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ReplayContractError(msg)

def _resolve(source: str) -> Path:
    p = Path(source)
    return p if p.is_absolute() else ROOT / p

def load_series(source: str = DEFAULT_SOURCE) -> Dict[str, object]:
    path = _resolve(source)
    _require(path.exists(), "回放源不在盘上：%s（慢层回放源是 %s，不是已发表的 "
                            "traces/trace_profile.json）" % (path, DEFAULT_SOURCE))
    doc = json.loads(path.read_text(encoding="utf-8"))
    schema = doc.get("schema")
    _require(schema == SCHEMA, "回放源 schema 是 %r，要求 %r ⇒ 已发表的 29 分钟窗口 profile "
                               "（30 点、被 *RPC_RT 污染 40.8%%）不许当回放源" % (schema, SCHEMA))
    t = doc.get("temporal")
    _require(isinstance(t, dict), "回放源缺 temporal 段：%s" % path)
    load_mult = t.get("load_mult")
    stamps = t.get("timestamps_ms")
    n = t.get("n_timestamps")
    _require(isinstance(load_mult, list) and isinstance(stamps, list),
             "temporal.load_mult / timestamps_ms 必须是数组")
    _require(len(load_mult) == n == len(stamps),
             "点数不自洽：len(load_mult)=%d, n_timestamps=%r, len(timestamps_ms)=%d ⇒ "
             "曲线被截过或字段没同步" % (len(load_mult), n, len(stamps)))
    _require(n >= 2, "回放源只有 %r 个采样点 ⇒ 谈不上时间结构" % (n,))
    res = t.get("resolution_ms")
    diffs = {stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)}
    _require(diffs == {res}, "时间戳间距不恒为 resolution_ms=%r（实测 %s）⇒ 「按序回放」失去"
                             "意义，第 k 与 k+1 点之间不是一个分辨率" % (res, sorted(diffs)[:5]))
    bad = [i for i, v in enumerate(load_mult)
           if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0.0]
    _require(bad == [], "load_mult 有非正 / 非有限值，下标 %s ⇒ 负载倍率不可用" % bad[:5])
    eps, episode_ms = _episode_ratio(doc, int(res))
    return {
        "path": path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "resolution_ms": int(res),
        "n_timestamps": int(n),
        "load_mult": [float(v) for v in load_mult],
        "episodes_per_point": eps,
        "env_episode_ms": episode_ms,
    }

def _episode_ratio(doc: Dict[str, object], resolution_ms: int) -> Tuple[int, int]:
    d = doc.get("resolution_disclosure")
    _require(isinstance(d, dict), "回放源缺 resolution_disclosure ⇒ 分辨率对不对得上无从判断")
    for key in ("env_episode_ms", "trace_resolution_ms", "episodes_per_trace_sample"):
        _require(key in d, "resolution_disclosure 缺 %s" % key)
    _require(int(d["trace_resolution_ms"]) == resolution_ms,
             "披露里的 trace_resolution_ms=%r ≠ temporal.resolution_ms=%r"
             % (d["trace_resolution_ms"], resolution_ms))
    ratio = float(d["episodes_per_trace_sample"])
    _require(ratio == int(ratio) and int(ratio) >= 1,
             "episodes_per_trace_sample=%r 不是 ≥1 的整数 ⇒ 策略 %r 装不下整数个 episode，"
             "必须先决定并披露如何取整" % (ratio, POLICY))
    eps = int(ratio)
    episode_ms = int(d["env_episode_ms"])
    _require(resolution_ms == eps * episode_ms,
             "resolution_ms=%d ≠ episodes_per_point=%d × env_episode_ms=%d ⇒ episode 长度变了"
             "或曲线被重采样过，必须重新披露时间压缩" % (resolution_ms, eps, episode_ms))
    return eps, episode_ms

class ReplayPlan(object):

    __slots__ = ("source", "source_sha256", "resolution_ms", "n_timestamps", "episodes_per_point",
                 "env_episode_ms", "block_len", "n_blocks", "block", "start_offset", "n_episodes",
                 "n_points", "unused_tail_points", "indices", "values", "fingerprint")

    def __init__(self, **kw):
        for k in self.__slots__:
            object.__setattr__(self, k, kw.pop(k))
        assert not kw, kw

    def __setattr__(self, *a):
        raise AttributeError("ReplayPlan 不可变")

    def __repr__(self):
        return ("ReplayPlan(block=%d, start_offset=%d, n_episodes=%d, n_points=%d, fp=%s…)"
                % (self.block, self.start_offset, self.n_episodes, self.n_points,
                   self.fingerprint[:12]))

    def episode_load_mult(self) -> Tuple[float, ...]:
        return tuple(self.values[i - self.start_offset] for i in self.indices)

    def provenance(self) -> Dict[str, object]:
        return {
            "trace_index_fingerprint": self.fingerprint,
            "trace_block": self.block,
            "trace_start_offset": self.start_offset,
            "trace_replay": {
                "source": self.source,
                "source_sha256": self.source_sha256,
                "policy": POLICY,
                "resolution_ms": self.resolution_ms,
                "n_timestamps": self.n_timestamps,
                "episodes_per_point": self.episodes_per_point,
                "env_episode_ms": self.env_episode_ms,
                "block_len": self.block_len,

                "n_blocks": self.n_blocks,
                "unused_tail_points": self.unused_tail_points,
                "n_episodes": self.n_episodes,
                "n_points_consumed": self.n_points,
                "index_span": [self.indices[0], self.indices[-1]],
                "fingerprint_payload_version": FINGERPRINT_PAYLOAD_VERSION,
            },
        }

def _fingerprint(payload: Dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def plan(block: int, n_episodes: int, block_len: int,
         source: str = DEFAULT_SOURCE, start_offset: Optional[int] = None) -> ReplayPlan:
    s = load_series(source)
    n_ts, eps = s["n_timestamps"], s["episodes_per_point"]
    _require(isinstance(block_len, int) and block_len >= 1, "block_len=%r 必须是正整数" % (block_len,))
    _require(block_len <= n_ts, "block_len=%d > 序列点数 %d" % (block_len, n_ts))
    n_blocks = n_ts // block_len
    _require(isinstance(block, int) and 0 <= block < n_blocks,
             "block=%r 越界：block_len=%d 下只有 %d 个**完整**块（合法 0…%d）⇒ 尾部 %d 个点"
             "不可寻址，这是有意的（半块回放会让不同臂的曲线长度不同）"
             % (block, block_len, n_blocks, n_blocks - 1, n_ts - n_blocks * block_len))
    _require(isinstance(n_episodes, int) and n_episodes >= 1, "n_episodes=%r 必须是正整数" % (n_episodes,))
    off = block * block_len
    if start_offset is not None:
        _require(start_offset == off,
                 "start_offset=%r ≠ block*block_len=%d ⇒ 起点必须落在块边界上（块是最小单位）"
                 % (start_offset, off))
    n_points = -(-n_episodes // eps)
    _require(n_points <= block_len,
             "%d 个 episode 在「%s」策略下要 %d 个采样点，超过块长 %d ⇒ **不做取模、不截断**："
             "静默绕回会把「12 h 真实曲线」变成某一段被重复播放，而产物里看不出来"
             % (n_episodes, POLICY, n_points, block_len))

    assert off + n_points <= n_ts, (off, n_points, n_ts, block_len)
    values = tuple(s["load_mult"][off:off + n_points])
    indices = tuple(off + (e // eps) for e in range(n_episodes))
    payload = {
        "version": FINGERPRINT_PAYLOAD_VERSION,
        "source_sha256": s["sha256"],
        "resolution_ms": s["resolution_ms"],
        "policy": POLICY,
        "episodes_per_point": eps,
        "block_len": block_len,
        "block": block,
        "start_offset": off,
        "n_episodes": n_episodes,
        "n_points": n_points,
        "indices": list(indices),
        "values": list(values),
    }
    return ReplayPlan(
        source=source, source_sha256=s["sha256"], resolution_ms=s["resolution_ms"],
        n_timestamps=n_ts, episodes_per_point=eps, env_episode_ms=s["env_episode_ms"],
        block_len=block_len, n_blocks=n_blocks, block=block, start_offset=off,
        n_episodes=n_episodes, n_points=n_points,
        unused_tail_points=n_ts - n_blocks * block_len, indices=indices,
        values=values, fingerprint=_fingerprint(payload))

PROVENANCE_KEYS = ("trace_index_fingerprint", "trace_block", "trace_start_offset", "trace_replay")

def check_provenance(obj: Dict[str, object]) -> bool:
    present = [k for k in PROVENANCE_KEYS if k in obj]
    if not present:
        return False
    _require(len(present) == len(PROVENANCE_KEYS),
             "溯源块只写了一半：有 %s，缺 %s ⇒ 「关」必须是一个键都不写，"
             "半套键分不出「没回放」与「回放了但没记全」" % (present,
             [k for k in PROVENANCE_KEYS if k not in obj]))
    d = obj["trace_replay"]
    _require(isinstance(d, dict), "trace_replay 必须是对象")
    block, off, block_len = obj["trace_block"], obj["trace_start_offset"], d["block_len"]
    _require(off == block * block_len,
             "trace_start_offset=%r ≠ trace_block=%r × block_len=%r ⇒ 这两个字段里至少一个是"
             "填出来的，不是算出来的" % (off, block, block_len))
    n_points = -(-int(d["n_episodes"]) // int(d["episodes_per_point"]))
    _require(int(d["n_points_consumed"]) == n_points,
             "n_points_consumed=%r 与 ceil(n_episodes/episodes_per_point)=%d 不符"
             % (d["n_points_consumed"], n_points))
    _require(list(d["index_span"]) == [off, off + n_points - 1],
             "index_span=%r 与 [start_offset, start_offset+n_points-1]=%r 不符"
             % (d["index_span"], [off, off + n_points - 1]))
    _require(int(d["resolution_ms"]) == int(d["episodes_per_point"]) * int(d["env_episode_ms"]),
             "产物里 resolution_ms=%r ≠ episodes_per_point=%r × env_episode_ms=%r ⇒ 这一跑的"
             "时间压缩没被披露（策略 %r 要求恰好装下整数个 episode）"
             % (d["resolution_ms"], d["episodes_per_point"], d["env_episode_ms"], POLICY))
    _require(len(str(obj["trace_index_fingerprint"])) == 64,
             "trace_index_fingerprint 不是 64 位 sha256：%r" % (obj["trace_index_fingerprint"],))
    return True

CONFIG_FIELDS = ("trace_replay", "trace_replay_block", "trace_replay_block_len",
                 "trace_replay_episodes", "trace_replay_source")

UNSET = -1

def plan_from_env_config(c: object) -> Optional[ReplayPlan]:
    on = bool(c.trace_replay)
    declared = {"trace_replay_block": int(c.trace_replay_block),
                "trace_replay_block_len": int(c.trace_replay_block_len),
                "trace_replay_episodes": int(c.trace_replay_episodes)}
    source = c.trace_replay_source
    if not on:
        given = sorted(k for k, v in declared.items() if v != UNSET)
        if source is not None:
            given = given + ["trace_replay_source"]
        _require(not given,
                 "trace_replay=False 却声明了 %s ⇒ 这一跑**没有**回放，产物里一个溯源键都不会"
                 "写；半套声明必须当场抛，否则「旗标写了、忘了打开」跑出来的是标称臂而无人"
                 "发现" % (given,))
        return None
    missing = sorted(k for k, v in declared.items() if v == UNSET)
    _require(not missing,
             "trace_replay=True 却没声明 %s ⇒ 回放不许有默认块：默认值会把「忘了声明」跑成"
             "「悄悄回放第 0 块」，而产物里看不出来（本库默认值陷阱已第五例）" % (missing,))
    return plan(block=declared["trace_replay_block"],
                n_episodes=declared["trace_replay_episodes"],
                block_len=declared["trace_replay_block_len"],
                source=DEFAULT_SOURCE if source is None else source)

def level_for_episode(p: ReplayPlan, episode_index: int) -> float:
    _require(isinstance(episode_index, int) and episode_index >= 0,
             "episode_index=%r 必须是非负整数" % (episode_index,))
    _require(episode_index < p.n_episodes,
             "第 %d 个 episode 越过了声明的回放窗口（n_episodes=%d，块 %d、块长 %d）⇒ "
             "**不取模、不截断**。要跑更多 episode 就把 trace_replay_episodes 声明大 —— 那是"
             "换了一段更长的曲线，指纹会跟着变，而这正是它该变的时候"
             % (episode_index, p.n_episodes, p.block, p.block_len))
    return p.episode_load_mult()[episode_index]

def _selftest() -> int:
    s = load_series()
    print("源 %s\n  sha256 %s\n  %d 点 × %d ms，一点管 %d 个 episode（%d ms）"
          % (DEFAULT_SOURCE, s["sha256"][:16], s["n_timestamps"], s["resolution_ms"],
             s["episodes_per_point"], s["env_episode_ms"]))
    p = plan(block=0, n_episodes=100, block_len=120, start_offset=0)
    prov = p.provenance()
    assert check_provenance(prov) is True
    assert check_provenance({}) is False, "空产物必须读成「没回放」，不是抛错"
    print("  %r\n  尾部 %d 点不可寻址；每 episode 倍率前三个 %s"
          % (p, p.unused_tail_points, [round(v, 5) for v in p.episode_load_mult()[:3]]))
    print("  fingerprint %s" % prov["trace_index_fingerprint"])
    return 0

if __name__ == "__main__":
    import sys
    if "--selftest" not in sys.argv[1:]:
        print(__doc__)
        sys.exit(2)
    try:
        sys.exit(_selftest())
    except ReplayContractError as exc:
        print("契约被破坏：%s" % exc)
        sys.exit(3)
