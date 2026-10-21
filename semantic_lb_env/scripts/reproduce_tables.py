"""Recompute the paper's result tables and p-values from the per-seed results in this repository.

Statistics: exact two-sided paired Wilcoxon signed-rank test (zero differences dropped, tie-aware
null distribution; see ``stats_helpers.py``) with Holm correction inside each pre-specified
comparison family. The independent experimental unit is the training seed. Confidence intervals are
95% t intervals across seeds.

Families (focus ``sc_mppo`` unless stated; each family is Holm-corrected on its own):
  * strong baselines (Table 3 / Table A1), per protocol:
      P1 = cost vs {ppo_lagrangian, p3o, maskable_ppo_semantic, ppo_no_mask}               (m = 4)
      P2 = {sla_violation_rate, p95_latency} x the same four peers                           (m = 8)
      D1 = cost vs deepset_ppo (m = 1);  D2 = {sla_violation_rate, p95_latency} vs deepset_ppo (m = 2)
      D3 = deepset_ppo vs maskable_ppo_semantic on {cost, sla_violation_rate, p95_latency}   (m = 3)
  * single-node testbed (Table 6 / Table A6): the same P1 / P2 / D1 / D2 families on the testbed runs.
  * instance-scale ladder (Table A3): P1 / P2 per tier (15 / 30 / 45 instances) and protocol.
  * telemetry candidates (Fig. 11, deterministic protocol, maskable_ppo_semantic rows):
      P1 = A2 vs A0, P1b = A1 vs A0, P1c = A2 vs A1, each on
      {failure_rate, p95_latency, mean_candidate_set_size}                                  (m = 3)
CPO is a descriptive reference only (no tests): it did not pass the pre-specified calibration of its
constraint handling.

CSV parsing matches the analysis that produced the paper: the strong-baseline and scale-ladder families
were computed with pandas' default float parser, the testbed with ``float_precision="round_trip"``, and the
telemetry families with Python's ``float()``. The parsers can differ in the last bit of a float.

Usage (from ``semantic_lb_env/``)::

    PYTHONPATH=$(pwd):$(pwd)/scripts python scripts/reproduce_tables.py               # print + write JSON
    PYTHONPATH=$(pwd):$(pwd)/scripts python scripts/reproduce_tables.py --check FILE  # compare to a saved JSON
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ENV = HERE.parent
sys.path[:0] = [str(HERE)]
from stats_helpers import holm_correct_family, wilcoxon_exact  # noqa: E402

PROTOCOLS = ("stochastic", "deterministic")
SEEDS_MAIN = tuple(range(68, 84))
SEEDS_SCALE = tuple(range(100, 116))
SEEDS_TELEMETRY = tuple(range(70, 86))
PEERS = ("ppo_lagrangian", "p3o", "maskable_ppo_semantic", "ppo_no_mask")
QOS = ("sla_violation_rate", "p95_latency")
COST = "mean_normalized_cost"
FOCUS = "sc_mppo"
SIM_ROOTS = {"main": "results/reeval_sb16", "cpo": "results/reeval_sb16_cpo", "deepset_ppo": "results/reeval_sb16_deepset"}
TESTBED_ROOT = "results/dsb_testbed_run"
SCALE_ROOTS = {"t15": "results/reeval_scale_t15", "t30": "results/reeval_scale_t30", "t45": "results/reeval_scale_t45"}
TELEMETRY_ROOT = "../results/telemetry_mask_formal"
TELEMETRY_FAMILIES = {"P1": ("A2", "A0"), "P1b": ("A1", "A0"), "P1c": ("A2", "A1")}
TELEMETRY_ENDPOINTS = ("failure_rate", "p95_latency", "mean_candidate_set_size")
TELEMETRY_METHOD = "maskable_ppo_semantic"
DESCRIBED = (COST, "sla_violation_rate", "p95_latency", "semantic_match", "failure_rate")


def ci95(values):
    v = np.asarray(values, dtype=float)
    return float(stats.sem(v) * stats.t.ppf(0.975, len(v) - 1)) if len(v) > 1 else 0.0


def load_seed_tables(root, protocol, seeds, round_trip):
    """{seed: DataFrame of metrics.csv} for root/protocol/seed_XX/metrics.csv."""
    kw = {"float_precision": "round_trip"} if round_trip else {}
    out = {}
    for s in seeds:
        path = ENV / root / protocol / ("seed_%d" % s) / "metrics.csv"
        if not path.exists():
            raise SystemExit("missing %s" % path)
        out[s] = pd.read_csv(path, **kw).set_index("method")
    return out


def column(tables, method, metric):
    return pd.Series({s: float(t.loc[method, metric]) for s, t in tables.items()})


def paired(a, b):
    """Exact paired Wilcoxon of a vs b (pd.Series indexed by seed)."""
    idx = sorted(set(a.index) & set(b.index))
    x, y = a.loc[idx].to_numpy(dtype=float), b.loc[idx].to_numpy(dtype=float)
    res = wilcoxon_exact(x, y)
    return {"p_raw": float(res.p_raw), "gap": float(np.mean(x) - np.mean(y)), "n": len(idx)}


def holm(rows):
    p_holm, _ = holm_correct_family([r["p_raw"] for r in rows])
    for r, ph in zip(rows, p_holm):
        r["p_holm"] = float(ph)
    return rows


def describe(tables_by_method, metrics, seeds):
    out = {}
    for method, (tables, name) in tables_by_method.items():
        out[method] = {}
        for m in metrics:
            v = column(tables, name, m).loc[list(seeds)].to_numpy(dtype=float)
            out[method][m] = {"mean": float(v.mean()), "ci95": ci95(v)}
    return out


def strong_families(tables_main, tables_deepset, focus_tables=None):
    """P1 / P2 / D1 / D2 / D3 on one protocol. focus_tables defaults to tables_main."""
    ft = tables_main if focus_tables is None else focus_tables
    fam = {}
    fam["P1"] = holm([dict(metric=COST, peer=p, **paired(column(ft, FOCUS, COST), column(tables_main, p, COST)))
                      for p in PEERS])
    fam["P2"] = holm([dict(metric=m, peer=p, **paired(column(ft, FOCUS, m), column(tables_main, p, m)))
                      for m in QOS for p in PEERS])
    fam["D1"] = holm([dict(metric=COST, peer="deepset_ppo",
                           **paired(column(ft, FOCUS, COST), column(tables_deepset, "deepset_ppo", COST)))])
    fam["D2"] = holm([dict(metric=m, peer="deepset_ppo",
                           **paired(column(ft, FOCUS, m), column(tables_deepset, "deepset_ppo", m))) for m in QOS])
    fam["D3"] = holm([dict(metric=m, focus="deepset_ppo", peer="maskable_ppo_semantic",
                           **paired(column(tables_deepset, "deepset_ppo", m), column(tables_main, "maskable_ppo_semantic", m)))
                      for m in (COST,) + QOS])
    return fam


def simulation():
    out = {}
    for proto in PROTOCOLS:
        main = load_seed_tables(SIM_ROOTS["main"], proto, SEEDS_MAIN, round_trip=False)
        ds = load_seed_tables(SIM_ROOTS["deepset_ppo"], proto, SEEDS_MAIN, round_trip=False)
        cpo = load_seed_tables(SIM_ROOTS["cpo"], proto, SEEDS_MAIN, round_trip=True)
        rt = {k: load_seed_tables(v, proto, SEEDS_MAIN, round_trip=True) for k, v in SIM_ROOTS.items()}
        methods = {m: (rt["main"], m) for m in (FOCUS,) + PEERS + ("dqn_no_mask",)}
        methods["deepset_ppo"] = (rt["deepset_ppo"], "deepset_ppo")
        methods["cpo"] = (rt["cpo"], "cpo")
        out[proto] = {"describe": describe(methods, DESCRIBED, SEEDS_MAIN), "families": strong_families(main, ds)}
        del cpo
    return out


def testbed():
    out = {}
    for proto in PROTOCOLS:
        t = load_seed_tables(TESTBED_ROOT, proto, SEEDS_MAIN, round_trip=True)
        methods = {m: (t, m) for m in (FOCUS,) + PEERS + ("deepset_ppo", "cpo")}
        fam = strong_families(t, t)
        fam.pop("D3")
        out[proto] = {"describe": describe(methods, (COST,) + QOS, SEEDS_MAIN), "families": fam}
    return out


def scale_ladder():
    out = {}
    for tier, root in SCALE_ROOTS.items():
        out[tier] = {}
        for proto in PROTOCOLS:
            t = load_seed_tables(root, proto, SEEDS_SCALE, round_trip=False)
            fam = {"P1": holm([dict(metric=COST, peer=p, **paired(column(t, FOCUS, COST), column(t, p, COST))) for p in PEERS]),
                   "P2": holm([dict(metric=m, peer=p, **paired(column(t, FOCUS, m), column(t, p, m))) for m in QOS for p in PEERS])}
            means = {m: {k: float(column(t, m, k).mean()) for k in (COST,) + QOS} for m in (FOCUS,) + PEERS}
            out[tier][proto] = {"families": fam, "means": means}
    return out


def telemetry():
    def read(arm, seed):
        path = ENV / TELEMETRY_ROOT / ("%s_seed%d" % (arm, seed)) / "metrics.csv"
        rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["method"] == TELEMETRY_METHOD]
        assert len(rows) == 1, path
        return rows[0]
    table = {(arm, s): read(arm, s) for arm in ("A0", "A1", "A2") for s in SEEDS_TELEMETRY}
    out = {}
    for fid, (a, b) in TELEMETRY_FAMILIES.items():
        rows = []
        for m in TELEMETRY_ENDPOINTS:
            x = pd.Series({s: float(table[(a, s)][m]) for s in SEEDS_TELEMETRY})
            y = pd.Series({s: float(table[(b, s)][m]) for s in SEEDS_TELEMETRY})
            rows.append(dict(metric=m, focus=a, peer=b, **paired(x, y)))
        out[fid] = holm(rows)
    return out


def build():
    return {"simulation": simulation(), "testbed": testbed(), "scale_ladder": scale_ladder(), "telemetry": telemetry()}


def _fmt(x, d):
    return ("%%.%df" % d) % x


def print_tables(r):
    s = r["simulation"]["stochastic"]["describe"]
    print("Table 3 (simulation, stochastic protocol, mean +/- 95% CI, 16 seeds)")
    for m in (FOCUS,) + PEERS[:1] + ("p3o", "cpo", "maskable_ppo_semantic", "deepset_ppo", "ppo_no_mask", "dqn_no_mask"):
        d = s[m]
        print("  %-24s cost %s+/-%s  sla %s+/-%s  semantic %s+/-%s" % (
            m, _fmt(d[COST]["mean"], 3), _fmt(d[COST]["ci95"], 3), _fmt(d["sla_violation_rate"]["mean"], 4),
            _fmt(d["sla_violation_rate"]["ci95"], 4), _fmt(d["semantic_match"]["mean"], 3), _fmt(d["semantic_match"]["ci95"], 3)))
    for name in ("simulation", "testbed"):
        for proto in PROTOCOLS:
            fam = r[name][proto]["families"]
            print("%s / %s: Holm-corrected p" % (name, proto))
            for fid in sorted(fam):
                print("  %s: %s" % (fid, ", ".join("%s vs %s %s=%.3g" % (row.get("focus", FOCUS), row["peer"], row["metric"], row["p_holm"])
                                                for row in fam[fid])))
    for fid, rows in r["telemetry"].items():
        print("telemetry %s: %s" % (fid, ", ".join("%s %.3g" % (row["metric"], row["p_holm"]) for row in rows)))


def _compare(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append("%s/%s missing" % (path, k))
            else:
                _compare(a[k], b[k], "%s/%s" % (path, k), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append("%s length" % path)
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, "%s[%d]" % (path, i), out)
    elif a != b:
        out.append("%s: %r != %r" % (path, a, b))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/reproduced_tables.json")
    ap.add_argument("--check", default=None, help="compare the recomputed results to this JSON (exact equality)")
    args = ap.parse_args()
    r = json.loads(json.dumps(build()))
    if args.check:
        diff = _compare(r, json.loads(Path(args.check).read_text(encoding="utf-8")))
        print("identical" if not diff else "DIFFERENCES:\n  " + "\n  ".join(diff[:30]))
        sys.exit(0 if not diff else 1)
    print_tables(r)
    out = ENV / args.out
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
