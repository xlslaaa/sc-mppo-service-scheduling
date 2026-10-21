from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

QUANTILES = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]

def fit_dsb(csv_path: Path) -> Dict:
    df = pd.read_csv(csv_path)
    if "childMS" not in df.columns or "childDuration" not in df.columns:
        raise ValueError(f"{csv_path} missing expected columns childMS/childDuration")

    df = df[["childMS", "childDuration"]].dropna()
    df = df[df["childDuration"] > 0]
    df["ms"] = df["childDuration"].astype(float) / 1000.0

    services: Dict[str, Dict] = {}
    for name, group in df.groupby("childMS"):
        vals = group["ms"].to_numpy()
        q = np.quantile(vals, QUANTILES)
        services[str(name)] = {
            "count": int(vals.size),
            "latency_ms_quantiles": [round(float(x), 4) for x in q],
            "latency_ms_mean": round(float(vals.mean()), 4),
            "latency_ms_std": round(float(vals.std()), 4),
        }

    total = sum(s["count"] for s in services.values())
    for s in services.values():
        s["call_frequency"] = round(s["count"] / total, 6) if total else 0.0

    return {
        "quantile_levels": QUANTILES,
        "num_services": len(services),
        "total_spans": int(total),
        "services": services,
    }

def fit_alibaba_arrival(
    rtqps_dir: Path,
    max_files: int = 1,
    chunk_rows: int = 2_000_000,
    max_chunks: int = 20,
) -> Optional[Dict]:
    files = sorted(rtqps_dir.glob("*.csv"))
    if not files:
        return None
    files = files[:max_files]

    RATE_KEYS = ("MCR", "QPS", "qps", "RPC")

    value_samples: List[np.ndarray] = []
    per_ts_total: Dict[int, float] = {}
    n_chunks = 0
    for f in files:
        reader = pd.read_csv(
            f,
            usecols=["timestamp", "metric", "value"],
            chunksize=chunk_rows,
        )
        for chunk in reader:
            n_chunks += 1
            metric = chunk["metric"].astype(str)
            mask = metric.str.contains("|".join(RATE_KEYS), case=False, regex=True)
            sub = chunk.loc[mask, ["timestamp", "value"]]
            if sub.empty:
                if n_chunks >= max_chunks:
                    break
                continue
            vals = pd.to_numeric(sub["value"], errors="coerce")
            ts = pd.to_numeric(sub["timestamp"], errors="coerce")
            good = vals.notna() & ts.notna() & (vals > 0)
            vals = vals[good].to_numpy(dtype=float)
            ts = ts[good].to_numpy(dtype=np.int64)

            if vals.size > 200_000:
                sel = np.random.default_rng(0).choice(vals.size, 200_000, replace=False)
                value_samples.append(vals[sel])
            else:
                value_samples.append(vals)

            grp = pd.Series(vals, index=ts).groupby(level=0).sum()
            for t, v in grp.items():
                per_ts_total[int(t)] = per_ts_total.get(int(t), 0.0) + float(v)
            if n_chunks >= max_chunks:
                break
        if n_chunks >= max_chunks:
            break

    if not value_samples:
        return None

    arr = np.concatenate(value_samples)
    arr = arr[arr > 0]
    if arr.size == 0:
        return None

    q = np.quantile(arr, QUANTILES)

    ts_totals = np.asarray(sorted(per_ts_total.values()), dtype=float)
    ts_totals = ts_totals[ts_totals > 0]
    if ts_totals.size >= 5:
        ts_q = np.quantile(ts_totals, QUANTILES)
        ts_median = float(np.median(ts_totals))
        temporal_mult = [round(float(x / ts_median), 4) if ts_median > 0 else 1.0 for x in ts_q]
    else:
        temporal_mult = None

    return {
        "source_files": [f.name for f in files],
        "quantile_levels": QUANTILES,
        "call_rate_quantiles": [round(float(x), 4) for x in q],
        "call_rate_mean": round(float(arr.mean()), 4),
        "call_rate_std": round(float(arr.std()), 4),
        "num_samples": int(arr.size),
        "num_timestamps": int(ts_totals.size),

        "temporal_load_mult_quantiles": temporal_mult,

        "arrival_scale_p50_p95": round(float(q[QUANTILES.index(0.95)] / max(q[QUANTILES.index(0.50)], 1e-6)), 4),
    }

def main() -> None:
    p = argparse.ArgumentParser(description="Fit empirical distributions from real traces.")
    p.add_argument(
        "--dsb-csv",
        type=Path,
        default=Path("../DeathStarBench-master/ms_collecter/trace.csv"),
        help="DeathStarBench collected span CSV.",
    )
    p.add_argument(
        "--ali-rtqps-dir",
        type=Path,
        default=Path("../clusterdata-master/cluster-trace-microservices-v2021/data/MSRTQps"),
        help="Directory of Alibaba MSRTQps CSVs (optional; skipped if empty).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("traces/trace_profile.json"),
        help="Output profile path.",
    )
    args = p.parse_args()

    profile: Dict = {"sources": {}}

    if args.dsb_csv.exists():
        profile["latency"] = fit_dsb(args.dsb_csv)
        profile["sources"]["dsb"] = str(args.dsb_csv)
        print(
            f"[dsb] fitted {profile['latency']['num_services']} services "
            f"from {profile['latency']['total_spans']} spans"
        )
    else:
        print(f"[dsb] WARN: {args.dsb_csv} not found; skipping latency fit")

    arrival = fit_alibaba_arrival(args.ali_rtqps_dir) if args.ali_rtqps_dir.exists() else None
    if arrival is not None:
        profile["arrival"] = arrival
        profile["sources"]["alibaba_rtqps"] = str(args.ali_rtqps_dir)
        print(f"[alibaba] fitted arrival profile from {len(arrival['source_files'])} file(s)")
    else:
        print("[alibaba] no MSRTQps data yet; arrival profile omitted (env will fall back)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2))
    print(f"[out] wrote {args.out}")

if __name__ == "__main__":
    main()
