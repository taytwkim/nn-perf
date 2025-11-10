#!/usr/bin/env python3
import argparse, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

METRIC_SM   = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
METRIC_TIME = "gpu__time_duration.sum"  # ns

def to_float(v):
    try:
        return float(str(v).strip().replace("%", ""))
    except Exception:
        return np.nan

def weighted_avg(df: pd.DataFrame, metric_name: str, join_keys):
    sub = df[df["Metric Name"] == metric_name][join_keys + ["Metric Value"]].copy()
    sub = sub.rename(columns={"Metric Value": "metric"})
    sub["metric"] = sub["metric"].apply(to_float)
    sub = sub.dropna(subset=["metric"])

    tdf = df[df["Metric Name"] == METRIC_TIME][join_keys + ["Metric Value"]].copy()
    tdf = tdf.rename(columns={"Metric Value": "time_ns"})
    tdf["time_ns"] = tdf["time_ns"].apply(to_float)
    tdf = tdf.dropna(subset=["time_ns"])

    merged = pd.merge(sub, tdf, on=join_keys, how="inner")
    merged = merged[(merged["time_ns"] > 0) & merged["metric"].notna()]
    if merged.empty:
        return float("nan")

    return float((merged["metric"] * merged["time_ns"]).sum() / merged["time_ns"].sum())

def logspace(a, b, n):
    return [10 ** (math.log10(a) + i * (math.log10(b / a) / (n - 1))) for i in range(n)]

def main():
    ap = argparse.ArgumentParser(description="Build a single-point roofline from Nsight Compute CSV")
    ap.add_argument("csv", help="Nsight Compute --csv export with per-kernel rows")
    ap.add_argument("--peak-compute", type=float, required=True, help="Peak compute (TFLOP/s)")
    ap.add_argument("--peak-bw", type=float, required=True, help="Peak memory bandwidth (GB/s)")
    ap.add_argument("--label", default="Workload", help="Label for the plotted point")
    ap.add_argument("--out", default="roofline.png", help="Output PNG filename")
    ap.add_argument("--summary", default="roofline_summary.csv", help="Output CSV with computed stats")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Columns that together identify a kernel instance in this CSV
    join_keys = [c for c in ["Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC"] if c in df.columns]
    if not join_keys:
        raise SystemExit("Could not find kernel identity columns (Kernel Name/Context/Stream/Block/Grid/Device/CC).")

    sm_pct   = weighted_avg(df, METRIC_SM, join_keys)
    dram_pct = weighted_avg(df, METRIC_DRAM, join_keys)

    if not np.isfinite(sm_pct) or not np.isfinite(dram_pct):
        raise SystemExit("Failed to compute weighted averages. Check metric names present in the CSV.")

    peak_compute_gflops = args.peak_compute * 1000.0
    achieved_compute_gflops = (sm_pct / 100.0) * peak_compute_gflops
    achieved_bw_gbps        = (dram_pct / 100.0) * args.peak_bw
    if achieved_bw_gbps <= 0:
        raise SystemExit("Derived bandwidth is non-positive; check --peak-bw and CSV contents.")

    # Roofline point
    AI = achieved_compute_gflops / achieved_bw_gbps      # FLOP/byte
    Y  = achieved_compute_gflops                         # GFLOP/s
    knee_AI = peak_compute_gflops / args.peak_bw

    # Save numeric summary
    pd.DataFrame([{
        "label": args.label,
        "sm_pct_elapsed": sm_pct,
        "dram_pct_elapsed": dram_pct,
        "peak_compute_TFLOPS": args.peak_compute,
        "peak_bw_GBps": args.peak_bw,
        "AI_flop_per_byte": AI,
        "attained_GFLOP_per_s": Y,
        "knee_AI": knee_AI
    }]).to_csv(args.summary, index=False)

    # Build the roof (min(PeakCompute, AI * PeakBW)), log-log plot
    x_min = max(1e-3, min(AI/5, knee_AI/20))
    x_max = max(AI*5, knee_AI*2)
    xs = logspace(x_min, x_max, 300)
    ys = [min(peak_compute_gflops, x * args.peak_bw) for x in xs]

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=140)
    ax.set_xscale("log"); ax.set_yscale("log")

    ax.plot(xs, ys, linewidth=2, label="Theoretical roof")
    ax.hlines(peak_compute_gflops, x_min, x_max, linestyles="--", linewidth=1, label="Compute peak")
    ax.vlines(knee_AI, max(ys[0]*0.5, 1.0), peak_compute_gflops, linestyles=":", linewidth=1, label="Knee")

    ax.scatter([AI], [Y], s=40)
    ax.annotate(f"{args.label}\nAI={AI:.3f}, Y={Y:.1f} GFLOP/s\nSM%={sm_pct:.1f}, DRAM%={dram_pct:.1f}",
                (AI, Y), textcoords="offset points", xytext=(8,6), fontsize=8)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(max(1.0, ys[0]*0.5), peak_compute_gflops*1.5)
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Attainable Performance (GFLOP/s)")
    ax.set_title(f"Roofline (Peak: {args.peak_compute:.2f} TFLOP/s, {args.peak_bw:.0f} GB/s)")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out)

    print(f"SM% (elapsed, time-weighted):   {sm_pct:.2f}")
    print(f"DRAM% (elapsed, time-weighted): {dram_pct:.2f}")
    print(f"Point: AI={AI:.3f} FLOP/byte, Y={Y:.1f} GFLOP/s; Knee AI={knee_AI:.2f}")
    print(f"Saved plot: {args.out}")
    print(f"Saved summary: {args.summary}")

if __name__ == "__main__":
    main()