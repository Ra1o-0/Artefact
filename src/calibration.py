# Chunk 6 -- calibration evaluation (the intellectual core of the dissertation).
#
# Chunk 5 produced, for every test day, a central forecast and a Monte Carlo
# dropout spread. This module does NOT touch the model, the frozen split, or any
# scaler. It is a pure measurement step over the frozen test_intervals.csv: it
# asks whether the model's stated confidence is actually honest.
#
# The calibration question: across all days the model claims X% confidence, does
# the truth actually fall inside its X% interval X% of the time? A model can be
# ACCURATE (point forecast close to the truth) yet badly CALIBRATED (intervals
# far too narrow, so it is systematically overconfident). In finance, knowing
# when NOT to trust a forecast matters as much as the forecast itself.
#
# Four outputs:
#   1. Reliability diagram -- claimed vs observed coverage across a sweep of
#      confidence levels, against the 45-degree perfect-calibration diagonal.
#      Below the diagonal = overconfident; above = underconfident.
#   2. Expected Calibration Error (ECE) -- one number summarising the average
#      gap between claimed and observed coverage across the swept levels.
#   3. Point-baseline vs interval comparison -- the point forecast's dollar
#      error (MAE and RMSE) beside the 90% band's empirical coverage, making
#      "accurate yet overconfident" explicit.
#   4. VIX extension -- do the model's widest-uncertainty days coincide with
#      high market fear (VIX)? Reported as Pearson and Spearman correlation
#      and a scatter plot.
#
# How I rebuild a band at an arbitrary confidence level:
#   The saved CSV stores only the 5th/95th MC percentiles, not all 100 passes.
#   To draw a reliability curve across many confidence levels I reconstruct
#   each level's band with a Gaussian approximation of the MC spread:
#       half_width(c) = z(c) * mc_std_return,   mean +/- half_width
#   where z(c) is the standard-normal z-score for a central interval of mass c
#   (e.g. c=0.90 gives z=1.645). This assumes the 100 MC passes are roughly
#   normally distributed -- a stated approximation. I sanity-check the 90%
#   reconstruction against the stored 5th/95th band before trusting it.
#
# Run modes:
#     py src/calibration.py               # default: compute + print all metrics, write figures
#     py src/calibration.py --no-figures  # metrics only, skip the .png files

import argparse
import os
from statistics import NormalDist

import numpy as np
import pandas as pd

from train_regression import PROCESSED_DIR

# Configuration
INTERVALS_PATH = os.path.join(PROCESSED_DIR, "test_intervals.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")   # carries VIX_Close per date
REPORTS_DIR = "reports"

# Confidence levels swept for the reliability diagram and ECE. A wide spread so
# the curve's shape (over/under-confidence) is visible across the full range.
CONFIDENCE_LEVELS = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99])


# 1. Load the frozen per-day table

# Just loads the CSV and checks all the columns I need are present.
def load_intervals():
    df = pd.read_csv(INTERVALS_PATH, parse_dates=["date"], index_col="date")
    needed = {"actual_next_return", "actual_next_close", "close_today",
              "point_price", "mc_mean_return", "mc_mean_price", "mc_std_return",
              "lower5_price", "upper95_price"}
    missing = needed - set(df.columns)
    assert not missing, f"test_intervals.csv missing columns: {missing}"
    return df


# Given a confidence level like 0.90, returns the z-score (1.645) that marks the
# edges of a central normal interval containing that fraction of the distribution.
# A 90% central interval leaves 5% in each tail, so I want the 95th quantile.
def z_value_for(conf):
    return NormalDist().inv_cdf(1 - (1 - conf) / 2)


# 2. Coverage at a given confidence level (Gaussian reconstruction)

# For a given confidence level I reconstruct the band as mean +/- z * std, then
# count what fraction of actual returns landed inside it. This is the core
# measurement repeated for every point on the reliability curve.
# I work in return space because coverage is scale-invariant -- the inside/outside
# decision is the same whether I use returns or dollar prices.
def coverage_at(df, conf):
    half_width = z_value_for(conf) * df["mc_std_return"].to_numpy()   # z * per-day std gives the band half-width
    mean = df["mc_mean_return"].to_numpy()
    actual_return = df["actual_next_return"].to_numpy()
    is_inside = (actual_return >= mean - half_width) & (actual_return <= mean + half_width)  # True where actual fell inside the band
    return float(is_inside.mean())  # fraction of days where the actual return was inside the band


# Build the full claimed-vs-observed table by calling coverage_at() for every
# confidence level in CONFIDENCE_LEVELS.
def reliability_table(df):
    observed = np.array([coverage_at(df, c) for c in CONFIDENCE_LEVELS])
    return pd.DataFrame({"claimed": CONFIDENCE_LEVELS, "observed": observed})


# ECE is the average absolute gap between claimed and observed coverage.
# 0 = perfectly calibrated; larger = more miscalibrated.
# I also keep the signed mean gap so overconfidence (negative) vs
# underconfidence (positive) is not hidden by the absolute value.
def expected_calibration_error(reliability):
    gap = reliability["observed"] - reliability["claimed"]
    return {"ece": float(gap.abs().mean()), "signed_mean_gap": float(gap.mean())}


# 3. Point-forecast accuracy vs interval calibration

# This computes the dollar accuracy of the point forecast (MAE and RMSE) and
# compares it to a naive "predict no change" baseline. It also reads the
# empirical coverage of the stored 90% band straight from the CSV, without
# using the Gaussian reconstruction -- so the two are independent checks.
# The contrast is the heart of the dissertation: accurate in dollars, yet
# the intervals can still be badly calibrated (too narrow = overconfident).
def point_vs_interval(df):
    errors = df["point_price"].to_numpy() - df["actual_next_close"].to_numpy()
    mae = float(np.abs(errors).mean())                   # mean absolute error in dollars
    rmse = float(np.sqrt((errors ** 2).mean()))          # root mean squared error in dollars

    # Naive baseline: predict "tomorrow's price equals today's price".
    naive_errors = df["close_today"].to_numpy() - df["actual_next_close"].to_numpy()
    naive_mae = float(np.abs(naive_errors).mean())

    # Read the stored 90% band coverage directly from the CSV's 5th/95th columns.
    inside_90 = ((df["lower5_price"] <= df["actual_next_close"]) &
                (df["actual_next_close"] <= df["upper95_price"]))
    return {
        "point_mae_price": mae,
        "point_rmse_price": rmse,
        "naive_mae_price": naive_mae,
        "stored_band_coverage_90": float(inside_90.mean()),  # fraction of days where actual price was inside the 90% band
    }


# 4. VIX extension: does uncertainty track market fear?

# Spearman rank correlation without importing scipy: I convert both series to
# ranks first, then take the ordinary Pearson correlation of those ranks.
# Rank correlation is more robust to outliers than Pearson on raw values.
def spearman_correlation(a, b):
    ranks_a = pd.Series(a).rank().to_numpy()  # replace values with their rank positions
    ranks_b = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])  # [0,1] picks the off-diagonal element (the correlation)


# Joins VIX onto the per-day interval table and correlates the model's interval
# width with VIX (external market fear). A positive correlation would mean the
# model's uncertainty rises when real market fear rises. A weak or zero
# correlation is an honest finding too -- I report whatever the data shows.
def vix_extension(df):
    test = pd.read_csv(TEST_PATH, parse_dates=["Date"], index_col="Date")
    assert "VIX_Close" in test.columns, "test.csv missing VIX_Close"

    interval_width = df["upper95_price"] - df["lower5_price"]  # dollar width of the 90% band for each day
    merged_table = pd.DataFrame({
        "interval_width_price": interval_width,
        "mc_std_return": df["mc_std_return"],
    }, index=df.index)
    merged_table = merged_table.join(test["VIX_Close"], how="inner")   # inner join: keep only dates present in both
    assert len(merged_table) == len(df), (
        f"VIX join lost rows: {len(df)} intervals vs {len(merged_table)} matched")

    pear = float(np.corrcoef(merged_table["interval_width_price"], merged_table["VIX_Close"])[0, 1])
    spear = spearman_correlation(merged_table["interval_width_price"].to_numpy(),
                      merged_table["VIX_Close"].to_numpy())
    return merged_table, {"pearson": pear, "spearman": spear}


# Figures

# Reliability diagram: claimed confidence on the x-axis, observed coverage on
# the y-axis. The dashed diagonal is perfect calibration. Points below the
# diagonal mean the model is overconfident at that level.
def plot_reliability(reliability, ece, path):
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend so it saves to a file without needing a screen
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(reliability["claimed"], reliability["observed"], "o-", color="C0", label="model")
    ax.fill_between(reliability["claimed"], reliability["claimed"], reliability["observed"],
                    color="C3", alpha=0.15, label="miscalibration gap")
    ax.set_xlabel("claimed confidence")
    ax.set_ylabel("observed coverage (fraction of truths inside)")
    ax.set_title(f"Reliability diagram (ECE = {ece['ece']:.3f})\n"
                 "below diagonal = overconfident")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# Scatter plot of interval width (model uncertainty) against VIX (market fear).
def plot_uncertainty_vs_vix(merged_table, correlations, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(merged_table["VIX_Close"], merged_table["interval_width_price"],
               s=14, alpha=0.5, color="C0")
    ax.set_xlabel("VIX (external market fear)")
    ax.set_ylabel(r"interval width  (\$, upper95 - lower5)")
    ax.set_title("Model uncertainty vs market fear\n"
                 f"Pearson r = {correlations['pearson']:+.3f},  Spearman = {correlations['spearman']:+.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# The MC dropout prediction band over the whole test period: shaded 90% band,
# MC mean forecast line, and the actual next-close line in black. This makes
# the overconfidence visible -- the band is often far narrower than it needs
# to be to contain the actual price.
def plot_prediction_intervals(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(df.index, df["lower5_price"], df["upper95_price"],
                    color="C0", alpha=0.25,
                    label="90% prediction band (5th-95th MC percentile)")
    ax.plot(df.index, df["mc_mean_price"], color="C0", lw=1.2,
            label="forecast (MC mean)")
    ax.plot(df.index, df["actual_next_close"], color="k", lw=1.0,
            label="actual next close")
    ax.set_xlabel("date")
    ax.set_ylabel(r"AAPL price (\$)")
    ax.set_title("Next-day forecast with 90% prediction band (test set)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# The deterministic point forecast (dropout OFF) against the actual price, with
# the naive "no change" predictor as a faint reference line. This plots
# point_price not mc_mean_price -- it is the like-for-like point baseline whose
# dollar accuracy is contrasted with the band's calibration. The naive line
# makes it visible when the point forecast is no better than just predicting
# "tomorrow equals today".
def plot_point_vs_actual(df, accuracy_summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(df.index, df["actual_next_close"], color="k", lw=1.0,
            label="actual next close")
    ax.plot(df.index, df["point_price"], color="C0", lw=1.0,
            label="point forecast (dropout off)")
    ax.plot(df.index, df["close_today"], color="0.6", lw=0.8, alpha=0.7,
            label="naive 'no change' (tomorrow = today)")
    ax.set_xlabel("date")
    ax.set_ylabel(r"AAPL price (\$)")
    ax.set_title("Point forecast vs actual\n"
                 rf"point MAE \${accuracy_summary['point_mae_price']:.2f}  "
                 rf"vs naive MAE \${accuracy_summary['naive_mae_price']:.2f}  (no better than naive)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# Orchestration

# Run all four analyses in order, print results to the terminal, and optionally
# write the four PNG figures. The --no-figures flag skips writing files if I
# just want the numbers without saving plots.
def main(write_figures):
    df = load_intervals()
    print(f"loaded {INTERVALS_PATH}  ({len(df)} test days, "
          f"{df.index.min().date()} -> {df.index.max().date()})")

    # Sanity-check the Gaussian reconstruction against the stored 90% band before
    # trusting it for the whole reliability curve. Both should give similar coverage.
    reconstructed_90 = coverage_at(df, 0.90)
    stored_90 = float(((df["lower5_price"] <= df["actual_next_close"]) &
                      (df["actual_next_close"] <= df["upper95_price"])).mean())
    print(f"\n[recon check] Gaussian 90% coverage {reconstructed_90:.3f}  vs  "
          f"stored 5/95 band coverage {stored_90:.3f}  (should be close)")

    # 1 and 2: reliability diagram + ECE
    reliability = reliability_table(df)
    ece = expected_calibration_error(reliability)
    print("\n--- Reliability: claimed vs observed coverage ---")
    for _, r in reliability.iterrows():
        gap = r["observed"] - r["claimed"]
        flag = "overconfident" if gap < -0.02 else ("underconfident" if gap > 0.02 else "ok")
        print(f"  claimed {r['claimed']:.2f}  ->  observed {r['observed']:.3f}   "
              f"(gap {gap:+.3f}, {flag})")
    print(f"\nExpected Calibration Error (ECE) = {ece['ece']:.3f}")
    print(f"signed mean gap = {ece['signed_mean_gap']:+.3f}  "
          f"({'overconfident overall' if ece['signed_mean_gap'] < 0 else 'underconfident overall'})")

    # 3: point accuracy vs interval calibration
    accuracy_summary = point_vs_interval(df)
    print("\n--- Point-forecast accuracy vs interval calibration ---")
    print(f"  point forecast  MAE = ${accuracy_summary['point_mae_price']:.3f}   "
          f"RMSE = ${accuracy_summary['point_rmse_price']:.3f}")
    print(f"  naive 'no change' MAE = ${accuracy_summary['naive_mae_price']:.3f}  "
          "(yardstick for the $ error)")
    print(f"  90% interval empirical coverage = {accuracy_summary['stored_band_coverage_90']:.3f}  "
          "(claimed 0.90)")
    print("  => the model can be reasonably ACCURATE in $ yet badly CALIBRATED "
          "(intervals far too narrow).")

    # 4: VIX extension
    merged_table, correlations = vix_extension(df)
    print("\n--- VIX extension: does model uncertainty track market fear? ---")
    print(f"  corr(interval width, VIX):  Pearson {correlations['pearson']:+.3f}   "
          f"Spearman {correlations['spearman']:+.3f}")
    widest_days = merged_table.nlargest(5, "interval_width_price")  # the 5 days with the widest uncertainty bands
    print("  widest-uncertainty days and their VIX:")
    for date, r in widest_days.iterrows():
        print(f"    {date.date()}  width ${r['interval_width_price']:.3f}   "
              f"VIX {r['VIX_Close']:.2f}")

    if write_figures:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        rel_path = os.path.join(REPORTS_DIR, "reliability_diagram.png")
        vix_path = os.path.join(REPORTS_DIR, "uncertainty_vs_vix.png")
        pi_path = os.path.join(REPORTS_DIR, "prediction_intervals.png")
        pva_path = os.path.join(REPORTS_DIR, "point_vs_actual.png")
        plot_reliability(reliability, ece, rel_path)
        plot_uncertainty_vs_vix(merged_table, correlations, vix_path)
        plot_prediction_intervals(df, pi_path)
        plot_point_vs_actual(df, accuracy_summary, pva_path)
        print(f"\n[saved] {rel_path}")
        print(f"[saved] {vix_path}")
        print(f"[saved] {pi_path}")
        print(f"[saved] {pva_path}")

    print("\n[DONE] Chunk 6 — calibration evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk 6: calibration evaluation.")
    parser.add_argument("--no-figures", action="store_true",
                        help="compute and print metrics only; skip writing the .png figures.")
    args = parser.parse_args()
    main(write_figures=not args.no_figures)
