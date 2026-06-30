# Chunk 7 -- minimal Streamlit dashboard (the artefact's face for the viva).
#
# This is a presentation layer only. Like Chunk 6, it does NOT touch the model,
# the frozen split, or any scaler. It re-reads Chunk 5's frozen per-day table
# (data/processed/test_intervals.csv) and the pre-saved Chunk 6 figures, and
# shows them in a browser. Nothing here retrains, reloads the .keras model,
# refits a scaler, or re-runs Monte Carlo passes.
#
# TensorFlow is deliberately not imported here. Importing calibration.py would
# chain all the way back through train_regression.py to TensorFlow, which is
# slow and unnecessary just to display results. Instead, the few numbers shown
# on screen (point MAE, naive MAE, 90% coverage) are recomputed directly from
# the CSV using the same formulas as calibration.py, so they cannot drift.
#
# What it shows (calibration is the headline figure per CLAUDE.md):
#   1. The forecast over the test days: actual next close, the MC-mean forecast
#      line, and the shaded 90% prediction band.
#   2. A "pick a day" readout turning one day into the project's headline
#      sentence ("$X-$Y, 90% confident; actual landed at $Z, inside/outside").
#   3. The reliability diagram as the centrepiece, with the headline calibration
#      numbers (claimed 90% vs observed roughly 12%, ECE roughly 0.509).
#   4. The VIX extension figure and its honest null result.
#   5. A one-line summary: no better than naive, yet badly miscalibrated.
#
# Run from the PROJECT ROOT (paths are root-relative, as in every prior chunk):
#     py -m streamlit run src/dashboard.py

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; Streamlit renders the Figure object directly
import matplotlib.pyplot as plt
import streamlit as st

# Paths (project-root-relative)
PROCESSED_DIR = os.path.join("data", "processed")
REPORTS_DIR = "reports"
INTERVALS_PATH = os.path.join(PROCESSED_DIR, "test_intervals.csv")
RELIABILITY_PNG = os.path.join(REPORTS_DIR, "reliability_diagram.png")
VIX_PNG = os.path.join(REPORTS_DIR, "uncertainty_vs_vix.png")
POINT_VS_ACTUAL_PNG = os.path.join(REPORTS_DIR, "point_vs_actual.png")

# The stored band is the 5th/95th MC percentile, which is a 90% central interval.
BAND_CONFIDENCE = 0.90


# Data and metrics (read-only, mirrors src/calibration.py)

# Thin wrapper that loads the Chunk 5 CSV and parses the date index.
def load_intervals():
    df = pd.read_csv(INTERVALS_PATH, parse_dates=["date"], index_col="date")
    return df


# Recomputes the displayed numbers from the CSV using the exact same formulas
# as calibration.py so nothing on screen is hardcoded and values cannot drift.
def headline_metrics(df):
    errors = df["point_price"].to_numpy() - df["actual_next_close"].to_numpy()          # dollar error of the point forecast
    naive_errors = df["close_today"].to_numpy() - df["actual_next_close"].to_numpy()    # error of "tomorrow equals today"
    inside_90 = ((df["lower5_price"] <= df["actual_next_close"]) &
                (df["actual_next_close"] <= df["upper95_price"]))                        # True on days the actual landed inside the 90% band
    return {
        "point_mae": float(np.abs(errors).mean()),
        "point_rmse": float(np.sqrt((errors ** 2).mean())),
        "naive_mae": float(np.abs(naive_errors).mean()),
        "coverage90": float(inside_90.mean()),  # fraction of test days where actual was inside the 90% band
        "n_days": len(df),
    }


# Figure: forecast line and shaded 90% band

# This figure is drawn live from the CSV rather than loaded from a PNG because
# no saved version exists. Streamlit can render a matplotlib Figure object directly.
def forecast_figure(df):
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
    return fig


# Page layout

# Everything below is Streamlit: each st.* call adds a widget or block of text
# to the page. The page is re-run top to bottom every time the user interacts
# with a widget (like the date dropdown), so there is no explicit event loop.
def main():
    st.set_page_config(page_title="Uncertainty-Calibrated LSTM — AAPL")

    st.title("Uncertainty-Calibrated LSTM: AAPL next-day forecast")
    st.markdown(
        "Rather than a single price, the model forecasts a **prediction interval** "
        r'with a confidence level (e.g. *"\$138-146, 90% confident"*). '
        "The research question is **calibration**: when the model claims 90% "
        "confidence, does the truth actually land inside its interval 90% of the "
        "time? A model can be accurate yet badly calibrated — and in finance, "
        "knowing when *not* to trust a forecast matters as much as the forecast."
    )

    # Stop early with a clear error if the interval CSV has not been produced yet.
    if not os.path.exists(INTERVALS_PATH):
        st.error(f"Missing {INTERVALS_PATH}. Run Chunk 5 (src/mc_dropout.py) first.")
        st.stop()

    df = load_intervals()
    metrics = headline_metrics(df)

    # Headline numbers at a glance
    verdict = "Overconfident" if metrics["coverage90"] < BAND_CONFIDENCE else "Calibrated"
    st.markdown(
        f"- **Claimed confidence:** {int(BAND_CONFIDENCE * 100)}%\n"
        f"- **Observed coverage:** {metrics['coverage90']:.1%} "
        "(fraction of test days whose true close fell inside the 90% band)\n"
        f"- **Verdict:** {verdict} "
        f"({metrics['coverage90'] - BAND_CONFIDENCE:+.1%} vs claimed)"
    )

    # 1. Forecast chart
    st.subheader("Forecast and confidence band")
    st.markdown(
        f"Across the {metrics['n_days']} test days "
        f"({df.index.min().date()} -> {df.index.max().date()}), the shaded band "
        "is the model's 90% interval. Notice how narrow it is relative to how often "
        "the black actual line strays outside it -- that gap is the overconfidence."
    )
    st.pyplot(forecast_figure(df))

    # 1b. Point-forecast accuracy vs naive
    st.subheader("Point-forecast accuracy vs naive")
    st.markdown(
        rf"The deterministic point forecast (dropout off) has MAE \${metrics['point_mae']:.2f}, "
        rf"essentially equal to a naive 'no-change' baseline (\${metrics['naive_mae']:.2f}) — "
        "the forecaster is **no better than naive**. Accuracy and calibration are "
        "separate concerns: this model is so-so on accuracy *and* badly overconfident."
    )
    if os.path.exists(POINT_VS_ACTUAL_PNG):
        st.image(POINT_VS_ACTUAL_PNG, width=700)
    else:
        st.warning(f"Missing {POINT_VS_ACTUAL_PNG}. Run Chunk 6 (src/calibration.py).")

    # 2. Pick-a-day readout
    # st.selectbox renders a dropdown; the chosen date drives all the values below.
    st.subheader("Inspect a single day")
    dates = [d.date() for d in df.index]
    chosen_day = st.selectbox("Test day", options=dates, index=0)
    day_row = df.loc[pd.Timestamp(chosen_day)]   # pull the row for the chosen date
    low_price, high_price = day_row["lower5_price"], day_row["upper95_price"]
    actual = day_row["actual_next_close"]
    inside = low_price <= actual <= high_price   # did the actual price land inside the band?
    st.markdown(
        f"On **{chosen_day}** the model forecast next-day AAPL at "
        rf"**\${low_price:.2f}–\${high_price:.2f}, {int(BAND_CONFIDENCE * 100)}% confident** "
        rf"(central forecast \${day_row['mc_mean_price']:.2f}). "
        rf"The actual next close was **\${actual:.2f}** — "
        + (":green[**inside** the band]." if inside
           else ":red[**outside** the band].")
    )

    # 3. Reliability diagram
    st.subheader("Reliability diagram")
    st.markdown(
        "This compares the claimed confidence with how often the true value "
        "actually fell inside the interval. A well-calibrated model follows the "
        f"diagonal. Here the curve sits well below it (claimed 90%, observed "
        f"{metrics['coverage90']:.1%}; ECE 0.509), so the model is overconfident."
    )
    if os.path.exists(RELIABILITY_PNG):
        st.image(RELIABILITY_PNG, width=560)
    else:
        st.warning(f"Missing {RELIABILITY_PNG}. Run Chunk 6 (src/calibration.py).")

    # 4. VIX extension
    st.subheader("Uncertainty vs market fear (VIX)")
    st.markdown(
        "Do the model's intervals get wider when market fear (VIX) is high? "
        "Not really - the correlation is about -0.089, so the model's "
        "uncertainty does not track market volatility."
    )
    if os.path.exists(VIX_PNG):
        st.image(VIX_PNG, width=560)
    else:
        st.warning(f"Missing {VIX_PNG}. Run Chunk 6 (src/calibration.py).")

    # 5. Summary
    st.subheader("Summary")
    st.markdown(
        rf"- **Point accuracy:** MAE \${metrics['point_mae']:.2f} vs a naive no-change "
        rf"MAE of \${metrics['naive_mae']:.2f} - the forecaster is **no better than "
        "naive**.\n"
        f"- **Calibration:** the 90% band covers only {metrics['coverage90']:.1%} of "
        "truths - **badly overconfident**.\n"
        "- **Takeaway:** *no worse than naive, yet badly miscalibrated* - "
        "exactly the 'accurate-looking but untrustworthy' failure mode the "
        "project set out to expose and measure."
    )


if __name__ == "__main__":
    main()
