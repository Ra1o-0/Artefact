# Chunk 5 part 2 -- Monte Carlo dropout prediction intervals.
#
# I take the SAME trained regression LSTM from train_regression.py and turn its
# single-number forecast into a prediction interval by keeping dropout active at
# inference time and running 100 forward passes on each input.
#
# Normally Keras switches dropout off during inference, giving one fixed output.
# Here I force it to stay on. Because dropout randomly silences different neurons
# on each pass, the 100 outputs come out slightly different from each other.
# That spread is the model's uncertainty estimate:
#   - mean of the 100 outputs  -> the central forecast
#   - 5th and 95th percentiles -> the lower and upper edges of the 90% band
#
# The one pass with dropout OFF is also kept as the point-forecast baseline that
# Chunk 6 compares the interval against.
#
# The most important correctness check in the whole project lives here.
# If two passes on the same input come out identical, dropout is not actually
# active and the whole method is silently a no-op. I prove it is working by
# asserting two passes differ BEFORE producing any interval. If they do not,
# the script crashes loudly rather than producing a fake result.
#
# All outputs are converted from scaled-return space back to interpretable units
# using the train-only target scaler, then to price using today's actual close:
#     next_price = close_today * (1 + return_forecast)
#
# Run modes:
#     py src/mc_dropout.py               # default: prove dropout is active + show a few
#                                        #          example intervals, then STOP (no file)
#     py src/mc_dropout.py --stage full  # also write data/processed/test_intervals.csv

import argparse
import os

import numpy as np
import pandas as pd
import joblib

from train_baseline import make_sequences, set_seeds, LOOKBACK_DAYS
from train_regression import (
    load_regression_splits, PROCESSED_DIR, REGRESSION_MODEL_PATH, TARGET_SCALER_PATH,
)

# Configuration
MC_PASSES = 100                              # number of stochastic forward passes per input
BAND_LOW_PCT, BAND_HIGH_PCT = 5, 95         # gives a 90% central prediction band
INTERVALS_PATH = os.path.join(PROCESSED_DIR, "test_intervals.csv")


# 1. Build the aligned test sequences

# I build the 3D sequence array for the test split, and alongside it a table of
# actuals (today's close, actual next-day return, actual next close) so each
# forecast has something concrete to be judged against.
# The slice(LOOKBACK_DAYS - 1, None) alignment is needed because each sequence's
# label belongs to the LAST day of the window, not the first day.
def build_test_inputs():
    splits = load_regression_splits()
    test = splits["test"]
    feature_table = test["feature_table"]

    # Pass the returns to make_sequences so it can run; the y output is unused here.
    feature_windows, _ = make_sequences(feature_table, test["next_return"])

    last_day_slice = slice(LOOKBACK_DAYS - 1, None)  # align actuals to the last day of each window
    actuals_table = pd.DataFrame({
        "close_today": test["close_today"].iloc[last_day_slice].to_numpy(),
        "actual_next_return": test["next_return"].iloc[last_day_slice].to_numpy(),
    }, index=feature_table.index[last_day_slice])
    actuals_table.index.name = "date"
    actuals_table["actual_next_close"] = actuals_table["close_today"] * (1 + actuals_table["actual_next_return"])

    assert len(actuals_table) == len(feature_windows), "actuals/sequences misaligned"
    return feature_windows.astype("float32"), actuals_table


# 2. The dropout-is-active proof

# Two forward passes on the SAME input with training=True must produce different
# numbers. If they do not, dropout is silently off and every MC pass would be
# identical -- making the spread zero and the whole method meaningless.
# I run this check before producing any interval so the failure is obvious.
def prove_dropout_active(model, feature_windows):
    one_input = feature_windows[:1]                           # take just the first test input
    a = float(model(one_input, training=True).numpy().ravel()[0])   # training=True forces dropout to stay on
    b = float(model(one_input, training=True).numpy().ravel()[0])
    print("\n--- dropout-active proof (same input, dropout ON, scaled output) ---")
    print(f"pass A = {a:+.6f}")
    print(f"pass B = {b:+.6f}")
    print(f"differ by {abs(a - b):.6f}")
    if np.allclose(a, b):   # np.allclose returns True if the two values are numerically indistinguishable
        raise RuntimeError(
            "Dropout is NOT active at inference (two mc_passes_scaled identical) - MC dropout "
            "is silently broken. Ensure the model is called with training=True and "
            "that the architecture contains a Dropout layer.")
    print("OK: dropout is genuinely active at inference.")


# 3. Point forecast and MC interval

# One pass with training=False gives the deterministic point forecast (dropout off).
# Then n_passes passes with training=True are stacked into a matrix of shape
# (n_passes, n_inputs). The mean and percentiles across the passes axis give the
# interval. Everything here is still in scaled-return space; back_to_returns()
# undoes the target scaler to get actual return values.
def mc_forecast(model, feature_windows, target_scaler, n_passes=MC_PASSES):
    def back_to_returns(scaled_2d):
        return target_scaler.inverse_transform(scaled_2d.reshape(-1, 1)).ravel()  # undo the target scaler, flatten to 1D

    # Point forecast: dropout OFF (deterministic, same output every time).
    point_scaled = model(feature_windows, training=False).numpy().ravel()
    point_return = back_to_returns(point_scaled)

    # MC dropout: run n_passes stochastic forward passes and stack the results.
    mc_passes_scaled = np.stack([
        model(feature_windows, training=True).numpy().ravel() for _ in range(n_passes)
    ], axis=0)                               # shape (n_passes, n_inputs), still in scaled space

    mc_passes_returns = np.stack([back_to_returns(p) for p in mc_passes_scaled], axis=0)
    return {
        "point_return": point_return,
        "mc_mean_return": mc_passes_returns.mean(axis=0),                              # mean across the 100 passes
        "mc_std_return": mc_passes_returns.std(axis=0),                                # spread -- the uncertainty estimate
        "lower_return": np.percentile(mc_passes_returns, BAND_LOW_PCT, axis=0),        # 5th percentile -> lower band edge
        "upper_return": np.percentile(mc_passes_returns, BAND_HIGH_PCT, axis=0),       # 95th percentile -> upper band edge
    }


# Convert return-space forecasts to price units and bundle everything into one
# output table alongside the actuals. Price formula: close_today * (1 + return).
def assemble(actuals_table, forecasts):
    close_prices = actuals_table["close_today"].to_numpy()
    output_table = actuals_table.copy()
    output_table["point_return"] = forecasts["point_return"]
    output_table["point_price"] = close_prices * (1 + forecasts["point_return"])
    output_table["mc_mean_return"] = forecasts["mc_mean_return"]
    output_table["mc_mean_price"] = close_prices * (1 + forecasts["mc_mean_return"])
    output_table["mc_std_return"] = forecasts["mc_std_return"]
    output_table["lower_return"] = forecasts["lower_return"]
    output_table["upper_return"] = forecasts["upper_return"]
    output_table["lower5_price"] = close_prices * (1 + forecasts["lower_return"])
    output_table["upper95_price"] = close_prices * (1 + forecasts["upper_return"])
    output_table["interval_width_price"] = output_table["upper95_price"] - output_table["lower5_price"]  # how wide the band is in price units
    return output_table


# Verification helper

# Prints the first k rows of the interval table so I can see at a glance whether
# the actual price fell inside or outside the band each day. Also prints the
# overall empirical coverage as a preview -- the rigorous version is Chunk 6.
def show_examples(output_table, k=5):
    print(f"\n--- first {k} test-day {BAND_HIGH_PCT - BAND_LOW_PCT}% prediction intervals (£/$) ---")
    for date, row in output_table.head(k).iterrows():
        inside = row["lower5_price"] <= row["actual_next_close"] <= row["upper95_price"]  # did the actual price land in the band?
        print(f"{date.date()}  forecast {row['mc_mean_price']:7.2f}  "
              f"band [{row['lower5_price']:7.2f}, {row['upper95_price']:7.2f}]  "
              f"actual {row['actual_next_close']:7.2f}  "
              f"{'IN ' if inside else 'OUT'}")
    cov = float(((output_table['lower5_price'] <= output_table['actual_next_close']) &
                 (output_table['actual_next_close'] <= output_table['upper95_price'])).mean())  # fraction of days where actual was inside the band
    print(f"\nwhole-test empirical coverage of the {BAND_HIGH_PCT - BAND_LOW_PCT}% band: "
          f"{cov:.3f}  (claimed {(BAND_HIGH_PCT - BAND_LOW_PCT) / 100:.2f}; rigorous calibration "
          "is Chunk 6)")


# Orchestration

# Load the saved model and scaler, build test inputs, prove dropout is active,
# run the MC forecast, assemble the output table, and show examples.
# In "full" mode write the interval CSV to disk for Chunk 6 to read.
def main(stage):
    from tensorflow import keras

    model = keras.models.load_model(REGRESSION_MODEL_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)  # load the fitted target scaler saved in Chunk 5 part 1
    feature_windows, actuals_table = build_test_inputs()
    print(f"loaded model {REGRESSION_MODEL_PATH} | test inputs {feature_windows.shape}")

    # Prove the method is real BEFORE producing any interval.
    prove_dropout_active(model, feature_windows)

    # Seed so the set of 100 MC passes is reproducible run-to-run.
    # The passes still differ from each other -- that variation IS the uncertainty.
    set_seeds()
    forecasts = mc_forecast(model, feature_windows, target_scaler)
    output_table = assemble(actuals_table, forecasts)

    show_examples(output_table)

    if stage == "verify":
        print("\n[STOP] Intervals look sane? Re-run with `--stage full` to write "
              f"{INTERVALS_PATH} for Chunk 6.")
        return

    output_table.to_csv(INTERVALS_PATH)
    print(f"\n[saved] per-day intervals -> {INTERVALS_PATH}  ({len(output_table)} rows)")
    print("[DONE] Chunk 5 part 2 — MC-dropout intervals produced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk 5 part 2: MC-dropout intervals.")
    parser.add_argument(
        "--stage",
        choices=["verify", "full"],
        default="verify",
        help="'verify' (default) proves dropout is active + shows examples then stops; "
             "'full' also writes data/processed/test_intervals.csv.",
    )
    args = parser.parse_args()
    main(args.stage)
