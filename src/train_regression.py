# Chunk 5 part 1 -- regression LSTM (point-forecast baseline for the interval work).
#
# The dissertation pivoted from direction classification to uncertainty-calibrated
# forecasting. Calibration ("does the truth fall inside the X% interval X% of the
# time?") needs a continuous forecast -- a 0/1 classifier cannot produce one.
# So here I train a small regression LSTM that forecasts the next-day RETURN of
# AAPL. Its plain prediction (dropout switched off) is the point-forecast baseline.
# The second part of Chunk 5, mc_dropout.py, takes the SAME trained model and
# produces prediction intervals by keeping dropout active at inference.
#
# Why predict the return rather than the price level?
# Training prices (2015-21) sit far below test prices (2023-24). A price-level
# target scaled on training data only would force the model to extrapolate badly
# because prices are not stationary over time. Daily returns are roughly stable
# (stationary), which is the standard finance choice. A return still gives a
# price band downstream via: next_price = today_close * (1 + return).
#
# Leakage discipline carried forward from Chunks 1-3:
#   - Features come from the Chunk-2 CSVs, already scaled by the train-only
#     feature scaler. I never refit that scaler here.
#   - The only newly fitted object is a small target scaler for the return values,
#     fitted on training returns only and applied forward.
#   - Sequences are built within each split using Chunk 3's make_sequences, so
#     no window crosses the frozen boundary.
#   - VIX is not a model input -- it stays the external validation signal that
#     Chunk 6 checks the model's uncertainty against.
#
# Run modes (see the bottom of the file):
#     py src/train_regression.py               # default: build targets + sequences, verify, STOP
#     py src/train_regression.py --stage full  # also seed, train, and save model + target scaler

import argparse
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Reuse column groups, sequencing, seeding, and hyperparameters from earlier
# chunks so this model cannot drift out of step with the frozen pipeline.
from features import PRICE_FEATURES, SPLIT_NAMES
from train_baseline import (
    make_sequences, set_seeds, load_features,
    LOOKBACK_DAYS, RANDOM_SEED, LSTM_UNITS, DROPOUT_RATE, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
)

# Configuration
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
REGRESSION_MODEL_PATH = os.path.join(MODELS_DIR, "regression_lstm.keras")
TARGET_SCALER_PATH = os.path.join(MODELS_DIR, "target_scaler.pkl")


# 1. Raw Close prices and next-day returns

# I stitch the three raw split CSVs into one continuous Close series and compute
# the next-day return on that continuous series. Doing it continuously means the
# return at a split boundary (e.g. the last training day looking one day into val)
# is defined -- the same convention Chunk 2 used for the classifier target.
def load_raw_close():
    frames = []
    for name in SPLIT_NAMES:
        path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "Date"
        frames.append(df[["Close"]])

    closes = pd.concat(frames).sort_index()
    closes = closes[~closes.index.duplicated(keep="first")]  # drop any duplicate dates that appear in both splits

    close_table = pd.DataFrame(index=closes.index)
    close_table["close"] = closes["Close"].astype("float64")
    close_table["next_return"] = closes["Close"].shift(-1) / closes["Close"] - 1.0  # (tomorrow / today) - 1
    return close_table


# 2. Assemble per-split (features, raw target, close today)

# For each split I pair up the scaled price features from Chunk 2 with the
# matching next-day return values from load_raw_close(). The assertions check
# that every feature date has a defined return -- if alignment is broken this
# would silently produce NaN targets and corrupt training.
def load_regression_splits():
    feat_splits = load_features()   # name -> (feature_table[PRICE_FEATURES], classifier_target)
    raw = load_raw_close()

    splits = {}
    for name in SPLIT_NAMES:
        feature_table = feat_splits[name][0]
        next_return = raw["next_return"].reindex(feature_table.index)   # align returns to the feature dates
        close_today = raw["close"].reindex(feature_table.index)

        assert not next_return.isna().any(), (
            f"{name}: {int(next_return.isna().sum())} feature dates lack a next-day "
            "return — date alignment with raw Close is broken")
        assert not close_today.isna().any(), f"{name}: missing close_today after align"

        splits[name] = {
            "feature_table": feature_table,
            "next_return": next_return,
            "close_today": close_today,
        }
    return splits


# Re-assert the frozen ordering train < val < test on the feature dates.
# This catches any date-alignment bug introduced in the steps above.
def assert_chronological(splits):
    tr = splits["train"]["feature_table"].index
    va = splits["val"]["feature_table"].index
    te = splits["test"]["feature_table"].index
    assert tr.max() < va.min(), "train overlaps/post-dates val"
    assert va.max() < te.min(), "val overlaps/post-dates test"


# 3. Train-only target scaler

# Return values need scaling just like the input features do, so the model
# sees numbers of a similar magnitude. I fit only on training returns and
# apply the same fixed scale to val and test -- same principle as Chunk 2.
def fit_target_scaler(train_returns):
    scaler = StandardScaler()
    scaler.fit(np.asarray(train_returns, dtype="float64").reshape(-1, 1))  # reshape to 2D because StandardScaler expects a matrix
    return scaler


def scale_returns(scaler, returns):
    arr = np.asarray(returns, dtype="float64").reshape(-1, 1)
    return scaler.transform(arr).ravel().astype("float32")  # ravel() flattens back to a 1D array


# 4. Model definition

# Same small shape as the Chunk 3 baseline but with a linear output and MSE
# loss instead of sigmoid and binary cross-entropy. Linear activation means the
# output is an unbounded number (a scaled return), not a probability.
# The Dropout layer is what mc_dropout.py later keeps switched on at inference
# to produce a spread of predictions that forms the uncertainty interval.
def build_regression_model(n_features, window=LOOKBACK_DAYS):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(window, n_features)),
        layers.LSTM(LSTM_UNITS),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(1, activation="linear"),  # linear = no squashing, output is a raw number
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])  # MSE = mean squared error; MAE = mean absolute error
    return model


# 5. Training

# Identical setup to Chunk 3: fit on training sequences, monitor val loss,
# early-stop and restore the best weights seen. Test split is never touched.
def train(model, train_windows, train_labels, val_windows, val_labels):
    from tensorflow.keras.callbacks import EarlyStopping

    stopper = EarlyStopping(monitor="val_loss",
                            patience=PATIENCE,
                            restore_best_weights=True)  # rewind weights to the best-val-loss epoch on stopping

    history = model.fit(
        train_windows, train_labels,
        validation_data=(val_windows, val_labels),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[stopper],
        verbose=2,
    )
    return history


# 6. Save

# Save both the model and the target scaler so mc_dropout.py and Chunk 6 can
# reload them without retraining.
def save(model, target_scaler):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(REGRESSION_MODEL_PATH)
    joblib.dump(target_scaler, TARGET_SCALER_PATH)  # joblib.dump saves the scaler object to disk
    print(f"[saved] regression model -> {REGRESSION_MODEL_PATH}")
    print(f"[saved] target scaler    -> {TARGET_SCALER_PATH}")


# Verification helper

# Prints raw return statistics and a scaling sanity check (scaled training
# returns should be near mean 0 and std 1), then shows sequence shapes for
# each split so I can confirm everything looks right before training.
def verify(splits, target_scaler):
    print("\n--- next-day return target (raw, before scaling) ---")
    for name in SPLIT_NAMES:
        r = splits[name]["next_return"]
        print(f"{name:5s} : n={len(r):4d}  mean={r.mean():+.5f}  std={r.std():.5f}  "
              f"min={r.min():+.4f}  max={r.max():+.4f}")

    scaled_train_target = scale_returns(target_scaler, splits["train"]["next_return"])
    print(f"\nscaled TRAIN target: mean={scaled_train_target.mean():+.3e}  std={scaled_train_target.std():.4f}  "
          "(expect ~0 and ~1)")

    print(f"\n--- sequence build (within-split, window={LOOKBACK_DAYS}, "
          f"features={len(PRICE_FEATURES)}) ---")
    for name in SPLIT_NAMES:
        feature_table = splits[name]["feature_table"]
        scaled_returns = scale_returns(target_scaler, splits[name]["next_return"])
        X_seq, y_seq = make_sequences(feature_table, pd.Series(scaled_returns, index=feature_table.index))
        print(f"{name:5s} : {len(feature_table):4d} rows -> {len(y_seq):4d} sequences  |  "
              f"X shape {X_seq.shape}")
    print(f"(each split loses its first {LOOKBACK_DAYS - 1} rows: no full window.)")
    print("chronological boundary train < val < test: OK")


# Orchestration

# Same two-stage pattern as earlier chunks. "verify" builds targets and
# sequences and prints the sanity checks then stops; "full" also trains and saves.
def main(stage):
    splits = load_regression_splits()
    assert_chronological(splits)

    # Target scaler is fit on TRAIN returns ONLY, then applied forward.
    target_scaler = fit_target_scaler(splits["train"]["next_return"])

    verify(splits, target_scaler)

    if stage == "verify":
        print("\n[STOP] Targets/sequences sane? Re-run with `--stage full` to "
              "train and save the regression model.")
        return

    # Full stage: seed BEFORE building the model so weight initialisation is reproducible.
    set_seeds()

    def windows_for(name):
        feature_table = splits[name]["feature_table"]
        scaled_returns = scale_returns(target_scaler, splits[name]["next_return"])
        return make_sequences(feature_table, pd.Series(scaled_returns, index=feature_table.index))

    train_windows, train_labels = windows_for("train")
    val_windows, val_labels = windows_for("val")

    model = build_regression_model(n_features=len(PRICE_FEATURES))
    model.summary()

    history = train(model, train_windows, train_labels, val_windows, val_labels)

    val_mae = history.history["val_mae"][-1]
    epochs_run = len(history.history["loss"])
    print("\n--- training summary (scaled-return space; intervals are Chunk 5 pt 2) ---")
    print(f"epochs run (early-stopped): {epochs_run}")
    print(f"final val MAE (scaled)    : {val_mae:.4f}")

    save(model, target_scaler)
    print("\n[DONE] Chunk 5 part 1 — regression LSTM trained and saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk 5 part 1: regression LSTM.")
    parser.add_argument(
        "--stage",
        choices=["verify", "full"],
        default="verify",
        help="'verify' (default) builds targets + sequences, reports, then stops; "
             "'full' also seeds, trains (early-stopped), and saves model + scaler.",
    )
    args = parser.parse_args()
    main(args.stage)
