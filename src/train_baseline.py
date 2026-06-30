# Chunk 3 -- baseline LSTM (price features only).
#
# Here I train the baseline direction classifier: an LSTM that looks at a short
# window of recent days using only the six price-derived features from Chunk 2
# and predicts whether tomorrow's price goes up (1) or down (0). VIX is left
# out deliberately -- it gets added in Chunk 5's enhanced model.
#
# The key new idea in this chunk is sequencing. An LSTM does not take flat rows;
# it expects a 3D block shaped (samples, timesteps, features). So I slide a
# window of LOOKBACK_DAYS consecutive days across each split to produce the
# training samples. Each sample's label is the target of the last day in that
# window.
#
# Leakage discipline carried forward from Chunks 1 and 2:
#   - Sequences are built within each split independently. No window ever
#     contains rows from a different split, so the frozen boundary is respected.
#   - The scaler from Chunk 2 is already baked into the feature CSVs; I never
#     re-scale anything here.
#   - The test split is never passed to training or validation. Its sequences
#     are built only so the shape report is complete.
#
# Run modes (see the bottom of the file):
#     py src/train_baseline.py                 # default: build sequences + verify, then STOP
#     py src/train_baseline.py --stage full    # also set seeds, train, and save the model

import argparse
import os
import random

import numpy as np
import pandas as pd

# Reuse the canonical column groups / split names from Chunk 2 so the feature
# set is single-sourced and can never drift out of step with the saved CSVs.
from features import PRICE_FEATURES, TARGET, SPLIT_NAMES

# Configuration
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_lstm.keras")

# Each LSTM sample is LOOKBACK_DAYS consecutive days of features.
LOOKBACK_DAYS = 10

# Single fixed seed so the model trains identically every run.
RANDOM_SEED = 42

# Training hyperparameters -- kept small because the dataset (~1,750 train rows)
# is too small to justify a large model; early stopping handles overfitting.
LSTM_UNITS = 32
DROPOUT_RATE = 0.2
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 8   # stop training if val loss does not improve for this many epochs


# 0. Reproducibility

# I seed Python, NumPy, and TensorFlow all at once before any model operation.
# Without this, weight initialisation and dropout patterns vary between runs,
# making results impossible to reproduce in the viva.
def set_seeds(seed=RANDOM_SEED):
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# 1. Load the scaled feature splits

# I read the Chunk-2 feature CSVs and pull out only the price feature columns.
# VIX columns are present in the CSV but I simply do not select them here.
def load_features():
    splits = {}
    for name in SPLIT_NAMES:
        path = os.path.join(PROCESSED_DIR, f"{name}_features.csv")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "Date"
        splits[name] = (df[PRICE_FEATURES], df[TARGET])  # return features and labels separately
    return splits


# 2. Build sliding-window sequences (within a single split)

# This turns a flat table of rows into the 3D blocks an LSTM needs.
# Sample i is rows [i : i+window] of the features; its label is the target
# of the last day in that window. By building sequences within each split
# separately, no window ever crosses the frozen train/val/test boundary.
# The trade-off is that the first (window - 1) rows of each split are lost
# because they do not have enough preceding rows to fill a full window.
def make_sequences(feature_table, label_series, window=LOOKBACK_DAYS):
    feature_array = feature_table.to_numpy(dtype="float32")   # convert to NumPy for fast slicing
    label_array = label_series.to_numpy(dtype="float32")

    sequences, labels = [], []
    for i in range(len(feature_array) - window + 1):
        sequences.append(feature_array[i:i + window])       # a window-sized block of feature rows
        labels.append(label_array[i + window - 1])          # label is the target of the last row in the window

    feature_windows = np.asarray(sequences, dtype="float32")  # shape: (num_samples, window, num_features)
    window_labels = np.asarray(labels, dtype="float32")
    return feature_windows, window_labels


# 3. Model definition

# A deliberately small architecture: one LSTM layer reads the sequence, dropout
# randomly switches off some neurons during training to reduce overfitting, and
# a single sigmoid output gives a probability between 0 and 1 (the chance of
# the price going up tomorrow).
def build_model(feature_count, window=LOOKBACK_DAYS):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(window, feature_count)),   # expects (timesteps, features) per sample
        layers.LSTM(LSTM_UNITS),                      # reads the sequence and outputs a fixed-size vector
        layers.Dropout(DROPOUT_RATE),                 # randomly zeros neurons during training to reduce overfitting
        layers.Dense(1, activation="sigmoid"),        # squashes to a probability: above 0.5 = predict up
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",         # standard loss for a 0/1 classification problem
                  metrics=["accuracy"])
    return model


# 4. Training

# I fit on training sequences and monitor validation loss after each epoch.
# EarlyStopping halts training when val_loss stops improving and restores the
# weights from the best epoch seen, so the saved model is not the last one but
# the best one.
def train(model, train_windows, train_labels, val_windows, val_labels):
    from tensorflow.keras.callbacks import EarlyStopping

    stopper = EarlyStopping(monitor="val_loss",
                            patience=PATIENCE,
                            restore_best_weights=True)  # rewind to the best-val-loss checkpoint on stopping

    history = model.fit(
        train_windows, train_labels,
        validation_data=(val_windows, val_labels),  # val data is only used to monitor, never to update weights
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[stopper],
        verbose=2,
    )
    return history


# 5. Save

# Persist the trained model so Chunks 4 and 6 can reload it without retraining.
def save(model):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(BASELINE_MODEL_PATH)
    print(f"[saved] baseline model -> {BASELINE_MODEL_PATH}")


# Verification helper

# Prints the shape of the sequences and the proportion of up-days in each split
# so I can check the windowing produced sensible numbers before training.
def verify(windows_by_split):
    print("\n--- sequence build (within-split, "
          f"window={LOOKBACK_DAYS}, features={len(PRICE_FEATURES)}) ---")
    for name in SPLIT_NAMES:
        feature_windows, window_labels, row_count = windows_by_split[name]
        up_rate = float(window_labels.mean())
        print(f"{name:5s} : {row_count:4d} rows -> {len(window_labels):4d} sequences  |  "
              f"X shape {feature_windows.shape}  |  up-rate {up_rate:.3f}")
    print(f"(each split loses its first {LOOKBACK_DAYS - 1} rows: no full window.)")


# Orchestration

# Same two-stage pattern as Chunks 1 and 2. "verify" builds and inspects the
# sequences then stops; "full" also seeds, trains, and saves the model.
def main(stage):
    splits = load_features()

    windows_by_split = {}
    for name in SPLIT_NAMES:
        feature_table, label_series = splits[name]
        feature_windows, window_labels = make_sequences(feature_table, label_series)
        windows_by_split[name] = (feature_windows, window_labels, len(feature_table))

    verify(windows_by_split)

    if stage == "verify":
        print("\n[STOP] Sequences look sane? Re-run with `--stage full` to "
              "train and save the baseline model.")
        return

    # Full stage: seed BEFORE building the model so weight initialisation is reproducible.
    set_seeds()

    train_windows, train_labels, _ = windows_by_split["train"]
    val_windows, val_labels, _ = windows_by_split["val"]

    model = build_model(feature_count=len(PRICE_FEATURES))
    model.summary()

    history = train(model, train_windows, train_labels, val_windows, val_labels)

    # Quick sanity print only -- rigorous metrics are Chunk 4's job. Test set untouched.
    train_accuracy = history.history["accuracy"][-1]
    val_accuracy = history.history["val_accuracy"][-1]
    epochs_run = len(history.history["loss"])
    print("\n--- training summary (sanity only; full metrics in Chunk 4) ---")
    print(f"epochs run (early-stopped): {epochs_run}")
    print(f"final train accuracy      : {train_accuracy:.3f}")
    print(f"final val accuracy        : {val_accuracy:.3f}")

    save(model)
    print("\n[DONE] Chunk 3 baseline LSTM complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk 3 baseline LSTM.")
    parser.add_argument(
        "--stage",
        choices=["verify", "full"],
        default="verify",
        help="'verify' (default) builds sequences + reports shapes then stops; "
             "'full' also seeds, trains (early-stopped), and saves the model.",
    )
    args = parser.parse_args()
    main(args.stage)
