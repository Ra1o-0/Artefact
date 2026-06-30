# Chunk 4 -- evaluation module for the baseline LSTM.
#
# Chunk 3 trained the model but only printed a last-epoch accuracy, which is
# not the real accuracy. Early stopping rewound the weights to the best
# validation epoch, so the model on disk is different from what was printed.
# Here I load the saved model and report proper held-out metrics.
#
# What I report:
#   - accuracy, precision, recall, and F1 (all with "up" as the positive class)
#   - a 2x2 confusion matrix showing right and wrong predictions for each direction
#   - a majority-class baseline: always predict the most common class in the
#     training labels. This is the trivial bar a useful model must clear.
#
# Leakage discipline:
#   - The majority class is taken from TRAIN labels only. Using the test
#     distribution to pick it would be peeking at the answer.
#   - Test sequences are built the same way as training sequences, so the
#     labels are directly comparable.
#   - Nothing is refit or re-scaled here.
#
# This module is designed to be reusable: the metric helpers return dicts and
# evaluate() takes a model path and feature list as arguments, so Chunk 6 can
# call the same code for the VIX-enhanced model and compare the two.
#
# Run modes (see the bottom of the file):
#     py src/evaluate.py                 # baseline (price-only) model on TEST
#     py src/evaluate.py --split val     # same, on the validation split instead

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Import the same feature columns and sequencing logic used in Chunk 3 so that
# what we evaluate is built identically to what the model was trained on.
from features import PRICE_FEATURES, TARGET, PROCESSED_DIR
from train_baseline import make_sequences, LOOKBACK_DAYS, BASELINE_MODEL_PATH

# A predicted probability of 0.5 or above is classified as "up".
DECISION_THRESHOLD = 0.5

# "up" (1) is the positive class for precision, recall, and F1.
POSITIVE_LABEL = 1


# 1. Load one split's features and labels

# Parametrised by `features` so the same function works for the price-only
# model now and the price+VIX model in Chunk 6.
def load_split(features, split):
    path = os.path.join(PROCESSED_DIR, f"{split}_features.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df[features], df[TARGET]  # return feature table and label column separately


# 2. Metric helpers (return dicts so Chunk 6 can build a comparison table)

# Standard binary classification metrics. zero_division=0 keeps precision and
# recall defined even if the model never predicts one of the classes, rather
# than crashing.
def compute_metrics(actual_labels, predicted_labels):
    return {
        "accuracy": accuracy_score(actual_labels, predicted_labels),
        "precision": precision_score(actual_labels, predicted_labels, pos_label=POSITIVE_LABEL,
                                     zero_division=0),
        "recall": recall_score(actual_labels, predicted_labels, pos_label=POSITIVE_LABEL,
                               zero_division=0),
        "f1": f1_score(actual_labels, predicted_labels, pos_label=POSITIVE_LABEL,
                       zero_division=0),
        "n": int(len(actual_labels)),
        "n_up": int(np.sum(actual_labels == 1)),    # count of actual up-days in the split
        "n_down": int(np.sum(actual_labels == 0)),  # count of actual down-days
    }


# The majority-class baseline is the simplest possible strategy: always predict
# whichever direction was more common in the training data. I score it on the
# test labels to get a fair comparison against the model.
def majority_baseline(train_label_values, actual_labels):
    train_up_rate = float(np.mean(train_label_values))                     # fraction of up-days in training
    majority_class = 1 if train_up_rate >= 0.5 else 0                     # pick whichever class appeared more often
    predicted_labels = np.full_like(actual_labels, fill_value=majority_class)  # predict that same class for every test row
    return {
        "majority_class": majority_class,
        "train_up_rate": train_up_rate,
        "accuracy": accuracy_score(actual_labels, predicted_labels),
    }


# A 2x2 matrix: rows are actual classes, columns are predicted classes,
# ordered [down(0), up(1)]. The diagonal is correct predictions.
def confusion(actual_labels, predicted_labels):
    return confusion_matrix(actual_labels, predicted_labels, labels=[0, 1])


# 3. Report formatting

# Prints all the metrics, the confusion matrix with labels, and a verdict on
# whether the model beats the majority-class baseline.
def print_report(split, metrics, confusion_counts, baseline):
    print(f"\n--- evaluation: baseline LSTM on {split.upper()} split ---")
    print(f"sequences evaluated : {metrics['n']}  "
          f"(actual up {metrics['n_up']} / down {metrics['n_down']})")
    print(f"threshold           : {DECISION_THRESHOLD}  (prob >= threshold -> up)")

    print("\nmetrics (positive class = up = 1):")
    print(f"  accuracy  : {metrics['accuracy']:.3f}")
    print(f"  precision : {metrics['precision']:.3f}")
    print(f"  recall    : {metrics['recall']:.3f}")
    print(f"  f1        : {metrics['f1']:.3f}")

    # Labelled 2x2: rows = actual, cols = predicted, order [down, up].
    true_down, false_up = int(confusion_counts[0, 0]), int(confusion_counts[0, 1])
    false_down, true_up = int(confusion_counts[1, 0]), int(confusion_counts[1, 1])
    print("\nconfusion matrix (rows = actual, cols = predicted):")
    print(f"                 pred down   pred up")
    print(f"  actual down  : {true_down:9d} {false_up:9d}")
    print(f"  actual up    : {false_down:9d} {true_up:9d}")
    print(f"  (cells sum to {true_down + false_up + false_down + true_up}, = sequences evaluated)")

    print("\nmajority-class baseline (trivial bar to clear):")
    class_name = "up" if baseline["majority_class"] == 1 else "down"
    print(f"  train up-rate    : {baseline['train_up_rate']:.3f} "
          f"-> majority class = {class_name} ({baseline['majority_class']})")
    print(f"  baseline accuracy: {baseline['accuracy']:.3f}  "
          f"(always predict {class_name})")
    accuracy_gap = metrics["accuracy"] - baseline["accuracy"]
    verdict = "beats" if accuracy_gap > 0 else ("ties" if accuracy_gap == 0 else "below")
    print(f"  model {verdict} baseline by {accuracy_gap:+.3f} accuracy")


# 4. Main evaluation function

# Loads the saved model, builds sequences the same way Chunk 3 did, gets
# predictions, runs all the metric helpers, and prints the report. Returns a
# dict so Chunk 6 can call this for both models and compare the results.
def evaluate(model_path, features, split="test"):
    from tensorflow import keras

    # Build sequences for the chosen split using the same function as Chunk 3.
    feature_table, label_series = load_split(features, split)
    feature_windows, window_labels = make_sequences(feature_table, label_series)
    actual_labels = window_labels.astype(int)

    # The majority baseline needs training labels to decide which class to predict.
    # I window them the same way for consistency, though only the label values matter.
    train_feature_table, train_labels = load_split(features, "train")
    _, train_window_labels = make_sequences(train_feature_table, train_labels)
    train_label_values = train_window_labels.astype(int)

    # Load the saved model (best-val-epoch weights, not the last epoch from Chunk 3).
    model = keras.models.load_model(model_path)
    predicted_probabilities = model.predict(feature_windows, verbose=0).ravel()  # ravel() flattens to a 1D array
    predicted_labels = (predicted_probabilities >= DECISION_THRESHOLD).astype(int)  # apply threshold to get 0/1 predictions

    metrics = compute_metrics(actual_labels, predicted_labels)
    confusion_counts = confusion(actual_labels, predicted_labels)
    baseline = majority_baseline(train_label_values, actual_labels)

    print_report(split, metrics, confusion_counts, baseline)

    return {
        "split": split,
        "metrics": metrics,
        "confusion_matrix": confusion_counts,
        "baseline": baseline,
    }


def main(split):
    evaluate(BASELINE_MODEL_PATH, PRICE_FEATURES, split=split)
    print("\n[DONE] Chunk 4 evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk 4 evaluation module.")
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Which held-out split to evaluate the baseline model on "
             "(default: test).",
    )
    args = parser.parse_args()
    main(args.split)
