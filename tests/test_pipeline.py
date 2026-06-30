# Automated tests for the uncertainty-calibrated LSTM pipeline.
#
# This suite backs Chapter 5 (Testing) of the dissertation. There are two kinds:
#
#   Unit tests -- exercise individual pure functions in isolation, on tiny
#   synthetic inputs with hand-computed expected outputs. They depend on no
#   files and no model, so they run fast.
#
#   Verification and integration tests -- re-check the project's key invariants
#   over the FROZEN artefacts on disk (the chronological split, the train-only
#   scaler, and the per-day interval table). They only READ those files: nothing
#   here retrains a model, refits a scaler, or regenerates any result, so
#   running the suite cannot perturb the pipeline or change any reported number.
#
# Run from the project root:
#     py -m pytest tests/ -v
#
# Each test's comment names the dissertation section it supports.

import os

import numpy as np
import pandas as pd
import pytest

# These imports work because conftest.py puts src/ on the path.
# None of them pull in TensorFlow at import time (TF is imported lazily inside
# the functions that need it), so the unit tests stay fast.
from train_baseline import make_sequences
from features import engineer, build_target, TARGET, ALL_FEATURES, PRICE_FEATURES
from calibration import (
    z_value_for, coverage_at, expected_calibration_error, spearman_correlation,
    load_intervals, point_vs_interval, reliability_table,
)

PROCESSED_DIR = os.path.join("data", "processed")


# Unit tests -- pure functions, synthetic inputs, known answers

def test_make_sequences_shapes_and_label_alignment():
    # Section 5.2: the sliding window must produce (n - window + 1) samples,
    # each of shape (window, n_features), and each label must be the target of
    # the LAST day of its window -- not the first.
    feature_table = pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": [10, 11, 12, 13, 14]})
    label_series = pd.Series([0, 1, 0, 1, 0])

    feature_windows, window_labels = make_sequences(feature_table, label_series, window=3)

    # 5 rows, window 3 -> 3 sequences.
    assert feature_windows.shape == (3, 3, 2)
    assert window_labels.shape == (3,)
    # Label of sequence i is label_series[i + window - 1] -> label_series[2], label_series[3], label_series[4].
    assert list(window_labels) == [0, 1, 0]
    # First window is rows 0, 1, 2 of both columns.
    assert feature_windows[0].tolist() == [[0, 10], [1, 11], [2, 12]]


def test_engineer_daily_return_and_volume_change():
    # Section 5.2: the engineered daily return and volume change must equal the
    # simple percentage change of Close and Volume, and look only backward
    # (so the first row is NaN because there is no prior day to compare against).
    df = pd.DataFrame(
        {"Close": [100.0, 110.0, 121.0], "Volume": [10, 20, 40],
         "VIX_Close": [20.0, 22.0, 21.0]},
        index=pd.date_range("2020-01-01", periods=3),
    )
    out = engineer(df)

    assert np.isnan(out["ret_1"].iloc[0])               # no prior day -> undefined
    assert out["ret_1"].iloc[1] == pytest.approx(0.10)  # 100 -> 110 is a 10% return
    assert out["ret_1"].iloc[2] == pytest.approx(0.10)  # 110 -> 121 is also 10%
    assert out["volume_chg"].iloc[1] == pytest.approx(1.0)   # 10 -> 20 is 100% increase
    assert out["vix_chg"].iloc[2] == pytest.approx(21 / 22 - 1)


def test_build_target_is_strict_next_day_up():
    # Section 5.2: target is 1 only when tomorrow's close is STRICTLY higher.
    # A flat day (same price) counts as 0, and the final row has no next day
    # so its target is undefined (NaN).
    df = pd.DataFrame(
        {"Close": [100.0, 101.0, 100.0, 100.0]},
        index=pd.date_range("2020-01-01", periods=4),
    )
    out = build_target(df)

    assert out[TARGET].iloc[0] == 1      # 100 -> 101  (up)
    assert out[TARGET].iloc[1] == 0      # 101 -> 100  (down)
    assert out[TARGET].iloc[2] == 0      # 100 -> 100  (flat counts as not-up)
    assert pd.isna(out[TARGET].iloc[3])  # no next day -> undefined


def test_z_for_standard_normal_quantiles():
    # Section 5.2.5: z_value_for() must return the known standard-normal
    # quantiles. A 90% central interval leaves 5% in each tail, so the z is
    # the 95th quantile = 1.645. A 50% interval gives z = 0.674.
    assert z_value_for(0.90) == pytest.approx(1.6448536, abs=1e-5)
    assert z_value_for(0.50) == pytest.approx(0.6744898, abs=1e-5)


def test_coverage_at_matches_constructed_coverage():
    # Section 5.4: with mean 0 and std 1, the 90% band is +/-1.645. I place
    # 9 actuals safely inside and 1 far outside (value 5.0), so coverage must
    # come out as exactly 0.9.
    df = pd.DataFrame({
        "mc_mean_return": [0.0] * 10,
        "mc_std_return": [1.0] * 10,
        "actual_next_return": [0.0] * 9 + [5.0],  # last one is well outside +/-1.645
    })
    assert coverage_at(df, 0.90) == pytest.approx(0.9)


def test_expected_calibration_error_is_mean_gap():
    # Section 5.4: ECE is the mean ABSOLUTE gap between claimed and observed.
    # The signed mean gap must keep its sign so you can tell which direction
    # the miscalibration goes (negative = overconfident).
    rel = pd.DataFrame({"claimed": [0.5, 0.9], "observed": [0.4, 0.6]})
    out = expected_calibration_error(rel)
    # gaps = -0.1, -0.3 -> absolute mean = 0.2, signed mean = -0.2
    assert out["ece"] == pytest.approx(0.2)
    assert out["signed_mean_gap"] == pytest.approx(-0.2)


def test_spearman_perfect_monotonic():
    # Section 5.5: the dependency-free Spearman helper must give +1 for a
    # perfectly increasing relationship and -1 for a perfectly decreasing one.
    a = np.array([1, 2, 3, 4, 5])
    assert spearman_correlation(a, np.array([10, 20, 30, 40, 50])) == pytest.approx(1.0)
    assert spearman_correlation(a, np.array([50, 40, 30, 20, 10])) == pytest.approx(-1.0)


# Verification and integration tests -- over the FROZEN artefacts (read-only)

def _read_split(name):
    # Helper to load one of the frozen Chunk 1 split CSVs by name.
    return pd.read_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"),
                       index_col=0, parse_dates=True)


def test_chronological_split_has_no_leakage():
    # Section 5.2.2: every test date must post-date every val date, and every
    # val date must post-date every train date. This is the frozen no-leakage
    # boundary asserted in Chunk 1 and re-verified here over the saved files.
    train, val, test = _read_split("train"), _read_split("val"), _read_split("test")
    assert train.index.max() < val.index.min(), "train overlaps/post-dates val"
    assert val.index.max() < test.index.min(), "val overlaps/post-dates test"


def test_scaler_was_fit_on_train_only():
    # Section 5.2.2: the saved scaled feature files prove the StandardScaler
    # was fitted on training data only. Training features should be centred
    # near mean 0 and std 1. Val features should drift away from 0 because
    # the val distribution was not used when fitting the scale.
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train_features.csv"),
                        index_col=0, parse_dates=True)
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "val_features.csv"),
                      index_col=0, parse_dates=True)

    assert train[ALL_FEATURES].mean().abs().max() < 1e-6      # train features centred on 0
    assert train[ALL_FEATURES].std().sub(1).abs().max() < 0.05  # train feature std close to 1
    assert val[ALL_FEATURES].mean().abs().max() > 0.05         # val features drift (scaler not refit on them)


def test_intervals_table_is_well_formed():
    # Section 5.2.4: the frozen per-day interval table must exist, carry all
    # the required columns, and have the expected 368 test rows.
    df = load_intervals()
    assert len(df) == 368
    for col in ["close_today", "actual_next_close", "actual_next_return",
                "point_price", "mc_mean_price", "lower5_price", "upper95_price"]:
        assert col in df.columns


def test_interval_bounds_are_ordered():
    # Section 5.2.4: on every test day the lower band edge must sit at or
    # below the upper band edge. A valid prediction interval can never have
    # its lower bound above its upper bound.
    df = load_intervals()
    assert (df["lower5_price"] <= df["upper95_price"]).all()
    assert (df["lower_return"] <= df["upper_return"]).all()


def test_price_reconstruction_is_consistent():
    # Section 5.2.4: the dollar prices in the interval table must equal the
    # return-space values mapped through price = close_today * (1 + return).
    # This checks the assembly step in mc_dropout.py did not introduce errors.
    df = load_intervals()
    recon_actual = df["close_today"] * (1 + df["actual_next_return"])
    recon_point = df["close_today"] * (1 + df["point_return"])
    assert np.allclose(recon_actual, df["actual_next_close"])  # np.allclose checks all values are numerically close
    assert np.allclose(recon_point, df["point_price"])


def test_headline_calibration_numbers_reproduce():
    # Sections 5.3 and 5.4: the dissertation's headline numbers must reproduce
    # from the frozen table. Claimed 90% gives roughly 12% observed coverage
    # (badly overconfident), ECE roughly 0.509, and the point forecast is no
    # better than just predicting "tomorrow equals today".
    df = load_intervals()
    pv = point_vs_interval(df)
    ece = expected_calibration_error(reliability_table(df))

    assert pv["stored_band_coverage_90"] == pytest.approx(0.120, abs=0.01)
    assert pv["point_mae_price"] == pytest.approx(1.99, abs=0.05)
    assert pv["naive_mae_price"] == pytest.approx(1.98, abs=0.05)
    # No genuine edge over naive: point MAE within a few pence of naive.
    assert abs(pv["point_mae_price"] - pv["naive_mae_price"]) < 0.10
    assert ece["ece"] == pytest.approx(0.509, abs=0.02)
    assert ece["signed_mean_gap"] < 0   # negative signed gap means overconfident overall


# Behavioural test -- MC dropout must be genuinely stochastic at inference

def test_mc_dropout_is_active_at_inference():
    # Section 5.2.3: the most important correctness check in the whole project.
    # Two forward passes with dropout ON (training=True) on the SAME input
    # must produce different outputs. If they are identical then dropout is
    # silently switched off and every MC pass would be the same, making the
    # spread zero and the entire interval method meaningless.
    # pytest.importorskip skips this test automatically if TensorFlow is not
    # installed, so the fast unit tests above are never held up by it.
    pytest.importorskip("tensorflow")
    from tensorflow import keras
    from mc_dropout import build_test_inputs
    from train_regression import REGRESSION_MODEL_PATH

    model = keras.models.load_model(REGRESSION_MODEL_PATH)
    feature_windows, _ = build_test_inputs()
    x0 = feature_windows[:1]   # take just the first test input
    a = float(model(x0, training=True).numpy().ravel()[0])
    b = float(model(x0, training=True).numpy().ravel()[0])
    assert not np.isclose(a, b), "two dropout-ON passes identical -> dropout is a no-op"
