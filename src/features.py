# Chunk 2 -- feature engineering and train-only scaler.
#
# Raw prices are too noisy to feed straight into a model, so here I create
# better input signals: one-day returns, moving-average ratios, rolling
# volatility, volume change, and two VIX columns. I also build the next-day
# direction label (1 = price goes up, 0 = price goes down or flat).
#
# The two big leakage rules I follow here:
#   - Every feature looks only at past or current days, never at future rows.
#   - The StandardScaler (which normalises the feature values) is fitted on
#     training data only, then applied to val and test. If I fitted it on
#     everything I would be letting future data influence the training scale.
#
# Run modes (see the bottom of the file):
#     py src/features.py                 # default: engineer + verify, then STOP
#     py src/features.py --stage full    # also scale + assert + save

import argparse
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Configuration
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"

# Longest rolling window; the leading rows with no full window are dropped.
MAX_WINDOW = 10

# Column groups recorded for downstream chunks.
#   Chunk 3 (baseline) uses PRICE_FEATURES only.
#   Chunk 5 (enhanced) uses PRICE_FEATURES + VIX_FEATURES.
PRICE_FEATURES = ["ret_1", "ma5_ratio", "ma10_ratio", "vol_5", "vol_10", "volume_chg"]
VIX_FEATURES = ["vix_level", "vix_chg"]
ALL_FEATURES = PRICE_FEATURES + VIX_FEATURES
TARGET = "target"

SPLIT_NAMES = ["train", "val", "test"]


# 1. Load the frozen splits as one continuous, labelled series

# I stack all three splits into one continuous timeline so that rolling
# calculations (like a 10-day moving average) can look back across the
# train/val boundary without hitting a gap. That is fine -- a val row
# looking back into training days is using past data, which is not leakage.
# Each row keeps a _split tag so I can slice them apart again afterwards.
def load_aligned():
    parts = []
    for name in SPLIT_NAMES:
        path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        part = pd.read_csv(path, index_col=0, parse_dates=True)
        part.index.name = "Date"
        part["_split"] = name  # tag each row so we know which split it came from
        parts.append(part)

    combined = pd.concat(parts).sort_index()  # stack all three and sort by date

    # Sanity: stacking the splits must reproduce a strictly increasing,
    # duplicate-free date index (i.e. the original continuous series).
    assert combined.index.is_monotonic_increasing, "combined dates not ordered"  # dates must run oldest to newest
    assert not combined.index.has_duplicates, "duplicate dates across splits"    # no date should appear twice
    return combined


# 2. Feature engineering (all backward-looking)

# Every column here uses only past or current values, so no future information
# leaks into a row. pct_change() gives the percentage move from the previous row;
# rolling(n).mean() and rolling(n).std() look back over the last n rows.
def engineer(df):
    df = df.copy()  # work on a copy so the original frame is not mutated

    # Price-derived features
    df["ret_1"] = df["Close"].pct_change()                              # one-day percentage return
    df["ma5_ratio"] = df["Close"] / df["Close"].rolling(5).mean() - 1  # how far above/below the 5-day average, as a fraction
    df["ma10_ratio"] = df["Close"] / df["Close"].rolling(10).mean() - 1
    df["vol_5"] = df["ret_1"].rolling(5).std()                          # rolling 5-day volatility (spread of recent returns)
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["volume_chg"] = df["Volume"].pct_change()                        # how much trading volume changed from yesterday

    # VIX-derived features (external market fear signal)
    df["vix_level"] = df["VIX_Close"]         # raw VIX level for that day
    df["vix_chg"] = df["VIX_Close"].pct_change()  # how much VIX moved from the previous day

    return df


# The target is the only forward-looking column -- it has to be, because it is
# what we are trying to predict. shift(-1) pulls the next row's close into the
# current row so I can compare it to today's close.
def build_target(df):
    df = df.copy()
    next_close = df["Close"].shift(-1)                          # shift(-1) moves the next row's value into the current row
    df[TARGET] = (next_close > df["Close"]).astype("float")    # 1.0 if price goes up tomorrow, 0.0 if not
    df.loc[next_close.isna(), TARGET] = pd.NA                  # the very last row has no next day, so mark it missing
    return df


# 3. Drop warmup and unlabelled rows

# The first MAX_WINDOW-1 rows do not have enough history for the longest
# rolling window so their feature values are NaN. The final row has no
# next-day target. I drop anything missing in either features or target.
def drop_warmup(df):
    needed = ALL_FEATURES + [TARGET]
    before = len(df)
    cleaned_rows = df.dropna(subset=needed).copy()             # drop any row with a missing value in features or target
    cleaned_rows[TARGET] = cleaned_rows[TARGET].astype(int)    # convert 0.0/1.0 to integer 0/1

    dropped_lead = (df.index < cleaned_rows.index.min()).sum()
    dropped_tail = (df.index > cleaned_rows.index.max()).sum()
    print("\n--- warmup / target drop ---")
    print(f"rows before        : {before}")
    print(f"dropped (leading)  : {dropped_lead}  (no full {MAX_WINDOW}-day window)")
    print(f"dropped (trailing) : {dropped_tail}  (no next-day target)")
    print(f"rows after         : {len(cleaned_rows)}")
    return cleaned_rows


# 4. Re-split at the frozen boundary and re-assert

# I use the _split tag added in load_aligned() to slice the rows back into
# their original sets. I then re-run the chronological assertions from Chunk 1
# so any bug introduced during feature engineering would be caught here.
def split_back(df):
    train = df[df["_split"] == "train"].drop(columns="_split")  # filter by tag, then remove the tag column
    val = df[df["_split"] == "val"].drop(columns="_split")
    test = df[df["_split"] == "test"].drop(columns="_split")

    assert len(train) + len(val) + len(test) == len(df), "splits lost rows"
    assert train.index.max() < val.index.min(), "train overlaps/post-dates val"
    assert val.index.max() < test.index.min(), "val overlaps/post-dates test"

    def describe(part, label):
        bal = part[TARGET].mean()  # proportion of days where the price went up
        print(f"{label:5s} : {len(part):4d} rows  |  "
              f"{part.index.min().date()}  ->  {part.index.max().date()}  |  "
              f"up-rate {bal:.3f}")

    print("\n--- re-split (frozen boundary) + class balance ---")
    describe(train, "train")
    describe(val, "val")
    describe(test, "test")
    print("leakage assertions passed: all test dates post-date train and val.")
    return train, val, test


# 5. Scale (fit on TRAIN only)

# StandardScaler subtracts the mean and divides by the standard deviation so
# all features end up on roughly the same numeric scale. I fit it only on
# training rows so val/test statistics never influence the scale -- then I
# apply the same fixed scale to all three splits.
def scale(train, val, test):
    scaler = StandardScaler().fit(train[ALL_FEATURES])  # learn mean and std from training rows only

    def apply(part):
        out = part.copy()
        out[ALL_FEATURES] = scaler.transform(part[ALL_FEATURES])  # apply the fixed scale without refitting
        return out

    train_scaled, val_scaled, test_scaled = apply(train), apply(val), apply(test)

    # After scaling, training features should have mean near 0 and std near 1.
    # Val/test will not match exactly because the scaler was not fit on them.
    means = train_scaled[ALL_FEATURES].mean().abs().max()
    stds = train_scaled[ALL_FEATURES].std().sub(1).abs().max()
    print("\n--- scaling (StandardScaler, fit on TRAIN only) ---")
    print(f"train scaled |mean| max   : {means:.2e}  (expect ~0)")
    print(f"train scaled |std-1| max  : {stds:.2e}  (expect ~0)")
    return train_scaled, val_scaled, test_scaled, scaler


# 6. Save

# I save the scaled feature CSVs and the fitted scaler object so later chunks
# can load them directly without repeating any of this work.
def save(train, val, test, scaler):
    os.makedirs(MODELS_DIR, exist_ok=True)
    columns_to_keep = ALL_FEATURES + [TARGET]
    for part, name in [(train, "train"), (val, "val"), (test, "test")]:
        path = os.path.join(PROCESSED_DIR, f"{name}_features.csv")
        part[columns_to_keep].to_csv(path)
        print(f"[saved] {name} -> {path}")

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)  # joblib.dump saves a Python object to disk, like pickle
    print(f"[saved] scaler -> {scaler_path}")


# Verification helper -- prints a preview of the engineered data before scaling
# so I can eyeball whether the feature values look sensible.
def verify(df):
    print("\n--- engineered features (head, pre-scaling) ---")
    cols = ["Close", "VIX_Close"] + ALL_FEATURES + [TARGET]
    print(df[cols].head(12).to_string())
    print("\nmissing values per engineered column:")
    print(df[ALL_FEATURES + [TARGET]].isna().sum().to_string())


# Orchestration

# Same two-stage pattern as Chunk 1: "verify" stops after printing so I can
# check things look right before committing to writing files.
def main(stage):
    combined = load_aligned()
    combined = engineer(combined)
    combined = build_target(combined)
    verify(combined)

    cleaned_rows = drop_warmup(combined)
    train, val, test = split_back(cleaned_rows)

    if stage == "verify":
        print("\n[STOP] Verification complete. Confirm features look sane, then "
              "re-run with `--stage full` to scale and save.")
        return

    train_scaled, val_scaled, test_scaled, scaler = scale(train, val, test)
    save(train_scaled, val_scaled, test_scaled, scaler)
    print("\n[DONE] Chunk 2 pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk 2 feature pipeline.")
    parser.add_argument(
        "--stage",
        choices=["verify", "full"],
        default="verify",
        help="'verify' (default) engineers + verifies then stops; "
             "'full' also scales (fit on train), asserts, and saves.",
    )
    args = parser.parse_args()
    main(args.stage)
