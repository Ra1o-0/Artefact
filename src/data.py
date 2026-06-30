# Chunk 1 -- data download, verification, alignment, and frozen chronological split.
#
# I download daily AAPL and VIX price data from yfinance, check it is clean,
# join the two series on the dates they share, then split everything into a
# strictly time-ordered train / validation / test set (70 / 15 / 15 percent).
#
# The whole point of the chronological split is to prevent look-ahead leakage --
# the model only ever learns from the past to predict a genuinely unseen future.
# I enforce this with explicit assertions in chronological_split().
#
# Run modes (see the bottom of the file):
#     py src/data.py                 # default: download + verify, then STOP
#     py src/data.py --stage full    # also align + split + assert + save

import argparse
import os

import pandas as pd
import yfinance as yf

# Fixed configuration (frozen for reproducibility)
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# Test fraction is the remainder (0.15) so the three always sum to 1.0.

# Paths are relative to the project root (run the script from there).
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")


# 1. Download + verify

# I load one ticker at a time and save a local CSV cache on the first run.
# That way every later run (including offline) gets the exact same data,
# which matters for reproducibility in the viva.
def load_ticker(ticker, cache_path):
    if os.path.exists(cache_path):
        print(f"[cache] reading {ticker} from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)  # read CSV, treating the first column as the row index and converting date strings to real date objects
    else:
        print(f"[download] fetching {ticker} from yfinance ...")
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,   # AAPL Close accounts for the 2020 4:1 split
            progress=False,     # suppresses the yfinance download progress bar
        )
        # yfinance can return MultiIndex columns like ('Close', 'AAPL').
        # We download one ticker at a time, so drop the redundant ticker level.
        if isinstance(df.columns, pd.MultiIndex):  # MultiIndex means each column has two labels stacked; we only want one
            df.columns = df.columns.get_level_values(0)  # keep just the top label, e.g. 'Close' not ('Close', 'AAPL')
        df.to_csv(cache_path)
        print(f"[cache] saved {ticker} to {cache_path}")

    df.index = pd.to_datetime(df.index)  # make sure the row labels are proper date objects, not plain strings
    df.index.name = "Date"               # name the index column so it is labelled clearly when saved to CSV
    df = df.sort_index()                 # guarantee rows are in ascending date order, just in case
    return df


# Quick sanity check -- just prints the row count, date range, and any missing values.
# Nothing will break downstream if this looks fine; it is purely for my own eyes.
def verify(df, name):
    print(f"\n--- {name} ---")
    print(f"rows           : {len(df)}")
    print(f"date range     : {df.index.min().date()}  ->  {df.index.max().date()}")
    print(f"columns        : {list(df.columns)}")
    print("missing values :")
    print(df.isna().sum().to_string())


# 2. Align on shared trading dates

# AAPL and VIX do not always have data on exactly the same days (rare holiday
# differences). An inner join keeps only dates that appear in both, so there
# are no gaps or mismatched rows when I later feed them into the model together.
def align(aapl, vix):
    vix_close = vix[["Close"]].rename(columns={"Close": "VIX_Close"})  # rename so it does not clash with AAPL's own Close column
    aligned = aapl.join(vix_close, how="inner")  # inner join: only keep dates that exist in both AAPL and VIX

    print("\n--- alignment ---")
    print(f"AAPL rows before   : {len(aapl)}")
    print(f"VIX rows before    : {len(vix)}")
    print(f"aligned rows after : {len(aligned)}")
    print(f"AAPL dropped       : {len(aapl) - len(aligned)}")
    print(f"VIX dropped        : {len(vix) - len(aligned)}")
    return aligned


# 3. Frozen chronological split + leakage assertions

# I slice the data by row position, not at random, so the split is always
# oldest-to-newest. The three assertions below are the actual leakage check --
# if any test date turned out to be earlier than a training date, the script
# would crash loudly rather than silently producing a dishonest model.
def chronological_split(df, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    df = df.sort_index()  # defensive: guarantee ascending date order
    row_count = len(df)
    train_rows = int(row_count * train_frac)   # int() truncates the decimal so we get a whole number of rows
    val_rows = int(row_count * val_frac)

    train = df.iloc[:train_rows]                         # iloc selects rows by position (0 to train_rows-1)
    val = df.iloc[train_rows:train_rows + val_rows]      # next block of rows immediately after train
    test = df.iloc[train_rows + val_rows:]               # everything left over becomes test

    # Chronological-order / integrity assertions
    assert len(train) + len(val) + len(test) == row_count, "splits do not cover all rows"  # assert crashes the script if the condition is false
    assert train.index.max() < val.index.min(), "train overlaps/post-dates val"             # latest train date must be before earliest val date
    assert val.index.max() < test.index.min(), "val overlaps/post-dates test"               # same check between val and test

    def describe(part, label):
        print(f"{label:5s} : {len(part):4d} rows  |  "
              f"{part.index.min().date()}  ->  {part.index.max().date()}")

    print("\n--- chronological split (70 / 15 / 15, no shuffle) ---")
    describe(train, "train")
    describe(val, "val")
    describe(test, "test")
    print("leakage assertions passed: all test dates post-date train and val.")

    return train, val, test


# I save the three splits to CSV so every later chunk can load exactly the same
# boundaries without rerunning this script. The split is frozen from this point on.
def save_splits(train, val, test):
    os.makedirs(PROCESSED_DIR, exist_ok=True)  # create the folder if it does not already exist
    for part, name in [(train, "train"), (val, "val"), (test, "test")]:  # loop over all three splits at once
        path = os.path.join(PROCESSED_DIR, f"{name}.csv")
        part.to_csv(path)
        print(f"[saved] {name} -> {path}")


# Orchestration

# In "verify" mode I just download and print then stop, so I can check the data
# looks sensible before committing to writing files. "full" mode does everything.
def main(stage):
    os.makedirs(RAW_DIR, exist_ok=True)

    aapl = load_ticker("AAPL", os.path.join(RAW_DIR, "AAPL.csv"))
    vix = load_ticker("^VIX", os.path.join(RAW_DIR, "VIX.csv"))

    verify(aapl, "AAPL")
    verify(vix, "VIX")

    if stage == "verify":
        print("\n[STOP] Verification complete. Confirm the data is clean, then "
              "re-run with `--stage full` to align and split.")
        return

    aligned = align(aapl, vix)
    train, val, test = chronological_split(aligned)
    save_splits(train, val, test)
    print("\n[DONE] Chunk 1 pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk 1 data pipeline.")
    parser.add_argument(
        "--stage",
        choices=["verify", "full"],
        default="verify",
        help="'verify' (default) downloads + verifies then stops; "
             "'full' also aligns, splits, asserts, and saves.",
    )
    args = parser.parse_args()
    main(args.stage)
