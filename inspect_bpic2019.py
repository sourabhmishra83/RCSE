from pathlib import Path

import pandas as pd
import pm4py


# =========================================================
# CONFIGURATION
# =========================================================

from pathlib import Path

# Downloads folder
DOWNLOADS = Path.home() / "Downloads"

# IMPORTANT:
# Replace the filename below with the exact name of your XES file
XES_FILE = DOWNLOADS / "BPI_Challenge_2019.xes"

# Results will be created here:
OUTPUT_DIR = DOWNLOADS / "bpic2019_output"

SAMPLE_SIZE = 10_000
RANDOM_SEED = 42


# =========================================================
# SETUP
# =========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not XES_FILE.exists():
    raise FileNotFoundError(
        f"XES file not found:\n{XES_FILE}\n\n"
        "Update XES_FILE at the top of the script."
    )

print(f"Reading XES file:\n{XES_FILE}\n")


# =========================================================
# LOAD XES
# =========================================================

df = pm4py.read_xes(str(XES_FILE))

# Some PM4Py versions already return a DataFrame.
# Others may return an EventLog.
if not isinstance(df, pd.DataFrame):
    df = pm4py.convert_to_dataframe(df)


# =========================================================
# BASIC INFORMATION
# =========================================================

print("=" * 80)
print("DATASET SHAPE")
print("=" * 80)

print(f"Rows / events : {len(df):,}")
print(f"Columns       : {len(df.columns):,}")

print("\n" + "=" * 80)
print("COLUMNS")
print("=" * 80)

for col in df.columns:
    print(col)


# =========================================================
# ACTIVITY DISTRIBUTION
# =========================================================

activity_col = None

# Standard XES activity field
if "concept:name" in df.columns:
    activity_col = "concept:name"

if activity_col is not None:
    print("\n" + "=" * 80)
    print("ACTIVITY COUNTS")
    print("=" * 80)

    activity_counts = (
        df[activity_col]
        .value_counts(dropna=False)
        .rename_axis("activity")
        .reset_index(name="count")
    )

    print(activity_counts.to_string(index=False))

    activity_counts.to_csv(
        OUTPUT_DIR / "bpic2019_activity_counts.csv",
        index=False,
    )

else:
    print(
        "\nWARNING: Could not find standard activity column 'concept:name'."
    )


# =========================================================
# CASE-ID CHECK
# =========================================================

case_candidates = [
    "case:concept:name",
    "case_id",
    "Case ID",
]

case_col = next(
    (col for col in case_candidates if col in df.columns),
    None,
)

print("\n" + "=" * 80)
print("CASE INFORMATION")
print("=" * 80)

if case_col:
    print(f"Case column    : {case_col}")
    print(f"Unique cases  : {df[case_col].nunique(dropna=True):,}")
else:
    print("Could not automatically identify the case-id column.")


# =========================================================
# SCHEMA SUMMARY
# =========================================================

print("\n" + "=" * 80)
print("BUILDING SCHEMA SUMMARY")
print("=" * 80)

summary_rows = []

for col in df.columns:
    series = df[col]

    summary_rows.append(
        {
            "column": col,
            "dtype": str(series.dtype),
            "rows": len(series),
            "non_null": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "missing_pct": round(series.isna().mean() * 100, 4),
            "unique_values": int(series.nunique(dropna=True)),
        }
    )

schema_summary = pd.DataFrame(summary_rows)

schema_output = OUTPUT_DIR / "bpic2019_schema_summary.csv"

schema_summary.to_csv(
    schema_output,
    index=False,
)

print(schema_summary.to_string(index=False))


# =========================================================
# RANDOM SAMPLE
# =========================================================

print("\n" + "=" * 80)
print("CREATING SAMPLE")
print("=" * 80)

sample_n = min(SAMPLE_SIZE, len(df))

sample_df = df.sample(
    n=sample_n,
    random_state=RANDOM_SEED,
)

sample_output = OUTPUT_DIR / "bpic2019_sample.csv"

sample_df.to_csv(
    sample_output,
    index=False,
)

print(f"Sample rows: {sample_n:,}")


# =========================================================
# OPTIONAL: FIRST 100 CASES
# =========================================================
# Random event samples are useful for schema inspection,
# but for process analysis it is also useful to preserve
# complete event traces for a small number of cases.

if case_col:
    unique_cases = (
        df[case_col]
        .dropna()
        .drop_duplicates()
        .head(100)
    )

    case_sample_df = df[
        df[case_col].isin(unique_cases)
    ].copy()

    case_sample_output = (
        OUTPUT_DIR / "bpic2019_100_complete_cases.csv"
    )

    case_sample_df.to_csv(
        case_sample_output,
        index=False,
    )

    print(
        f"Complete-case sample rows: "
        f"{len(case_sample_df):,}"
    )


# =========================================================
# COMPLETE
# =========================================================

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)

print(f"\nFiles created in:\n{OUTPUT_DIR}")

print("\nPrimary files to upload:")
print(f"1. {sample_output}")
print(f"2. {schema_output}")

if activity_col:
    print(
        f"3. {OUTPUT_DIR / 'bpic2019_activity_counts.csv'}"
    )

if case_col:
    print(
        f"4. "
        f"{OUTPUT_DIR / 'bpic2019_100_complete_cases.csv'}"
    )