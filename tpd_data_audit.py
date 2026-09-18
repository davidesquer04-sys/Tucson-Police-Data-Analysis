import sys
import pandas as pd

def load_data(path):
    "loads the data from a CSV file and converts the EventDate column to datetime"
    df = pd.read_csv(path)
    df["EventDate"] = pd.to_datetime(df["EventDate"], errors='coerce')
    return df

def structural_overview(df):
    "provides a structural overview of the DataFrame"
    print("1.Structure")
    print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
    print("\nDtypes:")
    print(df.dtypes)

def missingness_report(df):
    "reports the missingness of each column in the DataFrame"
    print("\n2.Missingness")
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    report = report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)
    print(report if not report.empty else "No missing values found.")

def duplicate_check(df):
    "checks for duplicates in the DataFrame"
    print("\n3.Duplicates / Uniqueness")
    full_dupes = df.duplicated().sum()
    print(f"Fully duplicated rows: {full_dupes}")
    dupe_ids = df["EventID"].duplicated().sum()
    print(f"Duplicate EventID values: {dupe_ids} (should be 0 if EventID is a true primary key)")
    dupe_obj = df["OBJECTID"].duplicated().sum()
    print(f"Duplicate OBJECTID values: {dupe_obj}")

def temporal_coverage(df):
    "analyzes the temporal coverage of the EventDate column"
    print("\n4.Temporal Coverage")
    print(f"Date range: {df['EventDate'].min().date()} to {df['EventDate'].max().date()}")
    all_days = pd.date_range(df["EventDate"].min(), df["EventDate"].max(), freq="D")
    present_days = pd.to_datetime(df["EventDate"].dt.date.unique())
    missing_days = sorted(set(all_days.date) - set(present_days.date))
    print(f"Calendar days in range: {len(all_days)}   Days with zero events: {len(missing_days)}")
    if missing_days:
        print(f"  First few missing dates: {missing_days[:5]}")

def categorical_summary(df: pd.DataFrame, columns: list, top_n: int = 8) -> None:
    print("5. CATEGORICAL FIELD SUMMARIES")
    for col in columns:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique(dropna=True)
        print(f"\n--- {col}  ({n_unique} unique values) ---")
        print(df[col].value_counts(dropna=False).head(top_n))

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "TPDOpenData_PoliceActivity_2025.csv"
    df = load_data(path)
    structural_overview(df)
    missingness_report(df)
    duplicate_check(df)
    temporal_coverage(df)
    categorical_summary(df)

main()
