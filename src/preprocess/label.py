import pandas as pd
import numpy as np

INPUT_CSV = "T1DiabetesGranada/original/Glucose_measurements.csv"
OUTPUT_CSV = "T1DiabetesGranada/labeled/granada_labeled.csv"

PATIENT_COL = "Patient_ID"
DATE_COL = "Measurement_date"
TIME_COL = "Measurement_time"
VALUE_COL = "Measurement"

HYPO_THRESHOLD = 70.0
HYPER_THRESHOLD = 180.0
T_STEPS = 4


def glucose_to_class_series(s: pd.Series) -> pd.Series:
    return np.select(
        [s < HYPO_THRESHOLD, s > HYPER_THRESHOLD],
        ["hypo", "hyper"],
        default="normal"
    )


df = pd.read_csv(INPUT_CSV)

df["timestamp"] = pd.to_datetime(
    df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str),
    errors="coerce"
)

df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")

df = df.dropna(subset=[PATIENT_COL, "timestamp", VALUE_COL]).copy()
df = df.sort_values([PATIENT_COL, "timestamp"]).reset_index(drop=True)

df["current_class"] = glucose_to_class_series(df[VALUE_COL])

df["future_value_t"] = df.groupby(PATIENT_COL)[VALUE_COL].shift(-T_STEPS)
df["target_point_t"] = glucose_to_class_series(df["future_value_t"])

future_cols = []
for k in range(1, T_STEPS + 1):
    col = f"future_{k}"
    df[col] = df.groupby(PATIENT_COL)[VALUE_COL].shift(-k)
    future_cols.append(col)

future_mat = df[future_cols]

df["future_hypo_within_t"] = (future_mat.lt(HYPO_THRESHOLD)).any(axis=1).astype("Int64")
df["future_hyper_within_t"] = (future_mat.gt(HYPER_THRESHOLD)).any(axis=1).astype("Int64")

df["target_any_within_t"] = np.select(
    [
        df["future_hypo_within_t"] == 1,
        df["future_hyper_within_t"] == 1,
    ],
    [
        "hypo",
        "hyper",
    ],
    default="normal"
)

valid_future = future_mat.notna().all(axis=1)

df.loc[~valid_future, ["future_hypo_within_t", "future_hyper_within_t", "target_any_within_t"]] = pd.NA
df.loc[df["future_value_t"].isna(), "target_point_t"] = pd.NA

df = df.drop(columns=future_cols)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Salvato: {OUTPUT_CSV}")
print(df.head(20))