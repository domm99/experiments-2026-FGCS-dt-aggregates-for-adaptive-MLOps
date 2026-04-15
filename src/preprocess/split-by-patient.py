import os
import pandas as pd


input_file = "T1DiabetesGranada/original/Glucose_measurements.csv"
output_dir = "T1DiabetesGranada/split"

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file)

for patient_id, patient_df in df.groupby("Patient_ID"):
    output_file = os.path.join(output_dir, f"{patient_id}.csv")
    patient_df.to_csv(output_file, index=False)