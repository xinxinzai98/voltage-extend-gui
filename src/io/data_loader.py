import pandas as pd
import os

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    df = pd.read_excel(file_path, skiprows=1)
    df.columns = ['Time', 'Voltage']
    return df

def save_results(processed_df, final_df, file_path):
    base = os.path.splitext(file_path)[0]
    processed_df[['Time', 'Voltage', 'is_outlier', 'Voltage_interpolated']].to_excel(f"{base}_processed.xlsx", index=False)
    final_df[['Time', 'Voltage', 'is_extended']].to_excel(f"{base}_final.xlsx", index=False)
    return f"Processed data saved to: {base}_processed.xlsx, Extended data saved to: {base}_final.xlsx"