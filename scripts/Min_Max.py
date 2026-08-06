import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Configuration
# Default paths for Demo dataset
INPUT_EXCEL = "./full_dataset_tables/roi_10_color_features_with_ratio.xlsx"
OUTPUT_EXCEL = "./full_dataset_tables/normalized_data_1319.xlsx"

def normalize_color_features(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: Input file not found '{input_path}'")
        return

    df = pd.read_excel(input_path)

    color_features = [
        "R_mean", "G_mean", "B_mean", 
        "H_circular_mean_deg", "S_mean", "V_mean", 
        "L_mean", "a_mean", "b_mean", "ExR_mean"
    ]

    missing_cols = [col for col in color_features if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing specified columns: {missing_cols}")
        return

    scaler = MinMaxScaler()
    df[color_features] = scaler.fit_transform(df[color_features])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)

    print(f"Min-Max normalization complete. Output saved to: {output_path}")

if __name__ == "__main__":
    normalize_color_features(INPUT_EXCEL, OUTPUT_EXCEL)
