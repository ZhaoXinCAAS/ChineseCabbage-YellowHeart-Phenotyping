import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ==================== Configuration Excel file path ====================
INPUT_EXCEL = r"H:\Cabbage_Heading_Segmentation_Dataset\Github\data\Auto_segment_Yellow_heart\True_Yellow_heart\roi_10_color_features_with_ratio.xlsx"   # Original table path
OUTPUT_EXCEL = r"H:\Cabbage_Heading_Segmentation_Dataset\Github\data\Auto_segment_Yellow_heart\True_Yellow_heart\normalized_data.xlsx"  # New table path saved as after normalization
# ============================================================

def normalize_color_features(input_path, output_path):
    # 1. read raw Excel table
    df = pd.read_excel(input_path)
    
    # 2. definition 10 color feature name
    color_features = [
        "R_mean", "G_mean", "B_mean", 
        "H_circular_mean_deg", "S_mean", "V_mean", 
        "L_mean", "a_mean", "b_mean", "ExR_mean"
    ]
    
    # Check if these columns exist in the table
    missing_cols = [col for col in color_features if col not in df.columns]
    if missing_cols:
        print(f"❌ Error:The following columns are missing from the table,Please check whether the column name format is fully aligned: {missing_cols}")
        return

    print("📊 Executing data Min-Max Normalized calculation [0, 1]...")

    # 3. Yes 10 color features execute globally Min-Max normalization [0, 1]
    scaler = MinMaxScaler()
    df[color_features] = scaler.fit_transform(df[color_features])

    # 4. Save the entire processed data to a new Excel(Export other columns completely intact)
    df.to_excel(output_path, index=False)

    print("\n" + "="*50)
    print(f"🎉 Normalization completed successfully!")
    print(f" - number of features processed: {len(color_features)} color features(All features are scaled to [0, 1])")
    print(f" - processed Excel Saved to: {output_path}")
    print("="*50)

if __name__ == "__main__":
    normalize_color_features(INPUT_EXCEL, OUTPUT_EXCEL)