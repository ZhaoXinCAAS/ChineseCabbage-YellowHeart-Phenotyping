import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Configuration
# Default paths for Demo dataset
ROI_FEATURES_FILE = "./samples_images/results/roi_10_color_features_with_ratio.xlsx"
LABELED_ROOT = "./samples_images/Fisher_score_data/"
OUTPUT_METRICS_EXCEL = "./samples_images/results/fisher_weights_418.xlsx"

# Full dataset paths (uncomment when running Figshare dataset)
# ROI_FEATURES_FILE = "./data/Auto_segment_Yellow_heart/True_Yellow_heart/json/roi_10_color_features_with_ratio.xlsx"
# LABELED_ROOT = "./data/Fisher_score_data/"
# OUTPUT_METRICS_EXCEL = "./data/fisher_weights_418.xlsx"

FEATURES = [
    "H_circular_mean_deg", "S_mean", "V_mean", "L_mean", 
    "a_mean", "b_mean", "R_mean", "G_mean", "B_mean", "ExR_mean"
]
LABEL_ORDER = ["Light_yellow", "yellow", "Deep_yellow"]

# 1. Load data and map labels
df_roi = pd.read_excel(ROI_FEATURES_FILE)

def get_label(filename):
    if pd.isna(filename):
        return None
    fn = str(filename).strip()
    for label in LABEL_ORDER:
        folder = os.path.join(LABELED_ROOT, label)
        if not os.path.exists(folder):
            continue
        if os.path.exists(os.path.join(folder, fn)):
            return label
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            if os.path.exists(os.path.join(folder, fn + ext)):
                return label
    return None

df_roi["label"] = df_roi["filename"].apply(get_label)
df_fisher = df_roi.dropna(subset=["label"]).copy()

# 2. Normalization
X = MinMaxScaler().fit_transform(df_fisher[FEATURES].values)
y = df_fisher["label"].values

# 3. Calculate Fisher Scores
total_mean = np.mean(X, axis=0)
classes = np.unique(y)
fisher_dict = {}

for i, feat in enumerate(FEATURES):
    num, den = 0.0, 0.0
    for cls in classes:
        cls_feat = X[y == cls, i]
        n_k = len(cls_feat)
        if n_k > 0:
            num += n_k * ((np.mean(cls_feat) - total_mean[i]) ** 2)
            den += n_k * np.var(cls_feat)
    fisher_dict[feat] = num / den if den >= 1e-10 else 0.0

# 4. Determine response direction and normalized weights
mask_light, mask_deep = (y == LABEL_ORDER[0]), (y == LABEL_ORDER[2])
dir_dict = {
    feat: ("Positive (+)" if np.mean(X[mask_deep, i]) > np.mean(X[mask_light, i]) else "Negative (-)")
    for i, feat in enumerate(FEATURES)
}

scores_arr = np.array([fisher_dict[f] for f in FEATURES])
norm_weights = scores_arr / scores_arr.sum()

# 5. Build results DataFrame
records = []
for i, feat in enumerate(FEATURES):
    direction_sign = 1 if "Positive" in dir_dict[feat] else -1
    records.append({
        "Feature_Name": feat,
        "Fisher_Score": round(fisher_dict[feat], 4),
        "Response_Direction": dir_dict[feat],
        "Final_Weight": round(norm_weights[i] * direction_sign, 4)
    })

df_result = pd.DataFrame(records)
df_result = df_result.sort_values(by="Fisher_Score", ascending=False).reset_index(drop=True)

# 6. Save results
os.makedirs(os.path.dirname(OUTPUT_METRICS_EXCEL), exist_ok=True)
df_result.to_excel(OUTPUT_METRICS_EXCEL, index=False)

print(f"Fisher score calculation complete. Output saved to: {OUTPUT_METRICS_EXCEL}")
