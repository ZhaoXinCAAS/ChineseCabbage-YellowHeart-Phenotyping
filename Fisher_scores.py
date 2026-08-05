import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ===================== Configuration area =====================
ROI_FEATURES_FILE = r"H:\Cabbage_Heading_Segmentation_Dataset\Github\data\418_Grade_3\roi_10_color_features_with_ratio.xlsx"
LABELED_ROOT = r"H:\Cabbage_Heading_Segmentation_Dataset\Github\data\418_Grade_3"

# The goal of exporting statistical tables Excel path
OUTPUT_METRICS_EXCEL = r"H:\Cabbage_Heading_Segmentation_Dataset\Github\data\418_Grade_3\fisher_weights.xlsx"

FEATURES = [
    "H_circular_mean_deg", "S_mean", "V_mean", "L_mean", 
    "a_mean", "b_mean", "R_mean", "G_mean", "B_mean", "ExR_mean"
]
LABEL_ORDER = ["Light_yellow", "yellow", "Deep_yellow"]
# =================================================

# 1. Read data and match tags
df_roi = pd.read_excel(ROI_FEATURES_FILE)

def get_label(filename):
    if pd.isna(filename): return None
    fn = str(filename).strip()
    for label in LABEL_ORDER:
        folder = os.path.join(LABELED_ROOT, label)
        if os.path.exists(folder):
            if os.path.exists(os.path.join(folder, fn)): return label
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                if os.path.exists(os.path.join(folder, fn + ext)): return label
    return None

df_roi["label"] = df_roi["filename"].apply(get_label)
df_fisher = df_roi.dropna(subset=["label"]).copy()

# 2. normalization
X = MinMaxScaler().fit_transform(df_fisher[FEATURES].values)
y = df_fisher["label"].values

# 3. Calculate Fisher Score
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

# 4. Judging positive and negative directions and calculating final weight
mask_light, mask_deep = (y == LABEL_ORDER[0]), (y == LABEL_ORDER[2])
dir_dict = {
    feat: ("forward (+)" if np.mean(X[mask_deep, i]) > np.mean(X[mask_light, i]) else "Negative (-)")
    for i, feat in enumerate(FEATURES)
}

scores_arr = np.array([fisher_dict[f] for f in FEATURES])
norm_weights = scores_arr / scores_arr.sum()

# 5. Build results table data frame (DataFrame)
records = []
for i, feat in enumerate(FEATURES):
    direction_sign = 1 if "forward" in dir_dict[feat] else -1
    records.append({
        "Feature name": feat,
        "Fisher Score": round(fisher_dict[feat], 4),
        "response direction": dir_dict[feat],
        "final weight": round(norm_weights[i] * direction_sign, 4)
    })

df_result = pd.DataFrame(records)

# press Fisher Score Descending order
df_result = df_result.sort_values(by="Fisher Score", ascending=False).reset_index(drop=True)

# 6. Export only this 10 statistical table of characteristics to Excel
os.makedirs(os.path.dirname(OUTPUT_METRICS_EXCEL), exist_ok=True)
df_result.to_excel(OUTPUT_METRICS_EXCEL, index=False)

print("🎉 Export successful!10characteristic Fisher The statistical table has been saved to:")
print(f"👉 {OUTPUT_METRICS_EXCEL}")