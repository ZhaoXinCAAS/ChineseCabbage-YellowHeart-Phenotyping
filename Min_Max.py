import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ==================== 配置 Excel 文件路径 ====================
INPUT_EXCEL = r"H:\Cabbage_Heading_Segmentation_Dataset\Github\data\Auto_segment_Yellow_heart\True_Yellow_heart\roi_10_color_features_with_ratio.xlsx"   # 原始表格路径
OUTPUT_EXCEL = r"H:\Cabbage_Heading_Segmentation_Dataset\Github\data\Auto_segment_Yellow_heart\True_Yellow_heart\normalized_data.xlsx"  # 归一化后另存为的新表格路径
# ============================================================

def normalize_color_features(input_path, output_path):
    # 1. 读取原始 Excel 表格
    df = pd.read_excel(input_path)
    
    # 2. 定义 10 个颜色特征名称
    color_features = [
        "R_mean", "G_mean", "B_mean", 
        "H_circular_mean_deg", "S_mean", "V_mean", 
        "L_mean", "a_mean", "b_mean", "ExR_mean"
    ]
    
    # 检查表格中是否存在这些列
    missing_cols = [col for col in color_features if col not in df.columns]
    if missing_cols:
        print(f"❌ 错误：表格中缺失以下列，请检查列名格式是否完全对齐: {missing_cols}")
        return

    print("📊 正在执行数据 Min-Max 归一化计算 [0, 1]...")

    # 3. 对 10 个颜色特征执行全局 Min-Max 归一化 [0, 1]
    scaler = MinMaxScaler()
    df[color_features] = scaler.fit_transform(df[color_features])

    # 4. 保存处理后的全量数据到新 Excel（完全保留其他列原样导出）
    df.to_excel(output_path, index=False)

    print("\n" + "="*50)
    print(f"🎉 归一化成功完成！")
    print(f" - 处理的特征数量: {len(color_features)} 个颜色特征（所有特征均缩放到 [0, 1]）")
    print(f" - 处理后的 Excel 已另存至: {output_path}")
    print("="*50)

if __name__ == "__main__":
    normalize_color_features(INPUT_EXCEL, OUTPUT_EXCEL)