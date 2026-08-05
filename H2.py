import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import hmean

# 忽略警告
warnings.filterwarnings('ignore')

# ===================== 1. 全局配置与文件路径 =====================
DATA_FILE_PATH = r"./samples_images/results/CYS_1319.xlsx"
OUTPUT_DIR = r"./samples_images/results/Plot/"

# 分组列（基因型/品系 ID，对应论文中的 germplasm ID / i-th accession）
GENOTYPE_COL = 'QR'

# 7 个表型特征映射（与论文及 Figure 6 完全一致）
TRAITS_MAP = {
    'S_mean': r'$\mathrm{S}$',
    'b_mean': r'$\mathrm{b^*}$',
    'ExR_mean': r'$\mathrm{ExR}$',
    'B_mean': r'$\mathrm{B}$',       # 自动兼容 '1-B'
    'Yellow_Ratio': 'Yellow Ratio',
    'Yellow_score': 'Yellow Score',
    'CYS': 'CYS'
}

# ===================== 2. LMM 混合效应模型计算 H^2 =====================
def compute_h2_with_lmm(df, genotype_col, traits_map):
    h2_results = {}
    print("=" * 75)
    print("🧬 正在运行线性混合效应模型 (LMM: Linear Mixed-Effects Model) 计算广义遗传力 H²...")
    print("-" * 75)
    print(f"{'Trait (性状)':<18} {'Obs (N)':<10} {'Accessions (k)':<15} {'σ²_G (遗传)':<12} {'σ²_E (残差)':<12} {'H²':<10}")
    print("-" * 75)

    for col, display_name in traits_map.items():
        actual_col = col
        if col not in df.columns:
            if col == 'B_mean' and '1-B' in df.columns:
                actual_col = '1-B'
            else:
                print(f"⚠️ 警告: 数据集中未找到列 [{col}]，跳过。")
                continue

        # 数据清洗
        df_clean = df.dropna(subset=[genotype_col, actual_col]).copy()
        df_clean[genotype_col] = df_clean[genotype_col].astype(str).str.strip()

        N = len(df_clean)
        k = df_clean[genotype_col].nunique()
        if k <= 1:
            continue

        # 1. 计算调和平均重复数 n_h (Harmonic mean of replicate numbers)
        replicate_counts = df_clean.groupby(genotype_col).size().values
        n_h = hmean(replicate_counts)

        # 2. 拟合 LMM 混合效应模型: Y_ij = μ + G_i + ε_ij
        # 固定效应: 截距 (1)
        # 随机效应: 品系分组 (groups=df_clean[genotype_col])
        try:
            model = smf.mixedlm(f'Q("{actual_col}") ~ 1', data=df_clean, groups=df_clean[genotype_col])
            result = model.fit(reml=True)  # REML 极小化受限极大似然估计

            # 3. 提取方差分量 σ²_G 和 σ²_E
            sigma2_G = max(0.0, float(result.cov_re.iloc[0, 0]))  # 随机效应方差 (Genotypic Variance)
            sigma2_E = float(result.scale)                         # 残差方差 (Residual Error Variance)

            # 4. 根据论文公式 (2-10) 计算 H^2
            # H^2 = σ²_G / (σ²_G + σ²_E / n_h)
            denominator = sigma2_G + (sigma2_E / n_h)
            H2 = sigma2_G / denominator if denominator > 0 else 0.0

            h2_results[display_name] = round(H2, 3)
            print(f"{actual_col:<18} {N:<10} {k:<15} {sigma2_G:<12.5f} {sigma2_E:<12.5f} {H2:.3f}")

        except Exception as e:
            print(f"❌ [{actual_col}] LMM 模型拟合失败: {e}")

    print("=" * 75)
    return h2_results

# ===================== 3. 高级 SCI 矢量图绘制函数 =====================
def plot_sci_heritability_bar(h2_results, output_dir):
    if not h2_results:
        print("❌ 错误：无有效 H² 数据用于绘图！")
        return

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['svg.fonttype'] = 'none'  # 矢量文字可编辑

    traits = list(h2_results.keys())
    h2_values = list(h2_results.values())

    # 还原论文 Fig. S1 专属色彩
    colors = ['#2B5B88', '#32B1C8', '#66CCB3', '#EAF3AD', '#F29913', '#D9531E', '#27A856'][:len(traits)]
    edge_colors = ['#1B3B58', '#1A7B8C', '#3B8C78', '#9AA362', '#A6680A', '#8C310D', '#186B36'][:len(traits)]

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    x = np.arange(len(traits))
    width = 0.45

    bars = ax.bar(x, h2_values, width=width, color=colors, edgecolor=edge_colors, linewidth=1.0, zorder=3)

    # 柱顶数值
    for bar, val in zip(bars, h2_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, 
            bar.get_height() + 0.006, 
            f'{val:.3f}', 
            ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#1E293B'
        )

    ax.set_ylim(0.60, 1.00)
    ax.set_yticks(np.arange(0.60, 1.01, 0.05))
    ax.set_ylabel(r'Broad-Sense Heritability ($H^2$)', fontsize=14, fontweight='bold', labelpad=10)

    ax.set_xticks(x)
    ax.set_xticklabels(traits, fontsize=12, rotation=28, ha='right')

    ax.tick_params(axis='y', labelsize=11.5, length=4, width=1.0)
    ax.tick_params(axis='x', length=0)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#CBD5E1', zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#334155')
    ax.spines['bottom'].set_color('#334155')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    svg_path = os.path.join(output_dir, "Figure_S1_Broad_Sense_Heritability_LMM.svg")
    png_path = os.path.join(output_dir, "Figure_S1_Broad_Sense_Heritability_LMM.png")

    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=600, bbox_inches='tight')
    plt.show()

    print("\n🎉 绘图成功！")
    print(f"👉 矢量图 (SVG) 已保存至: {svg_path}")

# ===================== 4. 主程序入口 =====================
if __name__ == "__main__":
    if os.path.exists(DATA_FILE_PATH):
        df_raw = pd.read_excel(DATA_FILE_PATH)
        if '1-B' in df_raw.columns and 'B_mean' not in df_raw.columns:
            df_raw['B_mean'] = df_raw['1-B']
            
        # 使用符合论文 2-9 和 2-10 式的 LMM 混合模型计算
        h2_computed_dict = compute_h2_with_lmm(df_raw, GENOTYPE_COL, TRAITS_MAP)
        
        # 绘图
        plot_sci_heritability_bar(h2_computed_dict, OUTPUT_DIR)
    else:
        print(f"❌ 找不到数据文件，请检查路径: {DATA_FILE_PATH}")
