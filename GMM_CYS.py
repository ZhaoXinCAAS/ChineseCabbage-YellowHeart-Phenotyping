import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import os
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ===================== 全局参数配置 =====================
DATA_FILE_PATH = r"./samples_images/results/CYS_1319.xlsx"
OUTPUT_DIR = r"./samples_images/results/Plot"  # 统一输出文件夹路径

# 自由模式：让数据自己说话，自动寻找最优聚类数
FORCE_COMPONENTS = None  
MAX_COMPONENTS = 8

# SCI 高级绘图参数设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none' # 确保导出 SVG 后文字完全可编辑
# ========================================================

def find_optimal_gmm_and_thresholds(file_path):
    print("⏳ [1/4] 正在读取表型数据并进行严格清洗...")
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 '{file_path}'")
        return

    if 'Yellow_score' not in df.columns:
        print("❌ 数据表中找不到 'Yellow_score' 列！")
        return

    # 自动确保输出文件夹存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_clean = df.dropna(subset=['Yellow_score']).copy()
    df_clean['Yellow_score'] = df_clean['Yellow_score'].clip(lower=0)
    X = df_clean['Yellow_score'].values.reshape(-1, 1)

    # ==========================================
    # 1. 计算 BIC，自由评估最佳聚类数量
    # ==========================================
    print(f"⏳ [2/4] 正在自由评估 1 到 {MAX_COMPONENTS} 个高斯分布分量 (BIC 准则)...")
    bics = []
    models = []
    
    for k in range(1, MAX_COMPONENTS + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=15) # 增加初始化次数确保稳定
        gmm.fit(X)
        bics.append(gmm.bic(X))
        models.append(gmm)
        
    optimal_k = np.argmin(bics) + 1
    best_k = FORCE_COMPONENTS if FORCE_COMPONENTS is not None else optimal_k
    print(f"🎯 探索完成！最优分类数为 K = {best_k}。")

    best_gmm = models[best_k - 1]
    
    weights = best_gmm.weights_
    means = best_gmm.means_.flatten()
    variances = best_gmm.covariances_.flatten()
    std_devs = np.sqrt(variances)

    sorted_indices = np.argsort(means)
    means = means[sorted_indices]
    std_devs = std_devs[sorted_indices]
    weights = weights[sorted_indices]

    # ==========================================
    # 2. 高精度求解相邻高斯分布曲线交点
    # ==========================================
    print("⏳ [3/4] 正在计算精细数学阈值 (符号跳变寻根法)...")
    thresholds = []
    x_max = X.max() * 1.05
    x_grid = np.linspace(0, x_max, 20000)
    
    for i in range(best_k - 1):
        pdf1 = weights[i] * norm.pdf(x_grid, means[i], std_devs[i])
        pdf2 = weights[i+1] * norm.pdf(x_grid, means[i+1], std_devs[i+1])
        
        diff = pdf1 - pdf2
        crossings = np.where(np.diff(np.sign(diff)))[0]
        
        if len(crossings) > 0:
            midpoint = (means[i] + means[i+1]) / 2.0
            closest_idx = crossings[np.argmin(np.abs(x_grid[crossings] - midpoint))]
            intersection_x = x_grid[closest_idx]
        else:
            intersection_x = (means[i] + means[i+1]) / 2.0
            
        thresholds.append(intersection_x)
        print(f"    ➤ 数学阈值 T{i+1}: {intersection_x:.4f}")

    # ==========================================
    # 3. 导出新数据表
    # ==========================================
    print("⏳ [4/4] 正在智能打标并导出...")
    bins = [-np.inf] + thresholds + [np.inf]
    labels = range(1, best_k + 1)
    df_clean['Scientific_Grade'] = pd.cut(df_clean['Yellow_score'], bins=bins, labels=labels)
    
    excel_name = os.path.basename(file_path).replace('.xlsx', f'_GMM_Final_{best_k}Classes.xlsx')
    output_excel = os.path.join(OUTPUT_DIR, excel_name)
    df_clean.to_excel(output_excel, index=False)

    # ==========================================
    # 4. 绘制顶级 SCI 多维验证图
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # 稍微增加高度让比例更协调

    # ---- (A) BIC 模型选择曲线 ----
    ax1 = axes[0]
    # 添加 SCI 标准 Panel 标签 (A)
    ax1.text(-0.12, 1.05, '(A)', transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top')
    
    ax1.plot(range(1, MAX_COMPONENTS + 1), bics, marker='o', linestyle='-', color='#2C7BB6', linewidth=2.5, markersize=8)
    ax1.plot(best_k, bics[best_k-1], marker='o', color='#D7191C', markersize=12, zorder=5) # 凸显最佳点
    ax1.axvline(x=best_k, color='#D7191C', linestyle='--', linewidth=1.8, label=f'Optimal K = {best_k}')
    
    ax1.set_title('Bayesian Information Criterion (BIC)', fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel('Number of Gaussian Components (K)', fontsize=14)
    ax1.set_ylabel('BIC Score', fontsize=14)
    ax1.set_xticks(range(1, MAX_COMPONENTS + 1))
    ax1.set_xlim(0.5, MAX_COMPONENTS + 0.5)
    
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(fontsize=12, frameon=False)
    
    sns.despine(ax=ax1, top=True, right=True)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)

    # ---- (B) GMM 分布与阈值 ----
    ax2 = axes[1]
    
    ax2.text(-0.12, 1.05, '(B)', transform=ax2.transAxes, fontsize=18, fontweight='bold', va='top')
    
    sns.histplot(df_clean['Yellow_score'], bins=65, stat='density', color='#D3D3D3', edgecolor='white', alpha=0.8, ax=ax2, label='Empirical Distribution')
    
    # 专属定制的高级渐变色系 (灰绿 -> 柔黄 -> 金黄 -> 橙 -> 绛红)
    base_colors = ['#8BA888', '#F2DD72', '#F2B705', '#F27405', '#BF212E', '#9467bd', '#8c564b']
    colors = base_colors[:best_k]
    
    for i in range(best_k):
        pdf = weights[i] * norm.pdf(x_grid, means[i], std_devs[i])
        ax2.plot(x_grid, pdf, color=colors[i], linewidth=3, alpha=0.9, label=f'Subpopulation {i+1}')
        ax2.fill_between(x_grid, pdf, alpha=0.2, color=colors[i])

    # 动态增加 Y 轴顶部留白 (Headroom)，防止拥挤
    current_y_max = ax2.get_ylim()[1]
    ax2.set_ylim(0, current_y_max * 1.15) 
    y_max = ax2.get_ylim()[1]

    for i, t in enumerate(thresholds):
        ax2.axvline(x=t, ymin=0, ymax=1, color='#404040', linestyle=':', linewidth=2)
        # 添加带白色半透明底框的标注，杜绝视觉干扰
        ax2.text(
            t + 0.008, y_max * 0.88, 
            r'$T_{' + str(i+1) + r'} = ' + f'{t:.3f}$', # 纯正 LaTeX 下标渲染
            color='#222222', fontweight='bold', fontsize=12,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2)
        )

    ax2.set_title(f'GMM Subpopulations & Derived Thresholds', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel('Comprehensive Yellow Score (CYS)', fontsize=14)
    ax2.set_ylabel('Probability Density', fontsize=14)
    ax2.legend(fontsize=11, loc='upper right', frameon=False)
    
    ax2.set_xlim(left=0, right=x_max)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    sns.despine(ax=ax2, top=True, right=True)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # 导出高精度矢量图与高清 PNG 至指定的 OUTPUT_DIR 目录
    out_svg = os.path.join(OUTPUT_DIR, 'GMM_Final_Analysis.svg')
    out_png = os.path.join(OUTPUT_DIR, 'GMM_Final_Analysis.png')
    
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=600, bbox_inches='tight')
    
    print(f"\n🎉 处理与绘图完成！")
    print(f"👉 标注 Excel 文件已保存至: {output_excel}")
    print(f"👉 矢量图已保存至: {out_svg}")
    print(f"👉 高清 PNG 已保存至: {out_png}")

if __name__ == "__main__":
    find_optimal_gmm_and_thresholds(DATA_FILE_PATH)
