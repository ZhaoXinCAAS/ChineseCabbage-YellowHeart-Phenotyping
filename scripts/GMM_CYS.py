import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

# Config
DATA_FILE_PATH = "./full_dataset_tables/CYS_1319.xlsx"
OUTPUT_DIR = "./full_dataset_tables/Plot"

TARGET_COL = 'CYS'
FORCE_COMPONENTS = None
MAX_COMPONENTS = 8

PHENOTYPE_LABELS = {
    1: ['Single Population'],
    2: ['White', 'Yellow'],
    3: ['White', 'Yellow', 'Deep Yellow'],
    4: ['White', 'Light Yellow', 'Yellow', 'Deep Yellow'],
    5: ['White', 'Light Yellow', 'Yellow', 'Deep Yellow', 'Extreme Yellow'],
    6: ['White', 'Light Yellow', 'Yellow', 'Deep Yellow', 'Extreme Yellow', 'Ultra Yellow']
}

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['svg.fonttype'] = 'none'

def find_optimal_gmm_and_thresholds(file_path):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: File not found '{file_path}'")
        return

    if TARGET_COL not in df.columns:
        print(f"Error: Column '{TARGET_COL}' not found in dataset.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df_clean = df.dropna(subset=[TARGET_COL]).copy()
    df_clean[TARGET_COL] = df_clean[TARGET_COL].clip(lower=0)
    X = df_clean[TARGET_COL].values.reshape(-1, 1)
    sample_size = len(X) 

    # 1. Model selection via BIC
    bics = []
    models = []
    
    for k in range(1, MAX_COMPONENTS + 1):
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=15)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        models.append(gmm)
        
    optimal_k = np.argmin(bics) + 1
    best_k = FORCE_COMPONENTS if FORCE_COMPONENTS is not None else optimal_k

    best_gmm = models[best_k - 1]
    
    weights = best_gmm.weights_
    means = best_gmm.means_.flatten()
    variances = best_gmm.covariances_.flatten()
    std_devs = np.sqrt(variances)

    sorted_indices = np.argsort(means)
    means = means[sorted_indices]
    std_devs = std_devs[sorted_indices]
    weights = weights[sorted_indices]

    # 2. Derive decision thresholds from component intersections
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

    # 3. Export graded dataset
    bins = [-np.inf] + thresholds + [np.inf]
    labels = range(1, best_k + 1)
    df_clean['Scientific_Grade'] = pd.cut(df_clean[TARGET_COL], bins=bins, labels=labels)
    
    excel_name = os.path.basename(file_path).replace('.xlsx', f'_CYS_GMM_{best_k}Classes.xlsx')
    output_excel = os.path.join(OUTPUT_DIR, excel_name)
    df_clean.to_excel(output_excel, index=False)

    # 4. Generate multi-panel validation plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: BIC selection curve
    ax1 = axes[0]
    ax1.text(-0.12, 1.05, '(A)', transform=ax1.transAxes, fontsize=18, fontweight='bold', va='top')
    
    ax1.plot(range(1, MAX_COMPONENTS + 1), bics, marker='o', linestyle='-', color='#2C7BB6', linewidth=2.5, markersize=8)
    ax1.plot(best_k, bics[best_k-1], marker='o', color='#D7191C', markersize=12, zorder=5)
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

    # Panel B: Subpopulation distributions and thresholds
    ax2 = axes[1]
    ax2.text(-0.12, 1.05, '(B)', transform=ax2.transAxes, fontsize=18, fontweight='bold', va='top')
    
    base_colors = ['#8BA888', '#F2DD72', '#F2B705', '#F27405', '#BF212E', '#9467bd', '#8c564b']
    colors = base_colors[:best_k]

    sub_labels = PHENOTYPE_LABELS.get(best_k, [f'Subpopulation {i+1}' for i in range(best_k)])

    for i in range(best_k):
        pdf = weights[i] * norm.pdf(x_grid, means[i], std_devs[i])
        ax2.plot(x_grid, pdf, color=colors[i], linewidth=3, alpha=0.9, label=sub_labels[i])
        ax2.fill_between(x_grid, pdf, alpha=0.2, color=colors[i])

    sns.histplot(df_clean[TARGET_COL], bins=65, stat='density', color='#D3D3D3', edgecolor='white', alpha=0.8, ax=ax2, label='Empirical Distribution')

    current_y_max = ax2.get_ylim()[1]
    ax2.set_ylim(0, current_y_max * 1.15) 
    y_max = ax2.get_ylim()[1]

    ax2.text(0.82, 0.90, f'n = {sample_size:,}', transform=ax2.transAxes, fontsize=13, fontweight='bold', color='#222222')

    offset_x = x_max * 0.01
    for i, t in enumerate(thresholds):
        y_pos = y_max * (0.88 - i * 0.08)
        ax2.axvline(x=t, ymin=0, ymax=1, color='#404040', linestyle=':', linewidth=2)
        ax2.text(
            t + offset_x, y_pos, 
            r'$T_{' + str(i+1) + r'} = ' + f'{t:.3f}$',
            color='#222222', fontweight='bold', fontsize=12,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2)
        )

    ax2.set_title('GMM Subpopulations & Derived Thresholds', fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel('Comprehensive Yellow Score (CYS)', fontsize=14)
    ax2.set_ylabel('Probability Density', fontsize=14)
    ax2.legend(fontsize=11, loc='center right', bbox_to_anchor=(0.98, 0.5), frameon=False)
    
    ax2.set_xlim(left=0, right=x_max)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    sns.despine(ax=ax2, top=True, right=True)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    out_svg = os.path.join(OUTPUT_DIR, 'CYS_GMM_Analysis.svg')
    out_png = os.path.join(OUTPUT_DIR, 'CYS_GMM_Analysis.png')
    
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.savefig(out_png, format='png', dpi=600, bbox_inches='tight')

if __name__ == "__main__":
    find_optimal_gmm_and_thresholds(DATA_FILE_PATH)