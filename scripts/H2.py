import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import hmean

warnings.filterwarnings('ignore')

# Configuration
# Default paths for Demo dataset
DATA_FILE_PATH = "./samples_images/results/CYS_1319.xlsx"
OUTPUT_DIR = "./samples_images/results/Plot/"

# Full dataset paths (uncomment when running Figshare dataset)
# DATA_FILE_PATH = "./data/CYS_1319.xlsx"
# OUTPUT_DIR = "./data/Plot/"

GENOTYPE_COL = 'QR'

TRAITS_MAP = {
    'S_mean': r'$\mathrm{S}$',
    'b_mean': r'$\mathrm{b^*}$',
    'ExR_mean': r'$\mathrm{ExR}$',
    'B_mean': r'$\mathrm{B}$',
    'Yellow_Ratio': 'Yellow Ratio',
    'Yellow_score': 'Yellow Score',
    'CYS': 'CYS'
}

def compute_h2_with_lmm(df, genotype_col, traits_map):
    h2_results = {}

    for col, display_name in traits_map.items():
        actual_col = col
        if col not in df.columns:
            if col == 'B_mean' and '1-B' in df.columns:
                actual_col = '1-B'
            else:
                print(f"Warning: Column [{col}] not found in dataset. Skipping.")
                continue

        df_clean = df.dropna(subset=[genotype_col, actual_col]).copy()
        df_clean[genotype_col] = df_clean[genotype_col].astype(str).str.strip()

        k = df_clean[genotype_col].nunique()
        if k <= 1:
            continue

        replicate_counts = df_clean.groupby(genotype_col).size().values
        n_h = hmean(replicate_counts)

        try:
            model = smf.mixedlm(f'Q("{actual_col}") ~ 1', data=df_clean, groups=df_clean[genotype_col])
            result = model.fit(reml=True)

            sigma2_G = max(0.0, float(result.cov_re.iloc[0, 0]))
            sigma2_E = float(result.scale)

            denominator = sigma2_G + (sigma2_E / n_h)
            H2 = sigma2_G / denominator if denominator > 0 else 0.0

            h2_results[display_name] = round(H2, 3)

        except Exception as e:
            print(f"Failed to fit LMM model for [{actual_col}]: {e}")

    return h2_results

def plot_sci_heritability_bar(h2_results, output_dir):
    if not h2_results:
        print("Error: No valid H² results to plot.")
        return

    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['svg.fonttype'] = 'none'

    traits = list(h2_results.keys())
    h2_values = list(h2_results.values())

    colors = ['#2B5B88', '#32B1C8', '#66CCB3', '#EAF3AD', '#F29913', '#D9531E', '#27A856'][:len(traits)]
    edge_colors = ['#1B3B58', '#1A7B8C', '#3B8C78', '#9AA362', '#A6680A', '#8C310D', '#186B36'][:len(traits)]

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    x = np.arange(len(traits))
    width = 0.45

    bars = ax.bar(x, h2_values, width=width, color=colors, edgecolor=edge_colors, linewidth=1.0, zorder=3)

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

    print(f"Heritability analysis complete. Figure saved to: {output_dir}")

if __name__ == "__main__":
    if os.path.exists(DATA_FILE_PATH):
        df_raw = pd.read_excel(DATA_FILE_PATH)
        if '1-B' in df_raw.columns and 'B_mean' not in df_raw.columns:
            df_raw['B_mean'] = df_raw['1-B']
            
        h2_computed_dict = compute_h2_with_lmm(df_raw, GENOTYPE_COL, TRAITS_MAP)
        plot_sci_heritability_bar(h2_computed_dict, OUTPUT_DIR)
    else:
        print(f"Error: Dataset file not found at '{DATA_FILE_PATH}'")
