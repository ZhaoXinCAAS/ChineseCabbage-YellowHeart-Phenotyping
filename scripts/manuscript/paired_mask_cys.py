"""Reproduce paired-mask CYS agreement, error decomposition, and reclassification."""

from argparse import ArgumentParser
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from scipy.stats import spearmanr, gaussian_kde
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# ----------------------- USER CONFIGURATION -----------------------
DEFAULT_GMM_WORKBOOK = Path("results/gmm/CYS_GMM_Analysis_Results.xlsx")
THRESHOLD_EQUALITY = 'lower'  # 'lower': CYS exactly at cutoff is assigned lower grade
EXPECTED_N = 150
FIG_WIDTH_CM = 17.0
FIG_HEIGHT_CM = 7.7
FONT_PT = 7.5
DPI = 600
FORMULA_TOL = 1e-5
BOOTSTRAP_N = 5000
SEED = 20261112
# Panel B layout: 25% of each IQR box overlaps the half-violin.
BOX_VIOLIN_OVERLAP = 0.25   # Fraction of the box width overlaid on violin, 0..1
# ---------------------------------------------------------------

REQUIRED = ['filename', 'Yellow_Ratio_Auto', 'YS_Auto', 'CYS_Auto',
            'Yellow_Ratio_Manual', 'YS_Manual', 'CYS_Manual']
C = {
    'ys': '#527F99', 'yr': '#C98A5B', 'interaction': '#7F919D',
    'navy': '#24435B', 'red': '#B96D60', 'ink': '#293B49',
    'pale': '#EAF1F5'
}
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': FONT_PT, 'axes.labelsize': FONT_PT,
    'xtick.labelsize': FONT_PT, 'ytick.labelsize': FONT_PT,
    'axes.linewidth': .65, 'xtick.major.width': .65,
    'ytick.major.width': .65, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'svg.fonttype': 'none',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})


def read_table(path):
    if not path.is_file():
        raise FileNotFoundError(f"Input does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, encoding="utf-8-sig")
    elif suffix == ".xlsx":
        df = pd.read_excel(path, engine="openpyxl")
    else:
        raise ValueError("Input must be .xlsx or .csv")
    df.columns = df.columns.astype(str).str.strip()
    return df.dropna(how="all").reset_index(drop=True)


def join_score_tables(auto_file, manual_file, cluster_column):
    tables = []
    for path, side in ((auto_file, "Auto"), (manual_file, "Manual")):
        table = read_table(path)
        if "YS" not in table:
            table = table.rename(columns={"Yellow_score": "YS"})
        required = ["filename", "Yellow_Ratio", "YS", "CYS"]
        if side == "Auto" and cluster_column is not None:
            required.append(cluster_column)
        missing = [column for column in required if column not in table]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        if table["filename"].isna().any():
            raise ValueError(f"Missing filename in {path}")
        table = table[required].copy()
        table["filename"] = table["filename"].astype(str).str.strip()
        names = {key: f"{key}_{side}" for key in ("Yellow_Ratio", "YS", "CYS")}
        tables.append(table.rename(columns=names))
    paired = tables[0].merge(
        tables[1], on="filename", how="outer", validate="one_to_one", indicator=True
    )
    if not paired["_merge"].eq("both").all():
        unmatched = paired.loc[paired["_merge"].ne("both"), "filename"].head(8).tolist()
        raise ValueError(f"Missing automatic/manual scores for filenames: {unmatched}")
    return paired.drop(columns="_merge")


def load_data(df, expected_n, cluster_column):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}')
    if df['filename'].isna().any():
        raise ValueError('Some nonempty rows have missing filenames')
    df['filename'] = df['filename'].astype(str).str.strip()
    if df['filename'].eq('').any():
        raise ValueError('Empty filenames found')
    if df['filename'].duplicated().any():
        dup = df.loc[df['filename'].duplicated(), 'filename'].head(8).tolist()
        raise ValueError(f'Duplicate filenames: {dup}')
    if len(df) != expected_n:
        raise ValueError(f'Expected {expected_n} images; found {len(df)}')
    for col in REQUIRED[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    bad = ~np.isfinite(df[REQUIRED[1:]].to_numpy(float))
    if bad.any():
        i, j = np.argwhere(bad)[0]
        raise ValueError(f'Invalid number: image={df.loc[i,"filename"]}, column={REQUIRED[j+1]}')
    for col in ['Yellow_Ratio_Auto', 'Yellow_Ratio_Manual']:
        if not df[col].between(0, 1).all():
            raise ValueError(f'{col} contains a value outside [0,1]')
    for side in ['Auto', 'Manual']:
        residual = (df[f'CYS_{side}'] -
                    df[f'YS_{side}'] * df[f'Yellow_Ratio_{side}']).abs()
        if residual.max() > FORMULA_TOL:
            k = int(residual.idxmax())
            raise ValueError(f'CYS != YS*YR for {side}: {df.loc[k,"filename"]}, '
                             f'absolute difference={residual.loc[k]:.8g}')
    if cluster_column is not None:
        if cluster_column not in df.columns or df[cluster_column].isna().any():
            raise ValueError(f'Missing/empty cluster identifier: {cluster_column}')
    return df


def read_gmm_thresholds(path):
    """Read T1–T4 in grade order from an archived or newly fitted GMM workbook."""
    if not path.is_file():
        raise ValueError(
            f"GMM workbook not found: {path}. Supply --gmm-workbook or --thresholds."
        )
    table = pd.read_excel(path, sheet_name="GMM_thresholds", engine="openpyxl")
    labels = [f"T{i}" for i in range(1, 5)]
    if not {"Threshold", "Value"}.issubset(table.columns):
        raise ValueError("GMM_thresholds must contain Threshold and Value columns.")
    if sorted(table["Threshold"].astype(str)) != labels:
        raise ValueError("GMM_thresholds must contain exactly one row each for T1–T4.")
    return table.set_index("Threshold").loc[labels, "Value"].to_numpy(dtype=float)


def make_grades(values, thresholds, threshold_equality):
    side = 'left' if threshold_equality == 'lower' else 'right'
    return np.searchsorted(thresholds, np.asarray(values), side=side) + 1


def bootstrap_means(values, rng, cluster_ids=None):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if cluster_ids is None:
        idx = rng.integers(0, n, size=(BOOTSTRAP_N, n))
        estimates = values[idx].mean(axis=1)
    else:
        # Draw independent biological units; retain all images in each sampled unit.
        codes, unique = pd.factorize(np.asarray(cluster_ids), sort=False)
        groups = [np.where(codes == k)[0] for k in range(len(unique))]
        estimates = np.empty(BOOTSTRAP_N)
        for b in range(BOOTSTRAP_N):
            selected = rng.integers(0, len(groups), size=len(groups))
            obs = np.concatenate([groups[k] for k in selected])
            estimates[b] = values[obs].mean()
    return np.percentile(estimates, [2.5, 97.5])


def style_open_axes(ax):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(.7)
        ax.spines[s].set_color(C['ink'])
    ax.tick_params(colors=C['ink'], direction='out', width=.65, pad=2.4)
    ax.grid(False, which='both')


def plot_violin_raincloud(ax, contributions, rng):
    """Raincloud in reference layout: left dots / central box / right half violin.

    Violin: Gaussian KDE of observed values, truncated to sample min/max.
    Boxplot: 25th–75th percentiles and median, 1.5×IQR whiskers.
    Every raw data point retained; horizontal jitter changes no data values.
    """
    names = ['YS', 'YR', 'Interaction']
    colors = [C['ys'], C['yr'], C['interaction']]

    ax.axhline(0, color=C['ink'], lw=.65, ls=(0, (3.5, 2.5)),
               alpha=.65, zorder=0)

    # Positions: raw observations to the left, box centered, half-violin
    # starts at the box right-hand quarter (25% overlap). KDE is drawn BEFORE the box,
    # so the thin black box outline and median remain fully visible.
    # All contributions share an identical vertical scale.
    for pos, (values, color) in enumerate(zip(contributions, colors), start=1):
        vals = np.asarray(values, dtype=float)
        box_x = pos - .07
        box_w = .215
        # 0.25 -> violin starts 25% of the box width to the left of its right edge.
        # This produces a restrained partial overlap without hiding the box.
        # 0.00 -> no overlap, 1.00 -> full box width overlaps.
        cloud_x = box_x + box_w * (0.5 - BOX_VIOLIN_OVERLAP)

        # (1) Half violin: filled smooth density on right, no text backgrounds.
        if np.ptp(vals) > 1e-12 and np.unique(vals).size >= 3:
            density = gaussian_kde(vals, bw_method=.30)
            y_grid = np.linspace(float(vals.min()), float(vals.max()), 300)
            dens = density(y_grid)
            if np.max(dens) > 0:
                cloud_width = .31 * dens / np.max(dens)
                ax.fill_betweenx(y_grid, cloud_x, cloud_x + cloud_width,
                                 facecolor=to_rgba(color, .34),
                                 edgecolor='none', zorder=1)
                ax.plot(cloud_x + cloud_width, y_grid, color=color,
                        linewidth=.58, alpha=.68, zorder=2)

        # (2) All observations to the left of the central box.
        # Gentle horizontal jitter only, not vertical displacement.
        dot_center = pos - .325
        jitter = rng.uniform(-.067, .067, size=len(vals))
        ax.scatter(dot_center + jitter, vals, s=5.2,
                   facecolors=color, edgecolors='none', alpha=.49,
                   zorder=3, rasterized=False)

        # (3) Solid thin black boxplot; colored face only (outline stays opaque).
        bp = ax.boxplot(
            [vals], positions=[box_x], widths=box_w, patch_artist=True,
            manage_ticks=False, showfliers=False, whis=1.5,
            boxprops=dict(facecolor=to_rgba(color, .42),
                          edgecolor='#222222', linewidth=.85, linestyle='-', zorder=5),
            medianprops=dict(color='#222222', linewidth=1.10,
                             linestyle='-', zorder=7),
            whiskerprops=dict(color='#222222', linewidth=.8,
                              linestyle='-', zorder=4),
            capprops=dict(color='#222222', linewidth=.8,
                          linestyle='-', zorder=4)
        )
        for artist in (bp['boxes'] + bp['medians'] + bp['whiskers'] + bp['caps']):
            artist.set_linestyle('-')

    ax.set_xlim(.49, 3.55)
    ax.set_xticks([1, 2, 3], names)
    ax.set_ylabel(r'Contribution to $\Delta$CYS', labelpad=4)
    ax.tick_params(axis='x', length=0, pad=4)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.margins(y=.075)
    style_open_axes(ax)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="Paired-mask score table, or automatic scores with --manual."
    )
    parser.add_argument("--manual", type=Path, help="Manual scores to pair with automatic scores.")
    parser.add_argument("output_dir", type=Path, help="Directory for figures and statistics.")
    threshold_source = parser.add_mutually_exclusive_group()
    threshold_source.add_argument(
        "--gmm-workbook", type=Path, default=DEFAULT_GMM_WORKBOOK,
        help=f"Workbook containing GMM_thresholds (default: {DEFAULT_GMM_WORKBOOK}).",
    )
    threshold_source.add_argument(
        "--thresholds", type=float, nargs=4, metavar=("T1", "T2", "T3", "T4"),
        help="Provide four explicit cutoffs instead of reading a workbook.",
    )
    parser.add_argument(
        "--threshold-equality", choices=["lower", "upper"], default=THRESHOLD_EQUALITY
    )
    parser.add_argument("--expected-n", type=int, default=EXPECTED_N)
    parser.add_argument("--cluster-column", help="Optional accession or biological unit column.")
    args = parser.parse_args()
    threshold_equality = args.threshold_equality
    cluster_column = args.cluster_column
    try:
        th = (
            np.asarray(args.thresholds, dtype=float)
            if args.thresholds is not None else read_gmm_thresholds(args.gmm_workbook)
        )
        if th.shape != (4,) or not np.isfinite(th).all() or not np.all(np.diff(th) > 0):
            raise ValueError("Four finite, increasing GMM cutoffs are required.")
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 <= BOX_VIOLIN_OVERLAP <= 1:
        raise ValueError('BOX_VIOLIN_OVERLAP must be between 0 and 1')
    if args.manual is None:
        table = read_table(args.input)
    else:
        table = join_score_tables(args.input, args.manual, cluster_column)
    df = load_data(table, args.expected_n, cluster_column)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    yr_a = df['Yellow_Ratio_Auto'].to_numpy(float)
    yr_m = df['Yellow_Ratio_Manual'].to_numpy(float)
    ys_a = df['YS_Auto'].to_numpy(float)
    ys_m = df['YS_Manual'].to_numpy(float)
    cys_a = df['CYS_Auto'].to_numpy(float)
    cys_m = df['CYS_Manual'].to_numpy(float)
    n = len(df)
    d_yr, d_ys = yr_a-yr_m, ys_a-ys_m
    d_cys = cys_a-cys_m
    contributions = [yr_m*d_ys, ys_m*d_yr, d_ys*d_yr]
    decomp_resid = d_cys - sum(contributions)
    if np.max(np.abs(decomp_resid)) > 2*FORMULA_TOL:
        raise ValueError('CYS error decomposition does not reconstruct Delta_CYS')
    if len(np.unique(cys_a)) < 2 or len(np.unique(cys_m)) < 2:
        raise ValueError('Spearman correlation undefined for constant CYS values')
    rho = float(spearmanr(cys_m, cys_a).statistic)
    mae, bias = np.mean(np.abs(d_cys)), np.mean(d_cys)
    rmse = np.sqrt(np.mean(d_cys**2))
    grade_m = make_grades(cys_m, th, threshold_equality)
    grade_a = make_grades(cys_a, th, threshold_equality)
    cm = confusion_matrix(grade_m, grade_a, labels=[1, 2, 3, 4, 5])
    assert int(cm.sum()) == n, 'Confusion-matrix count does not equal n'
    agreement = 100*float(np.mean(grade_m == grade_a))
    assert int(cm.trace()) == int((grade_m == grade_a).sum()), 'Agreement mismatch'
    kappa = float(cohen_kappa_score(grade_m, grade_a, weights='quadratic'))
    if not np.isfinite(kappa):
        warnings.warn('Quadratic-weighted kappa is undefined for this grade distribution')

    fig = plt.figure(figsize=(FIG_WIDTH_CM/2.54, FIG_HEIGHT_CM/2.54))
    # Fixed manual arrangement makes all three panels retain intended margins.
    ax_a = fig.add_axes([.073, .235, .252, .575])
    ax_b = fig.add_axes([.399, .235, .237, .575])
    ax_c = fig.add_axes([.751, .255, .222, .545])

    # (A) CYS agreement. No grids, no white text background.
    vmin = min(0., float(cys_a.min()), float(cys_m.min()))
    vmax = max(float(cys_a.max()), float(cys_m.max()), float(th[-1]))
    pad = max(.012, .045*(vmax-vmin))
    lo = vmin-pad if vmin < 0 else 0.0
    hi = vmax+pad
    ax_a.plot([lo,hi], [lo,hi], color=C['red'], lw=1.0,
              linestyle=(0,(4,2.5)), zorder=1)
    ax_a.scatter(cys_m, cys_a, s=9.8, color=C['ys'], alpha=.78,
                 edgecolor='white', linewidth=.18, zorder=2)
    ax_a.set_xlim(lo,hi)
    ax_a.set_ylim(lo,hi)
    ax_a.set_aspect('equal', adjustable='box')
    ax_a.set_xlabel('Manual-mask CYS', labelpad=4)
    ax_a.set_ylabel('Automated-mask CYS', labelpad=4)
    ax_a.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax_a.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax_a.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax_a.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax_a.text(.035, .965,
              f'n = {n}\nSpearman $\\rho$ = {rho:.3f}\nMAE = {mae:.4f}\nBias = {bias:+.4f}',
              transform=ax_a.transAxes, ha='left', va='top',
              fontsize=FONT_PT-.75, linespacing=1.24,
              color=C['ink'], zorder=5)  # Deliberately no bbox.
    style_open_axes(ax_a)

    # (B) Reference-style dots + boxplot + right half-violin.
    plot_violin_raincloud(ax_b, contributions, np.random.default_rng(SEED))

    # (C) Fixed-cutpoint grading confusion matrix.
    cmap = LinearSegmentedColormap.from_list('grade_blue',
        ['#F3F6F8', '#D5E3EC', '#94B8CC', '#477B9B', '#1F4B76'])
    im = ax_c.imshow(cm, cmap=cmap, vmin=0, vmax=max(1,int(cm.max())),
                     interpolation='nearest', aspect='equal')
    for i in range(5):
        for j in range(5):
            v = int(cm[i,j])
            ax_c.text(j,i,str(v), ha='center',va='center',
                      fontsize=FONT_PT-.2,
                      color='white' if im.norm(v) > .65 else C['ink'],
                      fontweight='bold' if i==j else 'normal')
    ax_c.set_xticks(range(5), [f'G{i}' for i in range(1,6)])
    ax_c.set_yticks(range(5), [f'G{i}' for i in range(1,6)])
    ax_c.set_xlabel('Automated-mask grade', labelpad=4)
    ax_c.set_ylabel('Manual-mask grade', labelpad=3)
    ax_c.set_xticks(np.arange(-.5,5,1), minor=True)
    ax_c.set_yticks(np.arange(-.5,5,1), minor=True)
    ax_c.grid(which='minor',color='white',linewidth=.9)
    ax_c.tick_params(which='minor', bottom=False, left=False)
    ax_c.tick_params(which='major', length=0, pad=3)
    for spine in ax_c.spines.values():
        spine.set_visible(False)

    for x, title in [(.073,'(A) CYS agreement'),
                     (.399,'(B) Error decomposition'),
                     (.751,'(C) Grade reclassification')]:
        fig.text(x, .917, title, fontsize=FONT_PT,
                 fontweight='bold', ha='left', va='top', color=C['ink'])
    fig.text(.861, .122, rf'Agreement = {agreement:.1f}%   |   $\kappa_w$ = {kappa:.2f}',
             fontsize=FONT_PT-.55, ha='center', va='center', color=C['ink'])

    prefix = out/'paired_mask_cys'
    fig.savefig(str(prefix)+'.svg', facecolor='white')
    fig.savefig(str(prefix)+'.pdf', facecolor='white')
    fig.savefig(str(prefix)+'_600dpi.png', dpi=DPI, facecolor='white')
    plt.close(fig)

    detail = df.copy()
    detail['Delta_YS'] = d_ys
    detail['Delta_YR'] = d_yr
    detail['Delta_CYS'] = d_cys
    for name, v in zip(['YS','YR','Interaction'], contributions):
        detail[f'Contribution_{name}'] = v
    detail['Decomposition_residual'] = decomp_resid
    detail['Manual_grade'] = grade_m
    detail['Auto_grade'] = grade_a
    detail['Grade_shift'] = grade_a-grade_m

    cluster_ids = df[cluster_column].to_numpy() if cluster_column else None
    rng = np.random.default_rng(SEED+1)
    rows = []
    def report(name, point, arr=None):
        if arr is None:
            rows.append({'Metric':name, 'Estimate':point, 'CI_low':np.nan, 'CI_high':np.nan})
        else:
            a,b = bootstrap_means(arr, rng, cluster_ids)
            rows.append({'Metric':name, 'Estimate':point, 'CI_low':float(a), 'CI_high':float(b)})
    report('N', n)
    report('CYS Spearman rho', rho)
    report('CYS MAE', float(mae), np.abs(d_cys))
    report('CYS bias (auto - manual)', float(bias), d_cys)
    report('CYS RMSE', float(rmse))
    report('Exact grade agreement (%)', agreement, 100*(grade_m==grade_a).astype(float))
    report('Reclassified images (n)', int(np.sum(grade_m!=grade_a)))
    report('Within-one-grade (%)', 100*float(np.mean(np.abs(grade_m-grade_a)<=1)),
           100*(np.abs(grade_m-grade_a)<=1).astype(float))
    report('Quadratic-weighted kappa', kappa)
    for name, v in zip(['YS','YR','Interaction'], contributions):
        report(f'{name} mean signed contribution', float(np.mean(v)), v)
        report(f'{name} mean absolute contribution', float(np.mean(np.abs(v))), np.abs(v))
    abs_diff = np.abs(contributions[1])-np.abs(contributions[0])
    report('Mean |YR contribution| minus |YS contribution|', float(np.mean(abs_diff)), abs_diff)

    crossing = []
    for i,t in enumerate(th,1):
        above_m = cys_m > t if threshold_equality=='lower' else cys_m>=t
        above_a = cys_a > t if threshold_equality=='lower' else cys_a>=t
        crossing.append({'Threshold':f'T{i}', 'Cutoff':t,
                         'Crossings_n':int(np.sum(above_m != above_a)),
                         'Upward_n':int(np.sum(~above_m & above_a)),
                         'Downward_n':int(np.sum(above_m & ~above_a))})
    cm_df = pd.DataFrame(cm, index=[f'Manual G{i}' for i in range(1,6)],
                         columns=[f'Auto G{i}' for i in range(1,6)])
    with pd.ExcelWriter(out/'paired_mask_statistics.xlsx', engine='openpyxl') as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name='Summary', index=False)
        pd.DataFrame(crossing).to_excel(writer, sheet_name='Threshold_crossings', index=False)
        cm_df.to_excel(writer, sheet_name='Grade_matrix')
        detail.to_excel(writer, sheet_name='Paired_images', index=False)
    print(f'Success: n={n}, rho={rho:.4f}, MAE={mae:.4f}, bias={bias:+.4f}')
    print(f'Agreement={agreement:.2f}%, kappa_w={kappa:.4f}, changed={(grade_m!=grade_a).sum()}')
    print('Figure:', str(prefix)+'.svg')
    print('Statistics:', str(out/'paired_mask_statistics.xlsx'))
    if cluster_column is None:
        print('NOTE: Bootstrap CI assumes independent images; use --cluster-column for repeated material.')


if __name__ == '__main__':
    main()
