"""Manhattan plot visualization for GWAS results using HoloViews."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import holoviews as hv
import numpy as np
import pandas as pd
from pandera.errors import SchemaError

from yheart.genetics.schemas import gwas_col, validate_gwas_table

# Shared presentation constants
Y_LABEL = "-log10(P)"
BOKEH_WIDTH = 1200
BOKEH_HEIGHT = 500


@dataclass(frozen=True, slots=True)
class ManhattanArgs:
    """Parsed command-line arguments."""

    input: str
    output: str
    chr_label: str
    sig_threshold: float
    suggestive_threshold: float
    title: str
    backend: str
    palette: str


def parse_args(argv: Sequence[str] | None = None) -> ManhattanArgs:
    p = argparse.ArgumentParser(description="Generate Manhattan plot from GWAS results.")
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="GWAS results CSV (must contain SNP, CHR, BP, P).",
    )
    p.add_argument(
        "--output",
        type=str,
        default="./manhattan_plot.png",
        help="Output plot file path (.png/.pdf for static, .html for interactive Bokeh).",
    )
    p.add_argument(
        "--chr-label",
        type=str,
        default="Chromosome",
        help="X-axis label.",
    )
    p.add_argument(
        "--sig-threshold",
        type=float,
        default=5e-8,
        help="Significance threshold line.",
    )
    p.add_argument(
        "--suggestive-threshold",
        type=float,
        default=1e-5,
        help="Suggestive threshold; set 0 to disable.",
    )
    p.add_argument(
        "--title",
        type=str,
        default="Manhattan Plot",
        help="Plot title.",
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=["auto", "matplotlib", "bokeh"],
        default="auto",
        help=(
            "Plotting backend: 'auto' (infer from output extension), "
            "'matplotlib' (static), or 'bokeh' (interactive HTML via HoloViews)."
        ),
    )
    p.add_argument(
        "--palette",
        type=str,
        default="Set2",
        help="Qualitative color palette (Set2, tab10, Dark2, Paired, GAPIT).",
    )
    return cast(ManhattanArgs, p.parse_args(argv))


def _as_element(value: object) -> hv.Element:
    """Narrow a HoloViews element across third-party typing boundaries."""
    return cast(hv.Element, value)


def lead_label_frame(
    lead_snps: list[pd.Series],
    total_x_len: float = 3.2e8,
    y_max: float = 10.0,
    w_box_ratio: float = 0.10,
    pad_y: float = 0.35,
    h_box: float = 0.40,
) -> pd.DataFrame | None:
    """Build the lead-SNP label table using connecting-line and text-orientation relaxation.

    Considers the relative vector between adjacent peaks and the horizontal orientation
    of text boxes:
    - Initial placement sits naturally directly above each peak (y + pad_y).
    - When two peaks are close along the connecting line (both delta_x and delta_y overlap),
      labels separate horizontally along the text orientation away from each other.
    - Preserves natural vertical positions without launching labels into the sky.
    - Enforces canvas boundaries so labels near edges are never clipped.
    """
    if not lead_snps:
        return None

    res = pd.DataFrame(lead_snps).sort_values(by="x").reset_index(drop=True)
    res["-log10P"] = res["-log10P"].astype(float)
    res["x"] = res["x"].astype(float)
    n = len(res)
    if n == 0:
        return None

    w_box = total_x_len * w_box_ratio

    # Initial natural placement directly above each peak
    lx = res["x"].to_numpy(dtype=float).copy()
    ly = res["-log10P"].to_numpy(dtype=float).copy() + pad_y

    # Iterative relaxation along the connecting line and text orientation
    for _ in range(5):
        changed = False
        for i in range(n - 1):
            dx = lx[i + 1] - lx[i]
            dy = abs(ly[i + 1] - ly[i])
            if dx < w_box and dy < h_box:
                overlap = w_box - dx
                shift = overlap / 2.0
                lx[i] -= shift
                lx[i + 1] += shift
                changed = True
        if not changed:
            break

    # Canvas boundary safety: ensure labels stay neatly within plot margins
    for i in range(n):
        half_w = w_box / 2.0
        if lx[i] + half_w > total_x_len:
            lx[i] = total_x_len - half_w
        if lx[i] - half_w < 0:
            lx[i] = half_w

    res["lx"] = lx
    res["ly"] = ly
    res["x_label"] = lx
    res["y_label"] = ly
    return res


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if not args.input:
        print("Error: --input is required.")
        sys.exit(1)

    try:
        df = pd.read_csv(args.input, dtype={"SNP": str, "CHR": str})
    except pd.errors.EmptyDataError:
        print("Error: Input CSV is empty.")
        sys.exit(1)

    if df.empty:
        print("Error: Input CSV is empty.")
        sys.exit(1)

    try:
        gdf = validate_gwas_table(df)
    except SchemaError as e:
        print(f"Error: GWAS table failed validation: {e}")
        sys.exit(1)

    gdf["P"] = gwas_col(gdf, "P").replace(0, 1e-300)
    gdf = gdf[gwas_col(gdf, "P") > 0].copy()
    if gdf.empty:
        print("Error: No valid P values (all zero or non-positive).")
        sys.exit(1)

    gdf["-log10P"] = -np.log10(gwas_col(gdf, "P"))
    chr_col = gwas_col(gdf, "CHR")

    # 1. Natural chromosome ordering (e.g. A01..A10, chr1..chr10, 1..22)
    def natural_chr_key(c: object) -> tuple[str, int, str]:
        s = str(c)
        prefix = re.split(r"\d+", s)[0]
        nums = re.findall(r"\d+", s)
        num = int(nums[0]) if nums else 0
        return (prefix, num, s)

    unique_chrs: list[str] = [str(c) for c in sorted(chr_col.unique(), key=natural_chr_key)]

    # 2. Cumulative genomic offsets
    chr_max_bp = gdf.groupby("CHR")["BP"].max().reindex(unique_chrs).fillna(0)
    offsets = chr_max_bp.cumsum().shift(fill_value=0)
    offset_map: dict[str, int] = {str(k): int(v) for k, v in offsets.items()}
    chr_max_map: dict[str, int] = {str(k): int(v) for k, v in chr_max_bp.items()}
    centers: dict[str, float] = {c: float(offset_map[c] + chr_max_map[c] / 2) for c in unique_chrs}

    mapped = chr_col.astype(str).map(offset_map).to_numpy()
    bp = gwas_col(gdf, "BP").to_numpy()
    gdf["x"] = bp + mapped

    # Ensure CHR is an ordered Categorical according to natural chromosome order
    gdf["CHR"] = pd.Categorical(gdf["CHR"].astype(str), categories=unique_chrs, ordered=True)

    # Determine backend
    is_html = args.output.lower().endswith(".html")
    backend = (
        "bokeh"
        if (args.backend == "bokeh" or (args.backend == "auto" and is_html))
        else "matplotlib"
    )
    hv.extension(backend)

    # Palette configuration: support custom GAPIT alternating colors or standard colormap names
    cmap: str | list[str]
    if args.palette.upper() == "GAPIT":
        cmap = ["#3C5587", "#89A8D0"]
    else:
        cmap = args.palette

    sig_color = "#E41A1C"  # Red for significant hits (GAPIT SIG_COLOR)
    suggest_color = "#FF7F00"  # Orange for suggestive line (GAPIT SUGGEST_COLOR)

    # 3. Peak lead SNP selection for annotation
    lead_snps: list[pd.Series] = []
    window_bp = 1_000_000

    def is_distinct_peak(cand: pd.Series, existing: list[pd.Series]) -> bool:
        for lead in existing:
            if (
                str(cand["CHR"]) == str(lead["CHR"])
                and abs(int(cand["BP"]) - int(lead["BP"])) < window_bp
            ):
                return False
        return True

    sig_mask = gdf["P"] <= args.sig_threshold
    if sig_mask.any():
        for _, cand in gdf[sig_mask].sort_values(by="P").iterrows():
            if is_distinct_peak(cand, lead_snps):
                lead_snps.append(cand)

    # If no significant SNPs exist, show top suggestive/general peaks
    if not lead_snps:
        suggest_mask = gdf["P"] <= args.suggestive_threshold
        for _, cand in gdf[suggest_mask].sort_values(by="P").iterrows():
            if is_distinct_peak(cand, lead_snps):
                lead_snps.append(cand)
            if len(lead_snps) >= 5:
                break
        if not lead_snps:
            for _, cand in gdf.sort_values(by="P").iterrows():
                if is_distinct_peak(cand, lead_snps):
                    lead_snps.append(cand)
                if len(lead_snps) >= 5:
                    break

    max_stat = float(gdf["-log10P"].max())
    y_max = max(max_stat + 1.8, -np.log10(args.sig_threshold) + 2.0)
    last_chr = unique_chrs[-1]
    total_x_len = max(
        float(gdf["x"].max() * 1.01),
        float(offset_map[last_chr] + chr_max_map[last_chr]),
    )

    x_range = (0.0, float(total_x_len))
    ylim = (-0.2, y_max + 0.5)

    # 4. Prepare plotting dataframe
    if backend == "bokeh" and len(gdf) > 150_000:
        # Extra processing for Bokeh interactive view:
        # Instead of 1D hard-threshold sampling (which creates horizontal density cliffs
        # at specific -log10P values and vertical voids in sparse centromeric regions),
        # we apply 2D screen-space grid downsampling across (x, -log10P).
        # This keeps the visual cloud silhouette smooth and void-free, guarantees all
        # suggestive/significant peaks (-log10P >= 4.0) are 100% preserved, and ensures
        # every retained point remains a real Bokeh glyph with active hover tooltips.
        high_mask = gdf["-log10P"] >= 4.0
        low_df = gdf[~high_mask].copy()

        n_x_bins = 2500
        n_y_bins = 100
        x_bins = np.linspace(0, total_x_len, n_x_bins)
        y_bins = np.linspace(0, y_max + 0.1, n_y_bins)

        low_df["x_bin"] = np.digitize(low_df["x"], x_bins)
        low_df["y_bin"] = np.digitize(low_df["-log10P"], y_bins)

        sampled_low = low_df.groupby(["x_bin", "y_bin"], as_index=False).first()
        sampled_low = sampled_low.drop(columns=["x_bin", "y_bin"])
        plot_df = pd.concat([sampled_low, gdf[high_mask]], ignore_index=True)
    else:
        plot_df = gdf.copy()

    # Cast fields for memory efficiency and tooltip precision
    plot_df["x"] = plot_df["x"].astype("int32")
    plot_df["-log10P"] = plot_df["-log10P"].round(3).astype("float32")
    plot_df["BP"] = plot_df["BP"].astype("int32")
    plot_df["P"] = plot_df["P"].astype("float64")
    plot_df["CHR"] = pd.Categorical(
        plot_df["CHR"].astype(str), categories=unique_chrs, ordered=True
    )

    # 5. Declarative HoloViews composition
    # Base chromosome scatter points with native cmap
    base_points = hv.Points(
        plot_df,
        kdims=["x", "-log10P"],
        vdims=["CHR", "SNP", "BP", "P"],
        label="Variants",
    )

    if backend == "bokeh":
        base_points = base_points.opts(
            color="CHR",
            cmap=cmap,
            size=4.5,
            alpha=0.75,
            line_color=None,
            tools=["hover"],
            show_legend=False,
        )
    else:
        base_points = base_points.opts(
            color="CHR",
            cmap=cmap,
            s=16,
            alpha=0.8,
            edgecolors="none",
            show_legend=False,
        )

    # Threshold lines
    layers: list[hv.Element] = [_as_element(base_points)]
    if backend == "bokeh":
        s_line_opts = {"color": suggest_color, "line_dash": "dashed", "line_width": 1.2}
        sig_line_opts = {"color": sig_color, "line_dash": "dashed", "line_width": 1.5}
    else:
        s_line_opts = {"color": suggest_color, "linestyle": "--", "linewidth": 1.0}
        sig_line_opts = {"color": sig_color, "linestyle": "--", "linewidth": 1.2}

    if args.suggestive_threshold > 0:
        layers.append(
            _as_element(hv.HLine(-np.log10(args.suggestive_threshold)).opts(**s_line_opts))
        )
    layers.append(_as_element(hv.HLine(-np.log10(args.sig_threshold)).opts(**sig_line_opts)))

    # Significant hits overlay
    if sig_mask.any():
        sig_df = gdf.loc[sig_mask, ["x", "-log10P", "SNP", "CHR", "BP", "P"]].copy()
        sig_df["x"] = sig_df["x"].astype("int32")
        sig_df["-log10P"] = sig_df["-log10P"].round(3).astype("float32")
        sig_df["BP"] = sig_df["BP"].astype("int32")
        sig_df["P"] = sig_df["P"].astype("float64")
        sig_df["CHR"] = pd.Categorical(
            sig_df["CHR"].astype(str), categories=unique_chrs, ordered=True
        )

        sig_pts = hv.Points(
            sig_df,
            kdims=["x", "-log10P"],
            vdims=["SNP", "CHR", "BP", "P"],
            label="Significant Hits",
        )
        if backend == "bokeh":
            sig_pts = sig_pts.opts(
                color=sig_color,
                size=9,
                alpha=0.95,
                line_color="#330000",
                line_width=1,
                tools=["hover"],
                show_legend=False,
            )
        else:
            sig_pts = sig_pts.opts(
                color=sig_color,
                s=36,
                alpha=0.95,
                edgecolors="#330000",
                linewidth=0.5,
                show_legend=False,
            )
        layers.append(_as_element(sig_pts))

    # Lead SNP text labels
    lead_df = lead_label_frame(lead_snps, total_x_len=total_x_len, y_max=y_max)
    if lead_df is not None:
        if backend == "bokeh":
            label_pts = hv.Labels(
                lead_df[["lx", "ly", "SNP"]],
                kdims=["lx", "ly"],
                vdims=["SNP"],
            ).opts(
                text_font_size="8pt",
                text_color="#111111",
                text_baseline="bottom",
                text_align="center",
            )
        else:
            label_pts = hv.Labels(
                lead_df[["lx", "ly", "SNP"]],
                kdims=["lx", "ly"],
                vdims=["SNP"],
            ).opts(
                size=7.5,
                color="#111111",
                verticalalignment="bottom",
                horizontalalignment="center",
            )
        layers.append(_as_element(label_pts))

    ticks = [(centers[c], c) for c in unique_chrs]
    manhattan_layout = hv.Overlay(layers)

    # 6. Backend-specific layout options and saving
    if backend == "bokeh":
        manhattan_layout = manhattan_layout.opts(
            title=args.title,
            xlabel=args.chr_label,
            ylabel=Y_LABEL,
            xlim=x_range,
            ylim=ylim,
            xticks=ticks,
            width=BOKEH_WIDTH,
            height=BOKEH_HEIGHT,
            show_legend=False,
        )
        hv.save(manhattan_layout, args.output, backend="bokeh", resources="cdn")
        print(f"Interactive Bokeh Manhattan plot saved to: {args.output}")
    else:
        manhattan_layout = manhattan_layout.opts(
            hv.opts.Overlay(
                title=args.title,
                xlabel=args.chr_label,
                ylabel=Y_LABEL,
                xlim=x_range,
                ylim=ylim,
                xticks=ticks,
                aspect=2.5,
                fig_inches=(14, 5.5),
                show_legend=False,
            )
        )
        hv.save(manhattan_layout, args.output, backend="matplotlib", dpi=300)
        print(f"Manhattan plot saved to: {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
