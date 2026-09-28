"""Manhattan plot visualization for GWAS results using HoloViews."""

from __future__ import annotations

import argparse
import math
import re
import sys
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import holoviews as hv
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from bokeh.core.properties import field as bokeh_field
from bokeh.models import ColumnDataSource, CustomJS, Plot, Range1d, Segment, Text
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
    namespace = p.parse_args(argv)
    args = ManhattanArgs(**vars(namespace))
    if not math.isfinite(args.sig_threshold) or not 0 < args.sig_threshold <= 1:
        p.error("--sig-threshold must be finite and in (0, 1].")
    if not math.isfinite(args.suggestive_threshold) or not 0 <= args.suggestive_threshold <= 1:
        p.error("--suggestive-threshold must be finite and in [0, 1].")
    return args


def _as_element(value: object) -> hv.Element:
    """Narrow a HoloViews element across third-party typing boundaries."""
    return cast(hv.Element, value)


def _bokeh_label_layout(plot: object, element: object) -> None:
    """Install collision-aware screen offsets that are recalculated as ranges change."""
    handles = getattr(plot, "handles", {})
    if not isinstance(handles, dict):
        return
    glyph = handles.get("glyph")
    source = handles.get("source")
    state = getattr(plot, "state", None)
    frame = getattr(element, "data", None)
    if (
        not isinstance(glyph, Text)
        or not isinstance(source, ColumnDataSource)
        or not isinstance(state, Plot)
        or not isinstance(frame, pd.DataFrame)
    ):
        return

    source_data = dict(source.data)
    source_data["_label_x"] = frame["x"].to_numpy()
    source_data["_label_y"] = frame["-log10P"].to_numpy()
    source_data["_label_text"] = frame["SNP"].astype(str).to_numpy()
    source_data["x_offset"] = frame["x_offset"].to_numpy()
    # Bokeh screen y offsets grow downward, while our layout uses upward-positive y.
    source_data["y_offset"] = -frame["y_offset"].to_numpy()
    source_data["text_align"] = frame["text_align"].to_numpy()
    source.data = source_data
    glyph.x_offset = bokeh_field("x_offset")
    glyph.y_offset = bokeh_field("y_offset")
    glyph.text_align = bokeh_field("text_align")

    x_range = getattr(state, "x_range", None)
    y_range = getattr(state, "y_range", None)
    if not isinstance(x_range, Range1d) or not isinstance(y_range, Range1d):
        return
    x_start, x_end = x_range.start, x_range.end
    y_start, y_end = y_range.start, y_range.end
    if not (
        isinstance(x_start, (int, float))
        and isinstance(x_end, (int, float))
        and isinstance(y_start, (int, float))
        and isinstance(y_end, (int, float))
    ):
        return
    xspan = float(x_end - x_start)
    yspan = float(y_end - y_start)
    width = float(BOKEH_WIDTH)
    height = float(BOKEH_HEIGHT)
    dx = frame["x_offset"].to_numpy(dtype=float)
    dy = frame["y_offset"].to_numpy(dtype=float)
    length = np.maximum(np.hypot(dx, dy), 1.0)
    edge_radius = 6.0
    x = frame["x"].to_numpy(dtype=float)
    y = frame["-log10P"].to_numpy(dtype=float)
    connector_source = ColumnDataSource(
        data={
            "x0": x + dx / length * edge_radius / width * xspan,
            "y0": y + dy / length * edge_radius / height * yspan,
            "x1": x + dx / width * xspan,
            "y1": y + dy / height * yspan,
        }
    )
    state.add_glyph(
        connector_source,
        Segment(
            x0="x0", y0="y0", x1="x1", y1="y1", line_color="#555555", line_alpha=0.7, line_width=0.8
        ),
    )
    callback = CustomJS(
        args={
            "source": source,
            "connectors": connector_source,
            "x_range": x_range,
            "y_range": y_range,
            "plot": state,
        },
        code="""
const d = source.data;
const xs = d._label_x, ys = d._label_y, names = d._label_text;
const width = plot.inner_width, height = plot.inner_height;
const xmin = Math.min(x_range.start, x_range.end), xmax = Math.max(x_range.start, x_range.end);
const ymin = Math.min(y_range.start, y_range.end), ymax = Math.max(y_range.start, y_range.end);
if (!(width > 0 && height > 0 && xmax > xmin && ymax > ymin)) return;
const placed = [], xo = new Array(xs.length), yo = new Array(xs.length);
const aligns = new Array(xs.length);
const order = Array.from(xs.keys()).sort((a, b) => ys[b] - ys[a]);
const candidates = [];
for (const dy of [20, 36, 52, 68, 84, -20, -36, -52]) {
  candidates.push({dx: 0, dy, align: "center"});
  for (const dx of [-18, 18, -34, 34, -50, 50])
    candidates.push({dx, dy, align: dx < 0 ? "right" : "left"});
}
for (const i of order) {
  const px = (xs[i] - xmin) / (xmax - xmin) * width;
  const py = (ys[i] - ymin) / (ymax - ymin) * height;
  const tw = Math.max(24, String(names[i]).length * 5.2), th = 13;
  let best = null;
  for (const candidate of candidates) {
      const {dx, dy, align} = candidate;
      const left = align === "left" ? px + dx : align === "right" ? px + dx - tw : px + dx - tw / 2;
      const right = left + tw;
      const bottom = py + dy, top = bottom + th;
      if (left < 4 || right > width - 4 || bottom < 4 || top > height - 4) continue;
      let overlap = 0;
      for (const r of placed)
        overlap += Math.max(0, Math.min(right, r[2]) - Math.max(left, r[0]))
          * Math.max(0, Math.min(top, r[3]) - Math.max(bottom, r[1]));
      const cost = overlap * 1000 + Math.abs(dx) + Math.abs(dy) * 0.15;
      if (best === null || cost < best.cost) best = {cost, dx, dy, align, left, bottom, right, top};
  }
  if (best === null) best = {
    dx: 0, dy: 20, align: "center", left: px - tw / 2,
    bottom: py + 20, right: px + tw / 2, top: py + 33,
  };
  xo[i] = best.dx;
  aligns[i] = best.align;
  // Bokeh screen y offsets grow downward; invert the upward-positive layout offset.
  yo[i] = -best.dy;
  placed.push([best.left, best.bottom, best.right, best.top]);
}
d.x_offset = xo;
d.y_offset = yo;
d.text_align = aligns;
const xspan = x_range.end - x_range.start, yspan = y_range.end - y_range.start;
const edgeRadius = 6;
connectors.data = {
  x0: xs.map((x, i) => {
    const dx = xo[i], dy = -yo[i], norm = Math.hypot(dx, dy) || 1;
    return x + dx / norm * edgeRadius / width * xspan;
  }),
  y0: ys.map((y, i) => {
    const dx = xo[i], dy = -yo[i], norm = Math.hypot(dx, dy) || 1;
    return y + dy / norm * edgeRadius / height * yspan;
  }),
  x1: xs.map((x, i) => x + xo[i] / width * xspan),
  y1: ys.map((y, i) => y + -yo[i] / height * yspan),
};
source.change.emit();
connectors.change.emit();
""",
    )
    x_range.js_on_change("start", callback)
    x_range.js_on_change("end", callback)
    y_range.js_on_change("start", callback)
    y_range.js_on_change("end", callback)
    state.js_on_change("inner_width", callback)
    state.js_on_change("inner_height", callback)


def _position_bokeh_labels(
    labels: pd.DataFrame,
    total_x_len: float,
    y_max: float,
) -> pd.DataFrame:
    """Greedily place labels in screen space, prioritizing the strongest peaks."""
    positioned = labels.copy()
    positioned["x_offset"] = 0.0
    positioned["y_offset"] = 0.0
    positioned["text_align"] = "center"
    if positioned.empty:
        return positioned

    width, height = float(BOKEH_WIDTH), float(BOKEH_HEIGHT)
    placed: list[tuple[float, float, float, float]] = []
    candidates = [
        (dx, dy, align)
        for dy in (20, 36, 52, 68, 84, -20, -36, -52)
        for dx, align in (
            (0, "center"),
            (-18, "right"),
            (18, "left"),
            (-34, "right"),
            (34, "left"),
            (-50, "right"),
            (50, "left"),
        )
    ]
    order = positioned.sort_values("-log10P", ascending=False).index
    for idx in order:
        row = positioned.loc[idx]
        anchor_x = float(row["x"]) / total_x_len * width
        anchor_y = float(row["-log10P"]) / y_max * height
        text_width = max(24.0, len(str(row["SNP"])) * 5.2)
        text_height = 13.0
        choices: list[tuple[float, float, float, float, str, float, float]] = []
        for dx, dy, align in candidates:
            left = (
                anchor_x + dx
                if align == "left"
                else anchor_x + dx - text_width
                if align == "right"
                else anchor_x + dx - text_width / 2
            )
            right = (
                anchor_x + dx + text_width
                if align == "left"
                else anchor_x + dx
                if align == "right"
                else anchor_x + dx + text_width / 2
            )
            bottom = anchor_y + dy
            top = bottom + text_height
            if left < 4 or right > width - 4 or bottom < 4 or top > height - 4:
                continue
            overlap = sum(
                max(0.0, min(right, r) - max(left, l)) * max(0.0, min(top, t) - max(bottom, b))
                for l, b, r, t in placed
            )
            choices.append((overlap, abs(dx) + abs(dy) * 0.15, dx, dy, align, left, right))

        if choices:
            _, _, dx, dy, align, left, right = min(choices)
            positioned.at[idx, "x_offset"] = dx
            positioned.at[idx, "y_offset"] = dy
            positioned.at[idx, "text_align"] = align
            placed.append((left, anchor_y + dy, right, anchor_y + dy + text_height))
        else:
            # Keep a readable default gap even for peaks near the plot ceiling.
            positioned.at[idx, "y_offset"] = 18.0
            positioned.at[idx, "text_align"] = "center"
    return positioned


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
    lx = np.clip(lx, w_box / 2.0, total_x_len - w_box / 2.0)

    res["lx"] = lx
    res["ly"] = ly
    res["x_label"] = lx
    res["y_label"] = ly
    return res


def _select_lead_snps(
    frame: pd.DataFrame, *, window_bp: int = 1_000_000, limit: int | None = None
) -> list[pd.Series]:
    """Choose strongest peaks; only adjacent accepted positions need distance checks."""
    ordered = frame.sort_values("P", kind="stable")
    positions: dict[str, list[int]] = {}
    selected: list[int] = []
    for row, (chromosome, position) in enumerate(
        zip(ordered["CHR"].astype(str), ordered["BP"], strict=True)
    ):
        bp = int(position)
        accepted = positions.setdefault(chromosome, [])
        slot = bisect_left(accepted, bp)
        if slot > 0 and bp - accepted[slot - 1] < window_bp:
            continue
        if slot < len(accepted) and accepted[slot] - bp < window_bp:
            continue
        accepted.insert(slot, bp)
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return [ordered.iloc[row] for row in selected]


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

    p_values = gwas_col(gdf, "P")
    valid = np.isfinite(p_values) & p_values.between(0, 1) & (gwas_col(gdf, "BP") >= 0)
    rejected = int((~valid).sum())
    if rejected:
        print(f"Warning: Dropping {rejected} rows with invalid P values or negative BP.")
    gdf = gdf[valid].copy()
    if gdf.empty:
        print("Error: No valid GWAS rows (P must be finite in [0, 1] and BP nonnegative).")
        sys.exit(1)

    gdf["-log10P"] = -np.log10(gwas_col(gdf, "P").clip(lower=1e-300))
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
    suggest_color = "#6A1B9A"  # Deep purple contrasts with the chromosome point cloud.

    # 3. Peak lead SNP selection for annotation
    sig_mask = gdf["P"] <= args.sig_threshold
    if sig_mask.any():
        lead_snps = _select_lead_snps(gdf.loc[sig_mask])
    else:
        suggestive = gdf.loc[gdf["P"] <= args.suggestive_threshold]
        lead_snps = _select_lead_snps(suggestive if not suggestive.empty else gdf, limit=5)

    max_stat = float(gdf["-log10P"].max())
    y_max = max(max_stat + 1.8, -np.log10(args.sig_threshold) + 2.0)
    last_chr = unique_chrs[-1]
    total_x_len = max(
        float(gdf["x"].max() * 1.01),
        float(offset_map[last_chr] + chr_max_map[last_chr]),
        1.0,
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
        # configured suggestive/significant peaks and P <= 1e-4 are preserved, and ensures
        # every retained point remains a real Bokeh glyph with active hover tooltips.
        high_mask = gdf["P"] <= max(1e-4, args.sig_threshold, args.suggestive_threshold)
        low_df = gdf[~high_mask].copy()

        n_x_bins = 2500
        n_y_bins = 100
        x_bins = np.linspace(0, total_x_len, n_x_bins)
        y_bins = np.linspace(0, y_max + 0.1, n_y_bins)

        low_df["x_bin"] = np.digitize(low_df["x"], x_bins)
        low_df["y_bin"] = np.digitize(low_df["-log10P"], y_bins)

        sampled_low = low_df.drop_duplicates(subset=["x_bin", "y_bin"])
        sampled_low = sampled_low.drop(columns=["x_bin", "y_bin"])
        plot_df = pd.concat([sampled_low, gdf[high_mask]], ignore_index=True)
    else:
        plot_df = gdf.copy()

    # Cast fields for memory efficiency and tooltip precision
    plot_df["x"] = plot_df["x"].astype("int64")
    plot_df["-log10P"] = plot_df["-log10P"].astype("float32")
    plot_df["BP"] = plot_df["BP"].astype("int64")
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

    layers: list[hv.Element] = [_as_element(base_points)]

    # Significant hits overlay
    if sig_mask.any():
        sig_df = gdf.loc[sig_mask, ["x", "-log10P", "SNP", "CHR", "BP", "P"]].copy()
        sig_df["x"] = sig_df["x"].astype("int64")
        sig_df["-log10P"] = sig_df["-log10P"].astype("float32")
        sig_df["BP"] = sig_df["BP"].astype("int64")
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
                show_legend=True,
            )
        else:
            sig_pts = sig_pts.opts(
                color=sig_color,
                s=36,
                alpha=0.95,
                edgecolors="#330000",
                linewidth=0.5,
                show_legend=True,
            )
        layers.append(_as_element(sig_pts))

    # Threshold lines in legend
    if backend == "bokeh":
        s_line_opts = {
            "color": suggest_color,
            "line_dash": "dashed",
            "line_width": 1.2,
            "show_legend": True,
        }
        sig_line_opts = {
            "color": sig_color,
            "line_dash": "dashed",
            "line_width": 1.5,
            "show_legend": True,
        }
    else:
        s_line_opts = {
            "color": suggest_color,
            "linestyle": "--",
            "linewidth": 1.0,
            "show_legend": True,
        }
        sig_line_opts = {
            "color": sig_color,
            "linestyle": "--",
            "linewidth": 1.2,
            "show_legend": True,
        }

    sig_y = -np.log10(args.sig_threshold)
    sig_line = hv.Curve(
        [(0.0, sig_y), (float(total_x_len), sig_y)],
        label=f"Significant (P={args.sig_threshold:g})",
    ).opts(**sig_line_opts)
    layers.append(_as_element(sig_line))

    if args.suggestive_threshold > 0:
        s_y = -np.log10(args.suggestive_threshold)
        s_line = hv.Curve(
            [(0.0, s_y), (float(total_x_len), s_y)],
            label=f"Suggestive (P={args.suggestive_threshold:g})",
        ).opts(**s_line_opts)
        layers.append(_as_element(s_line))

    # Keep the established static layout; apply screen-space collision placement
    # only to the interactive view, where it can be recalculated during zoom.
    if backend == "bokeh":
        lead_df = lead_label_frame(
            lead_snps,
            total_x_len=total_x_len,
            y_max=y_max,
            w_box_ratio=0.045,
            pad_y=y_max * 0.025,
            h_box=y_max * 0.035,
        )
    else:
        lead_df = lead_label_frame(lead_snps, total_x_len=total_x_len, y_max=y_max)
    if lead_df is not None:
        if backend == "bokeh":
            lead_df = _position_bokeh_labels(lead_df, total_x_len, y_max + 0.7)
            lead_df["lx"] = lead_df["x"] + lead_df["x_offset"] * total_x_len / BOKEH_WIDTH
            lead_df["ly"] = lead_df["-log10P"] + lead_df["y_offset"] * (y_max + 0.7) / BOKEH_HEIGHT
        else:
            lead_df["text_align"] = "center"
        if backend != "bokeh":
            static_width_px, static_height_px = 14.0 * 72.0, 5.5 * 72.0
            x_shift_px = (lead_df["lx"] - lead_df["x"]) / total_x_len * static_width_px
            y_shift_px = (
                (lead_df["ly"] - lead_df["-log10P"]) / (ylim[1] - ylim[0]) * static_height_px
            )
            shift_len = np.maximum(np.hypot(x_shift_px, y_shift_px), 1.0)
            segment_df = lead_df.assign(
                x0=lead_df["x"] + x_shift_px / shift_len * 6.0 / static_width_px * total_x_len,
                y0=lead_df["-log10P"]
                + y_shift_px / shift_len * 6.0 / static_height_px * (ylim[1] - ylim[0]),
            )
            connectors = hv.Segments(
                segment_df[["x0", "y0", "lx", "ly"]],
                kdims=["x0", "y0", "lx", "ly"],
            )
            connectors = connectors.opts(color="#666666", alpha=0.65, linewidth=0.65)
            layers.append(_as_element(connectors))
        if backend == "bokeh":
            # Keep each interactive label attached to its actual SNP coordinate.
            # Collision-aware pixel offsets are recalculated when the user zooms.
            label_pts = hv.Labels(
                lead_df[["x", "-log10P", "SNP", "x_offset", "y_offset", "text_align"]],
                kdims=["x", "-log10P"],
                vdims=["SNP", "x_offset", "y_offset", "text_align"],
            ).opts(
                text_font_size="7pt",
                text_color="#111111",
                text_baseline="bottom",
                text_align="center",
                show_legend=False,
                hooks=[_bokeh_label_layout],
            )
            layers.append(_as_element(label_pts))
        else:
            for alignment, group in lead_df.groupby("text_align", sort=False):
                label_pts = hv.Labels(
                    group[["lx", "ly", "SNP"]],
                    kdims=["lx", "ly"],
                    vdims=["SNP"],
                    label=f"Lead labels {alignment}",
                ).opts(
                    size=7.5,
                    color="#111111",
                    verticalalignment="bottom",
                    horizontalalignment=str(alignment),
                    show_legend=False,
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
            show_legend=True,
            legend_position="bottom",
            legend_opts={"location": "bottom_right", "orientation": "horizontal"},
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
                show_legend=True,
                legend_position="top_right",
            )
        )
        hv.save(manhattan_layout, args.output, backend="matplotlib", dpi=300)
        print(f"Manhattan plot saved to: {args.output}")


if __name__ == "__main__":
    main(sys.argv[1:])
