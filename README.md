# Chinese Cabbage Yellow-Heart Phenotyping

This repository accompanies the HPJ manuscript on quantitative grading of
yellow-heart appearance in Chinese cabbage. It covers head-region processing,
yellow-heart segmentation, color measurements, the Comprehensive Yellow Score
(CYS), and grading validation.

## Installation

Use Python 3.13 and install the locked dependencies with `uv sync`. For
development checks, use `uv sync --extra dev` and run
`uv run ruff check src tests`, `uv run pyrefly check`, and `uv run pytest`.

## Data and workflow

The manuscript scripts start from head images with polygon annotations.
The upstream acquisition workflow uses SAM 3 with the text prompt `cabbage`,
head cropping, and manual annotation refinement in X-AnyLabeling. If starting
from raw images, supply the original images and SAM 3 JSON to `crop-heads`
with `--input`, `--images`, and `--output` before manual refinement.

<p align="center">
  <img width="85%" alt="SAM 3 head-region segmentation demonstration" src="https://github.com/user-attachments/assets/2f37b9a9-11a8-4734-aed9-fb5d39ef488f" />
</p>

The Git repository includes compact annotated crops for a demo and the existing
study feature and CYS tables under `full_dataset_tables/`. Upstream raw photos,
SAM 3 outputs, and the full annotation and expert workbooks are supplied with
the study dataset. Follow the manuscript's Data and Code Availability statement
for access (Figshare DOI `10.6084/m9.figshare.33154715`).

Run the demo from the repository root using a new output directory.

```bash
uv run python scripts/manuscript/segment_yellow_heart.py \
  --input samples_images/results/annotations_head_region \
  --output results/demo/overlays --json-output results/demo/polygons
uv run python scripts/manuscript/extract_color_features.py \
  --input results/demo/polygons \
  --images samples_images/results/annotations_head_region \
  --output results/demo/features.xlsx
```

The demo checks the image-processing entry points. Use the study cohort and
archived normalization and grading inputs for the manuscript statistics.

Examples of head-region annotations and automated yellow-heart extraction.

<p align="center">
  <img width="31%" alt="Head annotation and yellow-heart extraction example 1" src="https://github.com/user-attachments/assets/491e65cc-eb52-494b-aa68-cd3a766d9327" />
  <img width="31%" alt="Head annotation and yellow-heart extraction example 2" src="https://github.com/user-attachments/assets/efe18338-9b2b-43fe-8f9d-ff1cc1ae6141" />
  <img width="31%" alt="Head annotation and yellow-heart extraction example 3" src="https://github.com/user-attachments/assets/9f7cceca-dcd7-404b-8321-dbfbd9706ac7" />
</p>

The package-level segmentation and evaluation commands also support engineering
checks. Use the commands below for the archived manuscript rules and tables.

## HPJ manuscript reproduction scripts

The scripts in [`scripts/manuscript/`](scripts/manuscript/) reproduce specific
analyses from the HPJ revision. Install the repository dependencies with
`uv sync`, then run the commands from the repository root. Pass inputs and
outputs explicitly. The paths below are examples relative to the current
working directory. Place the corresponding study inputs under `data/` or pass
their actual locations. Full study images and rating workbooks are distributed
separately.

Run each example as `uv run python scripts/manuscript/<command>`.

| Command and arguments | Input and output |
| --- | --- |
| `segment_yellow_heart.py --input data/heads --output results/segmentation/visualizations --json-output results/segmentation/json` | Head-region images and JSON polygons. Writes annotated JSON and contour overlays. |
| `extract_color_features.py --input results/segmentation/json --images data/heads --output results/features.xlsx` | Yellow-region JSON and original, unmarked images. Writes feature means, yellow-area ratios, and Min–Max normalized values. |
| `fisher_table3.py --input data/418_FisherScore_roi_10_color_features.xlsx --labels-dir data/Fisher_score_418 --output results/fisher.xlsx` | ROI feature workbook and phenotype subfolders. Writes ranking, direction and weight information. |
| `classifier_benchmark.py --input data/Fisher_score_LDA_SVM.xlsx --output results/classifier_benchmark.xlsx` | Labeled ten-feature workbook. Writes the five-classifier benchmark for Table S2. |
| `calculate_cys.py --input results/features.xlsx --output results/CYS.xlsx` | Normalized image features and yellow-area ratio. Reproduces the archived YS/CYS workbook with fixed manuscript weights. |
| `gmm_cys.py --input data/CYS.xlsx --output-dir results/gmm` | Table with a continuous `CYS` column. Use the archived input for the historical fit or a rebuilt workbook for a new fit. Writes `CYS_GMM_Analysis_Results.xlsx` and the statistical panels for Fig. 5A–C. |
| `cys_alternative_forms.py --input data/CYS_Add_Geo_Har.xlsx --output results/cys_forms.xlsx` | Workbook with `Yellow_Ratio`, `Yellow_score`, and `CYS`. Writes four CYS formulations. |
| `cys_form_comparison_table.py --input data/Experts_200_with_All_CYS_Models.xlsx --output results/cys_form_comparison.xlsx` | Workbook with the four formulations and `Final_Grade`. Writes the correlations for Table S4. |
| `paired_mask_cys.py data/Auto_Manual_YS_YR_CYS_segmentation_metrics_summary.xlsx results/fig6` | Automatic and manual YS/YR/CYS for 150 paired images. Writes Fig. 6 and the Table S6 statistics workbook. Reads unrounded T1–T4 from `results/gmm/CYS_GMM_Analysis_Results.xlsx` by default. Set `--gmm-workbook` to use the archived workbook or another fit, or supply `--thresholds T1 T2 T3 T4` directly. |
| `expert_agreement.py data/Five_Expert.xlsx results/expert_agreement.xlsx --fisher-features data/418_FisherScore_roi_10_color_features.xlsx` | Five expert ratings and algorithm grades. The optional Fisher feature table adds agreement by image overlap and a per-image membership sheet. |
| `plot_expert_agreement.py data/Five_Expert.xlsx results/fig7` | Same workbook. Writes the Fig. 7 panels as SVG, PDF and 600-dpi PNG. |
| `evaluate_paired_segmentation.py data/paired_annotations results/mask_overlap.xlsx` | JSON files carrying both `Yellow_area` and `Yellow_area_manual` polygons, optionally arranged in strata subdirectories. The same command accepts the flat low-light dataset. Writes image-level and grouped IoU/Dice. |
| `extract_paired_mask_features.py data/reference_annotations data/paired_annotations results/paired_features` | Uses automatic masks in the reference dataset to set the Min–Max ranges, then writes `automatic_mask_features.xlsx` and `manual_mask_features.xlsx`. Score each with `calculate_cys.py`, then pass the scored tables to `paired_mask_cys.py AUTO.xlsx OUT_DIR --manual MANUAL.xlsx`. |
| `plot_segmentation_ecdf.py data/segmentation_metrics.xlsx results/ecdf` | Per-image IoU with difficulty categories. Writes the stratified ECDF and its descriptive statistics. |

The manuscript scripts preserve their archived analysis rules. The package
commands described above support the repository's engineering workflow.
In particular:

- The archived mask-generation script applies `ExR > 0.15` OR `20 <= b <= 70`
  to the **encoded OpenCV 8-bit** Lab-b channel. This part of the rule corresponds
  to approximately `-108 <= b* <= -58` after subtracting 128. Section 3.2
  reports the encoded range. On the 1,319 retained images, the archived
  implementation reproduced all 1,319 contour
  vertex sequences. None of the 530,514,122 head-region pixels met the encoded
  b-channel criterion, so ExR determined candidate selection on these images.
  Historical extraction uses the head polygon as the area denominator and
  omits measurements without a yellow polygon. The archived YS and CYS were
  reproduced to floating-point precision by
  `YS = 0.2931 ExR_norm + 0.2595 S_norm + 0.2272 b*_norm + 0.2202 (1 - B_norm)`
  and `CYS = YS × YR`.
- The manuscript classifier benchmark uses stratified image-level folds
  to reproduce Table S2. Images from the same accession may occur in
  different folds.
- Preserve the full-precision GMM cut points from the generated workbook when
  grading. The archived 1,319-image run used approximately
  0.0675848797, 0.1434605076, 0.2327176455, and 0.3764637203. The main text
  displays the cut points to four decimal places. Grade assignments use the
  workbook cutoffs, giving counts of 367, 491, 261, 151 and 49. A score equal
  to a cutoff belongs to the lower grade by default. The 150 paired-mask
  grade pairs and 200 expert-image algorithm grades are unchanged.
- Matching by single-head filename (preserving the head identifier and
  ignoring image extension) identifies 75 expert images used for Fisher
  weighting and 125 outside that set. The 125-image subset has exact agreement
  of 88/125, within-one-grade agreement of 122/125 and quadratic-weighted
  kappa of 0.8493. This subset still comes from the same image collection and
  may share accessions with the Fisher cohort.
- For exact reproduction of the archived GMM fit, use its original `CYS.xlsx`
  input, including the original row order. The original normalized feature
  workbook and original GMM input have different row orders. A CYS workbook
  rebuilt from the feature table preserves that table's row order, which can
  affect initialization and change the fitted cut points.

## Citation

If you use this pipeline or data in your research, please cite this repository
and the Figshare dataset.
