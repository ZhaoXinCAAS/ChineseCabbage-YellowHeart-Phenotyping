# Chinese Cabbage Yellow-Heart Phenotyping

[![CI](https://github.com/ZhaoXinCAAS/ChineseCabbage-YellowHeart-Phenotyping/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ZhaoXinCAAS/ChineseCabbage-YellowHeart-Phenotyping/actions/workflows/ci.yml)

[Data guide](docs/data.md) · [Reproduction commands](#manuscript-reproduction-scripts) ·
[Contributing](CONTRIBUTING.md) · [Citation](CITATION.cff) · [License](LICENSE)

This repository accompanies the research manuscript on quantitative grading of
yellow-heart appearance in Chinese cabbage. It covers head-region processing,
yellow-heart segmentation, color measurements, the Comprehensive Yellow Score
(CYS), and grading validation.

## Installation

Clone the repository, open a shell in its root, and install the locked
environment with [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/ZhaoXinCAAS/ChineseCabbage-YellowHeart-Phenotyping.git
cd ChineseCabbage-YellowHeart-Phenotyping
uv sync --locked --python 3.13
```

See [Contributing](CONTRIBUTING.md#local-checks) for the developer environment
and the checks run by CI.

## Data and workflow

The manuscript scripts start from head images with polygon annotations.
The upstream acquisition workflow uses SAM 3 with the text prompt `cabbage`,
head cropping, and manual annotation refinement in X-AnyLabeling.

<p align="center">
  <img width="85%" alt="Yeqiu_predict" src="https://github.com/user-attachments/assets/2f37b9a9-11a8-4734-aed9-fb5d39ef488f" />
</p>

If starting from raw images, supply the original images and SAM 3 JSON to
`crop-heads` with `--input`, `--images`, and `--output` before manual refinement.

The Git repository includes compact annotated crops for a demo and the existing
study feature and CYS tables under `full_dataset_tables/`. Upstream raw photos,
SAM 3 outputs, and the full annotation and expert workbooks are supplied with
the study archive. See the [data guide](docs/data.md) for the bundled workbooks,
required study inputs, and archive access.

Run the demo from the repository root using a new output directory.

```bash
uv run --locked python scripts/manuscript/segment_yellow_heart.py \
  --input samples_images/results/annotations_head_region \
  --output results/demo/overlays --json-output results/demo/polygons
uv run --locked python scripts/manuscript/extract_color_features.py \
  --input results/demo/polygons \
  --images samples_images/results/annotations_head_region \
  --output results/demo/features.xlsx
uv run --locked python scripts/manuscript/calculate_cys.py \
  --input results/demo/features.xlsx --output results/demo/CYS.xlsx
```

The demo writes contour overlays, polygon JSON, a feature workbook, and a CYS
workbook to `results/demo/`. Demo scores use normalization fitted to the demo
images. Use the study cohort and archived inputs for the manuscript statistics.

Examples of head-region annotations and automated yellow-heart extraction.

<table border="0">
  <tr>
    <td width="33%" align="center" valign="middle">
      <img height="250px" src="https://github.com/user-attachments/assets/491e65cc-eb52-494b-aa68-cd3a766d9327" alt="Yeqiu_predict_3" />
    </td>
    <td width="33%" align="center" valign="middle">
      <img height="250px" src="https://github.com/user-attachments/assets/efe18338-9b2b-43fe-8f9d-ff1cc1ae6141" alt="Yeqiu_predict_2" />
    </td>
    <td width="33%" align="center" valign="middle">
      <img height="250px" src="https://github.com/user-attachments/assets/9f7cceca-dcd7-404b-8321-dbfbd9706ac7" alt="Yeqiu_predict_1" />
    </td>
  </tr>
</table>

The package-level segmentation and evaluation commands also support engineering
checks. Use the commands below for the archived manuscript rules and tables.

## Manuscript reproduction scripts

The scripts in [`scripts/manuscript/`](scripts/manuscript/) reproduce specific
analyses reported in the manuscript. Install the repository dependencies with
`uv sync --locked`, then run the commands from the repository root. Pass inputs and
outputs explicitly. The paths below are examples relative to the current
working directory. Place the corresponding study inputs under `data/` or pass
their actual locations. Full study images and rating workbooks are distributed
separately.

Run each example as `uv run --locked python scripts/manuscript/<command>`.

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

### Reproduction notes

- The archived mask workflow uses ExR and the encoded OpenCV Lab `b` channel.
  See `scripts/manuscript/segment_yellow_heart.py` for the exact criteria. The
  filled head polygon is the area-ratio denominator. Use the generated feature
  workbook and `CYS.xlsx` to inspect the per-image measurements and scores.
- The classifier benchmark reproduces Table S2 using stratified image-level
  folds. Images from the same accession may occur in different folds. See the
  generated `classifier_benchmark.xlsx` for the results.
- For the historical GMM fit, use the archived GMM input `CYS.xlsx` in its
  original row order. A workbook rebuilt from the feature table can have a
  different row order, which can affect initialization. Use the stored values
  in the `GMM_thresholds` sheet of `CYS_GMM_Analysis_Results.xlsx` for grading.
  A score equal to a cut point belongs to the lower grade. See `Grade_summary`
  in the same workbook for the grade counts.
- See the paired-mask output workbook for mask agreement and grade changes.
  In the expert agreement output workbook, `Fisher_Overlap` reports agreement
  by overlap status and `Fisher_Overlap_Images` records per-image membership.
  Matching uses the single-head filename, including the head identifier and
  ignoring the extension. Images outside the Fisher set still come from the
  same collection and may share accessions with that set.

## Citation

Cite the software using [CITATION.cff](CITATION.cff) or GitHub's **Cite this
repository** button. Record the commit used (`git rev-parse HEAD`) and cite the
study data separately using the archive metadata when available.

## License

The software is distributed under the [MIT License](LICENSE). See the
[data guide](docs/data.md#study-archive) for terms associated with the archive.
