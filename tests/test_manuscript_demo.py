"""Exercise the documented manuscript CLI workflow using the bundled demo."""

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_manuscript_demo(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts" / "manuscript"
    demo = root / "samples_images" / "results" / "annotations_head_region"
    overlays = tmp_path / "overlays"
    polygons = tmp_path / "polygons"
    features = tmp_path / "features.xlsx"
    scores = tmp_path / "CYS.xlsx"

    subprocess.run(
        [
            sys.executable,
            str(scripts / "segment_yellow_heart.py"),
            "--input",
            str(demo),
            "--output",
            str(overlays),
            "--json-output",
            str(polygons),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "extract_color_features.py"),
            "--input",
            str(polygons),
            "--images",
            str(demo),
            "--output",
            str(features),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts / "calculate_cys.py"),
            "--input",
            str(features),
            "--output",
            str(scores),
        ],
        check=True,
    )

    expected = {path.stem for path in demo.glob("*.json")}
    assert expected
    assert {path.stem for path in polygons.glob("*.json")} == expected
    result = pd.read_excel(scores)
    assert len(result) == len(expected)
    assert {Path(name).stem for name in result["filename"]} == expected
    assert result["Yellow_Ratio"].between(0, 1).all()
    assert result["CYS"].between(0, 1).all()
