"""Tests for deterministic stratified image sampling."""

from pathlib import Path

import pandas as pd
import pytest


from yheart.imaging.sample_images import sample_tree


def _tree(root: Path) -> None:
    for group, count in (("Grade_1", 5), ("Grade_2", 3)):
        folder = root / group
        folder.mkdir(parents=True)
        for i in range(count):
            (folder / f"{group}_{i}.png").write_bytes(b"image")


def test_samples_per_folder_with_group_prefix(tmp_path: Path):
    source = tmp_path / "images"
    _tree(source)
    manifest = sample_tree(str(source), str(tmp_path / "subset"), per_folder=2, seed=1)
    assert len(manifest) == 4
    assert set(manifest["group"]) == {"Grade_1", "Grade_2"}
    assert (manifest["group"] == "Grade_1").sum() == 2
    for row in manifest.itertuples():
        target = Path(str(row.output))
        assert target.exists()
        assert target.name.startswith(f"{row.group}_")


def test_same_seed_reproduces_selection(tmp_path: Path):
    source = tmp_path / "images"
    _tree(source)
    first = sample_tree(str(source), str(tmp_path / "a"), per_folder=2, seed=7)
    second = sample_tree(str(source), str(tmp_path / "b"), per_folder=2, seed=7)
    assert list(first["filename"]) == list(second["filename"])


def test_group_filter_and_cap(tmp_path: Path):
    source = tmp_path / "images"
    _tree(source)
    manifest = sample_tree(
        str(source), str(tmp_path / "subset"), per_folder=10, groups=["Grade_2"]
    )
    assert len(manifest) == 3
    assert set(manifest["group"]) == {"Grade_2"}
    with pytest.raises(ValueError, match="not found"):
        sample_tree(str(source), str(tmp_path / "x"), groups=["Grade_9"])


def test_output_inside_input_is_rejected(tmp_path: Path):
    source = tmp_path / "images"
    _tree(source)
    with pytest.raises(ValueError, match="outside"):
        sample_tree(str(source), str(source / "subset"))


def test_manifest_round_trips(tmp_path: Path):
    source = tmp_path / "images"
    _tree(source)
    manifest = sample_tree(str(source), str(tmp_path / "subset"), per_folder=1, seed=3)
    path = tmp_path / "manifest.csv"
    manifest.to_csv(path, index=False)
    assert list(pd.read_csv(path).columns) == ["group", "filename", "source", "output"]
