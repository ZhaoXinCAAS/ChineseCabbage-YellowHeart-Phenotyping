"""Tests for baseline_compare (Reviewer-1 Q5)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from yheart.scoring.baseline_compare import (
    build_models,
    cross_validate,
    run,
)
from yheart.scoring.fisher_scores import (
    FEATURES,
    LABEL_ORDER,
)


def _labeled_bundle(tmp_path: Path, n: int = 18, seed: int = 0) -> tuple[str, str]:
    rng = np.random.default_rng(seed)
    fnames: list[str] = []
    bvals: list[float] = []
    labels: list[str] = []
    for idx, lab in enumerate(LABEL_ORDER):
        for i in range(n):
            fn = f"{lab}_{i}.jpg"
            fnames.append(fn)
            bvals.append([0.1, 0.5, 0.9][idx] + rng.normal(0, 0.02))
            labels.append(lab)
    df = pd.DataFrame({"filename": fnames, "b_mean": np.clip(bvals, 0, 1)})
    for f in FEATURES:
        if f != "b_mean":
            df[f] = 0.3 + rng.normal(0, 0.01, len(df))
    feat = tmp_path / "feat.xlsx"
    df.to_excel(feat, index=False)
    root = tmp_path / "labels"
    for lab in LABEL_ORDER:
        d = root / lab
        d.mkdir(parents=True)
        for fn, lb in zip(fnames, labels):
            if lb == lab:
                (d / fn).write_text("")
    return str(feat), str(root)


def test_all_models_score_high_on_separable_data(tmp_path: Path):
    feat, root = _labeled_bundle(tmp_path)
    out = tmp_path / "m.csv"
    conf = tmp_path / "c.csv"
    assert run(feat, root, str(out), str(conf), 3, 42) is True
    m = pd.read_csv(out)
    assert set(m["model"]) == set(build_models())
    # Discriminant-style and tree models nail separable Gaussians; every
    # model must at least beat chance (1/3) by a wide margin.
    assert float(m.loc[m["model"] == "LDA(FDA)", "accuracy"].iloc[0]) > 0.9
    assert float(m.loc[m["model"] == "RF", "accuracy"].iloc[0]) > 0.9
    assert (m["accuracy"] > 0.6).all()
    assert (m["macro_f1"] > 0.6).all()
    c = pd.read_csv(conf)
    assert set(c["model"]) == set(build_models())
    # Every sample is predicted exactly once per model.
    n_samples = len(pd.read_excel(feat))
    assert c["count"].sum() == len(build_models()) * n_samples


def test_n_splits_shrinks_for_tiny_classes():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(v, 0.05, (3, len(FEATURES))) for v in (0.1, 0.5, 0.9)])
    y = np.array([L for L in LABEL_ORDER for _ in range(3)])
    metrics, _ = cross_validate(X, y, build_models(), n_splits=5, random_state=0)
    assert (metrics["n_splits"] == 3).all()


def test_no_labels_exits_clean(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    feat = tmp_path / "feat.xlsx"
    pd.DataFrame({
        "filename": ["x.jpg"],
        "b_mean": [0.1],
        **{f: [0.2] for f in FEATURES if f != "b_mean"},
    }).to_excel(feat, index=False)
    root = tmp_path / "empty"
    root.mkdir()
    assert (
        run(str(feat), str(root), str(tmp_path / "m.csv"), str(tmp_path / "c.csv"), 3, 42) is False
    )
    assert "No labeled samples" in capsys.readouterr().out
