"""Тесты ключевых гарантий пайплайна (работают без RDKit и без сети)."""
import numpy as np
import pandas as pd

from src import features as feat
from src import train


def test_forbidden_columns_guard():
    feat.assert_no_forbidden_columns(["Title", "strain", "Culture", "SMILES"])
    for col in ("CC50_uM", "IC50_uM_VACV"):
        try:
            feat.assert_no_forbidden_columns(["SMILES", col])
        except AssertionError:
            continue
        raise RuntimeError(f"{col} должен быть запрещён")


def test_model_input_has_only_allowed_columns():
    Xtr, y, Xte, idx = train.load_competition("data/raw/train_v2.csv", "data/raw/test.csv")
    assert list(Xtr.columns) == train.ALLOWED_COLS == list(Xte.columns)
    assert not set(feat.FORBIDDEN_COLUMNS) & set(Xtr.columns)
    assert len(Xtr) == len(y) and len(Xte) == len(idx)


def test_group_folds_keep_groups_together_and_cover_all():
    groups = np.repeat(np.arange(40), [1, 2, 3, 4] * 10)
    folds = train.group_folds(groups, 5, seed=0)
    seen = np.concatenate([te for _, te in folds])
    assert sorted(seen) == list(range(len(groups)))
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])


def test_group_folds_deterministic():
    g = np.arange(100) % 17
    a = train.group_folds(g, 5, seed=1)
    b = train.group_folds(g, 5, seed=1)
    assert all((x[1] == y[1]).all() for x, y in zip(a, b))


def test_fit_scale_is_bounded():
    y = np.array([0, 1, 10, 100, 100000.0])
    assert train.C_BOUNDS[0] <= train.fit_scale(np.array([1, 1, 1, 1, 1.0]), y) <= train.C_BOUNDS[1]
    assert train.fit_scale(np.zeros(5), y) == train.C_BOUNDS[0]  # нулевые предсказания -> нижняя граница (без раздувания)


def test_context_features_use_only_allowed_info():
    df = pd.DataFrame({"Title": ["sid1", "7b-x-1", "12", "bromantane"],
                       "strain": ["Vaccinia virus Copenhagen"] * 4, "Culture": ["Vero", "HeLa cells", "Vero", "BSC-40"],
                       "SMILES": ["CCO"] * 4})
    toks = feat.context_text(df)
    assert "t_sid" in toks[0] and "t_dashcode" in toks[1] and "t_number" in toks[2] and "t_name" in toks[3]
    assert "c_vero" in toks[0] and "s_copenhagen" in toks[0]


def test_end_to_end_fallback_submission_format(tmp_path):
    train.main(["--no-rdkit", "--out", str(tmp_path), "--n-jobs", "1", "--cv-repeats", "1", "--skip-cv"])
    sub = pd.read_csv(tmp_path / "submission.csv")
    sample = pd.read_csv("data/raw/sample_submit.csv")
    assert list(sub.columns) == list(sample.columns) == ["index", "SI VACV"]
    assert (sub["index"] == sample["index"]).all()
    assert np.isfinite(sub["SI VACV"]).all() and (sub["SI VACV"] >= 0).all()
