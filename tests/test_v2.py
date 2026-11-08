"""Тесты train_v2: аналоги, выбор параметров, формат выходных файлов (без RDKit и без сети)."""
import numpy as np
import pandas as pd

from src import train_v2 as t2


def test_neighbor_info_best_and_ties():
    k = np.array([[0.2, 0.95, 0.94, 0.5], [0.1, 0.3, 0.2, 0.25]])
    y = np.array([10.0, 100.0, 300.0, 5.0])
    s1, v, idx = t2.neighbor_info(k, y)
    assert s1.tolist() == [0.95, 0.3]
    assert v[0] == 200.0          # соседи 0.95 и 0.94 - "ничья" в пределах tol -> среднее 100 и 300
    assert v[1] == 100.0          # единственный лучший сосед
    assert idx.tolist() == [1, 1]


def test_combine_gating():
    p, v = np.full(3, 100.0), np.full(3, 1000.0)
    out = t2.combine(p, np.array([1.0, 0.85, 0.5]), v, wmax=0.4, s0=0.7)
    assert np.isclose(out[0], 0.6 * 100 + 0.4 * 1000)   # сходство 1.0 -> полный wmax
    assert out[2] == 100.0                               # ниже s0 - без изменений
    assert 100.0 < out[1] < out[0]                       # промежуточное сходство - промежуточный вес
    assert (t2.combine(p, np.ones(3), v, wmax=0.0, s0=0.7) == p).all()


def _synthetic(n=200, seed=0):
    rng = np.random.RandomState(seed)
    y = np.exp(rng.normal(3, 1.5, n))
    ens = np.log1p(y)[None, :] + rng.normal(0, 0.3, (2, n))
    return y, ens


def test_select_params_enables_helpful_analogs():
    y, ens = _synthetic()
    s1 = np.ones((2, len(y)))
    vnb = np.stack([y * np.exp(np.random.RandomState(1).normal(0, 0.05, len(y)))] * 2)  # почти точные аналоги
    params, table = t2.select_params(ens, s1, vnb, y)
    assert params["wmax"] == t2.DEFAULT_WMAX and params["cv_delta_win997_pct"] < 0
    assert min(table.values()) < 0


def test_select_params_disables_harmful_analogs_unless_forced():
    y, ens = _synthetic()
    s1 = np.ones((2, len(y)))
    vnb = np.stack([y[::-1] * 1000.0] * 2)  # вредные "аналоги"
    params, _ = t2.select_params(ens, s1, vnb, y)
    assert params["wmax"] == 0.0 and params["cv_delta_win997_pct"] > 100 * t2.MAX_CV_LOSS
    forced, _ = t2.select_params(ens, s1, vnb, y, wmax_override=0.3)
    assert forced["wmax"] == 0.3   # явное требование пользователя не отменяется


def test_scale_search_recovers_known_bias():
    # без тяжёлого хвоста (иначе winsorize-критерий намеренно смещает c вниз)
    y = np.random.RandomState(2).uniform(1, 100, 400)
    ens = np.log1p(y / 4.0)[None, :]            # модель систематически занижает в 4 раза
    params, _ = t2.select_params(ens, np.zeros((1, 400)), np.zeros((1, 400)), y)
    assert 3.5 <= params["c"] <= 4.5


def test_end_to_end_v2_files_and_format(tmp_path):
    t2.main(["--no-rdkit", "--skip-cv", "--n-jobs", "1", "--out", str(tmp_path)])
    sample = pd.read_csv("data/raw/sample_submit.csv")
    for name in ("submission.csv", "submission_no_analog.csv", "submission_analog_strong.csv"):
        sub = pd.read_csv(tmp_path / name)
        assert list(sub.columns) == ["index", "SI VACV"] and (sub["index"] == sample["index"]).all()
        assert np.isfinite(sub["SI VACV"]).all() and (sub["SI VACV"] >= 0).all()
    assert (tmp_path / "analog_changes.csv").exists() and (tmp_path / "cv_report_v2.json").exists()
