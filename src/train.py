"""Обучение модели SI VACV, кросс-валидация и генерация submission.csv.

Запуск (см. README.md):
    python -m src.train --train data/raw/train_v2.csv --test data/raw/test.csv \
        --sample data/raw/sample_submit.csv --out outputs

С внешними данными (результат collect_data.py + validate_data.py):
    python -m src.train ... --external data/external/vacv_external_clean.csv

Принципы:
  * Колонки CC50_uM / IC50_uM_VACV НЕ используются (запрещено правилами) - в признаки
    передаются только Title, strain, Culture, SMILES.
  * Цель моделируется как log1p(SI); итог = c * expm1(среднее по моделям), c подбирается по OOF.
  * Внешние данные идут только в обучающую часть; оценка CV - строго на строках соревнования.
    Внешние строки, совпадающие с молекулами валидационной свёртки, исключаются (нет утечки).
  * Всё детерминировано (seed), поэтому submission.csv воспроизводим.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR

from src import features as feat
from src.features import Featurizer, assert_no_forbidden_columns

TARGET = "SI VACV"
ALLOWED_COLS = ["Title", "strain", "Culture", "SMILES"]  # единственное, что видит Featurizer
WIN_Q = 0.99          # квантиль winsorize для устойчивых метрик/калибровки
C_BOUNDS = (1.0, 3.0)  # допустимый диапазон масштабного множителя


# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------
def load_competition(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    y = train[TARGET].to_numpy(dtype=float)
    X_train = train[ALLOWED_COLS].copy()
    X_test = test[ALLOWED_COLS].copy()
    assert_no_forbidden_columns(list(X_train.columns) + list(X_test.columns))
    return X_train, y, X_test, test["index"].to_numpy()


def load_external(path):
    """Внешняя БД (формат collect_data.py) -> формат признаков соревнования."""
    ext = pd.read_csv(path)
    need = {"smiles", "si"}
    if not need <= set(ext.columns):
        raise ValueError(f"В {path} нет колонок {need}")
    ext = ext[ext["si"].notna() & (ext["si"] >= 0)].copy()
    out = pd.DataFrame({
        "Title": ext.get("molecule_id", pd.Series("ext", index=ext.index)).astype(str),
        "strain": ext.get("virus_strain", pd.Series("Vaccinia virus", index=ext.index)).fillna("Vaccinia virus"),
        "Culture": ext.get("cell_line", pd.Series("", index=ext.index)).fillna(""),
        "SMILES": ext["smiles"].astype(str),
    })
    return out.reset_index(drop=True), ext["si"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
def group_folds(groups: np.ndarray, n_splits: int, seed: int):
    """Перемешанный GroupKFold: группы целиком попадают в одну свёртку."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    sizes = {g: (groups == g).sum() for g in uniq}
    order = sorted(uniq, key=lambda g: -sizes[g])  # жадная балансировка по размеру
    load = np.zeros(n_splits)
    assign = {}
    for g in order:
        k = int(np.argmin(load + rng.rand(n_splits) * 1e-6))
        assign[g] = k
        load[k] += sizes[g]
    fold_id = np.array([assign[g] for g in groups])
    return [(np.where(fold_id != k)[0], np.where(fold_id == k)[0]) for k in range(n_splits)]


# ---------------------------------------------------------------------------
# Ядра и модели
# ---------------------------------------------------------------------------
def minmax_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """MinMax (Tanimoto для счётчиков) ядро; sum(min)=(|a|+|b|-L1)/2."""
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    sa, sb = A.sum(1)[:, None], B.sum(1)[None, :]
    m = (sa + sb - cdist(A, B, "cityblock")) / 2.0
    return m / np.maximum(sa + sb - m, 1e-9)


class Data:
    """Все строки (train + external + test) в одном наборе признаков + лениво считаемые ядра."""

    def __init__(self, fs: feat.FeatureSet):
        self.fs = fs
        self.X = fs.tree_matrix()
        self._K = None

    @property
    def K(self):
        if self._K is None:
            t = time.time()
            k_mol = minmax_kernel(self.fs.mol_cnt, self.fs.mol_cnt)
            k_ctx = minmax_kernel(self.fs.ctx, self.fs.ctx)
            self._K = k_mol * (0.3 + k_ctx)
            print(f"  [kernel] gram {self._K.shape} за {time.time() - t:.0f}s", flush=True)
        return self._K


def make_models(seed: int, n_jobs: int):
    def et(D, tr, te, y, w):
        m = ExtraTreesRegressor(400, min_samples_leaf=2, max_features=0.3, random_state=seed, n_jobs=n_jobs)
        return m.fit(D.X[tr], y, sample_weight=w).predict(D.X[te])

    def rf(D, tr, te, y, w):
        m = RandomForestRegressor(400, min_samples_leaf=2, max_features=0.3, random_state=seed, n_jobs=n_jobs)
        return m.fit(D.X[tr], y, sample_weight=w).predict(D.X[te])

    def hgb(D, tr, te, y, w):
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                          min_samples_leaf=5, random_state=seed)
        return m.fit(D.X[tr], y, sample_weight=w).predict(D.X[te])

    def svr(D, tr, te, y, w):
        m = SVR(kernel="precomputed", C=3.0, epsilon=0.1)
        return m.fit(D.K[np.ix_(tr, tr)], y, sample_weight=w).predict(D.K[np.ix_(te, tr)])

    return {"ExtraTrees": et, "RandomForest": rf, "HistGB": hgb, "SVR_tanimoto": svr}


# ---------------------------------------------------------------------------
# Метрики и калибровка
# ---------------------------------------------------------------------------
def winsorize(y, q=WIN_Q):
    return np.clip(y, 0, np.quantile(y, q))


def fit_scale(p: np.ndarray, y: np.ndarray, bounds=C_BOUNDS) -> float:
    """Множитель c: argmin ||y_win - c*p||^2, обрезанный в bounds.
    Сложные калибровки (изотоника, степенная) на этих данных переобучаются под выбросы."""
    c = float((winsorize(y) * p).sum() / max((p * p).sum(), 1e-9))
    return float(np.clip(c, *bounds))


def metrics(pred_raw: np.ndarray, y: np.ndarray) -> dict:
    pred_raw = np.clip(pred_raw, 0, None)
    return {
        "MSE_raw": float(np.mean((pred_raw - y) ** 2)),
        "MSE_winsor99": float(np.mean((pred_raw - winsorize(y)) ** 2)),
        "RMSE_log1p": float(np.sqrt(np.mean((np.log1p(pred_raw) - np.log1p(y)) ** 2))),
        "Spearman": float(spearmanr(pred_raw, y)[0]),
    }


def run_cv(D: Data, n_comp: int, y_all: np.ndarray, is_ext: np.ndarray, ext_w: float,
           keys: np.ndarray, groups: np.ndarray, models: dict, folds: int, repeats: int,
           seed: int, use_ext: bool) -> dict:
    """OOF-предсказания (лог-шкала) на строках соревнования. Возвращает {model: [repeats x n_comp]}."""
    comp_idx = np.arange(n_comp)
    ext_idx = np.where(is_ext)[0] if use_ext else np.array([], dtype=int)
    y_log = np.log1p(y_all)
    oof = {k: np.zeros((repeats, n_comp)) for k in models}
    for r in range(repeats):
        for tr_c, te_c in group_folds(groups[comp_idx], folds, seed + r):
            held_keys = set(keys[te_c])
            ext_keep = np.array([i for i in ext_idx if keys[i] not in held_keys], dtype=int)
            tr = np.concatenate([comp_idx[tr_c], ext_keep])
            w = np.concatenate([np.ones(len(tr_c)), np.full(len(ext_keep), ext_w)])
            for name, fn in models.items():
                oof[name][r, te_c] = fn(D, tr, comp_idx[te_c], y_log[tr], w)
    return oof


def summarize(oof: dict, y: np.ndarray):
    reps = next(iter(oof.values())).shape[0]
    ens = np.mean([v for v in oof.values()], axis=0)  # repeats x n
    p_ens = np.expm1(ens)
    c = fit_scale(p_ens.mean(0), y)
    out = {"scale_c": c, "models": {}}
    for name, v in oof.items():
        out["models"][name] = {k: float(np.mean([metrics(np.expm1(v[r]), y)[k] for r in range(reps)]))
                               for k in ("MSE_raw", "MSE_winsor99", "RMSE_log1p", "Spearman")}
    for tag, mult in (("ensemble_c1", 1.0), ("ensemble_scaled", c)):
        out["models"][tag] = {k: float(np.mean([metrics(mult * p_ens[r], y)[k] for r in range(reps)]))
                              for k in ("MSE_raw", "MSE_winsor99", "RMSE_log1p", "Spearman")}
    out["baseline_const_mean"] = {"MSE_raw": float(np.var(y)), "MSE_winsor99": float(np.var(winsorize(y)))}
    return out, ens


def print_report(title: str, rep: dict):
    print(f"\n=== {title} ===")
    print(f"{'model':<18}{'MSE_raw':>12}{'MSE_win99':>12}{'RMSE_log':>10}{'Spearman':>10}")
    for name, m in rep["models"].items():
        print(f"{name:<18}{m['MSE_raw']:>12.0f}{m['MSE_winsor99']:>12.0f}{m['RMSE_log1p']:>10.3f}{m['Spearman']:>10.3f}")
    b = rep["baseline_const_mean"]
    print(f"{'const mean':<18}{b['MSE_raw']:>12.0f}{b['MSE_winsor99']:>12.0f}")
    print(f"scale c = {rep['scale_c']:.3f}")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/raw/train_v2.csv")
    ap.add_argument("--test", default="data/raw/test.csv")
    ap.add_argument("--sample", default="data/raw/sample_submit.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--external", default=None, help="CSV из collect_data.py/validate_data.py")
    ap.add_argument("--external-weight", type=float, default=0.5)
    ap.add_argument("--force-external", action="store_true", help="использовать внешние данные без сравнения по CV")
    ap.add_argument("--keep-test-overlap", action="store_true",
                    help="не удалять из внешних данных молекулы, присутствующие в test.csv")
    ap.add_argument("--no-rdkit", action="store_true", help="fallback без RDKit (только для проверки пайплайна)")
    ap.add_argument("--cv", choices=["scaffold", "random"], default="scaffold")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--skip-cv", action="store_true")
    args = ap.parse_args(argv)

    warnings.filterwarnings("ignore")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    use_rdkit = not args.no_rdkit
    if use_rdkit and not feat.HAVE_RDKIT:
        sys.exit("RDKit не найден. Установите: pip install rdkit  (или --no-rdkit для fallback-режима)")

    # ---- данные
    Xtr, y, Xte, test_index = load_competition(args.train, args.test)
    n_comp = len(Xtr)
    ext_df, y_ext = None, np.array([])
    if args.external:
        ext_df, y_ext = load_external(args.external)
        print(f"[data] внешних строк: {len(ext_df)}")
    parts = [Xtr] + ([ext_df] if ext_df is not None else [])
    train_df = pd.concat(parts, ignore_index=True)
    y_all = np.concatenate([y, y_ext]) if ext_df is not None else y.copy()
    is_ext = np.arange(len(train_df)) >= n_comp

    keys_all = np.array([feat.canonical_key(s) for s in train_df["SMILES"]])
    keys_test = np.array([feat.canonical_key(s) for s in Xte["SMILES"]])
    if ext_df is not None and not args.keep_test_overlap:
        overlap = is_ext & np.isin(keys_all, keys_test)
        if overlap.any():
            print(f"[data] удалено {overlap.sum()} внешних строк, молекулы которых есть в test.csv "
                  f"(используйте --keep-test-overlap, чтобы оставить)")
        keep = ~overlap
        train_df, y_all, is_ext, keys_all = train_df[keep].reset_index(drop=True), y_all[keep], is_ext[keep], keys_all[keep]

    if use_rdkit and args.cv == "scaffold":
        # одна молекула всегда имеет один каркас -> дубли автоматически в одной группе
        groups = pd.factorize(np.array([feat.scaffold_key(s) for s in train_df["SMILES"]]))[0]
    else:
        groups = pd.factorize(keys_all)[0]  # обычный KFold, но дубли молекул вместе

    # ---- признаки
    t0 = time.time()
    fz = Featurizer(use_rdkit=use_rdkit).fit(train_df)
    fs_all = fz.transform(pd.concat([train_df, Xte], ignore_index=True))
    D = Data(fs_all)
    n_train = len(train_df)
    print(f"[feat] ctx={fs_all.ctx.shape[1]} mol_cnt={fs_all.mol_cnt.shape[1]} "
          f"mol_desc={fs_all.mol_desc.shape[1]}  ({time.time() - t0:.0f}s)")
    models = make_models(args.seed, args.n_jobs)

    # ---- CV
    report = {"config": vars(args), "n_comp": int(n_comp), "n_ext": int(is_ext.sum())}
    use_ext = bool(ext_df is not None and is_ext.any())
    if not args.skip_cv:
        variants = {"without_external": False}
        if use_ext:
            variants["with_external"] = True
        scored = {}
        for tag, flag in variants.items():
            scheme = "по каркасам Мерка" if (use_rdkit and args.cv == "scaffold") else "случайные (дубли молекул вместе)"
            print(f"\n[cv] {tag}: {args.cv_folds}-fold x{args.cv_repeats}, свёртки: {scheme}", flush=True)
            oof = run_cv(D, n_comp, y_all, is_ext, args.external_weight, keys_all, groups, models,
                         args.cv_folds, args.cv_repeats, args.seed, flag)
            rep, ens = summarize(oof, y)
            print_report(tag, rep)
            report[f"cv_{tag}"] = rep
            scored[tag] = (rep, ens)
        if use_ext and not args.force_external:
            a = scored["without_external"][0]["models"]["ensemble_scaled"]["MSE_winsor99"]
            b = scored["with_external"][0]["models"]["ensemble_scaled"]["MSE_winsor99"]
            use_ext = b < a
            print(f"\n[cv] внешние данные: MSE_win99 {a:.0f} (без) vs {b:.0f} (с) -> "
                  f"{'ИСПОЛЬЗУЕМ' if use_ext else 'НЕ используем'}")
        chosen = "with_external" if use_ext else "without_external"
        rep, ens = scored[chosen]
        scale_c = rep["scale_c"]
        pd.DataFrame({"Title": Xtr["Title"], "SMILES": Xtr["SMILES"], "y_true": y,
                      "oof_pred": np.clip(scale_c * np.expm1(ens.mean(0)), 0, None)}
                     ).to_csv(out / "oof_predictions.csv", index=False)
    else:
        scale_c = float(np.mean(C_BOUNDS))
        print("[cv] пропущен, scale_c =", scale_c)
    report["use_external_in_final"] = use_ext
    report["scale_c"] = scale_c

    # ---- финальное обучение
    tr_idx = np.where(~is_ext | use_ext)[0]
    w = np.where(is_ext[tr_idx], args.external_weight, 1.0)
    te_idx = np.arange(n_train, n_train + len(Xte))
    y_log = np.log1p(y_all)[tr_idx]
    preds = []
    for name, fn in models.items():
        t = time.time()
        preds.append(fn(D, tr_idx, te_idx, y_log, w))
        print(f"[fit] {name} готово за {time.time() - t:.0f}s", flush=True)
    pred = np.clip(scale_c * np.expm1(np.mean(preds, axis=0)), 0, None)

    sub = pd.DataFrame({"index": test_index, TARGET: pred})
    sample_path = Path(args.sample)
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        assert list(sample.columns) == list(sub.columns), f"колонки {list(sample.columns)} != {list(sub.columns)}"
        assert len(sample) == len(sub) and (sample["index"].to_numpy() == sub["index"].to_numpy()).all()
    assert np.isfinite(sub[TARGET]).all()
    sub.to_csv(out / "submission.csv", index=False)
    print(f"\n[ok] {out / 'submission.csv'}  ({len(sub)} строк; mean={pred.mean():.2f}, max={pred.max():.1f})")

    report["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                          "pandas": pd.__version__,
                          "rdkit": __import__("rdkit").__version__ if use_rdkit else None}
    with open(out / "cv_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
