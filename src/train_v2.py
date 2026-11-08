"""v2: улучшения поверх v1. Файлы v1 (train.py, features.py ...) НЕ изменяются; результаты - в outputs_v2/.

Что изменилось относительно v1 и почему
1. CV по умолчанию СЛУЧАЙНАЯ (дубли молекул вместе), а не по каркасам. Сходство "тест -> ближайший сосед в
   train" статистически совпадает со сходством "train -> train", т.е. train/test разбиты случайно, и
   CV по каркасам была чрезмерно пессимистична.
2. Множитель масштаба c ищется в [1, 8] (в v1 верхняя граница 3.0 срабатывала в обоих CV-прогонах) по
   MSE с winsorize 99.7% (в v1 - 99%, что обрезало все значения > ~2500, а именно они дают почти всю ошибку).
3. Поправка по аналогу: если у соединения есть очень похожий сосед в обучении (сходство s1 >= s0), прогноз
   смещается к SI этого соседа: pred = (1-w)*модель + w*SI_соседа, w = wmax*clip((s1-s0)/(1-s0), 0, 1).
   Фингерпринты Morgan не различают стереоизомеры, поэтому эпимеры соединений-рекордсменов получают сходство 1.0.
   wmax по умолчанию 0.4 (для потерь вида MSE оптимально предсказывать p*S, а не 0 или S; p~0.4 - оценка
   вероятности, что эпимер рекордсмена тоже высокоактивен), s0 подбирается по CV; если CV ухудшает метрику более
   чем на 1% - аналоги выключаются.
4. Дополнительно сохраняются варианты submission (без аналогов / с усиленным доверием к аналогам)
   и таблица analog_changes.csv - какие тестовые строки были изменены и почему.

ОГРАНИЧЕНИЕ: в обучении нет пар эпимеров, поэтому CV не может проверить доверие именно к стереоизомерам
(сходство ~1.0 при разной хиральности). Это экстраполяция; wmax можно задать вручную (--analog-wmax).

Запуск:
    python -m src.train_v2 --out outputs_v2
    python -m src.train_v2 --out outputs_v2 --external data/external/vacv_external_clean.csv
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

from src import features as feat
from src import train as v1

TARGET = v1.TARGET
WIN_Q = 0.997                                   # winsorize для выбора параметров
C_GRID = [round(x, 2) for x in np.arange(1.0, 8.01, 0.25)]
WMAX_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
S0_GRID = (0.7, 0.8, 0.9)
TIE_TOL = 0.03                                  # соседи в пределах tol от лучшего считаются "ничьёй"
DEFAULT_WMAX = 0.4                              # доверие к аналогу по умолчанию (см. docstring)
MAX_CV_LOSS = 0.01                              # аналоги выключаются, если CV ухудшает win99.7 более чем на 1%
FALLBACK = {"c": 4.0, "wmax": DEFAULT_WMAX, "s0": 0.8}   # для --skip-cv


# ---------------------------------------------------------------------------
# Аналоги
# ---------------------------------------------------------------------------
def neighbor_info(ksub: np.ndarray, y_train: np.ndarray, tol: float = TIE_TOL):
    """ksub: (цели x обучающие) сходство в [0,1]. Возвращает (s1, v, idx):
    s1 - лучшее сходство, v - средний SI соседей, чьё сходство в пределах tol от лучшего, idx - индекс лучшего."""
    s1 = ksub.max(axis=1)
    mask = ksub >= (s1[:, None] - tol)
    v = (mask * y_train[None, :]).sum(axis=1) / mask.sum(axis=1)
    return s1, v, ksub.argmax(axis=1)


def analog_weight(s1: np.ndarray, wmax: float, s0: float) -> np.ndarray:
    return wmax * np.clip((s1 - s0) / (1.0 - s0), 0.0, 1.0)


def combine(model_raw: np.ndarray, s1: np.ndarray, v: np.ndarray, wmax: float, s0: float) -> np.ndarray:
    w = analog_weight(s1, wmax, s0)
    return np.clip((1.0 - w) * model_raw + w * v, 0.0, None)


# ---------------------------------------------------------------------------
# Метрики и подбор параметров
# ---------------------------------------------------------------------------
def score_set(pred: np.ndarray, y: np.ndarray) -> dict:
    """pred: (повторы x n). Усреднение по повторам."""
    win = np.clip(y, 0, np.quantile(y, WIN_Q))
    m1 = np.ones(len(y), bool)
    m1[np.argmax(y)] = False
    pred = np.clip(pred, 0, None)
    return {
        "MSE_raw": float(np.mean((pred - y[None, :]) ** 2)),
        "MSE_excl_top1": float(np.mean((pred[:, m1] - y[None, m1]) ** 2)),
        "MSE_win997": float(np.mean((pred - win[None, :]) ** 2)),
    }


def select_params(ens_log: np.ndarray, s1: np.ndarray, vnb: np.ndarray, y: np.ndarray, wmax_override=None):
    """ens_log: (повторы x n) среднее лог-прогнозов; s1, vnb - OOF-информация об аналогах.
    c  - минимум MSE_win997 по сетке C_GRID.
    wmax - DEFAULT_WMAX (или --analog-wmax); s0 - лучший по CV из S0_GRID при этом wmax.
    Аналоги включаются, если CV не ухудшает MSE_win997 более чем на MAX_CV_LOSS (явный --analog-wmax
    применяется всегда). Возвращает (params, grid_table) - таблица нужна для отчёта."""
    base = np.expm1(ens_log)
    c = C_GRID[int(np.argmin([score_set(c_ * base, y)["MSE_win997"] for c_ in C_GRID]))]
    ref = score_set(c * base, y)["MSE_win997"]

    def sc(w, s0):
        return score_set(np.stack([combine(c * base[r], s1[r], vnb[r], w, s0) for r in range(base.shape[0])]), y)["MSE_win997"]

    table = {f"wmax={w},s0={s0}": sc(w, s0) / ref - 1.0 for w in WMAX_GRID if w > 0 for s0 in S0_GRID}
    target = DEFAULT_WMAX if wmax_override is None else wmax_override
    s0_best, loss = 0.8, 0.0
    if target > 0:
        scores = {s0: sc(target, s0) for s0 in S0_GRID}
        s0_best = min(scores, key=scores.get)
        loss = scores[s0_best] / ref - 1.0
    enabled = target > 0 and (wmax_override is not None or loss <= MAX_CV_LOSS)
    params = {"c": float(c), "wmax": float(target if enabled else 0.0), "s0": float(s0_best),
              "cv_delta_win997_pct": float(100 * loss)}
    return params, table


def apply_params(ens_log, s1, vnb, p):
    base = p["c"] * np.expm1(ens_log)
    return np.stack([combine(base[r], s1[r], vnb[r], p["wmax"], p["s0"]) for r in range(base.shape[0])])


# ---------------------------------------------------------------------------
# CV
# ---------------------------------------------------------------------------
def run_cv(D, Kn, n_comp, y_all, is_ext, ext_w, keys, groups, models, folds, repeats, seed, use_ext):
    comp = np.arange(n_comp)
    ext_idx = np.where(is_ext)[0] if use_ext else np.array([], dtype=int)
    ylog = np.log1p(y_all)
    oof = {k: np.zeros((repeats, n_comp)) for k in models}
    s1 = np.zeros((repeats, n_comp))
    vnb = np.zeros((repeats, n_comp))
    for r in range(repeats):
        for tr_c, te_c in v1.group_folds(groups[comp], folds, seed + r):
            held = set(keys[te_c])
            ext_keep = np.array([i for i in ext_idx if keys[i] not in held], dtype=int)
            tr = np.concatenate([comp[tr_c], ext_keep])
            w = np.concatenate([np.ones(len(tr_c)), np.full(len(ext_keep), ext_w)])
            for name, fn in models.items():
                oof[name][r, te_c] = fn(D, tr, comp[te_c], ylog[tr], w)
            s, v, _ = neighbor_info(Kn[np.ix_(comp[te_c], tr)], y_all[tr])
            s1[r, te_c], vnb[r, te_c] = s, v
    ens = np.mean([oof[k] for k in models], axis=0)
    return ens, s1, vnb


def fmt(m: dict) -> str:
    return f"raw {m['MSE_raw']:>10.0f} | без топ-1 {m['MSE_excl_top1']:>8.0f} | win99.7 {m['MSE_win997']:>8.0f}"


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/raw/train_v2.csv")
    ap.add_argument("--test", default="data/raw/test.csv")
    ap.add_argument("--sample", default="data/raw/sample_submit.csv")
    ap.add_argument("--out", default="outputs_v2")
    ap.add_argument("--external", default=None)
    ap.add_argument("--external-weight", type=float, default=0.5)
    ap.add_argument("--force-external", action="store_true")
    ap.add_argument("--keep-test-overlap", action="store_true")
    ap.add_argument("--analog-wmax", type=float, default=None, help="задать доверие к аналогам вручную (0 = выключить)")
    ap.add_argument("--no-rdkit", action="store_true")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--cv-repeats", type=int, default=4)
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
        sys.exit("RDKit не найден (pip install rdkit) или используйте --no-rdkit только для проверки пайплайна")

    # ---- данные (та же подготовка, что в v1)
    Xtr, y, Xte, test_index = v1.load_competition(args.train, args.test)
    n_comp = len(Xtr)
    ext_df, y_ext = None, np.array([])
    if args.external:
        ext_df, y_ext = v1.load_external(args.external)
        print(f"[data] внешних строк: {len(ext_df)}")
    train_df = pd.concat([Xtr] + ([ext_df] if ext_df is not None else []), ignore_index=True)
    y_all = np.concatenate([y, y_ext]) if ext_df is not None else y.copy()
    is_ext = np.arange(len(train_df)) >= n_comp
    keys_all = np.array([feat.canonical_key(s) for s in train_df["SMILES"]])
    keys_test = np.array([feat.canonical_key(s) for s in Xte["SMILES"]])
    if ext_df is not None and not args.keep_test_overlap:
        overlap = is_ext & np.isin(keys_all, keys_test)
        if overlap.any():
            print(f"[data] удалено {int(overlap.sum())} внешних строк с молекулами из test.csv")
        keep = ~overlap
        train_df, y_all, is_ext, keys_all = train_df[keep].reset_index(drop=True), y_all[keep], is_ext[keep], keys_all[keep]
    groups = pd.factorize(keys_all)[0]  # случайные свёртки, дубли молекул вместе

    # ---- признаки, ядра
    t0 = time.time()
    fz = feat.Featurizer(use_rdkit=use_rdkit).fit(train_df)
    fs_all = fz.transform(pd.concat([train_df, Xte], ignore_index=True))
    D = v1.Data(fs_all)
    Kn = D.K / 1.3  # K = K_mol * (0.3 + K_ctx) <= 1.3  ->  сходство в [0, 1]
    n_train = len(train_df)
    models = v1.make_models(args.seed, args.n_jobs)
    print(f"[feat] ctx={fs_all.ctx.shape[1]} mol_cnt={fs_all.mol_cnt.shape[1]} mol_desc={fs_all.mol_desc.shape[1]} "
          f"({time.time() - t0:.0f}s)")

    report = {"config": vars(args), "n_comp": int(n_comp), "n_ext": int(is_ext.sum())}
    use_ext = bool(ext_df is not None and is_ext.any())

    # ---- CV и подбор параметров
    if not args.skip_cv:
        results = {}
        variants = {"without_external": False}
        if use_ext:
            variants["with_external"] = True
        for tag, flag in variants.items():
            print(f"\n[cv] {tag}: {args.cv_folds}-fold x{args.cv_repeats}, случайные свёртки", flush=True)
            ens, s1, vnb = run_cv(D, Kn, n_comp, y_all, is_ext, args.external_weight, keys_all, groups, models,
                                  args.cv_folds, args.cv_repeats, args.seed, flag)
            params, table = select_params(ens, s1, vnb, y, args.analog_wmax)
            results[tag] = (ens, s1, vnb, params)
            v1_like = score_set(3.0 * np.expm1(ens), y)
            only_c = score_set(params["c"] * np.expm1(ens), y)
            full = score_set(apply_params(ens, s1, vnb, params), y)
            const = {"MSE_raw": float(np.var(y)), "MSE_excl_top1": float(np.var(np.delete(y, np.argmax(y)))),
                     "MSE_win997": float(np.var(np.clip(y, 0, np.quantile(y, WIN_Q))))}
            print(f"  const (среднее)        : {fmt(const)}")
            print(f"  v1-настройка (c=3)     : {fmt(v1_like)}")
            print(f"  v2: c={params['c']:.2f}, без аналогов: {fmt(only_c)}")
            print(f"  v2: + аналоги (wmax={params['wmax']}, s0={params['s0']}): {fmt(full)}")
            print(f"  изменение win99.7 от аналогов при wmax={DEFAULT_WMAX if args.analog_wmax is None else args.analog_wmax}: "
                  f"{params['cv_delta_win997_pct']:+.2f}% (порог выключения +{100 * MAX_CV_LOSS:.0f}%)")
            best_cell = min(table, key=table.get)
            print(f"  лучшая ячейка сетки аналогов: {best_cell} ({100 * table[best_cell]:+.2f}%); "
                  f"остальные - в cv_report_v2.json")
            report[f"cv_{tag}"] = {"params": params, "const": const, "v1_like": v1_like, "v2_c_only": only_c,
                                   "v2_full": full, "analog_grid_delta_win997": table}
        if use_ext and not args.force_external:
            a = report["cv_without_external"]["v2_full"]["MSE_win997"]
            b = report["cv_with_external"]["v2_full"]["MSE_win997"]
            use_ext = b < a
            print(f"\n[cv] внешние данные: win99.7 {a:.0f} (без) vs {b:.0f} (с) -> {'ИСПОЛЬЗУЕМ' if use_ext else 'НЕ используем'}")
        chosen = "with_external" if use_ext else "without_external"
        params = results[chosen][3]
        ens, s1, vnb = results[chosen][:3]
        pd.DataFrame({"Title": Xtr["Title"], "SMILES": Xtr["SMILES"], "y_true": y,
                      "oof_pred_v2": apply_params(ens, s1, vnb, params).mean(0)}).to_csv(out / "oof_predictions_v2.csv", index=False)
    else:
        params = dict(FALLBACK)
        if args.analog_wmax is not None:
            params["wmax"] = args.analog_wmax
        print("[cv] пропущен, параметры по умолчанию:", params)
    report["use_external_in_final"] = use_ext
    report["params"] = params

    # ---- финальное обучение
    tr_idx = np.where(~is_ext | use_ext)[0]
    w = np.where(is_ext[tr_idx], args.external_weight, 1.0)
    te_idx = np.arange(n_train, n_train + len(Xte))
    ylog = np.log1p(y_all)[tr_idx]
    preds = []
    for name, fn in models.items():
        t = time.time()
        preds.append(fn(D, tr_idx, te_idx, ylog, w))
        print(f"[fit] {name} за {time.time() - t:.0f}s", flush=True)
    ens_te = np.mean(preds, axis=0)
    base = params["c"] * np.expm1(ens_te)
    s1, vnb, nn = neighbor_info(Kn[np.ix_(te_idx, tr_idx)], y_all[tr_idx])

    variants_out = {
        "submission.csv": combine(base, s1, vnb, params["wmax"], params["s0"]),
        "submission_no_analog.csv": np.clip(base, 0, None),
        "submission_analog_strong.csv": combine(base, s1, vnb, max(0.6, params["wmax"]), params["s0"]),
    }
    sample_path = Path(args.sample)
    sample = pd.read_csv(sample_path) if sample_path.exists() else None
    for name, pred in variants_out.items():
        sub = pd.DataFrame({"index": test_index, TARGET: pred})
        if sample is not None:
            assert list(sample.columns) == list(sub.columns) and (sample["index"].to_numpy() == sub["index"].to_numpy()).all()
        assert np.isfinite(sub[TARGET]).all()
        sub.to_csv(out / name, index=False)
        print(f"[ok] {out / name}: mean={pred.mean():.1f}  max={pred.max():.0f}")

    # ---- что именно изменила поправка по аналогу
    final = variants_out["submission.csv"]
    wts = analog_weight(s1, params["wmax"], params["s0"])
    train_titles = train_df["Title"].to_numpy()
    nn_idx = tr_idx[nn]
    diag = pd.DataFrame({
        "index": test_index, "Title": Xte["Title"].to_numpy(), "Culture": Xte["Culture"].to_numpy(),
        "model_pred": base, "analog_title": train_titles[nn_idx], "analog_SI": y_all[nn_idx], "sim": s1,
        "analog_mean_SI": vnb, "w": wts, "final_pred": final,
    })
    diag["change"] = diag["final_pred"] - diag["model_pred"]
    diag = diag[np.abs(diag["change"]) > 50].sort_values("change", key=np.abs, ascending=False)
    diag.to_csv(out / "analog_changes.csv", index=False)
    if len(diag):
        print(f"\n[analog] строк с изменением > 50: {len(diag)}. Крупнейшие:")
        print(diag.head(12)[["index", "Title", "model_pred", "analog_title", "analog_SI", "sim", "w", "final_pred"]]
              .round(2).to_string(index=False))

    report["versions"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                          "rdkit": __import__("rdkit").__version__ if use_rdkit else None}
    with open(out / "cv_report_v2.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
