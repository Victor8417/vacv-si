"""Валидация собранной базы (результат collect_data.py).

Проверки:
  1. схема: обязательные столбцы smiles, ic50_uM, cc50_uM, si, doi;
  2. SMILES разбирается RDKit (иначе строка удаляется), считается канонический ключ молекулы;
  3. размер молекулы (>=5 тяжёлых атомов, есть углерод);
  4. диапазоны: IC50/CC50 в [1e-5, 1e6] мкМ, SI > 0 и конечен;
  5. согласованность: si == cc50/ic50 (допуск 2% по log10) для строк, где есть оба значения;
  6. формат DOI (предупреждение, строки без DOI не удаляются - в задании "при наличии");
  7. точные дубликаты удаляются; противоречивые повторы (SI отличается >10x) только отмечаются;
  8. пересечение с train/test соревнования (только отчёт; test-пересечения исключает train.py).

Запуск:
    python -m src.validate_data --input data/external/vacv_external_raw.csv \
        --output data/external/vacv_external_clean.csv --train data/raw/train_v2.csv --test data/raw/test.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src import features as feat

REQUIRED = ["smiles", "ic50_uM", "cc50_uM", "si", "doi"]
DOI_OK = re.compile(r"^10\.\d{4,9}/\S+$")


def validate(df: pd.DataFrame, use_rdkit: bool = True):
    """Возвращает (чистый DataFrame, список проверок)."""
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "status": "OK" if ok else "WARN", "detail": detail})

    miss = [c for c in REQUIRED if c not in df.columns]
    add("schema", not miss, f"нет столбцов: {miss}" if miss else f"{len(df)} строк, столбцы на месте")
    if miss:
        raise ValueError(f"В файле нет обязательных столбцов: {miss}")
    df = df.copy()
    n0 = len(df)

    # 2-3. SMILES
    df["smiles"] = df["smiles"].astype(str).str.strip()
    if use_rdkit and feat.HAVE_RDKIT:
        mols = [feat.largest_fragment(s) for s in df["smiles"]]
        bad = np.array([m is None for m in mols])
        add("smiles_parse", not bad.any(), f"не разобрано RDKit: {int(bad.sum())}")
        small = np.array([(m is not None) and (m.GetNumHeavyAtoms() < 5 or
                                               not any(a.GetSymbol() == "C" for a in m.GetAtoms())) for m in mols])
        add("molecule_size_has_carbon", not small.any(), f"слишком мелкие/неорганические: {int(small.sum())}")
        df = df[~(bad | small)].copy()
        df["smiles_canonical"] = [feat.canonical_key(s) for s in df["smiles"]]
    else:
        add("smiles_parse", False, "RDKit недоступен - SMILES НЕ проверены")
        df["smiles_canonical"] = df["smiles"]

    # 4. диапазоны
    for c in ("ic50_uM", "cc50_uM", "si"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad_si = ~(np.isfinite(df["si"]) & (df["si"] > 0))
    bad_ic = df["ic50_uM"].notna() & ~df["ic50_uM"].between(1e-5, 1e6)
    bad_cc = df["cc50_uM"].notna() & ~df["cc50_uM"].between(1e-5, 1e6)
    bad = bad_si | bad_ic | bad_cc
    add("value_ranges", not bad.any(), f"вне диапазона/нечисловые: {int(bad.sum())}")
    df = df[~bad].copy()

    # 5. согласованность SI
    both = df["ic50_uM"].notna() & df["cc50_uM"].notna()
    dev = (np.log10(df["si"]) - np.log10(df["cc50_uM"] / df["ic50_uM"])).abs()
    incons = both & (dev > 0.02)
    add("si_equals_cc50_over_ic50", not incons.any(), f"расхождения >2% (по log10): {int(incons.sum())} -> удалены")
    df = df[~incons].copy()

    # 6. DOI
    doi = df["doi"].fillna("").astype(str)
    has = doi != ""
    bad_fmt = has & ~doi.str.match(DOI_OK)
    add("doi_format", not bad_fmt.any(), f"с DOI: {has.mean():.0%}; неверный формат: {int(bad_fmt.sum())}")

    # 7. дубликаты
    keycols = ["smiles_canonical", "doi", "cell_line", "ic50_uM", "cc50_uM"] if "cell_line" in df else \
        ["smiles_canonical", "doi", "ic50_uM", "cc50_uM"]
    dup = df.duplicated(subset=keycols)
    add("exact_duplicates", not dup.any(), f"удалено дублей: {int(dup.sum())}")
    df = df[~dup].copy()
    grp = [c for c in ["smiles_canonical", "doi", "cell_line"] if c in df]
    spread = df.groupby(grp)["si"].agg(lambda x: x.max() / x.min() if len(x) > 1 else 1.0)
    add("conflicting_repeats", not (spread > 10).any(), f"групп с разбросом SI >10x: {int((spread > 10).sum())} (оставлены)")

    add("final_rows", len(df) > 0, f"{len(df)} из {n0}" + ("" if len(df) else " - база пуста: проверьте доступ к сети и логи collect_data"))
    return df.reset_index(drop=True), checks


def overlap_report(clean: pd.DataFrame, train_path, test_path) -> dict:
    out = {}
    for tag, path in (("train", train_path), ("test", test_path)):
        if path and Path(path).exists():
            keys = {feat.canonical_key(s) for s in pd.read_csv(path)["SMILES"]}
            n = int(clean["smiles_canonical"].isin(keys).sum())
            out[tag] = n
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/external/vacv_external_raw.csv")
    ap.add_argument("--output", default="data/external/vacv_external_clean.csv")
    ap.add_argument("--report", default="data/external/validation_report.json")
    ap.add_argument("--train", default="data/raw/train_v2.csv")
    ap.add_argument("--test", default="data/raw/test.csv")
    ap.add_argument("--no-rdkit", action="store_true")
    args = ap.parse_args(argv)

    clean, checks = validate(pd.read_csv(args.input), use_rdkit=not args.no_rdkit)
    ov = overlap_report(clean, args.train, args.test)
    print(f"{'check':<28}{'status':<7}detail")
    for c in checks:
        print(f"{c['check']:<28}{c['status']:<7}{c['detail']}")
    print(f"\nпересечение с соревнованием (по канон. SMILES): {ov}")
    if ov.get("test"):
        print("  -> train.py по умолчанию исключает такие молекулы из обучения "
              "(иначе это утечка меток из публичной базы); --keep-test-overlap отключает это.")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)
    Path(args.report).write_text(json.dumps({"checks": checks, "overlap": ov, "n_rows": len(clean)},
                                            ensure_ascii=False, indent=2), "utf-8")
    print(f"\n[ok] {args.output} ({len(clean)} строк), отчёт: {args.report}")


if __name__ == "__main__":
    main()
