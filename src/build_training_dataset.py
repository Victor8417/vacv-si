"""Собирает итоговый обучающий набор в формате задания: smiles, ic50_uM, cc50_uM, si, doi (+ source).

  * строки соревнования (train_v2.csv): DOI в данных нет -> пусто;
  * внешние строки (vacv_external_clean.csv) - если train.py решил их использовать
    (см. outputs/cv_report.json -> use_external_in_final) или если указано --external yes.
Молекулы из test.csv исключаются из внешних данных (как и в train.py).

Запуск:
    python -m src.build_training_dataset --out data/processed/training_dataset.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src import features as feat

COLS = ["smiles", "ic50_uM", "cc50_uM", "si", "doi", "source"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/raw/train_v2.csv")
    ap.add_argument("--test", default="data/raw/test.csv")
    ap.add_argument("--external", default="data/external/vacv_external_clean.csv")
    ap.add_argument("--use-external", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--report", default="outputs/cv_report.json")
    ap.add_argument("--out", default="data/processed/training_dataset.csv")
    args = ap.parse_args(argv)

    tr = pd.read_csv(args.train)
    # CC50/IC50 здесь только ЭКСПОРТИРУЮТСЯ в таблицу данных (формат задания); в модель они не попадают.
    base = pd.DataFrame({"smiles": tr["SMILES"], "ic50_uM": tr["IC50_uM_VACV"], "cc50_uM": tr["CC50_uM"],
                         "si": tr["SI VACV"], "doi": "", "source": "competition_train"})
    parts = [base]

    use = args.use_external
    if use == "auto":
        rp = Path(args.report)
        use = "yes" if rp.exists() and json.loads(rp.read_text("utf-8")).get("use_external_in_final") else "no"
    if use == "yes" and Path(args.external).exists():
        ext = pd.read_csv(args.external)
        test_keys = {feat.canonical_key(s) for s in pd.read_csv(args.test)["SMILES"]}
        mask = ~ext["smiles"].map(feat.canonical_key).isin(test_keys)
        print(f"внешних строк: {len(ext)}, исключено из-за пересечения с test: {int((~mask).sum())}")
        ext = ext[mask]
        parts.append(ext[[c for c in COLS if c in ext.columns]])
    out = pd.concat(parts, ignore_index=True)[COLS]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[ok] {args.out}: {len(out)} строк\n{out['source'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
