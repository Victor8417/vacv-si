"""Признаки для предсказания SI VACV.

Три блока признаков:
  * ctx       - контекст эксперимента (культура клеток, штамм, происхождение записи);
  * mol_cnt   - счётчики подструктур (Morgan counts, либо char-nграммы SMILES в fallback-режиме);
  * mol_desc  - RDKit-дескрипторы + MACCS (пусто в fallback-режиме).

ВАЖНО: колонки CC50_uM и IC50_uM_VACV ЗАПРЕЩЕНЫ правилами соревнования
(SI = CC50 / IC50 -> прямая утечка цели). Модуль их никогда не читает, а
`assert_no_forbidden_columns` страхует от случайного использования.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FORBIDDEN_COLUMNS = ("CC50_uM", "IC50_uM_VACV")

try:  # RDKit опционален, чтобы можно было проверить пайплайн на «голой» машине
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except Exception:  # pragma: no cover
    HAVE_RDKIT = False


def assert_no_forbidden_columns(columns) -> None:
    bad = [c for c in columns if c in FORBIDDEN_COLUMNS]
    if bad:
        raise AssertionError(f"Запрещённые колонки попали в признаки: {bad}")


# --------------------------------------------------------------------------
# Молекулы
# --------------------------------------------------------------------------
def largest_fragment(smiles: str):
    """Mol самого крупного фрагмента (убирает соли/противоионы вроде '.N', '.Cl')."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return mol


def canonical_key(smiles: str) -> str:
    """Ключ молекулы для поиска дублей: канонический SMILES крупного фрагмента без стереохимии.
    Без RDKit - исходная строка."""
    if not HAVE_RDKIT:
        return str(smiles).strip()
    mol = largest_fragment(smiles)
    if mol is None:
        return str(smiles).strip()
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def scaffold_key(smiles: str) -> str:
    """Каркас Мерка (для CV по каркасам). Пустой каркас -> сам канонический SMILES."""
    mol = largest_fragment(smiles)
    if mol is None:
        return str(smiles)
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaf = ""
    return scaf if scaf else Chem.MolToSmiles(mol, isomericSmiles=False)


_MORGAN_CACHE: dict = {}


def _morgan_gen(radius: int, n_bits: int):
    key = (radius, n_bits)
    if key not in _MORGAN_CACHE:
        _MORGAN_CACHE[key] = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return _MORGAN_CACHE[key]


def morgan_counts(mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    gen = _morgan_gen(radius, n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    for idx, cnt in gen.GetCountFingerprint(mol).GetNonzeroElements().items():
        arr[idx] = cnt
    return arr


def maccs_bits(mol) -> np.ndarray:
    arr = np.zeros(167, dtype=np.float32)
    on = list(MACCSkeys.GenMACCSKeys(mol).GetOnBits())
    if on:
        arr[on] = 1.0
    return arr


def rdkit_descriptors(mol) -> np.ndarray:
    out = np.full(len(Descriptors._descList), np.nan, dtype=np.float64)
    for i, (_, fn) in enumerate(Descriptors._descList):
        try:
            out[i] = fn(mol)
        except Exception:
            pass
    return out


def rdkit_descriptor_names() -> list[str]:
    return [n for n, _ in Descriptors._descList]


# --------------------------------------------------------------------------
# Контекст эксперимента (разрешённые колонки: Title, strain, Culture)
# --------------------------------------------------------------------------
def _tokens(text: str, prefix: str) -> list[str]:
    text = str(text).lower()
    text = re.sub(r"\bcells?\b", " ", text)
    text = re.sub(r"[^a-z0-9\-]+", " ", text)
    return [f"{prefix}_{t}" for t in text.split() if len(t) >= 2]


def context_text(df: pd.DataFrame) -> list[str]:
    rows = []
    for title, strain, culture in zip(df["Title"], df["strain"], df["Culture"]):
        toks = _tokens(strain, "s") + _tokens(culture, "c")
        t = str(title).strip().lower()
        if t.startswith("sid"):
            toks.append("t_sid")  # запись из PubChem-скрининга
        elif re.fullmatch(r"\d+[a-z]?", t):
            toks.append("t_number")
        elif "-" in t:
            toks.append("t_dashcode")
        else:
            toks.append("t_name")
        rows.append(" ".join(toks))
    return rows


# --------------------------------------------------------------------------
# Основной класс
# --------------------------------------------------------------------------
@dataclass
class FeatureSet:
    ctx: np.ndarray
    mol_cnt: np.ndarray
    mol_desc: np.ndarray
    names: dict = field(default_factory=dict)

    def tree_matrix(self) -> np.ndarray:
        return np.hstack([self.ctx, self.mol_desc, self.mol_cnt]).astype(np.float32)

    def take(self, idx) -> "FeatureSet":
        return FeatureSet(self.ctx[idx], self.mol_cnt[idx], self.mol_desc[idx], self.names)


class Featurizer:
    """fit() на обучающих строках, transform() на любых."""

    def __init__(self, use_rdkit: bool = True, radius: int = 2, n_bits: int = 2048,
                 min_df: int = 3):
        if use_rdkit and not HAVE_RDKIT:
            raise RuntimeError("RDKit не установлен. Установите rdkit или запустите с --no-rdkit.")
        self.use_rdkit = use_rdkit
        self.radius, self.n_bits, self.min_df = radius, n_bits, min_df

    # ---- helpers
    def _mol_blocks(self, smiles: list[str]):
        if self.use_rdkit:
            cnt, desc = [], []
            n_bad = 0
            for smi in smiles:
                mol = largest_fragment(smi)
                if mol is None:
                    n_bad += 1
                    cnt.append(np.zeros(self.n_bits, np.float32))
                    desc.append(np.full(len(Descriptors._descList) + 167, np.nan))
                    continue
                cnt.append(morgan_counts(mol, self.radius, self.n_bits))
                desc.append(np.concatenate([rdkit_descriptors(mol), maccs_bits(mol)]))
            if n_bad:
                warnings.warn(f"{n_bad} SMILES не распарсились RDKit - признаки нулевые/NaN")
            return np.vstack(cnt), np.vstack(desc)
        # fallback: char-nграммы SMILES
        mat = self._ngram.transform(smiles).toarray().astype(np.float32)
        return mat, np.zeros((len(smiles), 0), np.float32)

    # ---- API
    def fit(self, df: pd.DataFrame) -> "Featurizer":
        from sklearn.feature_extraction.text import CountVectorizer

        self._ctx_vec = CountVectorizer(analyzer=str.split, binary=True, min_df=self.min_df)
        self._ctx_vec.fit(context_text(df))
        if not self.use_rdkit:
            self._ngram = CountVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=False,
                                          min_df=self.min_df)
            self._ngram.fit(df["SMILES"].astype(str).tolist())
        # статистика дескрипторов (медианы, постоянные колонки) - только по обучающим строкам
        if self.use_rdkit:
            _, desc = self._mol_blocks(df["SMILES"].astype(str).tolist())
            desc = np.where(np.isfinite(desc), desc, np.nan)
            desc = np.clip(desc, -1e6, 1e6)
            self._desc_median = np.nan_to_num(np.nanmedian(desc, axis=0), nan=0.0)
            filled = np.where(np.isnan(desc), self._desc_median, desc)
            self._desc_keep = filled.std(axis=0) > 1e-9
        return self

    def transform(self, df: pd.DataFrame) -> FeatureSet:
        ctx = self._ctx_vec.transform(context_text(df)).toarray().astype(np.float32)
        cnt, desc = self._mol_blocks(df["SMILES"].astype(str).tolist())
        if self.use_rdkit:
            desc = np.where(np.isfinite(desc), desc, np.nan)
            desc = np.clip(desc, -1e6, 1e6)
            desc = np.where(np.isnan(desc), self._desc_median, desc)[:, self._desc_keep]
            desc = desc.astype(np.float32)
        names = {"ctx": list(self._ctx_vec.get_feature_names_out())}
        return FeatureSet(ctx, cnt, desc, names)
