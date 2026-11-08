"""Сбор данных об активности малых молекул против вируса осповакцины (VACV) из открытых баз.

Источники
  1. ChEMBL REST API   https://www.ebi.ac.uk/chembl/api/data           (основной)
  2. PubChem BioAssay  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/  (поиск AID)
                       https://pubchem.ncbi.nlm.nih.gov/rest/pug/      (таблицы, SMILES)  (best effort)
  3. data/external/manual_literature.csv - ручное пополнение из статей (необязательно)

Результат: CSV со столбцами
  smiles, ic50_uM, cc50_uM, si, doi  + провенанс (source, molecule_id, document_id, cell_line, ...)
SI считается как CC50 / IC50 (как в соревновании). Если в статье есть только заявленный SI - он
сохраняется (si_source = reported).

Запуск:
    python -m src.collect_data --sources chembl,pubchem,manual --out data/external/vacv_external_raw.csv
Все ответы API кэшируются в data/external/cache (повторный запуск не ходит в сеть).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    requests = None

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except Exception:  # pragma: no cover
    HAVE_RDKIT = False

log = logging.getLogger("collect")

CHEMBL_HOST = "https://www.ebi.ac.uk"
CHEMBL_API = CHEMBL_HOST + "/chembl/api/data"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
VACV_TAXID = 10245

OUT_COLUMNS = ["smiles", "ic50_uM", "cc50_uM", "si", "doi", "source", "molecule_id", "document_id",
               "pubmed_id", "assay_ic50", "assay_cc50", "cell_line", "virus_strain", "si_source",
               "si_reported", "cell_matched", "censored", "n_ic50", "n_cc50"]

VIRAL_RE = re.compile(r"vaccinia|\bVACV\b|\bVV\b|\bWR\b strain", re.I)
TOX_RE = re.compile(r"cytotox|toxicity|viability|cell growth|\bCC50\b|\bCD50\b|\bTC50\b|proliferation", re.I)
ANTIVIRAL_RE = re.compile(r"antivir|inhibition of (the )?(virus|viral)|cytopathic|\bCPE\b|plaque|replicat|"
                          r"virus yield|infect", re.I)
CELL_RE = re.compile(r"\b(Vero(?:\s?E6|\s?76)?|HeLa|HFF|HEL(?:\s?299)?|BSC-?\d+|MRC-?5|A549|LLC-?MK2|PRK|"
                     r"CV-?1|RK-?13|MT-?4|CEM|HEp-?2|MDBK|L929|HEK-?293|Huh-?7|BGM|MA-?104)\b", re.I)
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+")

ANTIVIRAL_TYPES = {"IC50", "EC50"}
TOX_TYPES = {"CC50", "CD50", "TC50"}
SI_TYPES = {"SI", "SELECTIVITY INDEX", "SELECTIVITY RATIO", "SELECTIVITY"}


# ---------------------------------------------------------------------------
# HTTP + кэш
# ---------------------------------------------------------------------------
class CacheMiss(Exception):
    """Запрошен ответ только из кэша, а его там нет."""


class TooLarge(Exception):
    """Документ/выборка превышает лимит по числу записей."""


class Http:
    def __init__(self, cache_dir="data/external/cache", use_cache=True, pause=0.25,
                 connect_timeout=10, read_timeout=60, retries=2):
        if requests is None:
            raise RuntimeError("Нужен пакет requests: pip install requests")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache, self.pause = use_cache, pause
        self.timeout = (connect_timeout, read_timeout)
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "vacv-si-hackathon/1.0 (academic research)"
        retry = Retry(total=retries, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET",))
        self.s.mount("https://", HTTPAdapter(max_retries=retry))

    def _path(self, url, params, ext):
        h = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
        return self.cache_dir / f"{h}.{ext}"

    def get_json(self, url, params=None, cache_only=False):
        p = self._path(url, params, "json")
        if self.use_cache and p.exists():
            return json.loads(p.read_text("utf-8"))
        if cache_only:
            raise CacheMiss(url)
        r = self.s.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        p.write_text(json.dumps(data), "utf-8")
        time.sleep(self.pause)
        return data

    def get_text(self, url, params=None, max_bytes=40_000_000, max_seconds=120):
        p = self._path(url, params, "txt")
        if self.use_cache and p.exists():
            return p.read_text("utf-8")
        t0 = time.monotonic()
        r = self.s.get(url, params=params, timeout=self.timeout, stream=True)
        r.raise_for_status()
        chunks, size = [], 0
        for chunk in r.iter_content(1 << 20):
            chunks.append(chunk)
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"ответ больше {max_bytes} байт - пропускаем")
            if time.monotonic() - t0 > max_seconds:
                raise TimeoutError(f"скачивание дольше {max_seconds} с - пропускаем")
        text = b"".join(chunks).decode("utf-8", errors="replace")
        p.write_text(text, "utf-8")
        time.sleep(self.pause)
        return text


# ---------------------------------------------------------------------------
# Общие чистые функции (тестируются офлайн)
# ---------------------------------------------------------------------------
def molwt(smiles: str):
    if not HAVE_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol is not None else None


def to_uM(value, units, smiles=None):
    """Значение концентрации -> мкМ. Для массовых единиц нужен RDKit (молярная масса). None, если нельзя."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    u = (units or "").strip().lower().replace("µ", "u").replace("μ", "u").replace(" ", "")
    molar = {"nm": 1e-3, "um": 1.0, "mm": 1e3, "pm": 1e-6}
    if u in molar:
        return v * molar[u]
    mass = {"ug.ml-1": 1.0, "ug/ml": 1.0, "ugml-1": 1.0, "mg.ml-1": 1e3, "mg/ml": 1e3, "ng/ml": 1e-3, "ng.ml-1": 1e-3}
    if u in mass and smiles:
        mw = molwt(smiles)
        if mw:
            return v * mass[u] * 1000.0 / mw  # (мкг/мл) * 1000 / MW = мкМ
    return None


def guess_cell(assay_cell_type, description):
    if assay_cell_type:
        return str(assay_cell_type).strip()
    m = CELL_RE.search(description or "")
    return m.group(1) if m else ""


def gmean(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) == 1:  # без float-шума exp(log(x))
        return float(arr[0])
    return float(np.exp(np.mean(np.log(arr))))


def classify(rec: dict, assay: dict, vacv_assay_ids: set):
    """'ic50' (противовирусная активность) | 'cc50' (цитотоксичность) | 'si' | None."""
    t = (rec.get("standard_type") or rec.get("type") or "").upper().strip()
    desc = (assay or {}).get("description") or rec.get("assay_description") or ""
    viral = rec.get("assay_chembl_id") in vacv_assay_ids or bool(VIRAL_RE.search(desc))
    tox = bool(TOX_RE.search(desc))
    antiviral_words = bool(ANTIVIRAL_RE.search(desc))
    if t in SI_TYPES:
        return "si" if viral else None
    if t in TOX_TYPES:
        return "cc50"
    if t in ANTIVIRAL_TYPES:
        if viral and not (tox and not antiviral_words):
            return "ic50"
        if tox:
            return "cc50"
    return None


def _validity_ok(rec):
    c = rec.get("data_validity_comment")
    return (not c) or str(c).strip().lower() == "manually validated"


def chembl_to_rows(doc_id, doc_meta, assays, activities, vacv_assay_ids, keep_censored=False):
    """Чистая функция: активности одной статьи ChEMBL -> строки (IC50 из противовирусных ассеев,
    CC50 из ассеев цитотоксичности той же статьи для той же молекулы, SI)."""
    assays_by_id = {a["assay_chembl_id"]: a for a in assays}
    ic, cc, si = defaultdict(list), defaultdict(list), defaultdict(list)
    smiles_of, strain_of = {}, {}
    for rec in activities:
        if not _validity_ok(rec) or rec.get("standard_value") in (None, ""):
            continue
        assay = assays_by_id.get(rec.get("assay_chembl_id"), {})
        kind = classify(rec, assay, vacv_assay_ids)
        mol, smi = rec.get("molecule_chembl_id"), rec.get("canonical_smiles")
        if kind is None or not mol or not smi:
            continue
        rel = rec.get("standard_relation") or "="
        if rel != "=" and not keep_censored:
            continue
        if kind == "si":
            try:
                val = float(rec["standard_value"])
            except (TypeError, ValueError):
                continue
        else:
            val = to_uM(rec["standard_value"], rec.get("standard_units"), smi)
        if val is None or not val > 0:
            continue
        desc = assay.get("description") or rec.get("assay_description") or ""
        cell = guess_cell(assay.get("assay_cell_type"), desc)
        item = {"v": val, "rel": rel, "assay": rec.get("assay_chembl_id")}
        {"ic50": ic, "cc50": cc, "si": si}[kind][(mol, cell)].append(item)
        smiles_of[mol] = smi
        if kind in ("ic50", "si"):
            strain_of[(mol, cell)] = (assay.get("assay_strain") or desc)[:250]

    cc_by_mol = defaultdict(list)
    for (mol, cell) in cc:
        cc_by_mol[mol].append(cell)

    def agg(items):
        return gmean([i["v"] for i in items]), len(items), sorted({i["rel"] for i in items}), items[0]["assay"]

    rows = []
    keys = set(ic) | set(si)
    for (mol, cell) in sorted(keys):
        ic50 = cc50 = None
        n_ic = n_cc = 0
        rels, a_ic, a_cc, matched = set(), "", "", True
        if (mol, cell) in ic:
            ic50, n_ic, r, a_ic = agg(ic[(mol, cell)])
            rels |= set(r)
        # CC50: та же молекула, та же клетка; иначе - единственная цитотоксичность молекулы в статье
        cc_key = None
        if (mol, cell) in cc:
            cc_key = (mol, cell)
        elif len(cc_by_mol.get(mol, [])) == 1:
            cc_key, matched = (mol, cc_by_mol[mol][0]), False
        if cc_key:
            cc50, n_cc, r, a_cc = agg(cc[cc_key])
            rels |= set(r)
        si_rep = agg(si[(mol, cell)])[0] if (mol, cell) in si else None
        if ic50 and cc50:
            si_val, si_src = cc50 / ic50, "computed"
        elif si_rep is not None:
            si_val, si_src = si_rep, "reported"
        else:
            continue  # ни SI, ни пары IC50+CC50 - строка бесполезна
        rows.append({
            "smiles": smiles_of[mol], "ic50_uM": ic50, "cc50_uM": cc50, "si": si_val,
            "doi": (doc_meta or {}).get("doi") or "", "source": "chembl", "molecule_id": mol,
            "document_id": doc_id, "pubmed_id": (doc_meta or {}).get("pubmed_id") or "",
            "assay_ic50": a_ic, "assay_cc50": a_cc, "cell_line": cell,
            "virus_strain": strain_of.get((mol, cell), "Vaccinia virus"), "si_source": si_src,
            "si_reported": si_rep, "cell_matched": matched, "censored": rels != {"="} and bool(rels),
            "n_ic50": n_ic, "n_cc50": n_cc,
        })
    return rows


# ---------------------------------------------------------------------------
# ChEMBL (сеть)
# ---------------------------------------------------------------------------
def chembl_iter(http: Http, resource: str, key: str, params: dict, deadline=None, max_records=None,
                cache_only=False, label=""):
    """Постраничный обход ChEMBL.
    deadline    - момент time.monotonic(), после которого бросаем TimeoutError (проверка между страницами);
    max_records - если total_count первой страницы больше - TooLarge (до скачивания остальных страниц);
    cache_only  - не ходить в сеть: при отсутствии страницы в кэше бросается CacheMiss."""
    label = label or resource
    params = {**params, "format": "json", "limit": 1000}
    url = f"{CHEMBL_API}/{resource}.json"
    seen, n, page = {url}, 0, 0
    while url:
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError(f"{label}: превышен лимит времени на документ")
        data = http.get_json(url, params, cache_only=cache_only)
        meta = data.get("page_meta") or {}
        total = meta.get("total_count")
        page += 1
        if page == 1 and max_records and total and total > max_records:
            raise TooLarge(f"{label}: {total} записей > лимита {max_records}")
        items = data.get(key, [])
        n += len(items)
        if not cache_only and total and total > 1000:
            log.info("    %s: страница %d, получено %d из %s", label, page, n, total)
        yield from items
        nxt = meta.get("next")
        if not nxt:
            break
        url, params = CHEMBL_HOST + nxt, None
        if url in seen:
            log.warning("%s: ссылка на следующую страницу зациклилась - останавливаюсь", label)
            break
        seen.add(url)


# Типы активностей, которые вообще используются в classify(); по ним фильтруем на стороне сервера
FETCH_TYPES = sorted(ANTIVIRAL_TYPES | TOX_TYPES | {"SI", "Selectivity index", "Selectivity ratio", "Selectivity"})


def fetch_activities(http: Http, doc: str, deadline=None, max_records=50_000):
    """Активности документа. Возвращает (список, режим).
    1) если документ целиком уже лежит в кэше (прежняя нефильтрованная выгрузка) - берём из кэша без сети;
    2) иначе серверный фильтр по типам (IC50/EC50/CC50/CD50/TC50/SI) - на порядки меньше данных;
    3) если сервер отклонил фильтр __in (HTTP 400) - выгрузка без фильтра, но с лимитами."""
    try:
        acts = list(chembl_iter(http, "activity", "activities", {"document_chembl_id": doc}, cache_only=True))
        return acts, "cache"
    except CacheMiss:
        pass
    log.info("    %s: активности - загрузка из сети (фильтр по типам)", doc)
    params = {"document_chembl_id": doc, "standard_type__in": ",".join(FETCH_TYPES)}
    try:
        acts = list(chembl_iter(http, "activity", "activities", params, deadline, max_records, label=f"{doc} activities"))
        return acts, "filtered"
    except requests.HTTPError as e:
        if getattr(e.response, "status_code", None) != 400:
            raise
        log.warning("    %s: сервер не принял фильтр standard_type__in - качаю без фильтра", doc)
        acts = list(chembl_iter(http, "activity", "activities", {"document_chembl_id": doc}, deadline,
                                max_records, label=f"{doc} activities"))
        return acts, "unfiltered"


def find_vacv_assays(http: Http) -> dict:
    """Все ассеи ChEMBL, относящиеся к VACV (несколько независимых стратегий поиска, объединение)."""
    strategies = [
        ("tax_id=10245", {"assay_tax_id": VACV_TAXID}),
        ("description~vaccinia", {"description__icontains": "vaccinia"}),
        ("description~VACV", {"description__icontains": "VACV"}),
        ("assay_organism~vaccinia", {"assay_organism__icontains": "vaccinia"}),
    ]
    found = {}
    for name, params in strategies:
        n0 = len(found)
        try:
            for a in chembl_iter(http, "assay", "assays", params):
                found[a["assay_chembl_id"]] = a
        except Exception as e:  # сеть/синтаксис фильтра - не роняем весь сбор
            log.warning("стратегия %s не сработала: %s", name, e)
        log.info("ChEMBL assays, стратегия %-24s: +%d (всего %d)", name, len(found) - n0, len(found))
    try:
        for t in chembl_iter(http, "target", "targets", {"organism__icontains": "vaccinia"}):
            for a in chembl_iter(http, "assay", "assays", {"target_chembl_id": t["target_chembl_id"]}):
                found[a["assay_chembl_id"]] = a
        log.info("ChEMBL assays после поиска по таргетам: %d", len(found))
    except Exception as e:
        log.warning("поиск по таргетам не сработал: %s", e)
    return found


def collect_chembl(http: Http, keep_censored=False, max_docs=None, skip_docs=(), doc_timeout=300,
                   max_activities=50_000, checkpoint=None):
    """Возвращает (DataFrame, список пропущенных документов [(id, причина)]).
    Документ, который не скачался за doc_timeout секунд / слишком большой / упал - пропускается, сбор идёт дальше.
    Ctrl+C не теряет собранное: возвращается то, что успели собрать."""
    vacv = find_vacv_assays(http)
    docs = sorted({a["document_chembl_id"] for a in vacv.values() if a.get("document_chembl_id")})
    log.info("ChEMBL: %d VACV-ассеев в %d документах", len(vacv), len(docs))
    if max_docs:
        docs = docs[:max_docs]
    skip_docs = {d.strip() for d in skip_docs if d.strip()}
    rows, skipped = [], []
    try:
        for i, d in enumerate(docs, 1):
            if d in skip_docs:
                log.info("[%d/%d] %s: пропущен по --skip-docs", i, len(docs), d)
                continue
            deadline = time.monotonic() + doc_timeout
            try:
                assays = list(chembl_iter(http, "assay", "assays", {"document_chembl_id": d},
                                          deadline, label=f"{d} assays"))
                acts, mode = fetch_activities(http, d, deadline, max_activities)
                meta = next(iter(chembl_iter(http, "document", "documents", {"document_chembl_id": d},
                                             deadline, label=f"{d} document")), {})
            except KeyboardInterrupt:
                raise
            except Exception as e:  # TooLarge, TimeoutError, сеть - документ пропускаем, сбор продолжаем
                log.warning("[%d/%d] %s ПРОПУЩЕН: %s: %s", i, len(docs), d, type(e).__name__, e)
                skipped.append((d, f"{type(e).__name__}: {e}"))
                continue
            got = chembl_to_rows(d, meta, assays, acts, set(vacv), keep_censored)
            log.info("[%d/%d] %s: активностей=%d (%s) -> строк=%d (doi=%s)", i, len(docs), d, len(acts), mode,
                     len(got), meta.get("doi"))
            rows += got
            if checkpoint and i % 20 == 0:
                pd.DataFrame(rows, columns=OUT_COLUMNS).to_csv(checkpoint, index=False)
    except KeyboardInterrupt:
        log.warning("Прервано пользователем (Ctrl+C) - сохраняю собранное: %d строк. Кэш сохранён, "
                    "повторный запуск продолжит с места остановки.", len(rows))
    return pd.DataFrame(rows, columns=OUT_COLUMNS), skipped


# ---------------------------------------------------------------------------
# PubChem (best effort)
# ---------------------------------------------------------------------------
def _unit_from_header(col: str):
    m = re.search(r"(nM|[uµμ]M|mM)\b", col, re.I)
    return m.group(1) if m else None


def pubchem_table_to_rows(aid, df: pd.DataFrame, desc_text: str, doi: str = "") -> pd.DataFrame:
    """Таблица PubChem BioAssay (CSV) -> строки с ic50/cc50. Нужны ОБА столбца в одной таблице и
    определяемые единицы (строка RESULT_UNIT, заголовок столбца или единственная единица в описании)."""
    df = df.copy()
    units_row = None
    if "PUBCHEM_RESULT_TAG" in df.columns:
        tag = df["PUBCHEM_RESULT_TAG"].astype(str)
        meta = df[tag.str.startswith("RESULT_")]
        if (tag == "RESULT_UNIT").any():
            units_row = df[tag == "RESULT_UNIT"].iloc[0]
        df = df[~tag.str.startswith("RESULT_")]
    cols = list(df.columns)
    ic_cols = [c for c in cols if re.search(r"(?<![A-Za-z])(IC50|EC50)(?![A-Za-z0-9])", c, re.I)]
    cc_cols = [c for c in cols if re.search(r"(?<![A-Za-z])(CC50|CD50|TC50)(?![A-Za-z0-9])", c, re.I)]
    if not ic_cols or not cc_cols:
        return pd.DataFrame()
    desc_units = set(re.findall(r"\b(nM|[uµμ]M|mM)\b", desc_text or "", re.I))

    def unit_for(c):
        if units_row is not None and isinstance(units_row.get(c), str) and units_row.get(c).strip():
            return units_row[c]
        return _unit_from_header(c) or (next(iter(desc_units)) if len(desc_units) == 1 else None)

    ic_c, cc_c = ic_cols[0], cc_cols[0]
    u_ic, u_cc = unit_for(ic_c), unit_for(cc_c)
    if not u_ic or not u_cc:
        log.info("AID %s: единицы не определены - пропуск", aid)
        return pd.DataFrame()
    out = pd.DataFrame({
        "cid": pd.to_numeric(df.get("PUBCHEM_CID"), errors="coerce"),
        "sid": pd.to_numeric(df.get("PUBCHEM_SID"), errors="coerce"),
        "ic50": [to_uM(v, u_ic) for v in pd.to_numeric(df[ic_c], errors="coerce")],
        "cc50": [to_uM(v, u_cc) for v in pd.to_numeric(df[cc_c], errors="coerce")],
    })
    out = out.dropna(subset=["ic50", "cc50"])
    out = out[(out.ic50 > 0) & (out.cc50 > 0)]
    out["aid"], out["doi"] = aid, doi
    return out


def collect_pubchem(http: Http, max_aids=200) -> pd.DataFrame:
    res = http.get_json(f"{EUTILS}/esearch.fcgi",
                        {"db": "pcassay", "term": "vaccinia virus", "retmax": 1000, "retmode": "json"})
    aids = res["esearchresult"]["idlist"][:max_aids]
    log.info("PubChem: найдено %d AID по запросу 'vaccinia virus'", len(aids))
    parts = []
    for aid in aids:
        try:
            d = http.get_json(f"{PUG}/assay/aid/{aid}/description/JSON")
            descr = d["PC_AssayContainer"][0]["assay"]["descr"]
            text = " ".join([descr.get("name", "")] + list(descr.get("description", [])) +
                            list(descr.get("protocol", [])) + list(descr.get("comment", [])))
            if not VIRAL_RE.search(text):
                continue
            m = DOI_RE.search(text)
            csv_text = http.get_text(f"{PUG}/assay/aid/{aid}/CSV")
            tbl = pd.read_csv(io.StringIO(csv_text), low_memory=False)
            part = pubchem_table_to_rows(aid, tbl, text, m.group(0).rstrip(".,;)") if m else "")
            if len(part):
                log.info("AID %s: %d строк IC50+CC50", aid, len(part))
                parts.append(part)
        except Exception as e:
            log.warning("AID %s пропущен: %s", aid, e)
    if not parts:
        return pd.DataFrame(columns=OUT_COLUMNS)
    t = pd.concat(parts, ignore_index=True)
    cids = sorted(set(t["cid"].dropna().astype(int)))
    smiles = {}
    for i in range(0, len(cids), 200):
        chunk = ",".join(map(str, cids[i:i + 200]))
        for prop in ("SMILES", "CanonicalSMILES"):  # имя свойства менялось в 2023-24 гг.
            try:
                js = http.get_json(f"{PUG}/compound/cid/{chunk}/property/{prop}/JSON")
                for p in js["PropertyTable"]["Properties"]:
                    key = next((k for k in p if "SMILES" in k.upper()), None)
                    if key:
                        smiles[p["CID"]] = p[key]
                break
            except Exception as e:
                log.warning("PubChem property %s: %s", prop, e)
    t["smiles"] = t["cid"].map(lambda c: smiles.get(int(c)) if pd.notna(c) else None)
    t = t.dropna(subset=["smiles"])
    out = pd.DataFrame({
        "smiles": t["smiles"], "ic50_uM": t["ic50"], "cc50_uM": t["cc50"], "si": t["cc50"] / t["ic50"],
        "doi": t["doi"], "source": "pubchem",
        "molecule_id": [f"sid{int(s)}" if pd.notna(s) else f"cid{int(c)}" for s, c in zip(t["sid"], t["cid"])],
        "document_id": "AID" + t["aid"].astype(str), "pubmed_id": "", "assay_ic50": t["aid"].astype(str),
        "assay_cc50": t["aid"].astype(str), "cell_line": "", "virus_strain": "Vaccinia virus",
        "si_source": "computed", "si_reported": None, "cell_matched": True, "censored": False,
        "n_ic50": 1, "n_cc50": 1,
    })
    return out[OUT_COLUMNS]


# ---------------------------------------------------------------------------
# Ручное пополнение
# ---------------------------------------------------------------------------
def load_manual(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=OUT_COLUMNS)
    m = pd.read_csv(p)
    if m.empty:
        return pd.DataFrame(columns=OUT_COLUMNS)
    if "si" not in m or m["si"].isna().any():
        calc = m["cc50_uM"] / m["ic50_uM"]
        m["si"] = m["si"].fillna(calc) if "si" in m else calc
    m["source"] = "manual"
    for c in OUT_COLUMNS:
        if c not in m:
            m[c] = None
    return m[OUT_COLUMNS]


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="chembl,pubchem,manual")
    ap.add_argument("--out", default="data/external/vacv_external_raw.csv")
    ap.add_argument("--manual", default="data/external/manual_literature.csv")
    ap.add_argument("--cache-dir", default="data/external/cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--keep-censored", action="store_true", help="оставлять значения с > / <")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--skip-docs", default="", help="ID документов ChEMBL через запятую, которые не качать")
    ap.add_argument("--doc-timeout", type=int, default=300, help="лимит секунд на один документ ChEMBL")
    ap.add_argument("--max-activities", type=int, default=50_000, help="пропускать документы с большим числом активностей")
    ap.add_argument("--read-timeout", type=int, default=60, help="таймаут чтения одного HTTP-запроса, с")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    srcs = {s.strip() for s in args.sources.split(",")}
    http = (Http(args.cache_dir, use_cache=not args.no_cache, read_timeout=args.read_timeout)
            if srcs & {"chembl", "pubchem"} else None)
    frames = []
    if "chembl" in srcs:
        part, skipped = collect_chembl(http, args.keep_censored, args.max_docs, args.skip_docs.split(","),
                                       args.doc_timeout, args.max_activities,
                                       checkpoint=str(Path(args.out).with_suffix(".partial.csv")))
        frames.append(part)
        if skipped:
            sk = Path(args.out).with_name("skipped_docs.csv")
            pd.DataFrame(skipped, columns=["document_id", "reason"]).to_csv(sk, index=False)
            log.warning("Пропущено документов: %d (список и причины: %s). Чтобы исключить их явно: --skip-docs %s",
                        len(skipped), sk, ",".join(d for d, _ in skipped))
    if "pubchem" in srcs:
        try:
            frames.append(collect_pubchem(http))
        except Exception as e:
            log.error("PubChem полностью не удался: %s", e)
    if "manual" in srcs:
        frames.append(load_manual(args.manual))
    frames = [f for f in frames if len(f)]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUT_COLUMNS)
    out = out.drop_duplicates(subset=["smiles", "ic50_uM", "cc50_uM", "si", "doi", "cell_line"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    log.info("Готово: %d строк -> %s", len(out), args.out)
    if len(out):
        log.info("по источникам:\n%s", out["source"].value_counts().to_string())
        log.info("с DOI: %.0f%%", 100 * (out["doi"].fillna("") != "").mean())


if __name__ == "__main__":
    main()
