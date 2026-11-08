"""Офлайн-тесты логики сбора данных на синтетических ответах в формате ChEMBL / PubChem.
Они проверяют НАШ код разбора, но не реальное поведение API (для этого запустите collect_data.py)."""
import io
import math

import pandas as pd

from src import collect_data as cd


def act(mol, assay, typ, val, units="nM", rel="=", smi="CCO", doc="CHEMBL1"):
    return {"molecule_chembl_id": mol, "canonical_smiles": smi, "assay_chembl_id": assay,
            "standard_type": typ, "standard_value": val, "standard_units": units,
            "standard_relation": rel, "document_chembl_id": doc, "data_validity_comment": None}


ASSAYS = [
    {"assay_chembl_id": "A1", "description": "Antiviral activity against Vaccinia virus Copenhagen infected Vero cells",
     "assay_cell_type": "Vero", "assay_strain": None},
    {"assay_chembl_id": "A2", "description": "Cytotoxicity against Vero cells", "assay_cell_type": "Vero"},
    {"assay_chembl_id": "A3", "description": "Cytotoxicity against HeLa cells", "assay_cell_type": "HeLa"},
    {"assay_chembl_id": "A4", "description": "Inhibition of human kinase", "assay_cell_type": None},
]
DOC = {"doi": "10.1000/test.1", "pubmed_id": "123"}


def rows(acts, **kw):
    return cd.chembl_to_rows("CHEMBL1", DOC, ASSAYS, acts, {"A1"}, **kw)


def test_unit_conversion():
    assert cd.to_uM(500, "nM") == 0.5
    assert cd.to_uM(2, "uM") == 2
    assert cd.to_uM(2, "µM") == 2
    assert cd.to_uM(1, "mM") == 1000
    assert cd.to_uM(5, "unknown") is None
    assert cd.to_uM(None, "nM") is None


def test_classification():
    a1, a2, a4 = ASSAYS[0], ASSAYS[1], ASSAYS[3]
    assert cd.classify(act("m", "A1", "IC50", 1), a1, {"A1"}) == "ic50"
    assert cd.classify(act("m", "A1", "EC50", 1), a1, {"A1"}) == "ic50"
    assert cd.classify(act("m", "A2", "CC50", 1), a2, {"A1"}) == "cc50"
    assert cd.classify(act("m", "A2", "IC50", 1), a2, {"A1"}) == "cc50"   # IC50 в ассее цитотоксичности
    assert cd.classify(act("m", "A4", "IC50", 1), a4, {"A1"}) is None     # не про вирус и не про токсичность
    assert cd.classify(act("m", "A1", "SI", 10, units=None), a1, {"A1"}) == "si"


def test_si_computed_from_ic50_and_cc50():
    r = rows([act("M1", "A1", "IC50", 2000), act("M1", "A2", "CC50", 100000)])
    assert len(r) == 1
    x = r[0]
    assert math.isclose(x["ic50_uM"], 2.0) and math.isclose(x["cc50_uM"], 100.0)
    assert math.isclose(x["si"], 50.0) and x["si_source"] == "computed"
    assert x["doi"] == "10.1000/test.1" and x["cell_matched"] is True


def test_replicates_use_geometric_mean():
    r = rows([act("M1", "A1", "IC50", 1000), act("M1", "A1", "IC50", 4000), act("M1", "A2", "CC50", 20000)])
    assert math.isclose(r[0]["ic50_uM"], 2.0) and r[0]["n_ic50"] == 2


def test_censored_dropped_by_default_and_kept_on_flag():
    acts = [act("M1", "A1", "IC50", 1000), act("M1", "A2", "CC50", 50000, rel=">")]
    assert rows(acts) == []
    r = rows(acts, keep_censored=True)
    assert len(r) == 1 and r[0]["censored"] is True


def test_cell_mismatch_flagged_and_ambiguous_skipped():
    # единственная цитотоксичность в другой клетке -> пара с флагом cell_matched=False
    r = rows([act("M1", "A1", "IC50", 1000), act("M1", "A3", "CC50", 10000)])
    assert len(r) == 1 and r[0]["cell_matched"] is False
    # две разные цитотоксичности, ни одна не совпадает по клетке -> пара не строится
    acts = [act("M1", "A1", "IC50", 1000), act("M1", "A3", "CC50", 10000),
            {**act("M1", "A3", "CC50", 20000), "assay_chembl_id": "A5"}]
    assays = ASSAYS + [{"assay_chembl_id": "A5", "description": "Cytotoxicity against MT-4 cells", "assay_cell_type": "MT-4"}]
    assert cd.chembl_to_rows("CHEMBL1", DOC, assays, acts, {"A1"}) == []


def test_reported_si_only():
    r = rows([act("M1", "A1", "SI", 42, units=None)])
    assert len(r) == 1 and r[0]["si_source"] == "reported" and r[0]["si"] == 42


def test_invalid_data_skipped():
    bad = {**act("M1", "A1", "IC50", 1000), "data_validity_comment": "Potential transcription error"}
    assert rows([bad, act("M1", "A2", "CC50", 5000)]) == []
    assert rows([act("M1", "A1", "IC50", -5), act("M1", "A2", "CC50", 5000)]) == []
    assert rows([act("M1", "A4", "IC50", 5), act("M1", "A2", "CC50", 5000)]) == []


def test_pubchem_table_with_result_unit_row():
    csv = ("PUBCHEM_RESULT_TAG,PUBCHEM_SID,PUBCHEM_CID,IC50,CC50\n"
           "RESULT_TYPE,,,float,float\n"
           "RESULT_UNIT,,,um,um\n"
           "1,111,11,2.0,100.0\n"
           "2,222,22,,50.0\n"
           "3,333,33,4.0,40.0\n")
    out = cd.pubchem_table_to_rows(999, pd.read_csv(io.StringIO(csv)), "vaccinia assay", doi="10.1/x")
    assert len(out) == 2
    assert list(out["sid"]) == [111, 333] and list(out["cc50"]) == [100.0, 40.0]


def test_pubchem_without_units_is_skipped():
    csv = "PUBCHEM_SID,PUBCHEM_CID,IC50,CC50\n1,1,2.0,100.0\n"
    assert cd.pubchem_table_to_rows(1, pd.read_csv(io.StringIO(csv)), "no units mentioned").empty
    csv2 = "PUBCHEM_SID,PUBCHEM_CID,IC50 (nM),CC50 (nM)\n1,1,2000,100000\n"
    out = cd.pubchem_table_to_rows(1, pd.read_csv(io.StringIO(csv2)), "")
    assert math.isclose(out.iloc[0]["ic50"], 2.0) and math.isclose(out.iloc[0]["cc50"], 100.0)


def test_manual_loader_computes_missing_si(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("smiles,ic50_uM,cc50_uM,doi\nCCO,2,100,10.1/y\n")
    m = cd.load_manual(p)
    assert m.loc[0, "si"] == 50 and m.loc[0, "source"] == "manual"
