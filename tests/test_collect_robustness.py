"""Тесты устойчивости сбора ChEMBL (зависания, огромные документы, кэш) на подставном HTTP-клиенте."""
import json
import time

import requests

from src import collect_data as cd
from tests.test_collect import act


def page(key, items, nxt=None, total=None):
    return {"page_meta": {"next": nxt, "total_count": total if total is not None else len(items)}, key: items}


class FakeHttp:
    """Ведёт себя как Http: ответ из кэша, иначе handler (и сохраняет в кэш); cache_only -> CacheMiss."""

    def __init__(self, handler, cache=None):
        self.handler, self.cache, self.network_calls = handler, dict(cache or {}), []

    @staticmethod
    def key(url, params):
        return url + json.dumps(params or {}, sort_keys=True)

    def get_json(self, url, params=None, cache_only=False):
        k = self.key(url, params)
        if k in self.cache:
            return self.cache[k]
        if cache_only:
            raise cd.CacheMiss(url)
        self.network_calls.append((url, params))
        self.cache[k] = self.handler(url, params)
        return self.cache[k]


ACT_URL = cd.CHEMBL_API + "/activity.json"


def test_pagination_follows_next_links():
    pages = {
        0: page("activities", [{"i": 1}, {"i": 2}], nxt="/chembl/api/data/activity.json?offset=2", total=3),
        1: page("activities", [{"i": 3}], total=3),
    }
    h = FakeHttp(lambda url, p: pages[1] if "offset=2" in url else pages[0])
    got = list(cd.chembl_iter(h, "activity", "activities", {"document_chembl_id": "D"}))
    assert [x["i"] for x in got] == [1, 2, 3]


def test_too_large_stops_after_first_page():
    h = FakeHttp(lambda url, p: page("activities", [{"i": 1}], nxt="/chembl/api/data/x?offset=1", total=10 ** 6))
    try:
        list(cd.chembl_iter(h, "activity", "activities", {}, max_records=50_000))
    except cd.TooLarge:
        assert len(h.network_calls) == 1  # остальные страницы не качались
        return
    raise AssertionError("TooLarge не сработал")


def test_deadline_raises_timeout():
    h = FakeHttp(lambda url, p: page("activities", [{"i": 1}], nxt="/chembl/api/data/x?offset=1", total=5))
    try:
        list(cd.chembl_iter(h, "activity", "activities", {}, deadline=time.monotonic() - 1))
    except TimeoutError:
        return
    raise AssertionError("TimeoutError не сработал")


def test_pagination_cycle_guard():
    # сервер всё время отдаёт одну и ту же ссылку next -> не должно быть бесконечного цикла
    h = FakeHttp(lambda url, p: page("activities", [{"i": 1}], nxt="/chembl/api/data/activity.json?offset=1", total=9999))
    got = list(cd.chembl_iter(h, "activity", "activities", {}))
    assert 1 <= len(got) <= 3


def legacy_key(doc):
    return FakeHttp.key(ACT_URL, {"document_chembl_id": doc, "format": "json", "limit": 1000})


def test_fetch_activities_reuses_complete_cache_without_network():
    h = FakeHttp(lambda url, p: (_ for _ in ()).throw(AssertionError("сеть не должна вызываться")),
                 cache={legacy_key("D1"): page("activities", [act("M", "A", "IC50", 1)])})
    acts, mode = cd.fetch_activities(h, "D1")
    assert mode == "cache" and len(acts) == 1 and h.network_calls == []


def test_fetch_activities_incomplete_cache_falls_back_to_filtered_request():
    # в кэше только 1-я страница нефильтрованной выгрузки, следующей нет -> идём в сеть с фильтром по типам
    first = page("activities", [act("M", "A", "IC50", 1)], nxt="/chembl/api/data/activity.json?offset=1000", total=5000)
    seen = []

    def handler(url, params):
        seen.append(params)
        return page("activities", [act("M", "A", "CC50", 5)])

    h = FakeHttp(handler, cache={legacy_key("D2"): first})
    acts, mode = cd.fetch_activities(h, "D2")
    assert mode == "filtered" and len(acts) == 1
    assert "IC50" in seen[0]["standard_type__in"] and "CC50" in seen[0]["standard_type__in"]


def test_fetch_activities_falls_back_when_server_rejects_filter():
    def handler(url, params):
        if "standard_type__in" in (params or {}):
            resp = requests.Response()
            resp.status_code = 400
            raise requests.HTTPError("bad filter", response=resp)
        return page("activities", [act("M", "A", "IC50", 1)])

    acts, mode = cd.fetch_activities(FakeHttp(handler), "D3")
    assert mode == "unfiltered" and len(acts) == 1


def test_collect_chembl_skips_huge_and_user_listed_documents():
    assays = {
        "D_OK": [{"assay_chembl_id": "A1", "document_chembl_id": "D_OK", "assay_cell_type": "Vero",
                  "description": "Antiviral activity against Vaccinia virus in Vero cells"},
                 {"assay_chembl_id": "A2", "document_chembl_id": "D_OK", "assay_cell_type": "Vero",
                  "description": "Cytotoxicity against Vero cells"}],
        "D_BIG": [{"assay_chembl_id": "A3", "document_chembl_id": "D_BIG", "description": "Vaccinia virus assay"}],
        "D_SKIP": [{"assay_chembl_id": "A4", "document_chembl_id": "D_SKIP", "description": "Vaccinia virus assay"}],
    }

    def handler(url, params):
        params = params or {}
        if url.endswith("/assay.json"):
            if "assay_tax_id" in params:
                return page("assays", [assays[d][0] for d in assays])
            if "document_chembl_id" in params:
                return page("assays", assays[params["document_chembl_id"]])
            return page("assays", [])
        if url.endswith("/target.json"):
            return page("targets", [])
        if url.endswith("/document.json"):
            return page("documents", [{"doi": "10.1/ok", "pubmed_id": "1"}])
        if url.endswith("/activity.json"):
            if params["document_chembl_id"] == "D_BIG":
                return page("activities", [], total=10 ** 7)
            assert params["document_chembl_id"] != "D_SKIP", "D_SKIP не должен запрашиваться"
            return page("activities", [act("M1", "A1", "IC50", 2000), act("M1", "A2", "CC50", 100000)])
        raise AssertionError(url)

    df, skipped = cd.collect_chembl(FakeHttp(handler), skip_docs=["D_SKIP"], max_activities=50_000)
    assert len(df) == 1 and df.loc[0, "si"] == 50 and df.loc[0, "doi"] == "10.1/ok"
    assert [d for d, _ in skipped] == ["D_BIG"] and "TooLarge" in skipped[0][1]
