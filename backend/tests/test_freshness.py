import json

from reglens.corpus.freshness import check


def _manifest(tmp_path):
    m = {"as_of": "2026-10-02", "parts": {"1005": {"sections": None}, "1026": {"sections": ["1026.12", "1026.13"]}}}
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    return tmp_path / "manifest.json"


def v(identifier, d, part):
    return {"identifier": identifier, "amendment_date": d, "substantive": True, "part": part}


def test_current_when_nothing_in_scope_changed_after_snapshot(tmp_path):
    data = {
        "1005": [v("1005.10", "2025-10-01", "1005")],
        "1026": [v("1026.12", "2024-01-01", "1026"), v("1026.61", "2026-12-01", "1026")],
    }  # 1026.61 out of scope
    r = check(_manifest(tmp_path), fetch=data.__getitem__)
    assert r["current"] and r["parts"]["1026"]["latest_amendment"] == "2024-01-01"


def test_flags_in_scope_amendment_after_snapshot(tmp_path):
    data = {
        "1005": [v("1005.11", "2026-11-15", "1005")],
        "1026": [v("Supplement I to Part 1026", "2026-10-20", "1026")],
    }
    r = check(_manifest(tmp_path), fetch=data.__getitem__)
    assert not r["current"]
    assert r["parts"]["1005"]["amended_since_snapshot"][0]["identifier"] == "1005.11"
    assert r["parts"]["1026"]["amended_since_snapshot"][0]["identifier"] == "Supplement I to Part 1026"
