"""Invariants for the perfume atlas — pure helpers + integrity of the emitted JSON.

Run:  uv run --with pytest,numpy pytest perfumes/test_atlas.py -q
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

import build_atlas as B  # safe to import: main() is under __main__ guard

DATA = Path(__file__).resolve().parent.parent / "assets" / "data"


# ---- pure helpers -----------------------------------------------------------
def test_prettify():
    assert B.prettify("jean-paul-gaultier") == "Jean Paul Gaultier"
    assert (
        B.prettify("eau-de-toilette") == "Eau de Toilette"
    )  # known small words stay lower
    assert B.prettify("la-vie-est-belle") == "La Vie Est Belle"
    assert B.prettify("") == ""


def test_norm_gender():
    assert B.norm_gender("for women") == 0
    assert B.norm_gender("men") == 1
    assert B.norm_gender("unisex") == 2
    assert B.norm_gender("") == 3


def test_parse_list_strips_replacement_char():
    # the utf8-lossy � must not leak into notes
    assert B.parse_list("physcool�, pear") == ["physcool", "pear"]
    assert B.parse_list("Rose, UNKNOWN, unknown,  ") == ["rose"]
    assert B.parse_list(None) == []


def test_every_accord_maps_to_a_family():
    # guards future accord additions: an unmapped accord would silently mis-colour perfumes
    for accs in (a for _n, _c, a in B.MACRO):
        for a in accs:
            assert a in B.ACCORD2FAM
    assert len(B.MACRO) == 14


def test_pid_regex():
    assert (
        B.PID_RE.search(
            "https://www.fragrantica.com/perfume/Afnan/9am-70706.html"
        ).group(1)
        == "70706"
    )


def test_weighted_cosine_identity_holds_even_with_empty_accords():
    # the bug the review caught: a row with notes but no accords must still be unit-norm
    from scipy.sparse import csr_matrix, hstack
    from sklearn.preprocessing import normalize

    Xn = normalize(csr_matrix(np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]])))
    Xa = normalize(
        csr_matrix(np.array([[3.0, 0.0], [0.0, 0.0]]))
    )  # row 1 has no accords
    X = normalize(hstack([Xn * math.sqrt(0.7), Xa * math.sqrt(0.3)]).tocsr())
    rownorm = np.sqrt(X.multiply(X).sum(axis=1)).A1
    assert np.allclose(rownorm, 1.0, atol=1e-6)


# ---- emitted-JSON integrity (skips if the build hasn't been run) -------------
@pytest.fixture(scope="module")
def atlas():
    p = DATA / "perfumes-atlas.json"
    if not p.exists():
        pytest.skip("run build_atlas.py first")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def neighbors():
    p = DATA / "perfumes-neighbors.json"
    if not p.exists():
        pytest.skip("run build_atlas.py first")
    return json.loads(p.read_text())


def test_columns_are_aligned(atlas):
    n = atlas["meta"]["n"]
    for col in (
        "pid",
        "name",
        "brand",
        "year",
        "gender",
        "rating",
        "reviews",
        "fam",
        "x",
        "y",
        "notes",
        "accords",
    ):
        assert len(atlas[col]) == n, f"{col} length mismatch"


def test_positions_finite_and_families_valid(atlas):
    assert all(math.isfinite(v) for v in atlas["x"])
    assert all(math.isfinite(v) for v in atlas["y"])
    fam_ids = {f["id"] for f in atlas["families"]}
    assert set(atlas["fam"]) <= fam_ids
    assert sum(f["size"] for f in atlas["families"]) == atlas["meta"]["n"]


def test_neighbors_have_no_self_and_valid_indices(neighbors):
    n = len(neighbors["nbr"])
    for i, (nb, w) in enumerate(zip(neighbors["nbr"], neighbors["w"])):
        assert i not in nb, f"self-neighbour at {i}"
        assert all(0 <= j < n for j in nb)
        assert all(0.0 <= x <= 1.0001 for x in w)
        assert w == sorted(w, reverse=True), f"weights not descending at {i}"


def test_analyses_twins_are_cross_house():
    p = DATA / "perfume-analyses.json"
    if not p.exists():
        pytest.skip("run build_atlas.py first")
    an = json.loads(p.read_text())
    for t in an["twins"]:
        assert (
            t["a"]["brand"] != t["b"]["brand"]
        ), "headline twins must be from different houses"
        assert 0.5 < t["sim"] <= 1.0
