"""Closed-star (local) quotient presentation: agreement with the full QF-tree."""
import numpy as np
import pytest

from qfcore import FlagComplex, LocalQuotient


def _octahedron():
    edges = [(i, j) for i in range(6) for j in range(i + 1, 6)
             if {i, j} not in ({0, 1}, {2, 3}, {4, 5})]
    return FlagComplex.from_graph(6, edges, max_dim=None), edges


def test_partition_labels_are_exhaustive_and_consistent():
    K, edges = _octahedron()
    A = K.induced_subcomplex([0, 1])          # partners 0,1 are non-adjacent: nothing collapsed
    labels, counts = K.star_partition(A)
    assert sum(counts) == len(K) and counts[3] == 2
    # star = survivors with a vertex in {0,2}; frontier = their faces avoiding {0,2}
    for s, lab in enumerate(labels):
        simplex = K.simplex(s)
        meets = bool({0, 1} & set(simplex))
        if lab == 3:
            assert set(simplex) <= {0, 1}
        elif lab == 2:
            assert meets
        else:
            assert not meets


def test_octahedron_extremal_collapse_is_entirely_star():
    K, edges = _octahedron()
    A = K.flag_subcomplex([e for e in edges if e != (0, 2)], vertices=range(6))
    L = K.local_quotient(A)
    assert L.counts == (0, 0, 3, 23)          # loop edge + two lunes; no untouched cell
    assert L.verify()
    assert L.betti_numbers() == [1, 0, 1]
    assert L.storage_units('ideal') == K.storage_units(A)


def test_square_with_collapsed_edge():
    K = FlagComplex.from_graph(4, [(0, 1), (1, 2), (2, 3), (0, 3)], max_dim=1)
    A = K.induced_subcomplex([0, 1])
    L = K.local_quotient(A)
    assert L.counts == (1, 2, 2, 3)           # edge 23 untouched; vertices 2,3 frontier; 12, 03 star
    assert L.verify()
    assert L.betti_numbers() == K.quotient(A).betti_numbers() == [1, 1]
    assert L.betti_numbers(quotient=False) == K.quotient(A).betti_numbers(relative=True)


@pytest.mark.parametrize('seed,frac', [(0, 0.05), (1, 0.3), (2, 0.7)])
def test_random_rips_pairs_agree_with_full_quotient(seed, frac):
    rng = np.random.default_rng(seed)
    n = 400
    K = FlagComplex.from_points(rng.random((n, 2)), 0.08, max_dim=3, torus=True)
    A = K.induced_subcomplex(rng.choice(n, int(frac * n), replace=False))
    L = K.local_quotient(A)
    assert isinstance(L, LocalQuotient)
    assert L.verify()
    Q = K.quotient(A, word_index=False)
    assert L.betti_numbers() == Q.betti_numbers()
    assert L.betti_numbers(quotient=False) == Q.betti_numbers(relative=True)
    assert len(L.tree) == L.num_frontier + L.num_star + A.n_components
    assert L.storage_units('ideal') <= K.storage_units(A)


def test_localized_ball_has_thin_shell():
    rng = np.random.default_rng(7)
    n = 2000
    pts = rng.random((n, 2))
    K = FlagComplex.from_points(pts, float(np.sqrt(8 / (np.pi * n))), max_dim=3, torus=True)
    d = pts - 0.5
    A = K.induced_subcomplex(np.flatnonzero((d ** 2).sum(1) <= 0.3 ** 2))
    L = K.local_quotient(A)
    assert L.verify()
    shell = L.num_frontier + L.num_star
    assert shell < 0.2 * len(K)
    # In this sampled regime the ideal closed-star budget is below the cone budget.
    assert L.storage_units('ideal') < len(K) + A.num_simplices() + A.n_components


def test_empty_or_foreign_subcomplex_is_rejected():
    K, _ = _octahedron()
    K2, _ = _octahedron()
    A2 = K2.induced_subcomplex([0])
    with pytest.raises(ValueError):
        K.local_quotient(A2)


def test_storage_models_and_audit_counterexample():
    # Four collinear points at radius 1: K is the full tetrahedron. A = the edge on the
    # first two points, selected by a small ball. Localization does not make the
    # closed-star budget smaller than the cone budget here.
    pts = np.array([[0, 0], [0.1, 0], [0.8, 0], [0.9, 0]], float)
    K = FlagComplex.from_points(pts, 1.0, max_dim=None)
    A = K.induced_subcomplex([0, 1])
    L = K.local_quotient(A)
    assert L.counts == (0, 3, 9, 3)
    cone = len(K) + A.num_simplices() + A.n_components
    assert (L.storage_units('ideal'), L.storage_units('compact'), L.storage_units('retained'), cone) == (29, 32, 44, 19)
    assert L.storage_units('ideal') <= L.storage_units('compact') <= L.storage_units('retained')
    with pytest.raises(ValueError):
        L.storage_units('bytes')


@pytest.mark.parametrize('seed,frac', [(0, 0.05), (1, 0.4)])
def test_compact_presentation_reproduces_homology_without_source(seed, frac):
    rng = np.random.default_rng(seed)
    n = 300
    K = FlagComplex.from_points(rng.random((n, 2)), 0.09, max_dim=3, torus=True)
    A = K.induced_subcomplex(rng.choice(n, int(frac * n), replace=False))
    L = K.local_quotient(A)
    C = L.compact()
    assert len(C.untouched) == L.num_untouched + L.num_frontier
    assert C.storage_units() == L.storage_units('compact')
    Q = K.quotient(A, word_index=False)
    assert C.betti_numbers() == Q.betti_numbers()
    assert C.betti_numbers(quotient=False) == Q.betti_numbers(relative=True)


def test_local_quotient_keeps_its_source_snapshot():
    # Editing K after construction must not mix the new complex with the old
    # partition and local tree (copy-on-write snapshot semantics).
    K = FlagComplex.from_graph(4, [(0, 1), (1, 2), (2, 3)], max_dim=1)
    A = K.induced_subcomplex([0, 1])
    L = K.local_quotient(A)
    before = (L.betti_numbers(), L.betti_numbers(quotient=False), L.counts,
              L.storage_units('retained'), L.storage_units('compact'), L.storage_units('ideal'))
    n = len(K)
    K.insert([3, 4])
    assert len(K) == n + 2 and len(L.complex) == n
    after = (L.betti_numbers(), L.betti_numbers(quotient=False), L.counts,
             L.storage_units('retained'), L.storage_units('compact'), L.storage_units('ideal'))
    assert after == before
    assert L.verify()
    C = L.compact()
    assert C.betti_numbers() == L.betti_numbers()
    assert C.betti_numbers(quotient=False) == L.betti_numbers(quotient=False)
    with pytest.raises(ValueError):
        K.local_quotient(A)  # A belongs to the earlier snapshot of K


def test_compact_betti_lists_match_the_full_quotient():
    # A contains the top simplex: trailing zero degrees are kept, as in K.quotient(A).
    K = FlagComplex.from_graph(3, [(0, 1), (1, 2), (0, 2)], max_dim=None)
    A = K.induced_subcomplex([0, 1, 2])
    L = K.local_quotient(A)
    C = L.compact()
    for quotient in (True, False):
        full = K.quotient(A, word_index=False).betti_numbers(relative=not quotient)
        assert C.betti_numbers(quotient=quotient) == L.betti_numbers(quotient=quotient) == full
