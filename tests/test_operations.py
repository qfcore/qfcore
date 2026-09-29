"""Regression and independent algebraic checks for homology maps."""
import itertools
import numpy as np
import pytest
import qfcore as qf
from qfops import _ops as op

LIM = 2_000_000

def sphere():
    K = qf.FlagComplex.from_graph(5, [(0,1),(1,2),(2,3),(0,3),(0,4),(1,4),(2,4),(3,4)], max_dim=2)
    A = K.induced_subcomplex(range(4))
    return K, A, K.quotient(A, word_index=False)

def assert_chains(a, b):
    assert a._counts == b._counts
    for d in range(len(a._counts)):
        x, y = a.boundary_matrix(d), b.boundary_matrix(d)
        assert x.shape == y.shape
        np.testing.assert_array_equal(x.indices, y.indices)
        np.testing.assert_array_equal(x.indptr, y.indptr)

def independent_rank(columns):
    pivots = {}
    for col in columns:
        x = sum(1 << int(i) for i in col)
        while x:
            p = x.bit_length()-1
            if p in pivots: x ^= pivots[p]
            else:
                pivots[p] = x
                break
    return len(pivots)

def check_step(Q, R, f, src_h, dst_h):
    qc, rc = Q.chain_complex(), R.chain_complex()
    image = op.qf_images(Q._handle, R._handle, f.image.tolist())
    assert op.chain_map_valid(qc._handle, rc._handle, image, LIM)
    actual = op.induced(src_h, dst_h, image)
    inv = op.map_invariants(actual, dst_h.betti(), LIM)
    # Rank of induced H-map independently: rank(B + F(Z)) - rank(B).
    for d, cycles in enumerate(src_h.cycles()):
        b = rc.boundary_matrix(d+1) if d+1 < len(rc._counts) else None
        boundaries = [] if b is None else [b.indices[b.indptr[j]:b.indptr[j+1]].tolist() for j in range(b.shape[1])]
        mapped = []
        for z in cycles:
            odd = set()
            for j in z:
                t = image[d][j]
                if t >= 0:
                    if t in odd: odd.remove(t)
                    else: odd.add(t)
            mapped.append(sorted(odd))
        rank = independent_rank(boundaries+mapped)-independent_rank(boundaries)
        assert rank == inv[d][0]
        assert len(inv[d][1]) == src_h.betti()[d]-rank
        assert len(inv[d][2]) == dst_h.betti()[d]-rank
    return image, actual

def test_sphere_15_branches_source_free(tmp_path):
    K, A, Q = sphere()
    Q.save(tmp_path/'sphere.qft')
    Q = qf.QFTree.load(tmp_path/'sphere.qft')
    assert not Q.has_source_archive
    H = op.Homology(Q.chain_complex()._handle, LIM)
    for r in range(1, 5):
        for edges in itertools.combinations(Q.cell_ids(1), r):
            R, f = Q.collapse(edges, close=True, return_map=True)
            RH = op.Homology(R.chain_complex()._handle, LIM)
            assert RH.betti() == [1, 0, r]
            _, m = check_step(Q, R, f, H, RH)
            assert op.map_invariants(m, RH.betti(), LIM)[2][0] == 1
            if r == 4: assert m[2] == [[0,1,2,3]]

def test_equal_betti_different_maps():
    edges=[(0,1),(1,2),(2,3),(3,0),(0,4),(4,5),(5,6),(6,0)]
    K=qf.FlagComplex.from_graph(7, edges, max_dim=2)
    Q=K.quotient(K.induced_subcomplex([0]),word_index=False)
    H=op.Homology(Q.chain_complex()._handle,LIM)
    maps=[]
    for cycle in (edges[:4],edges[4:]):
        ids=[Q.find_by_source(e) for e in cycle]
        R,f=Q.collapse(ids,close=True,return_map=True)
        RH=op.Homology(R.chain_complex()._handle,LIM)
        assert RH.betti()==[1,1]
        _,m=check_step(Q,R,f,H,RH)
        maps.append(m[1])
    assert sorted(maps)==sorted([[[0],[]],[[],[0]]])


def test_degenerate_cases_and_limits():
    for n in (0,1,3):
        K=qf.FlagComplex.from_graph(n,[],max_dim=2)
        for vs in ([],list(range(n))):
            A=K.induced_subcomplex(vs);Q=K.quotient(A,word_index=False)
            data=op.TreeData(K._verts,K._off);state=op.TreeState(data,A.mask)
            c=Q.chain_complex();assert_chains(c,qf.ChainComplex(state.snapshot().chain()))
            H=op.Homology(c._handle,LIM);assert H.betti()==c.betti_numbers()
            R,f=Q.collapse([],close=True,return_map=True)
            _,m=check_step(Q,R,f,H,op.Homology(R.chain_complex()._handle,LIM))
            assert m==[[[i] for i in range(x)] for x in H.betti()]
    with pytest.raises(ValueError): op.Homology(c._handle,0)
    K,A,Q=sphere()
    with pytest.raises(ValueError):op.select_sources(Q._handle,[0])

def test_chain_map_negative_control():
    c=qf.ChainComplex(op.chain_from_columns([2,1],[[[],[]],[[0,1]]]))
    assert op.dd_zero(c._handle,LIM)
    assert op.chain_map_valid(c._handle,c._handle,[[0,1],[0]],LIM)
    assert not op.chain_map_valid(c._handle,c._handle,[[0,0],[0]],LIM)


def test_public_induced_map_api():
    from qfops.api import induced_homology_map
    _,_,Q=sphere()
    R,f=Q.collapse(Q.cell_ids(1),close=True,return_map=True)
    result=induced_homology_map(f)
    assert result.source_betti==(1,0,1)
    assert result.target_betti==(1,0,4)
    assert result.ranks==(1,0,1)
    assert result.kernel_dimensions==(0,0,0)
    assert result.cokernel_dimensions==(0,0,3)
    assert result.matrices[2]==((0,1,2,3),)
