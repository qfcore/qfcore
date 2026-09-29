import random
import pytest
from qfnext import CellMap
from qfnext import _native
from qfnext.controls import triangulated_torus,presentation_space,compare_polygon_cups,simplicial_complex,bouquet


def test_identical_matrices_different_rings():
    torus,_,_,_=presentation_space((1,2,-1,-2))
    wedge,_,_,_=presentation_space(())
    assert torus.homology().betti==wedge.homology().betti==(1,2,1)
    assert all(torus.boundary(c)==wedge.boundary(c) for c in torus.cell_ids())
    a,b=torus.cohomology(),wedge.cohomology()
    assert a.cup(1,0,1,1)==a.cup(1,1,1,0)==(0,)
    assert a.cup(1,0,1,0)==a.cup(1,1,1,1)==()
    assert all(not v for v in b.table().values())
    assert compare_polygon_cups(torus)>0
    assert compare_polygon_cups(wedge)>0


def test_rp2_inverse_letter_correction():
    for word in [(1,1),(-1,-1),(1,-1),(1,1,1,1)]:
        q,_,_,_=presentation_space(word,rank=1)
        ring=q.cohomology()
        assert ring.betti==(1,1,1)
        assert ring.cup(1,0,1,0)==((0,) if word in [(1,1),(-1,-1)] else ())
        compare_polygon_cups(q)


def test_triangulated_torus_tree_naturality():
    q=triangulated_torus(4);a=q.clone()
    receipt=q.collapse(q.fundamental_group()[0].tree_edges)
    ar,br=a.cohomology(),q.cohomology()
    f=CellMap(a,q,receipt.image)
    pull=_native.pullback(ar._handle,br._handle,f.chain_images())
    assert ar.betti==br.betti==(1,2,1)
    for p,i,qdeg,j in br.table(positive_only=False):
        lhs=set()
        for k in br.cup(p,i,qdeg,j):lhs.symmetric_difference_update(pull[p+qdeg][k])
        rhs=set()
        for x in pull[p][i]:
            for y in pull[qdeg][j]:rhs.symmetric_difference_update(ar.cup(p,x,qdeg,y))
        assert lhs==rhs


def test_polygon_diagonal_against_independent_fan_80_cases():
    rng=random.Random(20260921)
    for _ in range(80):
        q,v,edges=bouquet(rng.randrange(1,5))
        for k in range(rng.randrange(1,5)):
            word=[(rng.choice(edges),rng.choice([-1,1])) for j in range(rng.randrange(0,15))]
            q.attach_disk(word,basepoint=v)
        compare_polygon_cups(q)


def test_simplicial_degree_zero_unit_and_higher_dimension():
    # Boundary of the 4-simplex = S^3: exercises AW in higher degree.
    q,_=simplicial_complex([tuple(j for j in range(5) if j!=i) for i in range(5)])
    r=q.cohomology()
    assert r.betti==(1,0,0,1)
    assert r.cup(0,0,3,0)==r.cup(3,0,0,0)==(0,)
    # All source cells can be collapsed without constructing fake positive cells.
    q.collapse(q.cell_ids())
    assert q.cohomology().betti==(1,)


def test_nonloop_polygon_against_fan():
    q,ids=simplicial_complex([(0,1),(1,2),(0,2)])
    q.attach_disk([(ids[(0,1)],1),(ids[(1,2)],1),(ids[(0,2)],-1)])
    assert q.cohomology().betti==(1,0,0)
    compare_polygon_cups(q)


def test_public_ring_map_and_different_dimensions():
    q=triangulated_torus(4);old=q.clone()
    receipt=q.collapse(q.fundamental_group()[0].tree_edges)
    result=CellMap(old,q,receipt.image).cohomology(verify_products=True)
    assert result['products_verified']
    before=q.clone();receipt=q.collapse(q.cell_ids())
    result=CellMap(before,q,receipt.image).cohomology(verify_products=True)
    assert result['source_ring'].betti==(1,2,1)
    assert result['target_ring'].betti==(1,0,0)


def test_chain_map_is_not_a_certificate_of_cellular_map():
    a,_,_,_=presentation_space((1,2,-1,-2))
    b,_,_,_=presentation_space(())
    assert a.homology().betti==b.homology().betti
    # All chain matrices and integer boundaries coincide, but attaching words do not.
    with pytest.raises(ValueError,match='attaching'):
        CellMap(a,b).cohomology()
    with pytest.raises(ValueError,match='attaching'):
        CellMap(a,b).homology()


def test_three_torus_exterior_ring_and_quotient_naturality():
    from itertools import product,permutations
    from qfnext import EditableQF
    side=4
    def label(v):return sum((v[k]%side)*side**k for k in range(3))
    tetra=[]
    for base in product(range(side),repeat=3):
        for order in permutations(range(3)):
            point=list(base);vertices=[label(point)]
            for axis in order:
                point[axis]+=1;vertices.append(label(point))
            tetra.append(tuple(vertices))
    q,_=simplicial_complex(tetra);ring=q.cohomology()
    assert ring.betti==(1,3,3,1)
    assert all(not ring.cup(1,i,1,i) for i in range(3))
    columns=[ring.cup(1,i,1,j) for i in range(3) for j in range(i+1,3)]
    pivots={}
    for column in columns:
        v=sum(1<<i for i in column)
        while v:
            p=v.bit_length()-1
            if p in pivots:v^=pivots[p]
            else:pivots[p]=v;break
    assert len(pivots)==3
    triple=set()
    for k in ring.cup(1,0,1,1):triple.symmetric_difference_update(ring.cup(2,k,1,2))
    assert triple=={0}
    old=q.clone();change=q.collapse(q.fundamental_group()[0].tree_edges)
    assert CellMap(old,q,change.image).cohomology(verify_products=True)['products_verified']
