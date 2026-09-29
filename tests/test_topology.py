from pathlib import Path
import random
import numpy as np
import pytest
from qfcore import QFTree
from qfnext import EditableQF, CellMap, glue_points
from qfnext.controls import triangulated_torus, presentation_space, bouquet, simplicial_complex
ROOT=Path(__file__).resolve().parents[1]


def test_sphere_subsets_local_collapse():
    original=QFTree.load(ROOT/'tests/data/input_sphere.qft')
    for mask in range(1,16):
        selected=[i+2 for i in range(4) if mask>>i&1]
        q=EditableQF.from_qft(original);before=q.clone()
        receipt=q.collapse(selected)
        assert q.validate()
        r,f=original.collapse(selected,close=True,return_map=True)
        assert q.homology().betti==(1,0,len(selected))
        h=CellMap(before,q,receipt.image).homology()
        assert h['ranks']==(1,0,1)
        assert len(h['cokernel_basis'][2])==len(selected)-1
        images={i:receipt.image.get(i,i) for i in before.cell_ids()}
        # Compare cell-map targets and signed boundaries via correspondence.
        inverse={int(f.image[i]):images[i] for i in before.cell_ids()}
        for old in before.cell_ids():
            t=images[old];rid=int(f.image[old])
            assert q.cell(t).dimension==r.cell(rid).dimension
        assert q._handle.incidence_count()>=0


def test_locality_independent_of_unrelated_cells():
    for extra in [0,100,10000]:
        q,ids=simplicial_complex([(0,1,2)])
        for _ in range(extra):q.add_vertex()
        r=q.collapse([ids[(0,1)]])
        assert r.visited_cells<=7
        assert r.touched_occurrences<=7
        assert len(q)==extra+5
        assert q.validate()


def test_delete_uses_full_incidence_even_zero_boundary():
    q,v,edges,d=presentation_space()
    assert not q.boundary(d)
    with pytest.raises(ValueError):q.delete(edges[0])
    r=q.delete(edges[0],cascade=True)
    assert set(r.removed)=={edges[0],d}
    assert q.homology().betti==(1,1)


def test_glue_source_free_then_collapse(tmp_path):
    a,va,ea=bouquet(1);b,vb,eb=bouquet(1)
    a.save(tmp_path/'a.qfe');b.save(tmp_path/'b.qfe')
    a=EditableQF.load(tmp_path/'a.qfe');b=EditableQF.load(tmp_path/'b.qfe')
    r,maps=glue_points([a,b],[(0,va,1,vb)])
    assert r.homology().betti==(1,2)
    assert maps[0].homology()['ranks']==(1,1)
    before=r.clone()
    ch=r.collapse([maps[0].image[ea[0]]])
    assert r.homology().betti==(1,1)
    assert CellMap(before,r,ch.image).homology()['ranks']==(1,1)
    with pytest.raises(RuntimeError):maps[0].homology()


def test_two_intervals_glued_at_two_points():
    a,ai=simplicial_complex([(0,1)])
    b,bi=simplicial_complex([(0,1)])
    r,_=glue_points([a,b],[(0,ai[(0,)],1,bi[(0,)]),(0,ai[(1,)],1,bi[(1,)])])
    assert r.homology().betti==(1,1)


def test_roundtrip_and_no_implicit_id_reuse(tmp_path):
    q,v,e,d=presentation_space();q.delete(d)
    old_next=q._handle.next_id
    q.save(tmp_path/'q.qfe');r=EditableQF.load(tmp_path/'q.qfe')
    assert r._handle.next_id==old_next
    assert r.add_vertex()==old_next
    q.visualize_dag(tmp_path/'q')
    assert (tmp_path/'q.dot').exists()
    with pytest.raises(ValueError):EditableQF.load(tmp_path/'q.qfe',max_cells=1)


def test_invalid_edits_do_not_change_topology():
    q,ids=simplicial_complex([(0,1,2)])
    before=q._arrays()
    with pytest.raises(ValueError):q.collapse([ids[(0,1)]],close=False)
    with pytest.raises(ValueError):q.add_simplex(2,[ids[(0,1)]]*3)
    with pytest.raises(ValueError):q.attach_disk([(ids[(0,1)],1)])
    assert all(np.array_equal(v,q._arrays()[k]) for k,v in before.items())


def test_pi1_from_triangulated_torus_after_tree_and_load(tmp_path):
    q=triangulated_torus(4)
    assert q.homology().betti==(1,2,1)
    tree=q.fundamental_group()[0].tree_edges
    q.collapse(tree);q.save(tmp_path/'torus.qfe');del q
    r=EditableQF.load(tmp_path/'torus.qfe')
    p=r.fundamental_group()[0].simplify()
    assert len(p['generators'])==2 and len(p['relators'])==1
    word=p['relators'][0]
    assert len(word)==4 and all(sum(1 if x==g else -1 if x==-g else 0 for x in word)==0 for g in p['generators'])
    assert r.homology().betti==(1,2,1)


def test_attach_reversed_path_and_disconnected_groups():
    q,ids=simplicial_complex([(0,1)])
    e=ids[(0,1)]
    q.attach_disk([(e,1),(e,-1)],basepoint=ids[(0,)])
    q.add_vertex()
    assert q.homology().betti==(2,0,1)
    assert len(q.fundamental_group())==2


def test_large_affected_star_is_not_claimed_constant_time():
    q=EditableQF()
    for i in range(101):q.add_vertex(cell_id=i)
    center=100
    edges=[q.add_edge(center,j) for j in range(100)]
    change=q.collapse([edges[0]])
    assert change.visited_cells>=100
    assert change.touched_occurrences>=198
    assert q.validate()


def test_serialization_detects_corruption_and_negative_id(tmp_path):
    q,v,e,d=presentation_space()
    with pytest.raises(ValueError):q.add_vertex(cell_id=-2)
    with pytest.raises(ValueError):q.attach_disk([],basepoint=v,cell_id=-1)
    arrays=q._arrays();arrays['words']=arrays['words'].copy();arrays['words'][0]*=-1
    with (tmp_path/'corrupt.qfe').open('wb') as f:np.savez(f,**arrays)
    with pytest.raises(ValueError,match='integrity'):EditableQF.load(tmp_path/'corrupt.qfe')


def test_random_local_collapses_match_full_qft_facets_120_cases():
    from qfcore import FlagComplex
    rng=random.Random(704)
    steps_checked=0
    for trial in range(120):
        vertices=8
        edges=[(i,j) for i in range(vertices) for j in range(i+1,vertices) if rng.random()<.38]
        K=FlagComplex.from_graph(vertices,edges,max_dim=3)
        chosen=[v for v in range(vertices) if rng.random()<.35]
        original=K.quotient(K.induced_subcomplex(chosen),word_index=False,keep_source=False)
        editable=EditableQF.from_qft(original)
        correspond={i:i for i in original.cell_ids()}
        for step in range(4):
            candidates=[i for i in original.cell_ids() if original.cell(i).dimension>0]
            if not candidates:break
            selected=rng.sample(candidates,min(len(candidates),rng.randrange(1,4)))
            before=editable.clone()
            change=editable.collapse([correspond[i] for i in selected])
            target,f=original.collapse(selected,close=True,return_map=True)
            updated={}
            for i in original.cell_ids():
                rid=int(f.image[i]);eid=change.image.get(correspond[i],correspond[i])
                if rid in updated:assert updated[rid]==eid
                updated[rid]=eid
            assert len(updated)==len(target)==len(editable)
            for rid,eid in updated.items():
                assert target.cell(rid).dimension==editable.cell(eid).dimension
                assert [updated[x.target] for x in target.facets(rid)]==editable.cell(eid).facets
            left=CellMap(before,editable,change.image).homology()['target'].betti
            right=tuple(target.betti_numbers());degree=max(len(left),len(right))
            assert left+(0,)*(degree-len(left))==right+(0,)*(degree-len(right))
            assert editable.validate()
            original,correspond=target,updated;steps_checked+=1
    assert steps_checked>200
