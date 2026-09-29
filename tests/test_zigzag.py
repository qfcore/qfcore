import math
import random
from collections import Counter
import pytest
from qfnext import EditableQF,CellMap,ZigzagSession,map_zigzag
from qfnext.controls import bouquet,presentation_space,simplicial_complex


def reference_diagram(states,directions):
    maxdim=max(max(q.dimension(),0) for q in states)
    hs=[q.homology(max_dimension=maxdim) for q in states]
    arrows=[]
    for k,d in enumerate(directions):
        s,t=(k,k+1) if d>0 else (k+1,k)
        from qfops import _ops
        f=CellMap(states[s],states[t]).chain_images(maxdim)
        assert _ops.chain_map_valid(hs[s]._chain,hs[t]._chain,f,1000000)
        arrows.append(_ops.induced(hs[s]._handle,hs[t]._handle,f))
    out=[]
    for degree in range(maxdim+1):
        dims=[h.betti[degree] for h in hs]
        for b,e,m in map_zigzag(dims,[a[degree] for a in arrows],directions):
            out += [(degree,float(b),math.inf if e==len(states) else float(e))]*m
    return tuple(sorted(out))


def test_loop_disk_deletion_reinsertion_exact_barcode():
    q,v,edges=bouquet(1)
    zz=ZigzagSession(q);states=[zz.space.clone()];directions=[]
    disk=zz.attach_disk([(edges[0],1)],basepoint=v)
    states.append(zz.space.clone());directions.append(1)
    zz.delete(disk);states.append(zz.space.clone());directions.append(-1)
    zz.attach_disk([(edges[0],1)],basepoint=v,cell_id=disk)
    states.append(zz.space.clone());directions.append(1)
    assert zz.barcode()==((0,0.,math.inf),(1,0.,1.),(1,2.,3.))
    assert zz.barcode()==reference_diagram(states,directions)


def test_boundary_cancellation_does_not_allow_illegal_topological_removal():
    q,v,edges,d=presentation_space()
    zz=ZigzagSession(q)
    with pytest.raises(ValueError):zz.delete(edges[0])
    removed=zz.delete(edges[0],cascade=True)
    assert removed==(d,edges[0])
    assert zz.space.homology().betti==(1,1)


def test_disconnected_empty_and_out_of_band_changes():
    q=EditableQF();zz=ZigzagSession(q)
    assert zz.barcode()==()
    a=zz.add_vertex();b=zz.add_vertex();e=zz.add_edge(a,b)
    zz.delete(e);zz.delete(a);zz.delete(b)
    assert zz.space.homology().betti==()
    zz.space.add_vertex()
    with pytest.raises(RuntimeError):zz.barcode()


def test_random_edit_streams_against_global_oracle():
    rng=random.Random(61209)
    for trial in range(30):
        q,v,edges=bouquet(2)
        zz=ZigzagSession(q);states=[zz.space.clone()];directions=[];disks=[]
        for step in range(7):
            if disks and rng.random()<.5:
                zz.delete(disks.pop(rng.randrange(len(disks))));directions.append(-1)
            else:
                word=[(rng.choice(edges),rng.choice([-1,1])) for _ in range(rng.randrange(0,7))]
                disks.append(zz.attach_disk(word,basepoint=v));directions.append(1)
            states.append(zz.space.clone())
        assert zz.barcode()==reference_diagram(states,directions)


def apply(cols,vector):
    out=0
    for i,c in enumerate(cols):
        if vector>>i&1:out ^= c
    return out


def inverse(cols):
    piv={}
    for i,c in enumerate(cols):
        v,t=c,1<<i
        while v:
            p=v.bit_length()-1
            if p not in piv:piv[p]=(v,t);break
            v ^=piv[p][0];t ^=piv[p][1]
    out=[]
    for j in range(len(cols)):
        v,t=1<<j,0
        while v:
            p=v.bit_length()-1
            v ^=piv[p][0];t ^=piv[p][1]
        out.append(t)
    return out


def test_map_zigzag_recovers_scrambled_interval_modules_100_cases():
    rng=random.Random(77319)
    for trial in range(100):
        n=rng.randrange(2,8);intervals=[]
        for _ in range(rng.randrange(0,7)):
            a=rng.randrange(n);b=rng.randrange(a+1,n+1);intervals.append((a,b))
        active=[[j for j,(a,b) in enumerate(intervals) if a<=i<b] for i in range(n)]
        dims=list(map(len,active));directions=[rng.choice([-1,1]) for _ in range(n-1)]
        basis=[]
        for dim in dims:
            cols=[1<<i for i in range(dim)]
            for _ in range(15):
                if dim<2:break
                a,b=rng.sample(range(dim),2);cols[a]^=cols[b]
            basis.append(cols)
        maps=[]
        for k,d in enumerate(directions):
            s,t=(k,k+1) if d>0 else (k+1,k)
            f=[1<<active[t].index(j) if j in active[t] else 0 for j in active[s]]
            inv=inverse(basis[t]);new=[apply(inv,apply(f,v)) for v in basis[s]]
            maps.append([[j for j in range(dims[t]) if v>>j&1] for v in new])
        expected=Counter(intervals)
        result={(a,b):m for a,b,m in map_zigzag(dims,maps,directions)}
        assert result==dict(expected)


def test_maps_not_just_betti():
    # Same three vector spaces; different maps yield different barcodes.
    assert map_zigzag([1,1,1],[[[0]],[[0]]],[1,-1])==((0,3,1),)
    assert map_zigzag([1,1,1],[[[]],[[]]],[1,-1])==((0,1,1),(1,2,1),(2,3,1))
    with pytest.raises(ValueError):map_zigzag([1,1],[[[1]]],[1])
