"""Deterministic spaces and an independent fan subdivision for word attachments."""
from itertools import combinations
from .api import EditableQF


def simplicial_complex(maximal_simplices):
    simplices = set()
    for simplex in maximal_simplices:
        simplex = tuple(sorted(map(int, simplex)))
        if len(set(simplex)) != len(simplex) or not simplex:
            raise ValueError('nonempty simplex with distinct vertices required')
        for k in range(1,len(simplex)+1):
            simplices.update(combinations(simplex,k))
    q, ids = EditableQF(), {}
    for s in sorted(simplices,key=lambda s:(len(s),s)):
        ids[s] = q.add_vertex() if len(s)==1 else q.add_simplex(
            len(s)-1,[ids[s[:i]+s[i+1:]] for i in range(len(s))])
    return q,ids


def triangulated_torus(side=4):
    if side < 4:
        raise ValueError('side >= 4 avoids small-grid identification ambiguities')
    vertex=lambda i,j:(i%side)*side+(j%side)
    triangles=[]
    for i in range(side):
        for j in range(side):
            a,b,c,d=vertex(i,j),vertex(i+1,j),vertex(i+1,j+1),vertex(i,j+1)
            triangles += [(a,b,c),(a,c,d)]
    return simplicial_complex(triangles)[0]


def bouquet(rank=2):
    q=EditableQF();v=q.add_vertex();edges=[q.add_edge(v,v) for _ in range(rank)]
    return q,v,edges


def presentation_space(word=(1,2,-1,-2),rank=2):
    q,v,edges=bouquet(rank)
    disk=q.attach_disk([(edges[abs(w)-1],1 if w>0 else -1) for w in word],basepoint=v)
    return q,v,edges,disk


def fan_subdivide_polygons(q):
    """Replace each polygon by ordered triangle cells, independently of word cups.

    Returns (triangular model, cellular chain images). An original polygon maps
    to the sum of its fan triangles over F2, all other original cells identically.
    """
    r=q.clone();images={i:(i,) for i in q.cell_ids()}
    for f in q.cell_ids(2):
        c=q.cell(f)
        if c.kind!=2:
            continue
        r.delete(f)
        if not c.word:
            t=r.add_simplex(2,[c.base]*3)
            images[f]=(t,)
            continue
        center=r.add_vertex()
        current=c.base;vertices=[]
        for w in c.word:
            vertices.append(current)
            edge=q.cell(abs(w)-1)
            current=edge.facets[0 if w>0 else 1]
        spokes=[r.add_edge(center,v) for v in vertices]
        triangles=[]
        for j,w in enumerate(c.word):
            left,right=spokes[j],spokes[(j+1)%len(spokes)]
            facets=[abs(w)-1,right,left] if w>0 else [abs(w)-1,left,right]
            triangles.append(r.add_simplex(2,facets))
        images[f]=tuple(triangles)
    r.validate()
    return r,images


def triangle_chain_model_equal(q,r,images):
    for c in q.cell_ids():
        lhs=set()
        for t in images[c]:
            lhs.symmetric_difference_update(r.boundary(t))
        rhs=set()
        for f in q.boundary(c):
            rhs.symmetric_difference_update(images[f])
        if lhs!=rhs:
            return False
    return True


def compare_polygon_cups(q):
    """Compare the whole cup-product table under the fan chain equivalence."""
    r,images=fan_subdivide_polygons(q)
    if not triangle_chain_model_equal(q,r,images):
        raise AssertionError('fan chain equivalence boundary mismatch')
    a,b=q.cohomology(),r.cohomology()
    if a.betti!=b.betti:
        raise AssertionError('fan changed Betti numbers')
    index_b=[{c:i for i,c in enumerate(basis)} for basis in b.cell_basis]
    def pull(degree, cochain):
        values=set(cochain);out=[]
        for j,c in enumerate(a.cell_basis[degree]):
            if sum(index_b[degree][t] in values for t in images[c])%2:
                out.append(j)
        return out
    pullbacks={}
    for d,cs in enumerate(b.cocycles):
        for i,c in enumerate(cs):
            pullbacks[d,i]=a._handle.coordinates(d,pull(d,c))
    for p,i,qdeg,j in b.table(positive_only=False,max_products=100000):
        result=b._handle.cochain_product(p,list(b.cocycles[p][i]),qdeg,list(b.cocycles[qdeg][j]))
        left=a._handle.coordinates(p+qdeg,pull(p+qdeg,result))
        right=set()
        for x in pullbacks[p,i]:
            for y in pullbacks[qdeg,j]:
                right.symmetric_difference_update(a.cup(p,x,qdeg,y))
        if sorted(right)!=left:
            raise AssertionError(('fan cup naturality failed',p,i,qdeg,j,left,right))
    return len(b.table(positive_only=False))
