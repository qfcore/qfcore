# API guide

A space and the algebra computed from it are separate objects: homology,
cohomology and chain complexes are snapshots computed on request.
`qfcore.QFTree` is immutable; `qfnext.EditableQF` is mutable.

Section and proposition numbers refer to the preprint *A data structure for
quotient flag complexes* (Sorokin, Levin, Beketov, Ayzenberg).

## Closed-star (local) quotient

```python
from qfcore import FlagComplex
K = FlagComplex.from_graph(n, edges, max_dim=3)
A = K.induced_subcomplex(vertices)
labels, counts = K.star_partition(A)   # 0 untouched, 1 frontier, 2 star, 3 collapsed
L = K.local_quotient(A)
L.tree                                  # QFTree on the closed star, source ids into K
L.chain_complex(quotient=True)          # or quotient=False for relative chains
L.storage_units('retained')             # K + local records (what L holds)
L.storage_units('compact')              # untouched+frontier subcomplex + local records
L.storage_units('ideal')                # record-sharing reference budget
L.verify()                              # agreement with K.quotient(A)
C = L.compact()                         # CompactLocalQuotient without K or A
C.betti_numbers(quotient=True)
```

Only star simplices (simplices outside `A` with a vertex in `V(A)`) get new
attaching data. Frontier simplices are their faces disjoint from `V(A)`; they
are ordinary simplices, stored so that every facet of a local record is itself a
local cell. Untouched simplices stay in the simplex tree and are not copied.

## Load a saved quotient

```python
from qfcore import QFTree
from qfnext import EditableQF
Q = EditableQF.from_qft(QFTree.load('circle.qft'))
Q.save('circle.qfe')
Q = EditableQF.load('circle.qfe')
```

`.qft` and `.qfe` are different formats. A `.qfe` file stores stable IDs, ordered
local facets and signed polygon words. Both formats store the quotient itself;
neither needs the original pair `(K, A)`, and neither can reconstruct it.

## Change a space and keep its map

```python
from qfnext import CellMap
before = Q.clone()
edge = Q.cell_ids(1)[0]
receipt = Q.collapse([edge], close=True)
f = CellMap(before, Q, receipt.image)
h = f.homology()
```

`collapse` works in place. The returned receipt lists the removed, changed and
created IDs, the image of each cell (unchanged cells map to themselves and are
omitted), and counters of the records that were touched. The cost of taking a
full snapshot is counted separately. `CellMap` checks the ordered attaching
data, which is a stronger condition than `dF = Fd`. Changing either endpoint
invalidates an existing map, so keep clones of the spaces a map refers to.

`delete(id)` requires a maximal cell. `delete(id, cascade=True)` also removes all
cofaces, in a valid order. The reverse incidences record every attaching
occurrence, including those whose coefficients cancel modulo two.

## Point gluing and attaching a disc

```python
from qfnext import glue_points
R, inclusions = glue_points([Q1, Q2], [(0, vertex1, 1, vertex2)])
R.attach_disk([(edge1, 1), (edge2, -1)], basepoint=vertex1_in_R)
```

The word passed to `attach_disk` must be a closed path in `R`. IDs change during
gluing; get the new IDs from the returned inclusion maps. A disc attached by a
constant map has an empty word and an explicit basepoint. Gluing along general
subcomplexes and attaching higher-dimensional cells along arbitrary maps are not
supported.

## Orbit spaces of simplicial group actions

```python
from qfnext import orbit_space
R = orbit_space(K, generators, invariant=A, subdivide='auto', search_limit=100_000)
R.space             # EditableQF: one cell per orbit, one point per component of A/G
R.ordering          # 'labels', 'orbits', 'search' or 'subdivision'
R.subdivided        # True when the barycentric subdivision was glued
R.basepoints        # IDs of the component points
R.representatives   # cell ID -> one simplex of its orbit, vertices in the order of its faces
                    # (a chain of simplices of K if subdivided)
R.cell_of(simplex)  # the cell or component point containing a simplex (or chain)
R.source_simplices  # number of simplices of the complex that was glued
```

`K` is a `qfcore.FlagComplex`, a GUDHI-style simplex tree or an iterable of
simplices (faces are added). `generators` are vertex permutations, as mappings
(vertices not listed are fixed) or as sequences indexed by vertex; a single
permutation may be passed on its own. Each generator must be a simplicial
automorphism and `A` (simplices, or a `qfcore.Subcomplex` of `K`) must be
invariant, otherwise `ValueError` is raised. Orbits are computed from the
generators; the group is not enumerated.

A *local order* of `K` is a total order of the vertices of every simplex whose
restriction to each face is the order of that face. When the group preserves one,
that is, every element maps each simplex to its image increasingly, the simplices
of `K` are glued directly, one cell per orbit of simplices, with faces taken in
the local order. With `subdivide='auto'` three local orders are tried: the vertex
labels (`R.ordering == 'labels'`); the order of the vertex orbits, numbered by
their smallest vertex, which is preserved when no simplex has two vertices in one
orbit (`'orbits'`); and an invariant orientation of the edges inside the vertex
orbits without cyclic triangles, completed by orienting every other edge from the
smaller orbit to the larger (`'search'`). The last is a not-all-equal
satisfiability problem with one variable per orbit of such edges and one clause per
orbit of triangles inside an orbit; it is solved by backtracking with
propagation, which gives up after `search_limit` conflicts (`search_limit=0`
disables it). No local order is preserved when an element reverses an edge or
rotates a triangle. In that case, or when the search fails, the barycentric
subdivision is glued, with its vertices (the simplices of `K`) ordered by
dimension (`'subdivision'`). `subdivide=False` raises `ValueError` instead, and
`subdivide=True` always subdivides.

The pointed realization of `R.space` is the orbit space |K|/G with each
component of the image of |A| collapsed to its own point (Section 3.3 of the
preprint). A subdivided result is strictly graded and its cells form a regular
CW decomposition, also after the collapse. A result glued in a local order has
one cell per orbit of simplices of `K` outside `A` and can be much smaller, but
it need not be regular: for the lens space L(5, 2) from a join of two 5-cycles it
has 24 cells with loops, against 528 after subdivision. `max_simplices` bounds
the size of the complex that is glued.

## Cohomology and fundamental group

`Q.cohomology()` returns an immutable snapshot over F₂. `H.betti`, `H.cocycles`,
`H.cup(p, i, q, j)` and `H.table()` give the dimensions, representative cocycles
and products. A product is returned as the list of basis indices in degree
`p + q` with nonzero coefficient. `fill_limit`, `max_dimension` and
`max_products` bound the cost of large computations.
`f.cohomology(verify_products=True)` also checks that the induced map preserves
cup products.

`Q.fundamental_group()` returns one finite presentation per connected component,
read from the ordered boundaries after a maximal forest is chosen. It does not
decide whether a presentation defines the trivial group or whether two
presentations define isomorphic groups.

## Zigzag

```python
from qfnext import EditableQF, ZigzagSession
circle = EditableQF()
v = circle.add_vertex()
e = circle.add_edge(v, v)
z = ZigzagSession(circle)
disc = z.attach_disk([(e, 1)], basepoint=v)
z.delete(disc)
z.attach_disk([(e, 1)], basepoint=v)
print(z.barcode())
```

The session clones its input by default and advances one step per inserted or
deleted cell, using GUDHI's streaming zigzag persistence. Changes made to the
space outside the session are detected and rejected. A quotient map cannot be
expressed as insertions and deletions. For zigzags of arbitrary maps use
`map_zigzag`, which takes a small diagram of homology maps and processes it as
a whole.

## Visualization

`Q.visualize_dag(path, render=False)` writes the ordered attaching occurrences
as a Graphviz DOT file; `render=True` also runs Graphviz to produce an image.
Repeated arrows and distinct cells with equal words are all drawn. `max_cells`
limits the size of the drawing. Drawing does not compute homology.
