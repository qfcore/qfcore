# Mathematical details and validation

This document describes what the implementation computes and how it is tested.
Section, proposition and corollary numbers refer to the preprint *A data
structure for quotient flag complexes* (Sorokin, Levin, Beketov, Ayzenberg).

## 1. What is stored

A simplex-type d-cell keeps an ordered (d+1)-tuple of face occurrences. A face
may be a codimension-one simplex cell or a constant vertex. Local identities
`d_i d_j = d_(j-1) d_i` for i<j are checked, interpreting further faces of a
constant vertex as that vertex. Two distinct cells are never identified because
they have equal endpoints, words or boundary columns.

A polygon-type 2-cell keeps a basepoint and an ordered finite closed signed edge
word. An edge token is `sign*(stable_edge_id+1)`, so ID zero is not confused with
an empty token. Parallel edge occurrences remain distinct occurrences. A
constant attaching map has an empty word and an explicit basepoint.

Polygon cells extend the source-labelled QFT representation; they are not
simplices of the original flag complex. The `.qft` format does not store them;
the `.qfe` format stores the full records, including the edge words.
Higher-dimensional simplex cells cannot have polygons as codimension-one faces,
and cells of dimension above two can only be simplex-type cells.

The signed boundary is the alternating sum of surviving codimension-one simplex
faces, or the exponent sum of edges in a polygon word. Integer signs are kept
for inspection and for pi1 words; homology, cohomology and zigzag are computed
over F2.

## 2. Local update correctness and cost

The mutable layer stores reverse *topological* dependencies, before coefficient
cancellation. For a subcomplex B, connected components are found in its incidence
graph. Each component is collapsed to a retained minimum-ID vertex. Every removed
cell receives that vertex as its image. Surviving cells retain their stable IDs.

Only surviving cells whose direct attaching data reference a removed cell need
rewriting. Higher cofaces keep the same stable facet targets; the changed lower
attaching map propagates through that reference. Polygon words drop steps whose
edges were collapsed and change basepoints through the quotient map.

The work of a collapse depends on the selected closure, the reverse incidences
and the attaching occurrences that are examined or rewritten, so one selected
vertex with a large star can already be expensive. Full copies, sorting all IDs,
validation of the whole object, chain assembly, ring construction and
serialization are global operations. The native counters report the records and
occurrences that were touched; they do not measure CPU time.

Deletion is legal only for topologically maximal cells; `cascade=True` deletes
the upward closure in decreasing dimension. A polygon with zero F2 boundary still
prevents deletion of an edge occurring in its attaching word.

## 3. Cup products

For ordered simplex cells use Alexander–Whitney:

    (a cup b)(sigma) = a(sigma[0,...,p]) * b(sigma[p,...,p+q]).

Front and back faces are obtained through the retained ordered local face maps,
not from a vertex set. A positive-dimensional evaluation on a collapsed vertex
is zero. Degree-zero cocycles provide componentwise units.

For polygons and degree-one cocycles, first choose cohomologous representatives
vanishing on a deterministic maximal forest. For a based word
`w = e_1^eps_1 ... e_m^eps_m`, put A_i=a(e_i), B_i=b(e_i), values in F2. The value
of the based word diagonal is

    sum_{i<j} A_i B_j + sum_{eps_i=-1} A_i B_i.

The inverse-letter correction is essential: for RP2 with relator a*a, the square
of the H1 generator is nonzero, whereas a*a^-1 gives zero. Cochains are reduced to
cohomology coordinates only after checking they are cocycles. H1 forest
normalization is a coboundary change, not deletion of topological information.

The formula is tested against an independent construction: every polygon is
replaced by a fan of ordered triangles, covering negatively oriented edges and
degenerate constant maps. The original 2-cell maps to the sum of the fan triangles. Check dF=Fd,
Betti agreement, and equality of every cup product under the induced pullback.
The oracle uses simplex Alexander–Whitney, not the polygon prefix formula.
Eighty random word-presentation spaces and analytic RP2/torus/wedge controls are
included. A triangulated 3-torus additionally checks a nonzero triple product,
rank-three H1×H1→H2, and naturality after maximal-tree collapse.

The Alexander–Whitney product and Magnus-type formulas for cup products are
standard; see the references below. The Suciu–Wang paper is cited as
background. The F₂ implementation is validated by the fan construction above.

## 4. Maps and pi1

The public `CellMap` accepts the represented parameter-preserving cellular maps:
ordered facets must commute; polygon words must map to the target word after
collapsing edge steps; a collapsed positive cell and all its dependencies must
map to the same vertex. This is stronger than equality of boundary matrices.

After a maximal forest is chosen, nonforest edges are group generators. An ordered
triangle contributes the boundary path `(d2, d0, -d1)` with constant facets omitted.
A polygon contributes its signed word. Removing forest edges gives a
finite presentation in each connected component. Tietze simplification eliminates
a generator appearing exactly once in a relation; all replacements are logged.
It is not a decision algorithm for arbitrary group isomorphism.

The two four-cell models with one vertex, two loops and one 2-cell attached by
[a,b] or constantly have identical integral cellular chain matrices. Their cup
products and relators differ. The implementation rejects the alleged identity
cell map between them even though it would pass dF=Fd; this case is covered by
`tests/test_cup.py`.

## 5. Zigzag

The online insertion/deletion consumer is the bundled GUDHI
`Filtered_zigzag_persistence_with_storage`. Cells are supplied in valid order,
with F2 boundary IDs; deletion is additionally guarded by full topological
incidence. The initial snapshot is placed at t=0 and elementary edits at 1,2,... .
An infinite bar is a class that is still unpaired at the last observed step.

A quotient map is not treated as a deletion. For a small zigzag of arbitrary
homology maps, the separate reference algorithm computes, for every connected
subinterval I, the generalized rank

    r(I) = rank(lim(V|I) -> colim(V|I)).

The inverse limit is the kernel of compatibility equations; the colimit is the
quotient by the arrow relations. The canonical map takes one compatible
coordinate and embeds it into the colimit. Taking the sum of all coordinates
would be wrong, in particular over F2. Interval multiplicities follow by two-sided
finite differences of r. Diagram size and sparse fill are bounded by parameters.

Independent checks include 100 random interval-decomposable diagrams after random
basis changes and orientations, and 30 cell-edit streams compared with
snapshot homology maps. This reference implementation processes the whole
diagram at once and is meant for small diagrams; for cell insertions and
deletions use the GUDHI streaming consumer.

## 6. Orbit spaces

For a group G acting simplicially on K and a G-invariant subcomplex A,
`orbit_space` returns a QF object whose pointed realization is (|K|/G)/_SC(|A|/G),
with one of two sources X (Section 3.3 of the preprint).

*Local orders.* A local order of K is a total order of the vertices of every
simplex, restricted to the faces; it makes K a semisimplicial set K^o whose i-th
face omits the i-th vertex. If every element of G maps each simplex to its image
increasingly, G acts on K^o by semisimplicial automorphisms, and X = K^o glued by
the orbit relation, with A collapsed componentwise, has one cell per orbit of
simplices of K outside A. Realization commutes with the colimits involved, so the
pointed realization is the orbit space (Proposition 3.26). Such a local order
exists if and only if, for every vertex orbit P, the edges with both ends in P
have a G-invariant orientation without a cyclic triangle of K: orienting every
other edge from the smaller orbit to the larger, in any fixed order of the
orbits, then gives a local order, because a triangle with vertices in two or
three orbits has a source or a sink, or is ordered by its orbits. In particular,
if no simplex has two vertices in one orbit, the order of the orbits is
preserved. The conditions inside the orbits are a not-all-equal satisfiability
problem: a variable for each orbit of edges inside an orbit (none exists if an
element reverses such an edge), and for each orbit of triangles inside an orbit
the clause that its three edges, oriented cyclically, are neither all forward nor
all backward (a rotated triangle gives the clause x, x, x, which fails). The
quotient can be non-regular: the lens space L(5, 2) from a join of two 5-cycles
has 24 cells, and its tetrahedra have vertex words (a, a, b, b).

*Subdivision.* Otherwise X is the barycentric subdivision sd K, whose vertices are
the simplices of K ordered by dimension; a group element sends a chain of faces
to a chain with the same dimensions, so it preserves the vertex order of every
simplex of sd K and the orbit relation satisfies d_i(s) ~ d_i(t) whenever s ~ t
(Corollary 3.23). The subdivision sd A is invariant, hence saturated, and sd K
and sd A are order complexes, hence flag, so the object is strictly graded. It is
also regular: two distinct subchains of a chain have different sets of dimensions
and lie in different orbits, an element mapping a subchain to itself fixes it,
and the subchains in sd A form one initial face, crushed to one point, so each
closed cell is a simplex with one face crushed, a ball (Proposition 3.24).

Each orbit of simplices of X outside A_X is one cell. Its facets are the cells
of the orbits of its facets, or the point of the component of the image of A_X
that contains a collapsed facet; well-definedness is the gluing axiom, and the
native insertion re-checks the semisimplicial identities of every cell.

The tests compare random invariant complexes with an independent model: the
orbit simplicial complex of the second barycentric subdivision, on which every
simplicial action is regular, with a cone on each component of the image of A.
Known spaces (RP^2, RP^3, lens spaces, the torus and the Klein bottle, told
apart by cup squares, the seven-vertex torus modulo Z_7, the real moment-angle
complex of a pentagon and its small cover) are checked through homology over F_2,
F_3 and F_5 and through fundamental-group presentations. Every ordering is
compared with the subdivided presentation on random cyclic actions, and the
not-all-equal solver with brute force.

## Primary references

- GUDHI Zigzag Persistence documentation, with storage-wrapper example:
  https://gudhi.inria.fr/doc/3.13.0rc2/group__zigzag__persistence.html
- GUDHI filtered zigzag class documentation:
  https://gudhi.inria.fr/doc/3.13.0rc2/class_gudhi_1_1zigzag__persistence_1_1_filtered__zigzag__persistence.html
- SageMath finite simplicial sets: ordered face maps, Alexander–Whitney,
  quotients and fundamental-group presentations:
  https://doc.sagemath.org/html/en/reference/topology/sage/topology/simplicial_set.html
- A. I. Suciu, H. Wang, *Cup products, lower central series, and holonomy Lie algebras*,
  Journal of Pure and Applied Algebra 223 (2019), 3359–3385:
  https://arxiv.org/abs/1701.07768
  https://doi.org/10.1016/j.jpaa.2018.11.006
- T. K. Dey, W. Kim, F. Mémoli, *Computing Generalized Rank Invariant for 2-Parameter Persistence
  Modules via Zigzag Persistence and Its Applications*, SoCG 2022:
  https://doi.org/10.4230/LIPIcs.SoCG.2022.34

The zigzag reference routine computes the generalized rank directly from the
definition rank(lim → colim). SageMath is listed as related software; qfcore
has not been benchmarked against it.
