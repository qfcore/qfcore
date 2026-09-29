# qfcore

qfcore implements QF-trees, a data structure for quotient flag complexes, in
C++17 with a Python interface. It also provides editable cell complexes with
polygonal 2-cells, and algorithms that work on both: homology and cohomology
over F₂, cup products, fundamental-group presentations, cellular maps, zigzag
persistence and orbit spaces of simplicial group actions.

Every cell stores its ordered attaching data. Two cells with the same vertices
or the same boundary column remain different cells, and cup products,
fundamental groups and cellular maps are computed from the attaching data. For
example, the torus and the wedge S¹ ∨ S¹ ∨ S², each built from one vertex, two
loops and one disc, have the same cellular chain complex; qfcore tells them apart
by the cup product and by π₁ (see `examples/cup_pi1_and_zigzag.py`).

The mathematics is described in the preprint
[*A data structure for quotient flag complexes*](https://arxiv.org/abs/2609.32733)
by Konstantin Sorokin, Aleksandr Levin, Maxim Beketov and Anton Ayzenberg.

## Installation

```bash
python -m pip install qfcore
```

Python 3.10 or newer is required; the only runtime dependency is NumPy. Wheels
are built for CPython 3.10–3.14 on Linux (x86_64, aarch64) and macOS (x86_64,
arm64). Native Windows is not supported; use WSL2.

Building from source requires a C++17 compiler, the Python development headers
and the Boost headers (Boost 1.71 or newer). On Debian/Ubuntu:

```bash
sudo apt-get install build-essential python3-dev libboost-dev
python -m pip install .
```

On macOS, install the Xcode command-line tools and `brew install boost`. If
Boost is installed elsewhere, set `BOOST_INCLUDE` to the directory that contains
`boost/version.hpp`. The GUDHI headers needed for the build are included in
`vendor/`. See [docs/INSTALLATION.md](https://github.com/qfcore/qfcore/blob/main/docs/INSTALLATION.md)
for optional dependencies and platform details.

One distribution installs three import packages: `qfcore` (flag complexes and
QF-trees), `qfnext` (editable complexes, maps, cohomology, π₁, zigzag, orbit
spaces) and `qfops` (induced maps on the homology of QF-trees).

## Examples

### Quotient of a flag complex

```python
from qfcore import FlagComplex, QFTree

# A 4-cycle with the edge {0, 1} collapsed to a point is still a circle.
K = FlagComplex.from_graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)], max_dim=1)
A = K.induced_subcomplex([0, 1])
Q = K.quotient(A)
assert Q.betti_numbers() == [1, 1]

Q.save("circle.qft")
assert QFTree.load("circle.qft").betti_numbers() == [1, 1]
```

A `QFTree` is immutable and keeps no reference to `K` or `A`. Saving, loading,
traversal and further quotients do not compute homology. With
`K.quotient(A, keep_source=True)` the original pair is stored as well;
otherwise it cannot be recovered from the quotient.

### Editable complexes

```python
from qfnext import EditableQF

T = EditableQF()
v = T.add_vertex()
a, b = T.add_edge(v, v), T.add_edge(v, v)
T.attach_disk([(a, 1), (b, 1), (a, -1), (b, -1)], basepoint=v)  # the torus

H = T.cohomology()                 # over F₂
assert H.betti == (1, 2, 1)
assert H.cup(1, 0, 1, 1)           # the product of the two classes in H¹ is nonzero
print(T.fundamental_group())       # one finite presentation per component
T.save("torus.qfe")
```

Cells of an `EditableQF` have stable IDs. `collapse` replaces a subcomplex by
one point per component, `delete` removes a maximal cell (or, with
`cascade=True`, a cell together with its cofaces), and surviving cells keep
their IDs. An edit rewrites only the cells whose attaching data refer to the
changed cells. Homology and cohomology are computed as separate snapshots and
are recomputed after edits.

### Local quotients

A quotient by `A` changes only the simplices that have a vertex in `A`.
`K.local_quotient(A)` builds QF records for this closed star and leaves the rest
of `K` in its simplex tree:

```python
import numpy as np
from qfcore import FlagComplex

rng = np.random.default_rng(0)
points = rng.random((2000, 2))
K = FlagComplex.from_points(points, radius=0.035, max_dim=3, torus=True)
A = K.induced_subcomplex(np.flatnonzero(((points - 0.5) ** 2).sum(1) < 0.2 ** 2))

L = K.local_quotient(A)
L.counts                     # simplices that are untouched, frontier, star, collapsed
assert L.verify()            # the same chain complex as K.quotient(A)
C = L.compact()              # drops K and A, keeps the untouched part and the local records
print(C.betti_numbers())
```

`L.storage_units(...)` reports the storage of the retained, compact and ideal
(record-sharing) layouts.

### Orbit spaces

`orbit_space(K, generators, invariant=A)` builds the orbit space of a simplicial
group action. The group is given by generating vertex permutations and is not
enumerated. Each component of the image of an invariant subcomplex `A` is
collapsed to its own point.

```python
from qfnext import orbit_space

octahedron = [(a, b, c) for a in (0, 1) for b in (2, 3) for c in (4, 5)]
antipodal = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
P = orbit_space(octahedron, [antipodal])        # RP², 3 + 6 + 4 cells
assert P.space.homology().betti == (1, 1, 1)

n = 5                                           # the lens space L(5, 2)
join = [(i, (i + 1) % n, n + j, n + (j + 1) % n) for i in range(n) for j in range(n)]
g = {i: (i + 1) % n for i in range(n)} | {n + j: n + (j + 2) % n for j in range(n)}
L = orbit_space(join, [g])
assert len(L.space) == 24 and L.space.homology().betti == (1, 0, 0, 1)
(group,) = L.space.fundamental_group()
print(group.simplify()["relators"])             # relators a^5 in one generator a
```

If the group preserves a local vertex order of `K` (an order of the vertices of
each simplex, compatible with faces, that every group element maps
increasingly), the simplices of `K` are glued directly, one cell per orbit.
qfcore tries the vertex labels, the order of the vertex orbits and a bounded
search for such an order. Otherwise it glues the barycentric subdivision, which
gives a regular CW complex. `P.ordering` tells which case was used. For
L(5, 2) above the direct result has 24 cells; the subdivided one has 528.

## Features

| Area | What is available |
| --- | --- |
| Quotients | Componentwise quotients of flag complexes (`FlagComplex.quotient`), local quotients on the closed star (`local_quotient`) |
| Editing | Insert cells, delete a maximal cell or a cell with its cofaces, collapse a subcomplex; surviving cells keep their IDs |
| Gluing | Disjoint union with point identifications (`glue_points`), attaching a disc along a closed edge path (`attach_disk`) |
| Maps | Cellular maps checked against the ordered attaching data (`CellMap`); induced maps on homology and cohomology |
| Cup products | Alexander–Whitney for simplex cells and a word formula for polygonal 2-cells, over F₂ |
| Fundamental group | A finite presentation per connected component, with Tietze simplification |
| Zigzag persistence | `ZigzagSession` for cell insertions and deletions (GUDHI streaming zigzag); `map_zigzag` for small diagrams of arbitrary maps |
| Orbit spaces | `orbit_space` for a group given by vertex permutations |
| Files and drawing | Binary `.qft` for QF-trees, `.qfe` for editable complexes; `visualize_dag` writes Graphviz DOT |

## Limitations

- Homology, cohomology and cup products are computed over F₂. Integer
  incidence coefficients can be inspected, but there is no integral homology
  solver.
- Polygonal cells are 2-dimensional. Higher cells are simplex-type; attaching
  maps of higher-dimensional spheres are not supported.
- Gluing is supported along points and along the orbit relation of a group
  action, not along general subcomplexes.
- `fundamental_group` returns presentations; it does not decide whether two
  groups are isomorphic or whether a group is trivial.
- `map_zigzag` is a small reference implementation that processes the whole
  diagram at once.
- Cloning, saving, full validation and chain assembly take time proportional to
  the size of the complex, even after a local edit.

## Documentation

- [API guide](https://github.com/qfcore/qfcore/blob/main/docs/API.md)
- [Mathematical details and tests](https://github.com/qfcore/qfcore/blob/main/docs/MATHEMATICS.md)
- [QFT file format](https://github.com/qfcore/qfcore/blob/main/docs/QFT_FORMAT.md)
- [Installation](https://github.com/qfcore/qfcore/blob/main/docs/INSTALLATION.md)
- [Repository layout](https://github.com/qfcore/qfcore/blob/main/docs/ARCHITECTURE.md)

Runnable examples are in `examples/`.

## Development

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

See [CONTRIBUTING.md](https://github.com/qfcore/qfcore/blob/main/CONTRIBUTING.md).

## Citation

If you use qfcore in your work, please cite the preprint and state the qfcore
version:

> K. Sorokin, A. Levin, M. Beketov, A. Ayzenberg. *A data structure for quotient
> flag complexes.* Preprint,
> [arXiv:2609.32733](https://arxiv.org/abs/2609.32733), 2026.

```bibtex
@misc{sorokin2026datastructurequotientflag,
  title         = {A data structure for quotient flag complexes},
  author        = {Konstantin Sorokin and Aleksandr Levin and Maxim Beketov and Anton Ayzenberg},
  year          = {2026},
  eprint        = {2609.32733},
  archivePrefix = {arXiv},
  primaryClass  = {math.AT},
  url           = {https://arxiv.org/abs/2609.32733}
}
```

## License

qfcore is released under the MIT License. The repository contains GUDHI headers
(MIT), and the compiled extensions include code from pybind11 (BSD-3-Clause) and
Boost (BSL-1.0). See
[THIRD_PARTY_NOTICES.md](https://github.com/qfcore/qfcore/blob/main/THIRD_PARTY_NOTICES.md).
