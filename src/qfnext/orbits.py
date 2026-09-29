"""Orbit spaces of simplicial group actions as QF objects.

A group G acting simplicially on a finite complex K, together with a G-invariant
subcomplex A, gives a QF object whose pointed realization is |K|/G with each
component of the image of |A| collapsed to its own point (Section 3.3 of the
preprint "A data structure for quotient flag complexes").

* If G preserves a *local order* of K, a total order of the vertices of every
  simplex, restricted to its faces, that every element maps increasingly, the
  simplices of K are glued directly: one cell for each orbit of simplices of K.
  Three local orders are tried, in this order: the vertex labels; the order of the
  vertex orbits, which works when no simplex has two vertices in one orbit; and an
  invariant orientation of the edges inside the vertex orbits, found by a bounded
  search.  Edges between different orbits can always be oriented from the smaller
  orbit to the larger, so only the edges inside an orbit need a choice; the
  conditions are a not-all-equal satisfiability problem with one variable for each
  orbit of such edges and one clause for each orbit of triangles inside an orbit.
  The result need not be regular: the lens space L(5, 2) obtained from a join of two
  5-cycles has 24 cells, and its tetrahedra have two pairs of equal vertices.
* Otherwise X = sd K, the barycentric subdivision, whose vertices are the simplices
  of K ordered by dimension.  Every simplicial action preserves the vertex order of
  every simplex of sd K, the result is strictly graded, and its cells form a regular
  CW decomposition, also after the collapse.

The group is given by generators, as vertex permutations, and is never enumerated:
orbits are found by union-find over the generators.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .api import EditableQF

__all__ = ['OrbitSpace', 'orbit_space']

DEFAULT_MAX_SIMPLICES = 5_000_000
DEFAULT_SEARCH_LIMIT = 100_000
ORDERINGS = ('labels', 'orbits', 'search', 'subdivision')


def _vertex(x):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise TypeError('vertices must be integers')
    return int(x)


def _closure(simplices, name):
    out = set()
    for s in simplices:
        s = tuple(sorted(_vertex(v) for v in s))
        if not s or len(set(s)) != len(s):
            raise ValueError(f'{name}: nonempty simplices with distinct vertices required')
        if s in out:
            continue
        for k in range(1, len(s) + 1):
            out.update(combinations(s, k))
    return out


def _simplices_of(complex):
    """Simplices of a qfcore.FlagComplex, a GUDHI-style simplex tree, or an iterable."""
    if hasattr(complex, 'get_simplices'):
        return _closure((s for s, _ in complex.get_simplices()), 'complex')
    if isinstance(complex, Iterable):
        return _closure(complex, 'complex')
    raise TypeError('expected a FlagComplex, a simplex tree or an iterable of simplices')


def _invariant_of(invariant, complex, K):
    if invariant is None:
        return set()
    if hasattr(invariant, 'mask'):
        if not hasattr(complex, 'simplex'):
            raise TypeError('a Subcomplex invariant requires its FlagComplex as the complex')
        mask = np.asarray(invariant.mask)
        if len(mask) != len(complex):
            raise ValueError('subcomplex mask does not match the complex')
        return _closure((complex.simplex(int(i)) for i in np.flatnonzero(mask)), 'invariant')
    A = _closure(invariant, 'invariant')
    if not A <= K:
        raise ValueError('the invariant subcomplex is not contained in the complex')
    return A


def _generators_of(generators, vertices):
    if isinstance(generators, Mapping):
        generators = [generators]
    else:
        generators = list(generators)
        if generators and all(isinstance(g, (int, np.integer)) and not isinstance(g, (bool, np.bool_))
                              for g in generators):
            generators = [generators]          # a single permutation given as a sequence
    maps = []
    for number, g in enumerate(generators):
        if isinstance(g, Mapping):
            unknown = {_vertex(v) for v in g} - set(vertices)
            if unknown:
                raise ValueError(f'generator {number} moves {sorted(unknown)[:5]}, which are not vertices')
            image = {v: _vertex(g.get(v, v)) for v in vertices}
        else:
            seq = list(g)
            try:
                image = {v: _vertex(seq[v]) for v in vertices}
            except IndexError:
                raise ValueError(f'generator {number} does not define every vertex') from None
            if any(v < 0 for v in vertices):
                raise ValueError('sequence generators require nonnegative vertex labels')
        if set(image.values()) != set(vertices):
            raise ValueError(f'generator {number} is not a permutation of the vertices')
        maps.append(image)
    return maps


class _DisjointSets:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        parent = self.parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


class _SearchLimit(Exception):
    pass


def _nae_clause(literals):
    """Normalize the clause 'not all equal' on literals (variable, positive).

    Returns None if the clause always holds, () if it never holds, and otherwise the
    literals on distinct variables, sorted, with the first one positive (the clause
    is unchanged by complementing all of its literals).
    """
    by_var = {}
    for var, positive in literals:
        if by_var.get(var, positive) != positive:
            return None
        by_var[var] = positive
    if len(by_var) == 1:
        return ()
    clause = sorted(by_var.items())
    if not clause[0][1]:
        clause = [(var, not positive) for var, positive in clause]
    return tuple(clause)


def _solve_nae(nvars, clauses, limit):
    """Satisfy 'not all equal' clauses by backtracking with propagation.

    clauses: tuples of two or three literals (variable, positive) on distinct
    variables.  Returns a list of booleans, or None if the clauses cannot all hold;
    raises _SearchLimit when more than `limit` conflicts occur.  Components of the
    clause graph are solved separately, and the first choice in a component is not
    revisited, since complementing a component preserves its clauses.
    """
    occurs = [[] for _ in range(nvars)]
    for k, clause in enumerate(clauses):
        for var, _ in clause:
            occurs[var].append(k)
    value = [None] * nvars
    trail = []

    def propagate(start):
        queue = [start]
        while queue:
            x = queue.pop()
            for k in occurs[x]:
                free, free_count, seen = None, 0, set()
                for var, positive in clauses[k]:
                    if value[var] is None:
                        free, free_count = (var, positive), free_count + 1
                    else:
                        seen.add(value[var] == positive)
                if len(seen) == 2:
                    continue
                if free_count == 0:
                    return False
                if free_count == 1:
                    (common,) = seen
                    var, positive = free
                    value[var] = (not common) == positive
                    trail.append(var)
                    queue.append(var)
        return True

    conflicts = 0
    reached = [False] * nvars
    for root in range(nvars):
        if reached[root]:
            continue
        component, stack = [], [root]
        reached[root] = True
        while stack:
            x = stack.pop()
            component.append(x)
            for k in occurs[x]:
                for var, _ in clauses[k]:
                    if not reached[var]:
                        reached[var] = True
                        stack.append(var)
        component.sort(key=lambda var: (-len(occurs[var]), var))
        decisions = []          # [position in component, trail length, other value left]
        position = 0
        while True:
            while position < len(component) and value[component[position]] is not None:
                position += 1
            if position == len(component):
                break
            var = component[position]
            decisions.append([position, len(trail), bool(decisions)])
            value[var] = True
            trail.append(var)
            ok = propagate(var)
            while not ok:
                conflicts += 1
                if conflicts > limit:
                    raise _SearchLimit
                while decisions and not decisions[-1][2]:
                    decisions.pop()
                if not decisions:
                    return None
                position, mark, _ = decisions[-1]
                decisions[-1][2] = False
                while len(trail) > mark:
                    value[trail.pop()] = None
                var = component[position]
                value[var] = False
                trail.append(var)
                ok = propagate(var)
    return value


def _vertex_orbits(vertices, gens):
    """Orbit index of every vertex; orbits are numbered by their smallest vertex."""
    position = {v: i for i, v in enumerate(vertices)}
    sets = _DisjointSets(len(vertices))
    for g in gens:
        for v in vertices:
            sets.union(position[v], position[g[v]])
    number, orbit = {}, {}
    for v in vertices:               # vertices are sorted, roots are the smallest members
        orbit[v] = number.setdefault(sets.find(position[v]), len(number))
    return orbit


def _local_order(K, gens, orbit, search_limit):
    """A local order of K preserved by the generators.

    Returns (kind, order), where kind is 'orbits' or 'search' and order maps a sorted
    simplex to its vertices in the local order, or (None, reason) if none was found.
    """
    intra = sorted(s for s in K if len(s) == 2 and orbit[s[0]] == orbit[s[1]])
    forward = {}
    kind = 'orbits'
    if intra:
        kind = 'search'
        if search_limit == 0:
            return None, 'the search is disabled (search_limit=0)'
        arcs = {}
        for u, v in intra:
            arcs[(u, v)] = len(arcs)
            arcs[(v, u)] = len(arcs)
        sets = _DisjointSets(len(arcs))
        for g in gens:
            for (u, v), i in arcs.items():
                sets.union(i, arcs[(g[u], g[v])])
        variable, literal = {}, {}
        for u, v in intra:
            a, b = sets.find(arcs[(u, v)]), sets.find(arcs[(v, u)])
            if a == b:
                return None, f'an element of the group reverses the edge {(u, v)}'
            key = (min(a, b), max(a, b))
            var = variable.setdefault(key, len(variable))
            literal[(u, v)] = (var, a == key[0])
            literal[(v, u)] = (var, b == key[0])
        clauses = set()
        for s in K:
            if len(s) == 3 and orbit[s[0]] == orbit[s[1]] == orbit[s[2]]:
                a, b, c = s
                clause = _nae_clause([literal[(a, b)], literal[(b, c)], literal[(c, a)]])
                if clause == ():
                    return None, f'the triangle {s} cannot be ordered invariantly'
                if clause is not None:
                    clauses.add(clause)
        try:
            value = _solve_nae(len(variable), sorted(clauses), search_limit)
        except _SearchLimit:
            return None, f'the search exceeded search_limit={search_limit}'
        if value is None:
            return None, 'the edges inside the vertex orbits have no invariant orientation without a cyclic triangle'
        forward = {arc: value[var] == positive for arc, (var, positive) in literal.items()}

    def order(s):
        out = []
        groups = {}
        for v in s:
            groups.setdefault(orbit[v], []).append(v)
        for number in sorted(groups):
            members = groups[number]
            if len(members) > 1:
                rank = {v: sum(forward[(w, v)] for w in members if w != v) for v in members}
                if sorted(rank.values()) != list(range(len(members))):
                    raise RuntimeError('internal error: the orientation is not a local order')
                members = sorted(members, key=rank.__getitem__)
            out.extend(members)
        return tuple(out)

    ordered = {s: order(s) for s in K}
    for g in gens:                    # every generator maps the local order increasingly
        for s, o in ordered.items():
            image = tuple(g[v] for v in o)
            if ordered[tuple(sorted(image))] != image:
                raise RuntimeError('internal error: the local order is not invariant')
    return kind, ordered


@dataclass
class OrbitSpace:
    """Result of :func:`orbit_space`.

    ``space`` is the QF object as an :class:`EditableQF`; the basepoints are its
    vertices listed in ``basepoints``.  ``ordering`` says which simplices were glued
    and in which vertex order: ``'labels'``, ``'orbits'`` or ``'search'`` (the
    simplices of K, in a local order preserved by the group: the vertex labels, the
    order of the vertex orbits, or an invariant orientation found by search), or
    ``'subdivision'`` (the chains of sd K).  ``subdivided`` is True exactly in the
    last case.  ``representatives[c]`` is one simplex of X in the orbit that cell
    ``c`` represents, with its vertices in the order that defines the faces of ``c``:
    a vertex tuple of K when ``subdivided`` is False, and a chain of simplices of K
    (each a sorted vertex tuple) when it is True.  ``source_simplices`` is the number
    of simplices of X.
    """
    space: EditableQF
    subdivided: bool
    ordering: str
    basepoints: tuple
    representatives: dict
    source_simplices: int
    _cell_of_index: dict = field(repr=False)
    _index: dict = field(repr=False)
    _k_index: dict = field(repr=False)

    def cell_of(self, simplex):
        """Cell or basepoint of the orbit space that contains a simplex of X.

        For ``subdivided`` results, give a chain of simplices of K, or a single
        simplex of K for the vertex of sd K at its barycenter.  Otherwise give a
        simplex of K, with its vertices in any order.
        """
        if self.subdivided:
            chain = simplex
            if chain and isinstance(next(iter(chain)), (int, np.integer)):
                chain = (chain,)
            key = tuple(sorted(self._k_index[tuple(sorted(map(_vertex, s)))] for s in chain))
        else:
            key = tuple(sorted(map(_vertex, simplex)))
        if key not in self._index:
            raise KeyError('not a simplex of the complex that was acted on')
        return self._cell_of_index[self._index[key]]


def orbit_space(complex, generators, *, invariant=None, subdivide='auto',
                search_limit=DEFAULT_SEARCH_LIMIT,
                max_simplices=DEFAULT_MAX_SIMPLICES) -> OrbitSpace:
    """Orbit space of a simplicial group action, as a QF object.

    complex: a ``qfcore.FlagComplex``, a GUDHI-style simplex tree, or an iterable of
    simplices (vertex tuples; faces are added).  generators: vertex permutations
    generating the group, each a mapping (missing vertices are fixed) or a sequence
    indexed by vertex.  invariant: an optional G-invariant subcomplex, as simplices
    or as a ``qfcore.Subcomplex`` of ``complex``; each component of its image is
    collapsed to its own basepoint.

    subdivide: ``'auto'`` glues the simplices of K when the group preserves a local
    vertex order (the labels, the order of the vertex orbits, or an orientation found
    by a backtracking search that gives up after ``search_limit`` conflicts) and the
    barycentric subdivision otherwise; ``True`` always subdivides; ``False`` never
    does and raises ValueError when no preserved local order is found.
    ``search_limit=0`` disables the search.

    Raises ValueError if a generator is not a simplicial automorphism or the
    subcomplex is not invariant.  The group is not enumerated: orbits are computed
    from the generators.
    """
    if subdivide not in ('auto', True, False):
        raise ValueError("subdivide must be 'auto', True or False")
    if isinstance(search_limit, (bool, np.bool_)) or not isinstance(search_limit, (int, np.integer)) \
            or search_limit < 0:
        raise ValueError('search_limit must be a nonnegative integer')
    K = _simplices_of(complex)
    A = _invariant_of(invariant, complex, K)
    vertices = sorted(s[0] for s in K if len(s) == 1)
    gens = _generators_of(generators, vertices)

    order_preserving = True
    for number, g in enumerate(gens):
        for s in K:
            image = tuple(g[v] for v in s)
            if tuple(sorted(image)) not in K:
                raise ValueError(f'generator {number} is not a simplicial map')
            if order_preserving and any(image[i] > image[i + 1] for i in range(len(image) - 1)):
                order_preserving = False
        for s in A:
            if tuple(sorted(g[v] for v in s)) not in A:
                raise ValueError(f'the subcomplex is not invariant under generator {number}')

    ordering, local = 'subdivision', None
    if subdivide is not True:
        if order_preserving:
            ordering = 'labels'
        else:
            kind, found = _local_order(K, gens, _vertex_orbits(vertices, gens), int(search_limit))
            if kind is None:
                if subdivide is False:
                    raise ValueError(f'no local vertex order preserved by the group was found: {found}; '
                                     'use subdivide=True or subdivide="auto"')
            else:
                ordering, local = kind, found
    subdivided = ordering == 'subdivision'

    ks = sorted(K, key=lambda s: (len(s), s))
    k_index = {s: i for i, s in enumerate(ks)}
    if subdivided:
        # sd K: chains of simplices of K, as increasing tuples of indices into ks
        # (indices increase with dimension, so a chain lists its simplices by dimension).
        ending = []
        total = 0
        for i, s in enumerate(ks):
            chains = [(i,)]
            for k in range(1, len(s)):
                for f in combinations(s, k):
                    chains.extend(c + (i,) for c in ending[k_index[f]])
            total += len(chains)
            if total > max_simplices:
                raise ValueError(f'the barycentric subdivision exceeds max_simplices={max_simplices}')
            ending.append(chains)
        X = sorted((c for chains in ending for c in chains), key=lambda c: (len(c), c))
        del ending
        k_maps = [[k_index[tuple(sorted(g[v] for v in s))] for s in ks] for g in gens]
        act = lambda g, x: tuple(k_maps[g][i] for i in x)
        in_A = [ks[x[-1]] in A for x in X]
        index = {x: i for i, x in enumerate(X)}
        lookup = index
    else:
        if len(ks) > max_simplices:
            raise ValueError(f'the complex exceeds max_simplices={max_simplices}')
        # simplices of K with their vertices in the local order; the faces of an
        # ordered simplex are ordered simplices, and the generators map them increasingly
        X = ks if local is None else sorted((local[s] for s in ks), key=lambda x: (len(x), x))
        act = lambda g, x: tuple(gens[g][v] for v in x)
        in_A = [tuple(sorted(x)) in A for x in X]
        index = {x: i for i, x in enumerate(X)}
        lookup = index if local is None else {tuple(sorted(x)): i for i, x in enumerate(X)}

    parent = list(range(len(X)))

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for i, x in enumerate(X):
        for g in range(len(gens)):
            a, b = find(i), find(index[act(g, x)])
            if a != b:
                parent[max(a, b)] = min(a, b)

    # Components of the image of A_X: join vertex orbits along edge orbits of A_X.
    comp = {}

    def find_comp(r):
        root = r
        while comp[root] != root:
            root = comp[root]
        while comp[r] != root:
            comp[r], r = root, comp[r]
        return root

    for i, x in enumerate(X):
        if in_A[i] and len(x) == 1:
            comp.setdefault(find(i), find(i))
    for i, x in enumerate(X):
        if in_A[i] and len(x) == 2:
            a = find_comp(find(index[(x[0],)]))
            b = find_comp(find(index[(x[1],)]))
            if a != b:
                comp[max(a, b)] = min(a, b)

    space = EditableQF()
    basepoint = {}
    for r in sorted(comp):
        root = find_comp(r)
        if root not in basepoint:
            basepoint[root] = space.add_vertex()
    cell = {}
    representatives = {}
    for i, x in enumerate(X):            # X is sorted by dimension, so facets exist first
        if in_A[i] or find(i) != i:
            continue
        if len(x) == 1:
            c = space.add_vertex()
        else:
            facets = []
            for j in range(len(x)):
                f = index[x[:j] + x[j + 1:]]
                if in_A[f]:
                    facets.append(basepoint[find_comp(find(index[(X[f][0],)]))])
                else:
                    facets.append(cell[find(f)])
            c = space.add_simplex(len(x) - 1, facets)
        cell[i] = c
        representatives[c] = tuple(ks[j] for j in x) if subdivided else x
    space.validate()
    cell_of_index = {}
    for i in range(len(X)):
        r = find(i)
        cell_of_index[i] = basepoint[find_comp(find(index[(X[i][0],)]))] if in_A[i] else cell[r]
    return OrbitSpace(space=space, subdivided=subdivided, ordering=ordering,
                      basepoints=tuple(sorted(basepoint.values())),
                      representatives=representatives, source_simplices=len(X),
                      _cell_of_index=cell_of_index, _index=lookup, _k_index=k_index)
