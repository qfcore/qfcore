"""Orbit spaces of simplicial group actions (qfnext.orbit_space).

Known spaces are checked through homology over several primes, fundamental-group
presentations and cup squares.  Random invariant complexes are compared with an
independent model: the orbit simplicial complex of the second barycentric
subdivision, on which every simplicial action is regular, with a cone on each
component of the collapsed part.  The four orderings (vertex labels, vertex
orbits, searched invariant orientation, barycentric subdivision) are compared with
each other, and the not-all-equal solver behind the search with brute force.
"""
from itertools import combinations
import random

import pytest

from qfnext import orbit_space


# ------------------------------------------------------------------ helpers

def rank_mod_p(rows, p):
    """Rank over F_p of a list of sparse rows {column: value}."""
    pivots = {}
    rank = 0
    for row in rows:
        row = {c: v % p for c, v in row.items() if v % p}
        while row:
            col = min(row)
            if col not in pivots:
                inv = pow(row[col], p - 2, p)
                pivots[col] = {c: v * inv % p for c, v in row.items()}
                rank += 1
                break
            factor = row[col]
            for c, v in pivots[col].items():
                row[c] = (row.get(c, 0) - factor * v) % p
                if not row[c]:
                    del row[c]
    return rank


def betti_mod_p(space, p):
    """Betti numbers over F_p from the signed boundaries of the QF object."""
    dim = space.dimension()
    count = [len(space.cell_ids(d)) for d in range(dim + 1)]
    ranks = [0] * (dim + 2)
    for d in range(1, dim + 1):
        rows = [dict(space.boundary(c, signed=True)) for c in space.cell_ids(d)]
        ranks[d] = rank_mod_p(rows, p)
    return tuple(count[d] - ranks[d] - ranks[d + 1] for d in range(dim + 1))


def abelianization(presentation):
    """Orders of the cyclic factors of the abelianized group (0 for a free factor)."""
    simple = presentation.simplify()
    gens = list(simple['generators'])
    m = [[sum((1 if x > 0 else -1) for x in word if abs(x) == g) for g in gens]
         for word in simple['relators']]
    diagonal = []
    while m and m[0]:
        entries = [(abs(v), i, j) for i, row in enumerate(m) for j, v in enumerate(row) if v]
        if not entries:
            break
        _, i, j = min(entries)
        m[0], m[i] = m[i], m[0]
        for row in m:
            row[0], row[j] = row[j], row[0]
        pivot, clean = m[0][0], True
        for i in range(1, len(m)):
            q = m[i][0] // pivot
            m[i] = [x - q * y for x, y in zip(m[i], m[0])]
            clean = clean and not m[i][0]
        for j in range(1, len(m[0])):
            q = m[0][j] // pivot
            for row in m:
                row[j] -= q * row[0]
            clean = clean and not m[0][j]
        if clean:
            diagonal.append(abs(pivot))
            m = [row[1:] for row in m[1:]]
    free = len(gens) - len(diagonal)
    return tuple(sorted(d for d in diagonal if d != 1)) + (0,) * free


def closure(simplices):
    out = set()
    for s in simplices:
        s = tuple(sorted(s))
        for k in range(1, len(s) + 1):
            out.update(combinations(s, k))
    return out


def octahedron():
    return [(a, b, c) for a in (0, 1) for b in (2, 3) for c in (4, 5)]


ANTIPODAL = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}


def cross_polytope(k):
    facets = [()]
    for i in range(k):
        facets = [f + (v,) for f in facets for v in (2 * i, 2 * i + 1)]
    return facets


def join_cycles(n):
    a = [(i, (i + 1) % n) for i in range(n)]
    b = [(n + j, n + (j + 1) % n) for j in range(n)]
    return [x + y for x in a for y in b]


def torus(side=6):
    """Torus on a side x side grid, diagonals alternating by column strip."""
    v = lambda i, j: (i % side) * side + (j % side)
    out = []
    for i in range(side):
        for j in range(side):
            a, b, c, d = v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1)
            out += [(a, b, c), (a, c, d)] if i % 2 == 0 else [(a, b, d), (b, c, d)]
    return out


def grid_map(side, f):
    return {i * side + j: (f(i, j)[0] % side) * side + f(i, j)[1] % side
            for i in range(side) for j in range(side)}


# ------------------------------------------------------------- known spaces

def test_rotation_of_triangle_boundary_is_a_loop_or_a_bigon():
    # The rotation preserves the cyclic orientation of the three edges, so the
    # quotient is one vertex with a loop; the subdivision gives a bigon instead.
    r = orbit_space([(0, 1), (1, 2), (0, 2)], [{0: 1, 1: 2, 2: 0}])
    assert r.ordering == 'search' and not r.subdivided and r.source_simplices == 6
    assert len(r.space.cell_ids(0)) == 1 and len(r.space.cell_ids(1)) == 1
    (loop,) = r.space.cell_ids(1)
    assert len(set(r.space.cell(loop).facets)) == 1
    s = orbit_space([(0, 1), (1, 2), (0, 2)], [{0: 1, 1: 2, 2: 0}], subdivide=True)
    assert s.subdivided and s.ordering == 'subdivision' and s.source_simplices == 12
    assert len(s.space.cell_ids(0)) == 2 and len(s.space.cell_ids(1)) == 2
    for x in (r, s):
        assert x.space.homology().betti == (1, 1) and x.basepoints == ()


def test_order_preserving_action_needs_no_subdivision():
    hexagon = [(0, 2), (2, 4), (1, 4), (1, 3), (3, 5), (0, 5)]
    g = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
    r = orbit_space(hexagon, [g])
    assert not r.subdivided and r.ordering == 'labels'
    assert len(r.space) == 6 and r.space.homology().betti == (1, 1)
    s = orbit_space(hexagon, [g], subdivide=True)
    assert s.subdivided and len(s.space) == 12 and s.space.homology().betti == (1, 1)
    quarter = {0: 2, 2: 1, 1: 3, 3: 0}        # sends the triangle 024 to 214
    q = orbit_space(octahedron(), [quarter], subdivide=False)
    assert q.ordering == 'search' and q.space.homology().betti == (1, 0, 1)
    with pytest.raises(ValueError, match='reverses the edge'):
        orbit_space([(0, 1)], [{0: 1, 1: 0}], subdivide=False)


def test_relabelled_octahedron_uses_the_orbit_order():
    # After a relabelling the antipodal map reverses some edges, but no triangle has
    # two antipodal vertices, so ordering by orbits avoids the subdivision (13 cells
    # instead of 73).
    relabel = {0: 0, 1: 5, 2: 1, 3: 4, 4: 2, 5: 3}
    K = [tuple(relabel[v] for v in s) for s in octahedron()]
    g = {relabel[a]: relabel[b] for a, b in ANTIPODAL.items()}
    r = orbit_space(K, [g])
    assert r.ordering == 'orbits' and [len(r.space.cell_ids(d)) for d in range(3)] == [3, 6, 4]
    assert betti_mod_p(r.space, 2) == (1, 1, 1) and betti_mod_p(r.space, 3) == (1, 0, 0)
    assert r.cell_of((0, 1, 2)) == r.cell_of((5, 4, 3))
    s = orbit_space(K, [g], subdivide=True)
    assert len(s.space) == 73 and betti_mod_p(s.space, 2) == (1, 1, 1)


def test_projective_plane_and_its_equator_collapse():
    # The antipodal map preserves the vertex order of every triangle 0|1, 2|3, 4|5,
    # so no subdivision is needed: 3 vertices, 6 edges and 4 triangles.
    small = orbit_space(octahedron(), [ANTIPODAL])
    assert not small.subdivided and small.source_simplices == 26
    assert [len(small.space.cell_ids(d)) for d in range(3)] == [3, 6, 4]
    r = orbit_space(octahedron(), [ANTIPODAL], subdivide=True)
    assert r.source_simplices == 146 and len(r.space) == 73
    for x in (small, r):
        assert betti_mod_p(x.space, 2) == x.space.homology().betti == (1, 1, 1)
        assert betti_mod_p(x.space, 3) == (1, 0, 0)
        (group,) = x.space.fundamental_group()
        assert abelianization(group) == (2,)
    assert r.cell_of((0,)) == r.cell_of((1,)) != r.cell_of((2,))
    assert small.cell_of((0, 2, 4)) == small.cell_of((1, 3, 5))
    equator = [(0, 2), (1, 2), (1, 3), (0, 3)]
    q = orbit_space(octahedron(), [ANTIPODAL], invariant=equator)
    assert len(q.basepoints) == 1
    assert betti_mod_p(q.space, 2) == betti_mod_p(q.space, 3) == (1, 0, 1)


def test_rotations_and_reflections_of_the_octahedron():
    quarter = {0: 2, 2: 1, 1: 3, 3: 0}
    assert orbit_space(octahedron(), [quarter]).space.homology().betti == (1, 0, 1)
    assert orbit_space(octahedron(), [{0: 1, 1: 0}]).space.homology().betti == (1, 0, 0)
    both = orbit_space(octahedron(), [quarter, ANTIPODAL])     # contains a reflection
    assert both.space.homology().betti == (1, 0, 0)


def test_real_projective_three_space():
    antipodal = {v: v ^ 1 for v in range(8)}
    for subdivide in ('auto', True):
        r = orbit_space(cross_polytope(4), [antipodal], subdivide=subdivide)
        assert r.subdivided == (subdivide is True)
        assert betti_mod_p(r.space, 2) == (1, 1, 1, 1)
        assert betti_mod_p(r.space, 3) == (1, 0, 0, 1)


@pytest.mark.parametrize('n, q', [(5, 1), (5, 2), (7, 2), (7, 3)])
def test_lens_spaces_from_joins_of_cycles(n, q):
    # The rotation reverses the vertex order of the edge {0, n-1}, but it preserves
    # the cyclic orientations of both cycles: one cell per orbit of simplices of the
    # join, with loops, against the regular presentation of the subdivision.
    g = {i: (i + 1) % n for i in range(n)}
    g.update({n + j: n + (j + q) % n for j in range(n)})
    small = orbit_space(join_cycles(n), [g])
    assert small.ordering == 'search' and not small.subdivided
    assert [len(small.space.cell_ids(d)) for d in range(4)] == [2, n + 2, 2 * n, n]
    loops = [e for e in small.space.cell_ids(1) if len(set(small.space.cell(e).facets)) == 1]
    assert len(loops) == 2
    large = orbit_space(join_cycles(n), [g], subdivide=True)
    assert large.ordering == 'subdivision' and len(large.space) * n == large.source_simplices
    for r in (small, large):
        assert betti_mod_p(r.space, 2) == (1, 0, 0, 1)
        assert betti_mod_p(r.space, n) == (1, 1, 1, 1)
        (group,) = r.space.fundamental_group()
        assert abelianization(group) == (n,)
    for c, x in small.representatives.items():      # faces follow the local order
        if len(x) > 1:
            assert list(small.space.cell(c).facets) == [small.cell_of(x[:i] + x[i + 1:]) for i in range(len(x))]
    assert small.cell_of((0, 1, n, n + 1)) == small.cell_of((1, 2, n + q, n + (q + 1) % n))


def test_torus_and_klein_bottle_are_told_apart_by_cup_squares():
    side = 6
    shift = orbit_space(torus(side), [grid_map(side, lambda i, j: (i + 2, j))])
    glide = orbit_space(torus(side), [grid_map(side, lambda i, j: (i + 3, -j))])
    for r in (shift, glide):
        assert r.space.homology().betti == (1, 2, 1)
    squares = lambda r: [bool(ring.cup(1, i, 1, i)) for ring in [r.space.cohomology()]
                         for i in range(ring.betti[1])]
    assert not any(squares(shift))
    assert any(squares(glide))


def test_dihedral_group_folds_a_circle_to_an_interval():
    n = 5
    cycle = [(i, (i + 1) % n) for i in range(n)]
    rotation = {i: (i + 1) % n for i in range(n)}
    reflection = {i: (-i) % n for i in range(n)}
    r = orbit_space(cycle, [rotation])
    assert r.ordering == 'search' and len(r.space) == 2 and r.space.homology().betti == (1, 1)
    d = orbit_space(cycle, [rotation, reflection])    # the reflection reverses the edge {2, 3}
    assert d.ordering == 'subdivision' and d.space.homology().betti == (1, 0)


def test_actions_without_a_preserved_local_order():
    flip = orbit_space([(0, 1)], [{0: 1, 1: 0}])
    assert flip.ordering == 'subdivision' and len(flip.space) == 3
    turn = orbit_space([(0, 1, 2)], [[1, 2, 0]])
    assert turn.ordering == 'subdivision' and turn.space.homology().betti == (1, 0, 0)
    with pytest.raises(ValueError, match='cannot be ordered invariantly'):
        orbit_space([(0, 1, 2)], [[1, 2, 0]], subdivide=False)


def test_search_limit():
    n = 5
    g = {i: (i + 1) % n for i in range(n)} | {n + j: n + (j + 2) % n for j in range(n)}
    r = orbit_space(join_cycles(n), [g], search_limit=0)
    assert r.ordering == 'subdivision' and len(r.space) == 528
    assert orbit_space(join_cycles(n), [g], search_limit=1).ordering == 'search'
    with pytest.raises(ValueError, match='search_limit=0'):
        orbit_space(join_cycles(n), [g], subdivide=False, search_limit=0)
    for bad in (-1, True, 1.5):
        with pytest.raises(ValueError, match='search_limit'):
            orbit_space(join_cycles(n), [g], search_limit=bad)
    # the labels and the orbit order are tried before any search
    assert orbit_space(octahedron(), [ANTIPODAL], search_limit=0).ordering == 'labels'


def test_seven_vertex_torus():
    # The rotation of Z_7 preserves an orientation of the three edge orbits without
    # cyclic triangles, and the quotient is the one-vertex torus with 3 edges and 2
    # triangles.  Adding the multiplication by 2 identifies the three edge orbits and
    # forces the subdivision; the quotient by the group of order 21 is a sphere.
    torus7 = [tuple(sorted((i, (i + 1) % 7, (i + 3) % 7))) for i in range(7)]
    torus7 += [tuple(sorted((i, (i + 2) % 7, (i + 3) % 7))) for i in range(7)]
    rotation = {i: (i + 1) % 7 for i in range(7)}
    r = orbit_space(torus7, [rotation])
    assert r.ordering == 'search' and [len(r.space.cell_ids(d)) for d in range(3)] == [1, 3, 2]
    assert r.space.homology().betti == (1, 2, 1)
    ring = r.space.cohomology()
    assert any(ring.cup(1, i, 1, j) for i in range(2) for j in range(2))
    assert len(orbit_space(torus7, [rotation], subdivide=True).space) == 36
    m = orbit_space(torus7, [rotation, {i: (2 * i) % 7 for i in range(7)}])
    assert m.ordering == 'subdivision' and m.space.homology().betti == (1, 0, 1)


def test_not_all_equal_solver_against_brute_force():
    from itertools import product
    from qfnext.orbits import _SearchLimit, _nae_clause, _solve_nae
    holds = lambda x, c: len({x[v] == p for v, p in c}) == 2
    rng = random.Random(3)
    outcomes = set()
    for trial in range(400):
        if trial % 2:
            n = rng.randint(1, 8)
            raw = [[(rng.randrange(n), rng.random() < 0.5) for _ in range(rng.choice((2, 3)))]
                   for _ in range(rng.randint(0, 3 * n))]
        else:                             # near the threshold, where backtracking is needed
            n = rng.randint(8, 12)
            raw = [[(v, rng.random() < 0.5) for v in rng.sample(range(n), 3)] for _ in range(2 * n)]
        clauses = sorted({c for c in map(_nae_clause, raw) if c not in (None, ())})
        brute = any(all(holds(x, c) for c in clauses) for x in product((False, True), repeat=n))
        value = _solve_nae(n, clauses, 10 ** 6)
        assert (value is not None) == brute
        if value is not None:
            assert all(holds(value, c) for c in clauses)
        try:
            _solve_nae(n, clauses, 0)
            direct = True
        except _SearchLimit:
            direct = False
        outcomes.add((brute, direct))
    assert (True, False) in outcomes and (False, False) in outcomes
    assert _nae_clause([(0, True)] * 3) == () and _nae_clause([(0, True), (0, False), (1, True)]) is None
    assert _nae_clause([(1, False), (0, False), (0, False)]) == ((0, True), (1, True))
    # an odd cycle of 'different' constraints has no solution
    assert _solve_nae(3, [((0, True), (1, True)), ((0, True), (2, True)), ((1, True), (2, True))], 100) is None


def test_collapse_of_the_boundary_of_a_rotated_triangle():
    r = orbit_space([(0, 1, 2)], [[1, 2, 0]], invariant=[(0, 1), (1, 2), (0, 2)])
    assert len(r.basepoints) == 1 and r.space.homology().betti == (1, 0, 1)


def test_flag_complex_and_subcomplex_input():
    from qfcore import FlagComplex
    edges = [(a, b) for a, b in combinations(range(6), 2) if a // 2 != b // 2]
    K = FlagComplex.from_graph(6, edges)
    A = K.induced_subcomplex([0, 1, 2, 3])
    r = orbit_space(K, [ANTIPODAL], invariant=A)
    assert r.space.homology().betti == (1, 0, 1)


def test_invalid_actions_are_rejected():
    path = [(0, 1), (1, 2)]
    with pytest.raises(ValueError, match='simplicial'):
        orbit_space(path, [{0: 1, 1: 0}])
    with pytest.raises(ValueError, match='permutation'):
        orbit_space(path, [{0: 1}])
    with pytest.raises(ValueError, match='not vertices'):
        orbit_space(path, [{7: 7}])
    with pytest.raises(ValueError, match='invariant'):
        orbit_space(path, [{0: 2, 2: 0}], invariant=[(0,)])
    with pytest.raises(ValueError, match='max_simplices'):
        orbit_space(cross_polytope(4), [{v: v ^ 1 for v in range(8)}], subdivide=True, max_simplices=100)


def test_random_cyclic_actions_agree_with_the_subdivision():
    # Circulant complexes on Z_n: rotations put all vertices in one orbit, so the
    # search is exercised; a second generator (multiplication by a unit or a
    # reflection) sometimes forces the subdivision.
    seen = set()
    for seed in range(60):
        rng = random.Random(1000 + seed)
        n = rng.randint(4, 9)
        rotation = {i: (i + 1) % n for i in range(n)}
        gens = [rotation]
        if seed % 3 == 0:
            unit = rng.choice([u for u in range(2, n) if all((u * k) % n for k in range(1, n))] or [n - 1])
            gens.append({i: (unit * i) % n for i in range(n)})
        seeds = [tuple(sorted(rng.sample(range(n), rng.choice((2, 2, 3, 3, 4))))) for _ in range(rng.randint(1, 3))]
        K, todo = set(), list(seeds)
        while todo:
            s = todo.pop()
            if s in K:
                continue
            K.add(s)
            todo.extend(tuple(sorted(g[v] for v in s)) for g in gens)
        K = closure(K)
        A = set()
        if seed % 2:
            A, todo = set(), [rng.choice(sorted(s for s in K if len(s) <= 2))]
            while todo:
                s = todo.pop()
                if s in A:
                    continue
                A.add(s)
                todo.extend(tuple(sorted(g[v] for v in s)) for g in gens)
            A = closure(A)
        r = orbit_space(sorted(K), gens, invariant=sorted(A) or None)
        s = orbit_space(sorted(K), gens, invariant=sorted(A) or None, subdivide=True)
        seen.add(r.ordering)
        for p in (2, 3):
            got, expected = betti_mod_p(r.space, p), betti_mod_p(s.space, p)
            width = max(len(got), len(expected))
            assert got + (0,) * (width - len(got)) == expected + (0,) * (width - len(expected))
        if not r.subdivided:                    # one cell per orbit of simplices of K - A
            left, orbits = set(K - A), 0
            while left:
                todo, orbits = [left.pop()], orbits + 1
                while todo:
                    x = todo.pop()
                    for g in gens:
                        y = tuple(sorted(g[v] for v in x))
                        if y in left:
                            left.remove(y)
                            todo.append(y)
            assert len(r.space) - len(r.basepoints) == orbits
        if len(K) <= 40 and seed % 4 == 0:
            for p in (2, 3):
                expected = oracle_betti(K, A, gens, p)
                got = betti_mod_p(r.space, p)
                width = max(len(got), len(expected))
                assert got + (0,) * (width - len(got)) == expected + (0,) * (width - len(expected))
    assert {'search', 'subdivision'} <= seen


def test_results_are_deterministic():
    a = orbit_space(join_cycles(5), [{i: (i + 1) % 5 for i in range(5)} | {5 + j: 5 + (j + 2) % 5 for j in range(5)}])
    b = orbit_space(join_cycles(5), [{i: (i + 1) % 5 for i in range(5)} | {5 + j: 5 + (j + 2) % 5 for j in range(5)}])
    assert a.representatives == b.representatives
    assert [a.space.cell(c).facets for c in a.space.cell_ids()] == [b.space.cell(c).facets for c in b.space.cell_ids()]


def real_moment_angle_complex(faces, m):
    """Barycentric subdivision of the cubical real moment-angle complex of K.

    faces: the simplices of K on the vertices 0..m-1 (the empty face is added).
    A cube is a vector in {-1, 0, 1}^m whose zero set is a face of K; the vertices
    of the subdivision are the cubes, and its simplices are chains of cubes.
    Returns the simplices (tuples of cube indices, ordered by dimension) and the
    cube list, so that sign changes of coordinates act simplicially.
    """
    from itertools import product
    faces = {frozenset()} | {frozenset(f) for f in closure(faces)}
    cubes = sorted({tuple(0 if i in f else s[k] for k, i in enumerate(i for i in range(m)))
                    for f in faces for s in product((-1, 1), repeat=m)},
                   key=lambda c: (c.count(0), c))
    index = {c: i for i, c in enumerate(cubes)}
    def proper_faces(c):
        free = [i for i, x in enumerate(c) if x == 0]
        for k in range(len(free)):
            for fixed in combinations(free, len(free) - k):
                for signs in product((-1, 1), repeat=len(fixed)):
                    d = list(c)
                    for i, s in zip(fixed, signs):
                        d[i] = s
                    yield tuple(d)
    chains = {}
    for c in cubes:
        chains[c] = [(index[c],)] + [ch + (index[c],) for d in proper_faces(c) for ch in chains[d]]
    simplices = [ch for c in cubes for ch in chains[c]]
    return simplices, cubes


def reflection(cubes, coordinates):
    """The sign change of the given coordinates, as a permutation of the cubes."""
    index = {c: i for i, c in enumerate(cubes)}
    return {i: index[tuple(-x if k in coordinates else x for k, x in enumerate(c))]
            for i, c in enumerate(cubes)}


def test_real_moment_angle_complexes_and_small_covers():
    # Pentagon: the real moment-angle complex is the orientable surface of genus 5.
    pentagon = [(i, (i + 1) % 5) for i in range(5)]
    simplices, cubes = real_moment_angle_complex(pentagon, 5)
    Z = orbit_space(simplices, [])
    assert Z.space.homology().betti == (1, 10, 1)
    # Characteristic map (e1, e2, e1, e2, e1+e2); its kernel acts freely and the
    # quotient is a small cover: a nonorientable surface with h-vector (1, 3, 1).
    kernel = [reflection(cubes, {0, 2}), reflection(cubes, {1, 3}), reflection(cubes, {0, 1, 4})]
    M = orbit_space(simplices, kernel)
    assert not M.subdivided and len(M.space) * 8 == M.source_simplices
    assert betti_mod_p(M.space, 2) == (1, 3, 1)
    assert betti_mod_p(M.space, 3) == (1, 2, 0)
    # The whole group Z_2^5 folds the complex onto the pentagon, a disc.
    full = [reflection(cubes, {i}) for i in range(5)]
    assert orbit_space(simplices, full).space.homology().betti == (1, 0, 0)
    # Simplex boundaries: the small cover over a simplex is a real projective space.
    for n in (2, 3):
        boundary = list(combinations(range(n + 1), n))
        simplices, cubes = real_moment_angle_complex(boundary, n + 1)
        P = orbit_space(simplices, [reflection(cubes, set(range(n + 1)))])
        assert betti_mod_p(P.space, 2) == (1,) * (n + 1)
        assert betti_mod_p(P.space, 3) == ((1, 0, 0) if n == 2 else (1, 0, 0, 1))


# ------------------------------------------------- independent random oracle

def subdivide(simplices):
    """Barycentric subdivision; vertices are the simplices, chains listed by size."""
    simplices = sorted(simplices, key=lambda s: (len(s), s))
    by_top, out = {}, set()
    for s in simplices:
        chains = [(s,)]
        for k in range(1, len(s)):
            for t in combinations(s, k):
                chains += [c + (s,) for c in by_top[t]]
        by_top[s] = chains
        out.update(chains)
    return out


def oracle_betti(K, A, gens, p):
    """F_p Betti numbers of the orbit simplicial complex of sd^2 K, with cones on A."""
    X, AX = subdivide(subdivide(K)), subdivide(subdivide(A))
    maps = []
    for g in gens:
        m1 = {s: tuple(sorted(g[v] for v in s)) for s in K}
        m2 = {c: tuple(m1[s] for s in c) for c in subdivide(K)}
        maps.append(m2)
    vertices = {c[0] for c in X if len(c) == 1}
    orbit = {}
    for v in vertices:
        if v in orbit:
            continue
        todo, seen = [v], {v}
        while todo:
            w = todo.pop()
            for m in maps:
                u = m[w]
                if u not in seen:
                    seen.add(u)
                    todo.append(u)
        rep = min(seen, key=repr)
        for w in seen:
            orbit[w] = rep
    L = {frozenset(orbit[v] for v in c) for c in X}
    LA = {frozenset(orbit[v] for v in c) for c in AX}
    # componentwise cone on the image of A
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x
    for s in LA:
        s = sorted(s, key=repr)
        for x in s[1:]:
            parent[find(s[0])] = find(x)
    cones = set()
    for s in LA:
        apex = ('apex', find(next(iter(s))))
        cones.add(s | {apex})
        cones.add(frozenset({apex}))
    L |= cones
    faces = set()
    for s in L:
        s = sorted(s, key=repr)
        for k in range(1, len(s) + 1):
            faces.update(frozenset(c) for c in combinations(s, k))
    by_dim = {}
    for s in faces:
        by_dim.setdefault(len(s) - 1, []).append(tuple(sorted(s, key=repr)))
    top = max(by_dim)
    index = {d: {s: i for i, s in enumerate(by_dim[d])} for d in by_dim}
    ranks = [0] * (top + 2)
    for d in range(1, top + 1):
        rows = [{index[d - 1][s[:i] + s[i + 1:]]: (-1) ** i for i in range(len(s))} for s in by_dim[d]]
        ranks[d] = rank_mod_p(rows, p)
    return tuple(len(by_dim[d]) - ranks[d] - ranks[d + 1] for d in range(top + 1))


@pytest.mark.parametrize('seed', range(40))
def test_random_invariant_complexes_against_second_subdivision(seed):
    rng = random.Random(seed)
    n = rng.randint(4, 6)
    perm = list(range(n))
    rng.shuffle(perm)
    g = dict(enumerate(perm))
    gens = [g]
    if seed % 5 == 0:
        h = list(range(n))
        i, j = rng.sample(range(n), 2)
        h[i], h[j] = h[j], h[i]
        gens.append(dict(enumerate(h)))
    sizes = (2, 2, 3, 3, 4) if n <= 5 and seed % 4 == 0 else (2, 2, 3)
    seeds = [tuple(rng.sample(range(n), rng.choice(sizes))) for _ in range(rng.randint(1, 3))]
    def orbit_of(simplices):
        out, todo = set(), [tuple(sorted(s)) for s in simplices]
        while todo:
            s = todo.pop()
            if s in out:
                continue
            out.add(s)
            for m in gens:
                todo.append(tuple(sorted(m[v] for v in s)))
        return closure(out)
    K = orbit_of(seeds)
    A = set()
    if seed % 3:
        pick = rng.sample(sorted(s for s in K if len(s) <= 2), min(2, sum(len(s) <= 2 for s in K)))
        A = orbit_of(pick)
    vertices = {s[0] for s in K if len(s) == 1}
    gens = [{v: m[v] for v in vertices} for m in gens]
    if any(sorted(m[v] for v in vertices) != sorted(vertices) for m in gens):
        pytest.skip('seed orbit does not cover a permutation-invariant vertex set')
    r = orbit_space(K, gens, invariant=A or None)
    for p in (2, 3):
        expected = oracle_betti(K, A, gens, p)
        got = betti_mod_p(r.space, p)
        width = max(len(expected), len(got))
        assert got + (0,) * (width - len(got)) == expected + (0,) * (width - len(expected))
