"""Randomized differential tests against an independent F2 oracle.

Pairs (K, A): flag, capped and non-flag K; induced, flag-but-not-full and
arbitrary (possibly non-flag) A. The Betti numbers of K/_SC A are compared with
those of the cone model computed here from scratch, H_*(K, A) with a direct F2
computation, regularity with component fullness, strict gradedness with the
absence of survivors of dimension >= 2 whose boundary lies in A, and every
presentation of the library (QFTree, chain, cone model, LocalQuotient and its
compact form, .qft round trip) with the same oracle. Random edit sequences on
EditableQF (collapse, deletion, cascade deletion, insertion) are compared with
the cone model of the updated pair after every step.
"""

from __future__ import annotations
import itertools, random, tempfile
from pathlib import Path
import numpy as np
from qfcore import FlagComplex, QFTree
from qfnext import EditableQF
from qfnext.controls import simplicial_complex


# ---------------------------------------------------------------- F2 oracle
def _rank(rows):
    piv, r = {}, 0
    for x in rows:
        while x:
            h = x.bit_length() - 1
            if h in piv:
                x ^= piv[h]
            else:
                piv[h] = x
                r += 1
                break
    return r


def betti(cells):
    """cells: set of sorted tuples forming a chain basis; faces outside the set are dropped."""
    by_dim = {}
    for s in cells:
        by_dim.setdefault(len(s) - 1, []).append(s)
    if not by_dim:
        return []
    top = max(by_dim)
    index = {d: {s: i for i, s in enumerate(sorted(v))} for d, v in by_dim.items()}
    ranks = {}
    for d in range(1, top + 1):
        rows = []
        low = index.get(d - 1, {})
        for s in by_dim.get(d, []):
            x = 0
            for j in range(len(s)):
                f = s[:j] + s[j + 1:]
                if f in low:
                    x ^= 1 << low[f]
            rows.append(x)
        ranks[d] = _rank(rows)
    out = [len(by_dim.get(d, [])) - ranks.get(d, 0) - ranks.get(d + 1, 0) for d in range(top + 1)]
    return trim(out)


def trim(b):
    b = list(b)
    while b and b[-1] == 0:
        b.pop()
    return b


def components(A):
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for s in A:
        for v in s:
            parent.setdefault(v, v)
    for s in A:
        if len(s) == 2:
            a, b = find(s[0]), find(s[1])
            if a != b:
                parent[a] = b
    comps = {}
    for v in parent:
        comps.setdefault(find(v), set()).add(v)
    return list(comps.values())


def closure(simplices):
    out = set()
    for s in simplices:
        for k in range(1, len(s) + 1):
            out.update(itertools.combinations(s, k))
    return out


# ---------------------------------------------------------------- generators
def random_K(r):
    n = r.randint(3, 9)
    kind = r.choice(('flag', 'flag', 'capped', 'general'))
    if kind == 'general':
        tops = [tuple(sorted(r.sample(range(n), r.randint(1, min(4, n))))) for _ in range(r.randint(2, 7))]
        tops += [(v,) for v in range(n)]
        S = sorted(closure(tops), key=lambda s: (len(s), s))
        verts = np.array([v for s in S for v in s], dtype=np.uint32)
        off = np.cumsum([0] + [len(s) for s in S]).astype(np.int64)
        return FlagComplex(verts, off), kind
    p = r.uniform(0.25, 0.9)
    edges = [(a, b) for a in range(n) for b in range(a + 1, n) if r.random() < p]
    cap = None if kind == 'flag' else r.choice((1, 2))
    return FlagComplex.from_graph(n, edges, max_dim=cap), kind


def random_A(K, r):
    simplices = list(K.simplices())
    verts = [s[0] for s in simplices if len(s) == 1]
    edges = [s for s in simplices if len(s) == 2]
    mode = r.choice(('induced', 'flag', 'general', 'general'))
    if mode == 'induced':
        return K.induced_subcomplex(r.sample(verts, r.randint(1, len(verts)))), mode
    if mode == 'flag' and edges:
        chosen = r.sample(edges, r.randint(1, len(edges)))
        extra = r.sample(verts, r.randint(0, min(2, len(verts))))
        return K.flag_subcomplex(chosen, vertices=extra), mode
    pick = r.sample(simplices, r.randint(1, max(1, len(simplices) // 3)))
    cl = closure(pick)
    mask = np.array([s in cl for s in simplices], dtype=np.uint8)
    return K.subcomplex(mask), 'general'


# ---------------------------------------------------------------- one trial
def check_pair(K, A, tmp, r):
    simplices = list(K.simplices())
    Kset = set(simplices)
    mask = np.asarray(A.mask).astype(bool)
    Aset = {s for s, m in zip(simplices, mask) if m}
    comps = components(Aset)
    assert A.n_components == len(comps)

    # oracle: cone model and relative chains
    apex0 = 1 + max(v for s in simplices for v in s)
    cone = set(Kset)
    for i, C in enumerate(comps):
        c = apex0 + i
        cone.add((c,))
        for s in Aset:
            if set(s) <= C:
                cone.add(tuple(sorted(s + (c,))))
    ob = betti(cone)
    orel = betti(Kset - Aset)

    # regularity: every component of A is full in K
    full = all(all(not set(t) <= C or t in Aset for t in Kset) for C in comps)
    # strict gradedness: no survivor of dim >= 2 with boundary inside A
    skip = any(t not in Aset and len(t) >= 3 and all(t[:j] + t[j + 1:] in Aset for j in range(len(t)))
               for t in Kset)

    Q = K.quotient(A, word_index=r.random() < 0.5, keep_source=r.random() < 0.5)
    assert trim(Q.betti_numbers()) == ob, ('quotient betti', Q.betti_numbers(), ob)
    assert trim(Q.betti_numbers(relative=True)) == orel, ('relative betti', Q.betti_numbers(relative=True), orel)
    assert trim(K.quotient_betti(A).values()) == ob
    assert trim(K.relative_betti(A).values()) == orel
    assert trim(K.quotient(A, representation='chain').betti_numbers()) == ob
    assert trim(K.cone_model(A).betti_numbers()) == ob
    assert Q.validate()
    assert K.is_regular_collapse(A) == full, ('regular', K.is_regular_collapse(A), full)
    assert Q.is_regular() == full, ('Q.is_regular', Q.is_regular(), full)
    assert Q.is_strictly_graded() == (not skip), ('graded', Q.is_strictly_graded(), skip)
    assert K.storage_units(A) == len(comps) + sum(len(s) for s in Kset - Aset)

    labels, counts = K.star_partition(A)
    assert sum(counts) == len(K) and counts[3] == len(Aset)
    VA = {v for s in Aset for v in s}
    for s, lab in zip(simplices, labels):
        meets = bool(set(s) & VA)
        assert (lab == 3) == (s in Aset)
        assert (lab == 2) == (meets and s not in Aset)
    L = K.local_quotient(A)
    assert L.verify()
    assert trim(L.betti_numbers()) == ob and trim(L.betti_numbers(quotient=False)) == orel
    C = L.compact()
    assert trim(C.betti_numbers()) == ob and trim(C.betti_numbers(quotient=False)) == orel
    assert L.storage_units('ideal') <= L.storage_units('compact') <= L.storage_units('retained')

    p = Path(tmp) / 'q.qft'
    Q.save(p)
    R = QFTree.load(p)
    assert trim(R.betti_numbers()) == ob and len(R) == len(Q)
    assert trim(R.betti_numbers(relative=True)) == orel
    if Q.has_source_archive:
        K2, A2 = R.source_pair()
        assert list(K2.simplices()) == simplices
        assert np.array_equal(np.asarray(A2.mask).astype(bool), mask)
    return full, skip


def run(trials=500, seed=0):
    r = random.Random(seed)
    stats = dict(pairs=0, regular=0, nonregular=0, graded=0, skipping=0)
    kinds, modes = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(trials):
            K, kind = random_K(r)
            A, mode = random_A(K, r)
            if A.num_simplices() == 0:
                continue
            full, skip = check_pair(K, A, tmp, r)
            stats['pairs'] += 1
            stats['regular' if full else 'nonregular'] += 1
            stats['skipping' if skip else 'graded'] += 1
            kinds[kind] = kinds.get(kind, 0) + 1
            modes[mode] = modes.get(mode, 0) + 1
    return stats, kinds, modes

def cone_betti(K, A):
    comps = components(A)
    apex0 = 1 + max((v for s in K for v in s), default=0)
    cone = set(K)
    for i, C in enumerate(comps):
        c = apex0 + i
        cone.add((c,))
        for s in A:
            if set(s) <= C:
                cone.add(tuple(sorted(s + (c,))))
    return betti(cone)


def check(q, K, A, tmp):
    assert q.validate()
    hb = trim(q.homology().betti)
    ob = cone_betti(K, A)
    assert hb == ob, ('editable betti', hb, ob)
    assert trim(q.cohomology().betti) == hb
    p = Path(tmp) / 'e.qfe'
    q.save(p)
    assert trim(EditableQF.load(p).homology().betti) == hb
    assert trim(q.clone().homology().betti) == hb


def run_edits(trials=200, steps=6, seed=0):
    r = random.Random(seed)
    done = {'collapse': 0, 'delete': 0, 'cascade': 0, 'insert': 0}
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(trials):
            n = r.randint(3, 8)
            tops = [tuple(sorted(r.sample(range(n), r.randint(1, min(4, n))))) for _ in range(r.randint(2, 6))]
            K = closure(tops)
            q, ids = simplicial_complex(tops)
            A = set()
            check(q, K, A, tmp)
            for _ in range(steps):
                survivors = sorted(K - A, key=lambda s: (len(s), s))
                op = r.choice(('collapse', 'collapse', 'delete', 'cascade', 'insert'))
                if op == 'collapse' and survivors:
                    B = closure(r.sample(survivors, r.randint(1, min(3, len(survivors)))))
                    q.collapse([ids[s] for s in B - A], close=True)
                    A |= B
                elif op == 'delete':
                    maximal = [s for s in survivors if not any(set(s) < set(t) for t in K)]
                    if not maximal:
                        continue
                    s = r.choice(maximal)
                    q.delete(ids[s])
                    K.discard(s)
                elif op == 'cascade' and survivors:
                    s = r.choice(survivors)
                    q.delete(ids[s], cascade=True)
                    K -= {t for t in K if set(s) <= set(t)}
                elif op == 'insert':
                    verts = sorted({v for s in survivors if len(s) == 1 for v in s})
                    cands = [c for k in (2, 3) for c in itertools.combinations(verts, k)
                             if c not in K and all(c[:j] + c[j + 1:] in K and c[:j] + c[j + 1:] not in A
                                                   for j in range(len(c)))]
                    if not cands:
                        continue
                    s = r.choice(cands)
                    ids[s] = q.add_simplex(len(s) - 1, [ids[s[:j] + s[j + 1:]] for j in range(len(s))])
                    K.add(s)
                else:
                    continue
                done[op] += 1
                check(q, K, A, tmp)
    return done


def test_random_pairs_against_independent_oracle():
    stats, kinds, modes = run(300, 2026)
    assert stats['pairs'] > 250 and stats['nonregular'] > 0 and stats['skipping'] > 0
    assert set(kinds) == {'flag', 'capped', 'general'} and set(modes) == {'induced', 'flag', 'general'}


def test_random_edit_sequences_against_oracle():
    done = run_edits(60, 6, 2026)
    assert all(v > 0 for v in done.values())
