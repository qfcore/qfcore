"""Flag complexes and QF-trees. Storage and computation are in the C++17 extension.

    K = FlagComplex.from_graph(4, [(0, 1), (1, 2), (2, 3), (0, 3)], max_dim=1)
    A = K.induced_subcomplex([0, 1])
    Q = K.quotient(A)  # QF cell table (no coefficients) with an optional word trie
    Q.betti_numbers()

QFTree stores topology without coefficients. Homology is computed separately,
currently unfiltered and over F2 only. Arrays returned from native objects are
read-only views that keep their native owner alive.
"""
from __future__ import annotations

import operator
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from . import _native
except ImportError as exc:
    raise ImportError(
        'qfcore requires its compiled C++17 extension. Install a wheel with '
        '`python -m pip install qfcore`, or build from a source checkout with '
        '`python -m pip install .` (needs Boost headers). There is no Python fallback.'
    ) from exc

from ._version import __version__

CORE_API_TAG = '0.5.1'  # legacy tag formerly reported as qfcore.__version__ (pre-release builds)
DEFAULT_MAX_SIMPLICES = 10_000_000
__all__ = ['QFTree', 'QFCell', 'QFFacet', 'QFMap', 'FlagComplex', 'Subcomplex', 'ChainComplex', 'BoundaryMatrix', 'LocalQuotient', 'CompactLocalQuotient',
           'RipsComplex', 'HomologyCheck', 'validate_homology', 'backend_status',
           'native_available', 'gudhi_available', 'self_test']


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f'{name} must be an integer, not bool')
    try:
        n = operator.index(value)
    except TypeError as exc:
        raise TypeError(f'{name} must be an integer') from exc
    if n < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    return n


def _cap(value) -> int:
    return -1 if value is None else _integer(value, 'max_dimension')


def _labels(values: Iterable[int], *, name: str = 'vertices') -> np.ndarray:
    arr = np.asarray(values if isinstance(values, (np.ndarray, list, tuple)) else list(values))
    if arr.ndim != 1:
        raise ValueError(f'{name} must be one-dimensional')
    if arr.size:
        if arr.dtype.kind not in 'iu':
            raise TypeError(f'{name} must contain integers')
        if np.any(arr < 0) or np.any(arr > np.iinfo(np.uint32).max):
            raise ValueError(f'{name} must fit unsigned 32-bit vertex labels')
    return np.ascontiguousarray(arr, dtype=np.uint32)


def _edges(values) -> np.ndarray:
    arr = np.asarray(values if isinstance(values, (np.ndarray, list, tuple)) else list(values))
    if arr.size == 0:
        if arr.ndim > 2 or (arr.ndim == 2 and arr.shape[1] != 2):
            raise ValueError('edges must have shape (m, 2)')
        return np.empty(0, np.uint32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError('edges must have shape (m, 2)')
    return _labels(arr.reshape(-1), name='edge endpoints')


def _f2(field: int) -> None:
    if _integer(field, 'homology_coeff_field', 2) != 2:
        raise NotImplementedError('Only coefficients in F2 are implemented')


def native_available() -> bool:
    return True  # Import would have failed without the compiled extension.


def gudhi_available() -> bool:
    from importlib.util import find_spec
    return find_spec('gudhi') is not None


def backend_status() -> str:
    b = _native.build_info()
    return f"core={b['language']}; binding={b['binding']}; compiler={b['compiler']}; fallback=disabled"


class Subcomplex:
    """Immutable native subcomplex tied to one snapshot of its parent complex.

    `mask` is a read-only uint8 view. For a modified mask, use
    `K.subcomplex(new_mask)`; closure is checked before native computation.
    """
    __slots__ = ('_handle', '_parent_handle', '_data')

    def __init__(self, *args, **kwargs):
        raise TypeError('Construct a subcomplex using K.induced_subcomplex, K.flag_subcomplex or K.subcomplex')

    @classmethod
    def _wrap(cls, handle, parent_handle):
        obj = object.__new__(cls)
        obj._handle, obj._parent_handle, obj._data = handle, parent_handle, None
        return obj

    def _load(self):
        if self._data is None:
            m, c, n = _native.sub_data(self._handle)
            self._data = (np.frombuffer(m, np.uint8), np.frombuffer(c, np.int32), n)
        return self._data

    @property
    def mask(self) -> np.ndarray:
        return self._load()[0]

    @property
    def n_components(self) -> int:
        return self._load()[2]

    def num_simplices(self) -> int:
        return int(self.mask.sum())

    def component_labels(self) -> tuple[np.ndarray, np.ndarray]:
        """Actual vertex labels and component IDs; -1 means not in A."""
        v, _, _ = _native.arrays(self._parent_handle)
        nv = _native.info(self._parent_handle)[1]
        return np.frombuffer(v, np.uint32)[:nv], self._load()[1]

    def low_degree_correction(self) -> tuple[int, int]:
        """Return (t,c): touched components of K, and components of A."""
        return _native.correction(self._handle)

    def __array__(self, dtype=None, copy=None):
        arr = self.mask
        if dtype is not None and np.dtype(dtype) != arr.dtype:
            if copy is False:
                raise ValueError('dtype conversion requires a copy')
            return arr.astype(dtype)
        return arr.copy() if copy is True else arr

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, index):
        return self.mask[index]

    def sum(self, *args, **kwargs):
        return self.mask.sum(*args, **kwargs)

    def __repr__(self):
        return f'Subcomplex(simplices={self.num_simplices()}, components={self.n_components})'


class BoundaryMatrix:
    """Sparse F2 boundary in CSC format (compressed *columns*, not CSR).

    `indices` and `indptr` are immutable zero-copy NumPy views.
    """
    __slots__ = ('_handle', 'indices', 'indptr', 'shape')

    def __init__(self, handle):
        self._handle = handle
        i, o, rows, cols = _native.boundary_data(handle)
        self.indices = np.frombuffer(i, np.uint32)
        self.indptr = np.frombuffer(o, np.int64)
        self.shape = (rows, cols)

    def rank(self) -> int:
        return _native.boundary_rank(self._handle)

    def to_scipy(self):
        """Optional SciPy export, with one in each stored F2 position."""
        from scipy.sparse import csc_matrix
        return csc_matrix((np.ones(len(self.indices), np.uint8), self.indices, self.indptr), shape=self.shape)

    def __repr__(self):
        return f'BoundaryMatrix(shape={self.shape}, nnz={len(self.indices)}, field=F2)'


class ChainComplex:
    """Independent native F2 chain complex with counts and boundary matrices.

    Not a full QF-tree: characteristic/attaching maps and quotient presentations
    are not encoded. No parent complex is retained by this object.
    """
    __slots__ = ('_handle', '_counts', '_payload', 'is_quotient')

    def __init__(self, handle):
        self._handle = handle
        counts, self._payload, self.is_quotient = _native.chain_info(handle)
        self._counts = tuple(counts)

    def num_cells(self) -> int:
        return sum(self._counts)

    def count(self, dimension: int) -> int:
        d = _integer(dimension, 'dimension')
        return self._counts[d] if d < len(self._counts) else 0

    def dimension(self) -> int:
        return max((d for d, n in enumerate(self._counts) if n), default=-1)

    def betti_numbers(self, homology_coeff_field: int = 2) -> list[int]:
        _f2(homology_coeff_field)
        return _native.chain_betti(self._handle)

    def boundary_matrix(self, dimension: int) -> BoundaryMatrix:
        return BoundaryMatrix(_native.chain_boundary(self._handle, _integer(dimension, 'dimension')))

    def nbytes(self) -> int:
        """Native array payload, not RSS or all allocator overhead."""
        return self._payload

    def __len__(self):
        return self.num_cells()

    def __repr__(self):
        return f'ChainComplex(cells={len(self)}, dimension={self.dimension()}, field=F2)'


class FlagComplex:
    """Finite native simplicial complex, canonically ordered by dimension/vertices.

    `from_graph` creates a clique complex or an explicitly capped skeleton.
    `insert` closes the inserted simplex under faces, with copy-on-write
    snapshots. For large complexes, prefer the bulk constructors.
    """
    __slots__ = ('_handle', '_cache', '_limit')

    def __init__(self, verts=None, off=None, dims=None, *, max_simplices=DEFAULT_MAX_SIMPLICES):
        self._limit = _integer(max_simplices, 'max_simplices')
        self._cache = None
        if verts is None and off is None:
            if dims is not None:
                raise ValueError('dims requires vertices and offsets')
            self._handle = _native.empty()
            return
        if verts is None or off is None:
            raise ValueError('both vertices and offsets are required')
        v = _labels(verts)
        o = np.asarray(off)
        if o.ndim != 1 or o.dtype.kind not in 'iu':
            raise TypeError('offsets must be a one-dimensional integer array')
        if o.size and (np.any(o < 0) or np.any(o > np.iinfo(np.int64).max)):
            raise ValueError('offsets must fit nonnegative int64')
        o = np.ascontiguousarray(o, np.int64)
        if dims is not None:
            d = np.asarray(dims)
            if d.shape != (len(o)-1,) or not np.array_equal(d, np.diff(o)-1):
                raise ValueError('dimensions must equal diff(offsets)-1')
        self._handle = _native.from_arrays(v, o, self._limit)

    @classmethod
    def _wrap(cls, handle, limit=DEFAULT_MAX_SIMPLICES):
        obj = object.__new__(cls)
        obj._handle, obj._cache, obj._limit = handle, None, limit
        return obj

    @classmethod
    def from_graph(cls, n_vertices: int, edges, max_dim: int | None = 3,
                   use_gudhi: bool | None = None, *, max_simplices=DEFAULT_MAX_SIMPLICES):
        """Build in native code. None means full expansion, guarded by max_simplices.

        Explicit use_gudhi=True selects a real GUDHI constructor and then
        imports its simplices. It never silently falls back to another backend.
        """
        n = _integer(n_vertices, 'n_vertices')
        limit = _integer(max_simplices, 'max_simplices')
        cap = _cap(max_dim)
        e = _edges(edges)
        if use_gudhi is True:
            import gudhi
            if e.size and (np.any(e >= n) or np.any(e[::2] == e[1::2])):
                raise ValueError('invalid edge endpoint or self-loop')
            if n > limit:
                raise MemoryError('max_simplices limit exceeded by vertices')
            st = gudhi.SimplexTree()
            for v in range(n):
                st.insert([v])
            if cap != 0:
                for a, b in e.reshape(-1, 2):
                    st.insert([int(a), int(b)])
                st.expansion(n-1 if cap == -1 and n else max(cap, 0))
            return cls.from_simplex_tree(st, max_simplices=limit)
        return cls._wrap(_native.from_graph(n, e, cap, limit), limit)

    @classmethod
    def from_points(cls, points, radius: float, max_dim: int | None = 3,
                    torus: bool = False, use_gudhi: bool | None = None,
                    *, max_simplices=DEFAULT_MAX_SIMPLICES):
        pts = np.ascontiguousarray(points, np.float64)
        if pts.ndim != 2:
            raise ValueError('points must have shape (n_points, n_coordinates)')
        n, p = pts.shape
        if p == 0 and n:
            raise ValueError('nonempty point cloud needs coordinates')
        r = float(radius)
        if not np.isfinite(r) or r < 0 or not np.all(np.isfinite(pts)):
            raise ValueError('points and radius must be finite; radius must be nonnegative')
        cap, limit = _cap(max_dim), _integer(max_simplices, 'max_simplices')
        if use_gudhi is True:
            if torus:
                raise ValueError('explicit GUDHI construction does not implement the periodic metric here')
            import gudhi
            if n > limit:
                raise MemoryError('max_simplices limit exceeded by vertices')
            if cap == 0:
                return cls.from_graph(n, [], max_dim=0, use_gudhi=True, max_simplices=limit)
            st = gudhi.RipsComplex(points=pts, max_edge_length=r).create_simplex_tree(
                max_dimension=max(n-1, 0) if cap == -1 else cap)
            return cls.from_simplex_tree(st, max_simplices=limit)
        return cls._wrap(_native.from_points(pts.ravel(), n, p, r, cap, bool(torus), limit), limit)

    @classmethod
    def from_simplex_tree(cls, st, *, max_simplices=DEFAULT_MAX_SIMPLICES):
        """Explicit copying adapter; GUDHI filtrations are not retained."""
        n = st.num_simplices()
        if n > max_simplices:
            raise MemoryError('max_simplices limit exceeded')
        cells = []
        warned = False
        for s, filtration in st.get_simplices():
            if filtration != 0 and not warned:
                warnings.warn('Only the underlying complex is imported, not GUDHI filtration values', RuntimeWarning)
                warned = True
            cells.append(sorted(s))
        lengths = np.array([len(s) for s in cells], np.int64)
        offsets = np.concatenate((np.zeros(1, np.int64), np.cumsum(lengths)))
        flat = [v for s in cells for v in s]
        return cls(flat, offsets, max_simplices=max_simplices)

    _from_simplex_tree = from_simplex_tree

    def to_simplex_tree(self):
        """Copy the complex into a new unfiltered gudhi.SimplexTree."""
        import gudhi
        st = gudhi.SimplexTree()
        for simplex in self.simplices():
            st.insert(simplex)
        return st

    def _arrays(self):
        if self._cache is None:
            v, o, d = _native.arrays(self._handle)
            self._cache = (np.frombuffer(v, np.uint32), np.frombuffer(o, np.int64), np.frombuffer(d, np.int32))
        return self._cache

    @property
    def _verts(self): return self._arrays()[0]
    @property
    def _off(self): return self._arrays()[1]
    @property
    def dims(self): return self._arrays()[2]
    @property
    def max_dim(self): return self.dimension()
    @property
    def n_vertices(self):
        """Legacy label bound, max(vertex)+1; use num_vertices() for the count."""
        return _native.info(self._handle)[3]
    @property
    def vertices(self): return self.block(0).ravel()

    def num_simplices(self) -> int: return _native.info(self._handle)[0]
    def num_vertices(self) -> int: return _native.info(self._handle)[1]
    def dimension(self) -> int: return _native.info(self._handle)[2]
    def __len__(self): return self.num_simplices()

    def dim_range(self, d: int) -> tuple[int, int]:
        return _native.range(self._handle, _integer(d, 'dimension'))

    def block(self, d: int) -> np.ndarray:
        d = _integer(d, 'dimension')
        lo, hi = self.dim_range(d)
        return self._verts[self._off[lo]:self._off[hi]].reshape(hi-lo, d+1)

    def count(self, d: int) -> int:
        lo, hi = self.dim_range(d)
        return hi-lo

    def simplex(self, i: int) -> tuple[int, ...]:
        i = _integer(i, 'simplex index')
        if i >= len(self):
            raise IndexError('simplex index out of range')
        return tuple(map(int, self._verts[self._off[i]:self._off[i+1]]))

    def simplices(self):
        for i in range(len(self)):
            yield self.simplex(i)

    def get_simplices(self):
        """GUDHI-style convenience iterator, with constant filtration 0.0."""
        for s in self.simplices():
            yield list(s), 0.0

    def insert(self, simplex, filtration: float = 0.0) -> bool:
        if filtration != 0.0:
            raise NotImplementedError('This version does not implement nonconstant filtrations')
        handle, changed = _native.insert(self._handle, _labels(simplex), self._limit)
        if changed:
            self._handle, self._cache = handle, None
        return changed

    def nbytes(self) -> int:
        """Native array payload, not process RSS or the size of a QF trie."""
        return _native.memory(self._handle)[0]

    def allocated_nbytes(self) -> int:
        """sizeof(Complex) + native vector capacities; excludes allocator/Python overhead."""
        return _native.memory(self._handle)[1]

    def subcomplex(self, mask) -> Subcomplex:
        if isinstance(mask, Subcomplex):
            if not _native.sub_parent(mask._handle, self._handle):
                raise ValueError('subcomplex belongs to another complex or an earlier snapshot')
            return mask
        arr = np.asarray(mask)
        if arr.ndim != 1 or len(arr) != len(self):
            raise ValueError('mask must be one-dimensional, length num_simplices')
        if arr.dtype.kind not in 'biu' or np.any((arr != 0) & (arr != 1)):
            raise ValueError('mask entries must be integer or boolean 0/1')
        handle = _native.mask(self._handle, np.ascontiguousarray(arr, np.uint8))
        return Subcomplex._wrap(handle, self._handle)

    def _a(self, subcomplex):
        return None if subcomplex is None else self.subcomplex(subcomplex)._handle

    def induced_subcomplex(self, vertices) -> Subcomplex:
        return Subcomplex._wrap(_native.induced(self._handle, _labels(vertices)), self._handle)

    def flag_subcomplex(self, edges, *, vertices=()) -> Subcomplex:
        """Flag subcomplex in K, with optional isolated vertices explicitly retained."""
        return Subcomplex._wrap(_native.flag(self._handle, _edges(edges), _labels(vertices)), self._handle)

    def components(self, mask) -> tuple[np.ndarray, int]:
        """Legacy dense vertex-ID map; use A.component_labels() for sparse labels."""
        a = self.subcomplex(mask)
        labels, c = a.component_labels()
        if self.n_vertices > 10_000_000:
            raise MemoryError('sparse vertex IDs: use A.component_labels() instead of a dense map')
        out = np.full(max(self.n_vertices, 1), -1, np.int32)
        out[labels] = c
        return out, a.n_components

    def is_regular_collapse(self, mask) -> bool:
        """Component-fullness test; for flag complexes this is the regularity criterion."""
        return bool(_native.regular(self.subcomplex(mask)._handle))

    def storage_units(self, mask, ncomp: int | None = None) -> int:
        a = self.subcomplex(mask)
        if ncomp is not None and ncomp != a.n_components:
            raise ValueError('ncomp does not match the subcomplex')
        # A statistic on a read-only native view, not a representation RAM measurement.
        return int(a.n_components + np.sum(self.dims[a.mask == 0].astype(np.int64)+1))

    def cone_model(self, mask, *, max_simplices=None):
        limit = self._limit if max_simplices is None else _integer(max_simplices, 'max_simplices')
        return FlagComplex._wrap(_native.cone(self.subcomplex(mask)._handle, limit), limit)

    def betti_numbers(self, homology_coeff_field: int = 2) -> list[int]:
        _f2(homology_coeff_field)
        return _native.betti(self._handle, None, False)

    def relative_betti(self, mask) -> dict[int, int]:
        return dict(enumerate(_native.betti(self._handle, self._a(mask), False)))

    def quotient_betti(self, mask) -> dict[int, int]:
        return dict(enumerate(_native.betti(self._handle, self._a(mask), True)))

    def boundary_matrix(self, dimension: int, subcomplex=None, *, quotient=False) -> BoundaryMatrix:
        return BoundaryMatrix(_native.boundary(self._handle, self._a(subcomplex),
                                              _integer(dimension, 'dimension'), bool(quotient)))

    def _boundary(self, mask, d, quotient, vcomp=None, ncomp=0):
        b = self.boundary_matrix(d, mask, quotient=quotient)
        return b.indices, b.indptr, b.shape[1], b.shape[0]

    def quotient(self, subcomplex, *, representation: str = 'qf_tree',
                 word_index: bool = True, keep_source: bool = False):
        """Materialize a topological QFTree, without constructing chain matrices.

        representation='chain' explicitly selects the F2-only object.
        keep_source=True retains the exact original K,A as an extra archive;
        this is not needed for quotient topology, traversal, saving or collapse.
        """
        if not isinstance(word_index, bool) or not isinstance(keep_source, bool):
            raise TypeError('word_index and keep_source must be bool')
        a = self._a(subcomplex)
        if representation == 'qf_tree':
            return QFTree._wrap(_native.qft_new(self._handle, a, word_index, keep_source))
        if representation == 'chain':
            if keep_source:
                raise ValueError('source archive is only available for representation=qf_tree')
            return ChainComplex(_native.chain(self._handle, a, True))
        raise ValueError("representation must be 'qf_tree' or 'chain'")

    def to_qf_tree(self, *, word_index: bool = True, keep_source: bool = False):
        """Promote K to a full cell table with no collapsed components."""
        return self.quotient(None, word_index=word_index, keep_source=keep_source)

    def relative_chain_complex(self, subcomplex) -> ChainComplex:
        return ChainComplex(_native.chain(self._handle, self._a(subcomplex), False))

    def star_partition(self, subcomplex) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Label every simplex relative to A: 0 untouched, 1 frontier, 2 star, 3 collapsed.

        Star simplices are the survivors with a vertex in V(A); only their
        attaching data changes under the collapse. Frontier simplices are the
        faces of star simplices that are disjoint from V(A).
        """
        labels, counts = _native.star_partition(self._handle, self._a(subcomplex))
        return np.frombuffer(labels, np.uint8), tuple(int(c) for c in counts)

    def local_quotient(self, subcomplex, *, word_index: bool = False) -> LocalQuotient:
        """Closed-star presentation of K/_SC A: QF records only where attachments change."""
        return LocalQuotient(self, self.subcomplex(subcomplex), word_index=word_index)

    def __repr__(self):
        return f'FlagComplex(vertices={self.num_vertices()}, simplices={len(self)}, dimension={self.dimension()}, backend=C++)'


class LocalQuotient:
    """Hybrid presentation of a componentwise collapse K/_SC A.

    The simplices of K are partitioned into untouched, frontier, star and
    collapsed classes (see ``FlagComplex.star_partition``). The QF-tree is built
    only on the closed star of A, i.e. on the frontier and star simplices, with
    source ids pointing into K. Untouched simplices keep their implicit simplicial
    attachments and are never copied: the simplex tree (or ``FlagComplex``) that
    already holds K remains their representation.

    This is the local application of the QF-tree inside a simplex tree: the
    full quotient is recovered from ``K`` plus ``tree``. The object keeps the
    snapshot of ``K`` taken at construction (``self.complex``); later insertions
    into ``K`` produce a new snapshot and do not affect it. ``chain_complex`` assembles
    the relative or quotient F2 chain complex from the two parts in the same
    generator order as ``K.quotient(A, representation='chain')``, so the two can
    be compared column for column (``verify``).
    """
    __slots__ = ('complex', 'subcomplex', 'labels', 'counts', 'tree')

    def __init__(self, K: 'FlagComplex', A: 'Subcomplex', *, word_index: bool = False):
        if not isinstance(word_index, bool):
            raise TypeError('word_index must be bool')
        # FlagComplex.insert replaces the native handle (copy-on-write). Pin the
        # snapshot taken now in a separate wrapper, so that later edits of K do not
        # mix a new complex with the old mask, partition and local tree.
        snapshot = FlagComplex._wrap(K._handle, K._limit)
        self.complex, self.subcomplex = snapshot, snapshot.subcomplex(A)
        self.labels, self.counts = snapshot.star_partition(self.subcomplex)
        self.tree = QFTree._wrap(_native.qft_star_new(snapshot._handle, self.subcomplex._handle,
                                                      self.labels, word_index))

    @property
    def num_untouched(self) -> int: return self.counts[0]
    @property
    def num_frontier(self) -> int: return self.counts[1]
    @property
    def num_star(self) -> int: return self.counts[2]
    @property
    def num_collapsed(self) -> int: return self.counts[3]

    def storage_units(self, model: str = 'retained') -> int:
        """Storage-unit budget of the closed-star presentation under three conventions.

        All three use the convention of ``FlagComplex.storage_units``: d+1 units per
        QF record of a d-cell, one unit per simplex held in a simplex tree, one per
        collapsed component. With T the untouched, F the frontier and S the star
        simplices and R_X the record units of X:

        ``'retained'`` (what this object holds): the whole source K stays in memory
            next to the local records, |K| + R_{S+F} + c.
        ``'compact'`` (what ``compact()`` builds): the source is replaced by the
            subcomplex T+F, |T| + |F| + R_{S+F} + c.
        ``'ideal'``: the nonduplicating reference budget of a hybrid that shares
            every record, |T| + R_{S+F} + c (not a lower bound over all encodings).
        These are counts, not bytes or resident memory.
        """
        dims = self.complex.dims.astype(np.int64)
        local = int(np.sum(dims[(self.labels == 1) | (self.labels == 2)] + 1))
        c = self.subcomplex.n_components
        if model == 'retained':
            return len(self.complex) + local + c
        if model == 'compact':
            return self.num_untouched + self.num_frontier + local + c
        if model == 'ideal':
            return self.num_untouched + local + c
        raise ValueError("model must be 'retained', 'compact' or 'ideal'")

    def compact(self) -> 'CompactLocalQuotient':
        """Drop the source: keep only the untouched-plus-frontier subcomplex and the local tree."""
        return CompactLocalQuotient(self)

    def chain_complex(self, *, quotient: bool = True) -> ChainComplex:
        return ChainComplex(_native.star_chain(self.complex._handle, self.subcomplex._handle,
                                               self.tree._handle, self.labels, bool(quotient)))

    def betti_numbers(self, homology_coeff_field: int = 2, *, quotient: bool = True) -> list[int]:
        _f2(homology_coeff_field)
        return self.chain_complex(quotient=quotient).betti_numbers()

    def verify(self) -> bool:
        """Compare with the full QF-tree and the direct chain complexes of the same pair.

        Checks that (i) every star/frontier record agrees with the corresponding
        record of ``K.quotient(A)``, (ii) every untouched simplex has unchanged
        (fully simplicial) attaching data in the full tree, and (iii) the
        assembled quotient and relative chain complexes agree column for column
        with the direct ones.
        """
        full = self.complex.quotient(self.subcomplex, word_index=False)
        local = self.tree
        # (i)/(ii): compare facet records through source ids.
        full_by_source = {int(s): i for i, s in enumerate(full.source_ids) if s >= 0}
        f_facets, f_off = full.facet_targets, full.facet_offsets
        l_facets, l_off = local.facet_targets, local.facet_offsets
        f_src, l_src = full.source_ids, local.source_ids
        nc = local.n_components
        if nc != full.n_components:
            return False
        for i in range(nc, local.num_cells()):
            j = full_by_source[int(l_src[i])]
            lt = l_facets[l_off[i]:l_off[i+1]]
            ft = f_facets[f_off[j]:f_off[j+1]]
            if len(lt) != len(ft):
                return False
            for x, y in zip(lt, ft):
                if x < nc or y < nc:
                    if int(x) != int(y):
                        return False
                elif int(l_src[x]) != int(f_src[y]):
                    return False
        K = self.complex
        for s in np.flatnonzero(self.labels == 0):
            j = full_by_source[int(s)]
            for t in f_facets[f_off[j]:f_off[j+1]]:
                if t < nc:
                    return False
        # (iii): chain complexes.
        for quotient in (True, False):
            direct = ChainComplex(_native.chain(K._handle, self.subcomplex._handle, quotient))
            local_chain = self.chain_complex(quotient=quotient)
            if direct._counts != local_chain._counts:
                return False
            for d in range(len(direct._counts)):
                a, b = direct.boundary_matrix(d), local_chain.boundary_matrix(d)
                if a.shape != b.shape or not np.array_equal(a.indices, b.indices) \
                        or not np.array_equal(a.indptr, b.indptr):
                    return False
        return True

    def __repr__(self):
        u, f, s, c = self.counts
        return f'LocalQuotient(untouched={u}, frontier={f}, star={s}, collapsed={c}, records={len(self.tree)})'


class CompactLocalQuotient:
    """Source-free closed-star presentation: a simplicial complex of the untouched and
    frontier simplices (kept in a simplex tree or ``FlagComplex``) plus the local QF-tree.

    The frontier is present in both parts, which is what makes the two parts
    joinable: a star record's facet that is a frontier simplex is looked up by its
    vertex support. ``betti_numbers`` assembles the quotient or relative F2 chain
    complex from the two parts alone; the original ``K`` and ``A`` are not needed.
    """
    __slots__ = ('untouched', 'tree', 'n_star', 'n_frontier', 'n_untouched', 'source_dimension')

    def __init__(self, local: 'LocalQuotient'):
        K = local.complex
        keep = local.labels <= 1
        verts, off = K._verts, K._off
        starts = off[:-1][keep]
        ends = off[1:][keep]
        parts = [verts[a:b] for a, b in zip(starts, ends)]
        new_verts = np.concatenate(parts) if parts else np.zeros(0, np.uint32)
        new_off = np.concatenate(([0], np.cumsum([len(p) for p in parts]))).astype(np.int64)
        self.untouched = FlagComplex(np.ascontiguousarray(new_verts, np.uint32), new_off,
                                     max_simplices=K._limit)
        self.tree = local.tree
        self.n_untouched, self.n_frontier, self.n_star = local.counts[0], local.counts[1], local.counts[2]
        # Only the dimension of K is kept, so that Betti lists have the same length
        # as those of K.quotient(A) and LocalQuotient (trailing zero degrees included).
        self.source_dimension = K.dimension()

    def storage_units(self) -> int:
        dims = self.tree.dims.astype(np.int64)
        return int(len(self.untouched) + np.sum(dims[self.tree.n_components:] + 1) + self.tree.n_components)

    def _columns(self, quotient: bool):
        """Sparse F2 boundary columns of the assembled chain complex, generators ordered
        basepoints (degree 0 only), then untouched/frontier simplices, then star cells."""
        U, Q = self.untouched, self.tree
        nc = Q.n_components if quotient else 0
        dim = max(U.dimension(), Q.dimension())
        # frontier simplices appear in both parts: map tree cell -> simplex index in U
        u_index = {tuple(int(x) for x in U.simplex(i)): i for i in range(len(U))}
        u_pos = {}
        counts = [0] * (dim + 1)
        for i in range(len(U)):
            d = int(U.dims[i]); u_pos[i] = counts[d]; counts[d] += 1
        star_pos, star_cells = {}, []
        supports = Q.source_vertices; soff = Q.source_offsets
        for i in range(Q.n_components, len(Q)):
            sup = tuple(int(x) for x in supports[soff[i]:soff[i+1]])
            if sup in u_index:
                continue                      # frontier record: represented by U
            d = int(Q.dims[i]); star_pos[i] = counts[d]; counts[d] += 1; star_cells.append((i, d))
        if nc:
            counts[0] += nc
        shift = lambda d: nc if d == 1 else 0
        cols = [[] for _ in range(dim + 1)]
        # generator order per degree: untouched then star, as numbered above
        order = [[] for _ in range(dim + 1)]
        for i in range(len(U)):
            order[int(U.dims[i])].append(('u', i))
        for i, d in star_cells:
            order[d].append(('s', i))
        facets_arr, foff = Q.facet_targets, Q.facet_offsets
        for d in range(dim + 1):
            for kind, i in order[d]:
                col = []
                if d == 0:
                    cols[d].append(col); continue
                if kind == 'u':
                    simplex = [int(x) for x in U.simplex(i)]
                    for j in range(len(simplex)):
                        f = tuple(simplex[:j] + simplex[j+1:])
                        col.append(shift(d) + u_pos[u_index[f]])
                else:
                    for t in facets_arr[foff[i]:foff[i+1]]:
                        t = int(t)
                        if t < Q.n_components:
                            if quotient and d == 1:
                                col.append(t)
                        elif t in star_pos:
                            col.append(shift(d) + star_pos[t])
                        else:
                            sup = tuple(int(x) for x in supports[soff[t]:soff[t+1]])
                            col.append(shift(d) + u_pos[u_index[sup]])
                cols[d].append(col)
        if nc:
            cols[0] = [[] for _ in range(nc)] + cols[0]
        return counts, cols

    def betti_numbers(self, homology_coeff_field: int = 2, *, quotient: bool = True) -> list[int]:
        _f2(homology_coeff_field)
        from qfops import _ops  # lazy: qfops imports qfcore
        counts, cols = self._columns(bool(quotient))
        chain = _ops.chain_from_columns([int(c) for c in counts], cols)
        betti = list(_ops.Homology(chain, 10_000_000).betti())
        return betti + [0] * (self.source_dimension + 1 - len(betti))

    def __repr__(self):
        return (f'CompactLocalQuotient(untouched={self.n_untouched}, frontier={self.n_frontier}, '
                f'star={self.n_star}, source_simplices={len(self.untouched)}, records={len(self.tree)})')


class RipsComplex:
    """GUDHI-inspired constructor, with no persistence/filtration API implied."""
    def __init__(self, *, points, max_edge_length: float, torus: bool = False):
        self.points = np.array(points, dtype=np.float64, copy=True)
        self.max_edge_length, self.torus = float(max_edge_length), bool(torus)

    def create_complex(self, max_dimension: int | None = 3, *, max_simplices=DEFAULT_MAX_SIMPLICES) -> FlagComplex:
        return FlagComplex.from_points(self.points, self.max_edge_length, max_dim=max_dimension,
                                       torus=self.torus, max_simplices=max_simplices)


@dataclass(frozen=True)
class HomologyCheck:
    quotient_equals_cone: bool
    relative_matches_correction: bool
    touched_components: int
    subcomplex_components: int

    @property
    def agree(self) -> bool:
        return self.quotient_equals_cone and self.relative_matches_correction


def validate_homology(K: FlagComplex, A, direct: dict, quotient: dict, cone: dict) -> HomologyCheck:
    """All degrees checked, including the relative/absolute correction at 0 and 1."""
    a = K.subcomplex(A)
    t, c = a.low_degree_correction()
    top = max([1, *direct, *quotient, *cone])
    qc = all(quotient.get(d, 0) == cone.get(d, 0) for d in range(top+1))
    dr = (direct.get(0, 0) == quotient.get(0, 0)-t
          and direct.get(1, 0) == quotient.get(1, 0)+c-t
          and all(direct.get(d, 0) == quotient.get(d, 0) for d in range(2, top+1)))
    return HomologyCheck(qc, dr, t, c)


def self_test(verbose: bool = True) -> bool:
    K = FlagComplex.from_graph(4, [(0, 1), (1, 2), (2, 3), (0, 3)], max_dim=1)
    A = K.induced_subcomplex([0, 1])
    C = K.cone_model(A)
    b = K.betti_numbers()
    check = validate_homology(K, A, K.relative_betti(A), K.quotient_betti(A), dict(enumerate(C.betti_numbers())))
    ok = b == [1, 1] and check.agree and K.quotient(A).betti_numbers() == [1, 1]
    if verbose:
        print(backend_status())
        print(f'native self-test: {"PASS" if ok else "FAIL"}; cycle={b}; all-degree agreement={check.agree}')
    return ok

# Topology API; does not import qfcore.homology or construct any chain matrices.
from .qftree import QFCell, QFFacet, QFMap, QFTree
from .visualize import visualize_dag

__all__.append("visualize_dag")
