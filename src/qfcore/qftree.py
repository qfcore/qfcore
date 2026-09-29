"""Coefficient-independent source-labelled quotient objects backed by C++17.

The topology is stored as ordered local facet occurrences, component flags and
source supports. Trie words are only a secondary, many-to-one lookup index.
No chain matrices or ranks are constructed on create/save/load/traversal.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Iterable
import numpy as np
from . import _native, _integer, _labels, DEFAULT_MAX_SIMPLICES


@dataclass(frozen=True)
class QFCell:
    id: int
    dimension: int
    source_id: int | None
    source_vertices: tuple[int, ...]
    is_component: bool
    component_vertices: tuple[int, ...] = ()


@dataclass(frozen=True)
class QFFacet:
    local_index: int
    target: int
    is_constant: bool
    source_vertices: tuple[int, ...]
    orientation: int
    # For a genuine target, increasing source coordinates prescribe the
    # identity coordinate correspondence after deleting local_index.


class QFTree:
    """Immutable native QF presentation for componentwise subcomplex collapses.

    Build using ``K.quotient(A)`` or ``K.to_qf_tree()``. Arbitrary affine
    permutation gluings are not supported by this class. The object owns all
    data needed for its quotient topology without retaining K or A unless
    keep_source=True was explicitly requested.
    """
    __slots__ = ('_handle', '_data', '_info')

    def __init__(self, *args, **kwargs):
        raise TypeError('Use K.quotient(A), K.to_qf_tree(), or QFTree.load(path)')

    @classmethod
    def _wrap(cls, handle):
        obj = object.__new__(cls)
        obj._handle, obj._data = handle, None
        obj._info = _native.qft_info(handle)
        return obj

    @classmethod
    def load(cls, path, *, max_cells: int = DEFAULT_MAX_SIMPLICES,
             max_bytes: int = 2**31, word_index: bool | None = None) -> QFTree:
        """Load a standalone .qft; validate records/overlaps and restore indices.

        Limits also apply to a retained source archive. The file is a versioned
        binary table, not pickle. Its noncryptographic checksum detects ordinary
        corruption but is not a trust/authentication mechanism.
        """
        if word_index is not None and not isinstance(word_index, bool):
            raise TypeError('word_index must be bool or None')
        handle = _native.qft_load(os.fspath(path), _integer(max_cells, 'max_cells'),
                                 _integer(max_bytes, 'max_bytes'),
                                 -1 if word_index is None else int(word_index))
        return cls._wrap(handle)

    def save(self, path) -> Path:
        """Atomically replace a .qft file; no homology/chain computation.

        Native binary encoding is written to a same-directory temporary file,
        then renamed. Does not promise power-loss durability (no fsync).
        """
        target = Path(os.fsdecode(os.fspath(path)))
        fd, temporary = tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent)
        os.close(fd)
        try:
            _native.qft_save(self._handle, temporary)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target

    def _arrays(self):
        if self._data is None:
            buffers = _native.qft_data(self._handle)
            dtypes = (np.int32, np.int64, np.uint32, np.int64, np.uint32,
                      np.int64, np.uint32, np.int64, np.uint32, np.uint32)
            self._data = tuple(np.frombuffer(b, dtype) for b, dtype in zip(buffers, dtypes))
        return self._data

    @property
    def dims(self) -> np.ndarray: return self._arrays()[0]
    @property
    def source_ids(self) -> np.ndarray: return self._arrays()[1]
    @property
    def source_vertices(self) -> np.ndarray: return self._arrays()[2]
    @property
    def source_offsets(self) -> np.ndarray: return self._arrays()[3]
    @property
    def facet_targets(self) -> np.ndarray: return self._arrays()[4]
    @property
    def facet_offsets(self) -> np.ndarray: return self._arrays()[5]
    @property
    def n_components(self) -> int: return self._info[2]
    @property
    def has_source_archive(self) -> bool: return bool(self._info[6])
    @property
    def has_word_index(self) -> bool: return bool(self._info[5])
    @property
    def source_dimension(self) -> int: return self._info[4]
    @property
    def source_num_simplices(self) -> int: return self._info[3]

    def num_cells(self) -> int: return self._info[0]
    def num_vertices(self) -> int: return self.count(0)
    def dimension(self) -> int: return self._info[1]
    def __len__(self) -> int: return self.num_cells()

    def cell_ids(self, dimension: int | None = None) -> range:
        if dimension is None:
            return range(len(self))
        lo, hi = _native.qft_range(self._handle, _integer(dimension, 'dimension'))
        return range(lo, hi)

    def count(self, dimension: int) -> int:
        return len(self.cell_ids(dimension))

    def _id(self, cell_id) -> int:
        cell_id = _integer(cell_id, 'cell_id')
        if cell_id >= len(self):
            raise IndexError('QF cell ID out of range')
        return cell_id

    def cell(self, cell_id: int) -> QFCell:
        c = self._id(cell_id)
        source = tuple(map(int, self.source_vertices[self.source_offsets[c]:self.source_offsets[c+1]]))
        members = ()
        if c < self.n_components:
            vertices, offsets = self._arrays()[6:8]
            members = tuple(map(int, vertices[offsets[c]:offsets[c+1]]))
        return QFCell(c, int(self.dims[c]), None if c < self.n_components else int(self.source_ids[c]),
                      source, c < self.n_components, members)

    def cells(self, dimension: int | None = None):
        for cell_id in self.cell_ids(dimension):
            yield self.cell(cell_id)

    def facets(self, cell_id: int) -> tuple[QFFacet, ...]:
        c = self.cell(cell_id)
        targets = self.facet_targets[self.facet_offsets[c.id]:self.facet_offsets[c.id+1]]
        return tuple(QFFacet(i, int(t), int(t) < self.n_components,
                             c.source_vertices[:i] + c.source_vertices[i+1:],
                             -1 if i % 2 else 1) for i, t in enumerate(targets))

    def face(self, cell_id: int, deletions: Iterable[int]) -> int:
        """Compose deletions in CURRENT local coordinates, absorbing at points."""
        return _native.qft_face(self._handle, self._id(cell_id), _labels(deletions, name='local deletions'))

    def closure(self, cells: Iterable[int]) -> tuple[int, ...]:
        """Smallest face-closed set, including every encountered component point."""
        return tuple(_native.qft_closure(self._handle, _labels(cells, name='cell IDs')))

    def cofaces(self, cell_id: int, codimension: int | None = None) -> tuple[int, ...]:
        """Reachability cofaces, including self when codimension is None or zero.

        This implementation scans the table; the word trie is not a coface index.
        """
        codim = -1 if codimension is None else _integer(codimension, 'codimension')
        return tuple(_native.qft_cofaces(self._handle, self._id(cell_id), codim))

    def find_by_source(self, simplex: Iterable[int]) -> int | None:
        """Find a surviving cell by its original support; not by quotient vertices.

        Returns None for an absorbed simplex (including absorbed vertices).
        Use vertex_image for the current image of any original vertex.
        """
        result = _native.qft_find(self._handle, _labels(simplex))
        return None if result < 0 else result

    def vertex_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Read-only arrays (original vertex labels, current zero-cell IDs)."""
        return self._arrays()[8:10]

    def vertex_image(self, vertex: int) -> int:
        v = _integer(vertex, 'vertex')
        labels, images = self.vertex_map()
        i = int(np.searchsorted(labels, v))
        if i >= len(labels) or int(labels[i]) != v:
            raise KeyError(f'unknown source vertex {v}')
        return int(images[i])

    def word(self, cell_id: int) -> tuple[int, ...]:
        """Sorted unique vertex images, padded with their maximum to dim+1."""
        return tuple(_native.qft_word(self._handle, self._id(cell_id)))

    def lookup_word(self, word: Iterable[int]) -> tuple[int, ...]:
        """Return ALL cell IDs in the terminal bucket, never just one representative."""
        if not self.has_word_index:
            raise RuntimeError('word index disabled; use Q.with_word_index(True)')
        return tuple(_native.qft_lookup(self._handle, _labels(word, name='word')))

    def with_word_index(self, enabled: bool = True) -> QFTree:
        if not isinstance(enabled, bool):
            raise TypeError('enabled must be bool')
        return self._wrap(_native.qft_index(self._handle, enabled))

    def validate(self) -> bool:
        """Validate dimensions, local supports, constants and all codim-2 overlaps."""
        return _native.qft_validate(self._handle)

    def is_strictly_graded(self) -> bool:
        """Component-resolved shallowness, not an assumption inferred from a cap."""
        return _native.qft_graded(self._handle)

    def is_regular(self) -> bool:
        """Component-fullness criterion in this source-collapse regime."""
        return _native.qft_regular(self._handle)

    def collapse(self, cells: Iterable[int], *, close: bool = False,
                 return_map: bool = False):
        """Collapse each connected component of a QF cell subcomplex separately.

        The selection must be closed under faces; close=True adds the closure
        first. Returns a new table; the old one is unchanged. The whole table is
        rebuilt (for in-place local collapse use qfnext.EditableQF). Cell IDs may
        change; return_map=True returns (new_tree, QFMap).
        """
        if not isinstance(close, bool) or not isinstance(return_map, bool):
            raise TypeError('close and return_map must be bool')
        h, image_buffer = _native.qft_collapse(self._handle, _labels(cells, name='cell IDs'), close)
        result = self._wrap(h)
        if return_map:
            return result, QFMap(self, result, np.frombuffer(image_buffer, np.uint32))
        return result

    def source_pair(self):
        """Recover the exact K,A snapshot only when keep_source=True was used."""
        from . import FlagComplex
        if not self.has_source_archive:
            raise RuntimeError('source archive not retained; construct with keep_source=True')
        kh, mb = _native.qft_archive(self._handle)
        k = FlagComplex._wrap(kh, max(DEFAULT_MAX_SIMPLICES, self.source_num_simplices))
        return k, k.subcomplex(np.frombuffer(mb, np.uint8))

    def memory_breakdown(self) -> dict[str, int]:
        """Payload counts, not RSS or complete allocator/live-heap measurements."""
        return dict(topology_payload_bytes=self._info[7],
                    index_payload_bytes=self._info[8],
                    source_archive_payload_bytes=self._info[9],
                    trie_nodes=self._info[10])

    def nbytes(self) -> int:
        return sum(self._info[7:10])

    # Optional conveniences. These import the homology consumer only on demand.
    def chain_complex(self, *, relative: bool = False, homology_coeff_field: int = 2):
        from .homology import chain_complex
        return chain_complex(self, relative=relative, homology_coeff_field=homology_coeff_field)

    def betti_numbers(self, homology_coeff_field: int = 2, *, relative: bool = False) -> list[int]:
        from .homology import betti_numbers
        return betti_numbers(self, homology_coeff_field=homology_coeff_field, relative=relative)

    def boundary_matrix(self, dimension: int, *, relative: bool = False):
        from .homology import boundary_matrix
        return boundary_matrix(self, dimension, relative=relative)

    def visualize_dag(self, filename=None, **kwargs):
        """Graphviz of local facet incidences; see qfcore.visualize.visualize_dag."""
        from .visualize import visualize_dag
        return visualize_dag(self, filename, **kwargs)

    def __repr__(self):
        return (f'QFTree(cells={len(self)}, dimension={self.dimension()}, components={self.n_components}, '
                f'word_index={self.has_word_index}, source_archive={self.has_source_archive})')


@dataclass(frozen=True)
class QFMap:
    """Source-labelled pointed face morphism; absorbed cells map to points.

    Domain and codomain are kept to interpret IDs. On chains, a
    positive-dimensional absorbed generator maps to zero (not to a generator
    of a different degree).
    """
    domain: QFTree
    codomain: QFTree
    image: np.ndarray

    def __post_init__(self):
        # Copy even a read-only NumPy view: another alias might still mutate its
        # base array. The bytes-backed result cannot be made writable.
        arr = _labels(self.image, name='cell images')
        arr = np.frombuffer(arr.tobytes(), np.uint32)
        object.__setattr__(self, 'image', arr)
        if not self.validate():
            raise ValueError('not a compatible pointed face morphism')

    def __call__(self, cell_id: int) -> int:
        return int(self.image[self.domain._id(cell_id)])

    def validate(self) -> bool:
        return _native.qft_map_valid(self.domain._handle, self.codomain._handle, self.image)

    def then(self, following: QFMap) -> QFMap:
        if self.codomain is not following.domain:
            raise ValueError('composition requires the same intermediate QFTree snapshot')
        return QFMap(self.domain, following.codomain, following.image[self.image])

    def chain_images(self, dimension: int, *, relative: bool = False) -> tuple[int, ...]:
        """Target basis index for each source generator; -1 means zero."""
        return tuple(_native.qft_chain_image(self.domain._handle, self.codomain._handle, self.image,
                                            _integer(dimension, 'dimension'), bool(relative)))
