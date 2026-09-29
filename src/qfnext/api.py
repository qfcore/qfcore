"""Python interface; local topology, cohomology and streaming zigzag are in C++.

Every homology or cohomology result is a snapshot. Topology edits are local,
but rings and bases are recomputed from scratch. Saving, cloning, chain
assembly and basis construction take time proportional to the whole complex.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import os
import tempfile
import zipfile
import numpy as np
from qfops import _ops
from . import _native as n
from .presentations import GroupPresentation


def _int(x, name='ID'):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)):
        raise TypeError(f'{name} must be an integer')
    return int(x)


def _requested(cell_id):
    if cell_id is None:
        return -1
    value = _int(cell_id)
    if value < 0:
        raise ValueError('explicit cell IDs must be nonnegative')
    return value


class EditableQF:
    """Mutable native cells with stable IDs, ordered facets, and optional polygons."""
    def __init__(self, handle=None):
        self._handle = n.Editable() if handle is None else handle

    @classmethod
    def from_qft(cls, qft):
        from qfcore import QFTree
        if not isinstance(qft, QFTree):
            raise TypeError('expected qfcore.QFTree')
        return cls(n.Editable.from_qft(qft._handle))

    def __len__(self):
        return self._handle.size()

    @property
    def epoch(self):
        return self._handle.epoch

    def dimension(self):
        return self._handle.dimension()

    def cell_ids(self, dimension=None):
        return tuple(self._handle.ids(-1 if dimension is None else _int(dimension, 'dimension')))

    def cell(self, cell_id):
        return self._handle.cell(_int(cell_id))

    def add_vertex(self, *, cell_id=None):
        return self._handle.vertex(_requested(cell_id))

    def add_edge(self, tail, head, *, cell_id=None):
        return self.add_simplex(1, (head, tail), cell_id=cell_id)

    def add_simplex(self, dimension, facets, *, cell_id=None):
        """Insert an ordered simplex cell; facets may be constant point maps.

        Does not infer a simplex from a vertex set and does not merge parallel cells.
        """
        return self._handle.simplex(_int(dimension, 'dimension'), [_int(x) for x in facets],
                                    _requested(cell_id))

    def attach_disk(self, loop, *, basepoint=None, cell_id=None):
        """Attach a polygonal 2-cell along a closed signed edge path.

        loop is [(edge_id, +1/-1), ...]. The empty path is a constant attachment
        and requires a basepoint. Only 2-cells can be attached along paths.
        """
        steps = []
        for edge, sign in loop:
            edge, sign = _int(edge), _int(sign, 'orientation')
            if edge < 0 or sign not in (-1, 1):
                raise ValueError('expected nonnegative edge ID and sign +/-1')
            steps.append(sign * (edge + 1))
        if basepoint is None:
            if not steps:
                raise ValueError('constant attachment needs a basepoint')
            e = self.cell(abs(steps[0]) - 1)
            if e.dimension != 1:
                raise ValueError('path contains a nonedge')
            basepoint = e.facets[1 if steps[0] > 0 else 0]
        return self._handle.disk(steps, _int(basepoint),
                                 _requested(cell_id))

    def closure(self, cells):
        return tuple(self._handle.closure([_int(x) for x in cells]))

    def cofaces(self, cell_id):
        return tuple(self._handle.cofaces(_int(cell_id)))

    def delete(self, cell_id, *, cascade=False):
        return self._handle.erase(_int(cell_id), bool(cascade))

    def collapse(self, cells, *, close=True):
        """Local IN-PLACE collapse; receipt is a sparse map plus implicit identity.

        No old snapshot or homology basis is automatically retained. Take clone()
        explicitly before this operation when an induced map is needed.
        """
        return self._handle.collapse([_int(x) for x in cells], bool(close))

    def identify_vertices(self, pairs):
        return self._handle.identify_vertices([(_int(a), _int(b)) for a, b in pairs])

    def clone(self):
        """Full O(cells+incidences) snapshot, not copy-on-write."""
        return EditableQF(self._handle.clone())

    def validate(self):
        return self._handle.validate()

    def boundary(self, cell_id, *, signed=False):
        f = self._handle.signed_boundary if signed else self._handle.boundary
        return tuple(f(_int(cell_id)))

    def homology(self, *, max_dimension=None, fill_limit=10_000_000):
        degree = -1 if max_dimension is None else _int(max_dimension, 'max_dimension')
        chain, basis = self._handle.chain_data(degree)
        if not _ops.dd_zero(chain, fill_limit):
            raise ValueError('boundary squared is nonzero')
        h = _ops.Homology(chain, fill_limit)
        return HomologyResult(tuple(h.betti()), tuple(basis), h, chain)

    def cohomology(self, *, fill_limit=10_000_000, max_dimension=None):
        return CohomologyRing(self, fill_limit=fill_limit, max_dimension=max_dimension)

    def fundamental_group(self):
        """One finite presentation per connected component, from the 2-skeleton."""
        return tuple(GroupPresentation(p.root, tuple(p.generators),
                     tuple(tuple(w) for w in p.relators), tuple(p.tree_edges),
                     tuple(p.relator_cells)) for p in n.presentation(self._handle))

    def _arrays(self):
        ids = self.cell_ids()
        dims, kinds, bases, fs, fo, ws, wo = [], [], [], [], [0], [], [0]
        for i in ids:
            c = self.cell(i)
            dims.append(c.dimension); kinds.append(c.kind); bases.append(c.base)
            fs.extend(c.facets); fo.append(len(fs)); ws.extend(c.word); wo.append(len(ws))
        arr = {k: np.asarray(v, dtype='<i8') for k, v in {
            'ids': ids, 'dims': dims, 'kinds': kinds, 'bases': bases,
            'facets': fs, 'facet_offsets': fo, 'words': ws, 'word_offsets': wo,
            'format': [1], 'next_id': [self._handle.next_id]}.items()}
        digest = hashlib.sha256()
        for k in sorted(arr):
            digest.update(k.encode('ascii')); digest.update(arr[k].tobytes())
        arr['sha256'] = np.frombuffer(digest.digest(), dtype='u1')
        return arr

    def save(self, path):
        """Versioned numeric .qfe archive, SHA256 integrity, no pickle/no source K,A.

        Atomic rename, not guaranteed power-loss durability. O(N) serialization
        is not included in the locality claim.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as f:
                np.savez(f, **self._arrays())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return path

    @classmethod
    def load(cls, path, *, max_cells=5_000_000, max_bytes=2**31):
        with zipfile.ZipFile(path) as z:
            if len(z.infolist()) != 11 or sum(x.file_size for x in z.infolist()) > max_bytes:
                raise ValueError('invalid archive members or expanded size limit')
        with np.load(path, allow_pickle=False) as z:
            keys = {'ids','dims','kinds','bases','facets','facet_offsets','words','word_offsets',
                    'format','next_id','sha256'}
            if set(z.files) != keys:
                raise ValueError('unrecognized QFE fields')
            arr = {k: z[k] for k in keys}
        if any(v.ndim != 1 for v in arr.values()):
            raise ValueError('QFE arrays must be one-dimensional')
        if any(v.dtype != np.dtype('<i8') for k, v in arr.items() if k != 'sha256'):
            raise ValueError('unexpected QFE integer dtype')
        if arr['format'].tolist() != [1] or len(arr['next_id']) != 1:
            raise ValueError('unsupported QFE format')
        digest = hashlib.sha256()
        for k in sorted(keys - {'sha256'}):
            digest.update(k.encode('ascii')); digest.update(arr[k].tobytes())
        if arr['sha256'].dtype != np.dtype('u1') or digest.digest() != arr['sha256'].tobytes():
            raise ValueError('QFE integrity check failed')
        count = len(arr['ids'])
        if count > max_cells or any(len(arr[k]) != count for k in ('dims','kinds','bases')):
            raise ValueError('QFE cell count/shape limit')
        if len(set(map(int, arr['ids']))) != count:
            raise ValueError('duplicate cell IDs')
        for off, val in [('facet_offsets', 'facets'), ('word_offsets', 'words')]:
            o = arr[off]
            if len(o) != count + 1 or o[0] != 0 or o[-1] != len(arr[val]) or np.any(np.diff(o) < 0):
                raise ValueError('invalid QFE offsets')
        out = cls()
        for j in sorted(range(count), key=lambda j: (arr['dims'][j], arr['ids'][j])):
            f = arr['facets'][arr['facet_offsets'][j]:arr['facet_offsets'][j+1]].tolist()
            w = arr['words'][arr['word_offsets'][j]:arr['word_offsets'][j+1]].tolist()
            out._handle.insert_record(n.Cell(int(arr['ids'][j]), int(arr['dims'][j]),
                                            int(arr['kinds'][j]), f, int(arr['bases'][j]), w))
        out._handle.reserve_ids(int(arr['next_id'][0]))
        out.validate()
        return out

    def visualize_dag(self, path, *, max_cells=500, render=False):
        """Write every local facet/word occurrence, including parallel occurrences."""
        if len(self) > max_cells:
            raise ValueError('Graphviz cell limit exceeded; select a smaller example')
        path = Path(path).with_suffix('.dot')
        lines = ['digraph QF {', ' graph [rankdir=TB];', ' node [shape=box];']
        for i in self.cell_ids():
            c = self.cell(i)
            label = f'cell {i} | dim {c.dimension}' + (' | polygon' if c.kind == 2 else '')
            lines.append(f' c{i} [label="{label}"];')
            if c.kind == 2:
                for j, w in enumerate(c.word):
                    lines.append(f' c{i} -> c{abs(w)-1} [label="step {j}, sign {1 if w>0 else -1}"];')
                lines.append(f' c{i} -> c{c.base} [label="basepoint",style=dashed];')
            else:
                for j, f in enumerate(c.facets):
                    const = self.cell(f).dimension != c.dimension-1
                    lines.append(f' c{i} -> c{f} [label="d{j}",style={"dashed" if const else "solid"}];')
        lines.append('}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(lines)+'\n')
        if render:
            import subprocess
            subprocess.run(['dot','-Tsvg',str(path),'-o',str(path.with_suffix('.svg'))], check=True)
        return path


@dataclass
class HomologyResult:
    betti: tuple
    cell_basis: tuple
    _handle: object
    _chain: object

    @property
    def cycles(self):
        return tuple(tuple(tuple(self.cell_basis[d][i] for i in z) for z in degree)
                     for d, degree in enumerate(self._handle.cycles()))


class CellMap:
    """Cellular map with sparse changed images and implicit identity.

    Domain and target must be explicit frozen snapshots, not later-mutated objects.
    This class records epochs and rejects use after either object changes.
    """
    def __init__(self, source, target, image=None):
        self.source, self.target = source, target
        self.image = {} if image is None else {_int(a):_int(b) for a,b in image.items()}
        self._epochs = source.epoch, target.epoch

    def _check(self):
        if (self.source.epoch, self.target.epoch) != self._epochs:
            raise RuntimeError('map domain/target mutated; retain snapshots for a map')
        if not n.cellular_map_valid(self.source._handle,self.target._handle,self.image):
            raise ValueError('cell images violate ordered attaching data, even if dF=Fd')

    def chain_images(self, max_dimension=None):
        self._check()
        d = max(self.source.dimension(), self.target.dimension(), 0) if max_dimension is None else max_dimension
        return n.images(self.source._handle, self.target._handle, self.image, d)

    def homology(self, *, fill_limit=10_000_000):
        self._check()
        d = max(self.source.dimension(), self.target.dimension(), 0)
        a = self.source.homology(max_dimension=d, fill_limit=fill_limit)
        b = self.target.homology(max_dimension=d, fill_limit=fill_limit)
        f = self.chain_images(d)
        if not _ops.chain_map_valid(a._chain, b._chain, f, fill_limit):
            raise ValueError('cell images do not induce a chain map')
        matrices = _ops.induced(a._handle, b._handle, f)
        inv = _ops.map_invariants(matrices, b.betti, fill_limit)
        return {'source': a, 'target': b, 'matrices': matrices,
                'ranks': tuple(i[0] for i in inv), 'kernel_basis': tuple(i[1] for i in inv),
                'cokernel_basis': tuple(i[2] for i in inv)}


    def cohomology(self, *, verify_products=False, fill_limit=10_000_000):
        """Contravariant cohomology map, optionally checking all cup products.

        Uses the strict cellular-map check, which is stronger than dF = Fd. Both
        ring snapshots use the same maximum degree, including zero spaces.
        """
        self._check()
        degree=max(self.source.dimension(),self.target.dimension(),0)
        a=self.source.cohomology(max_dimension=degree,fill_limit=fill_limit)
        b=self.target.cohomology(max_dimension=degree,fill_limit=fill_limit)
        matrices=n.pullback(a._handle,b._handle,self.chain_images(degree))
        if verify_products:
            for p,i,q,j in b.table(positive_only=False):
                left=set()
                for k in b.cup(p,i,q,j):left.symmetric_difference_update(matrices[p+q][k])
                right=set()
                for x in matrices[p][i]:
                    for y in matrices[q][j]:right.symmetric_difference_update(a.cup(p,x,q,y))
                if left!=right:
                    raise AssertionError('cohomology map does not preserve cup products')
        return {'source_ring':a,'target_ring':b,'matrices':matrices,
                'products_verified':bool(verify_products)}


class CohomologyRing:
    """Cohomology and cup products over F2, immutable native snapshot.

    Alexander–Whitney on ordered simplex cells; a based polygon-word diagonal
    on attached 2-cells, including inverse-letter corrections. The ring is
    recomputed for each snapshot. Integer cup products are not implemented.
    """
    def __init__(self, space, *, fill_limit=10_000_000, max_dimension=None):
        self._handle = n.Cohomology(space._handle, _int(fill_limit,'fill_limit'),
                                   -1 if max_dimension is None else _int(max_dimension,'max_dimension'))
        self.betti = tuple(self._handle.betti())
        self.cell_basis = tuple(tuple(x) for x in self._handle.basis())
        self.cocycles = tuple(tuple(tuple(c) for c in d) for d in self._handle.cocycles())

    def cup(self, p, i, q, j):
        return tuple(self._handle.product(_int(p),_int(i),_int(q),_int(j)))

    def table(self, *, positive_only=True, max_products=100000):
        total = sum(a*b for p,a in enumerate(self.betti) for q,b in enumerate(self.betti)
                    if p+q < len(self.betti) and (not positive_only or (p and q)))
        if total > max_products:
            raise ValueError('cup table product limit; request selected products instead')
        return {(p,i,q,j): self.cup(p,i,q,j)
                for p,a in enumerate(self.betti) for q,b in enumerate(self.betti)
                if p+q < len(self.betti) and (not positive_only or (p and q))
                for i in range(a) for j in range(b)}

    def cochain_product(self, p, a, q, b):
        return tuple(self._handle.cochain_product(p,list(a),q,list(b)))


def glue_points(parts, identifications):
    """Disjoint union followed by specified point identifications.

    identifications = [(module_a, vertex_a, module_b, vertex_b), ...].
    Returns new space and maps from each input. Module IDs disambiguate origins.
    This is not arbitrary subcomplex/polygon-face gluing.
    """
    parts = tuple(parts)
    result = n.glue([p._handle for p in parts], [tuple(map(_int,t)) for t in identifications])
    out = EditableQF(result.object)
    return out, tuple(CellMap(p,out,image) for p,image in zip(parts,result.maps))


class ZigzagSession:
    """GUDHI streaming zigzag persistence over insertions and deletions of cells.

    Deletions are checked against the full incidence, before coefficients
    cancel. Collapse maps cannot be expressed as deletions; use map_zigzag for
    small diagrams of arbitrary maps. Time t labels the snapshot after edit t.
    """
    def __init__(self, space, *, clone=True):
        self.space = space.clone() if clone else space
        self._zz = n.Zigzag()
        self.time = 0
        for d in range(self.space.dimension()+1):
            for c in self.space.cell_ids(d):
                self._zz.insert(c,list(self.space.boundary(c)),d,0.)
        self._epoch = self.space.epoch

    def _check(self):
        if self.space.epoch != self._epoch:
            raise RuntimeError('topology changed outside the zigzag session')

    def _inserted(self, c):
        self.time += 1
        self._zz.insert(c,list(self.space.boundary(c)),self.space.cell(c).dimension,float(self.time))
        self._epoch = self.space.epoch
        return c

    def add_vertex(self, *, cell_id=None):
        self._check()
        return self._inserted(self.space.add_vertex(cell_id=cell_id))

    def add_edge(self, tail, head, *, cell_id=None):
        self._check()
        return self._inserted(self.space.add_edge(tail,head,cell_id=cell_id))

    def add_simplex(self, dimension, facets, *, cell_id=None):
        self._check()
        return self._inserted(self.space.add_simplex(dimension,facets,cell_id=cell_id))

    def attach_disk(self, loop, *, basepoint=None, cell_id=None):
        self._check()
        return self._inserted(self.space.attach_disk(loop,basepoint=basepoint,cell_id=cell_id))

    def delete(self, cell_id, *, cascade=False):
        self._check()
        order = self.space._handle.removal_order(_int(cell_id),bool(cascade))
        for c in order:
            self.space.delete(c)
            self.time += 1
            self._zz.remove(c,float(self.time))
        self._epoch = self.space.epoch
        return tuple(order)

    def barcode(self):
        self._check()
        return tuple((int(d),float(b),float(e)) for d,b,e in self._zz.bars())


def map_zigzag(dimensions, matrices, directions, *, max_states=64,
               max_total_dimension=512, fill_limit=1_000_000):
    """Exact reference decomposition of a single-degree finite zigzag module.

    directions[k] = +1 for V[k]->V[k+1], -1 for V[k]<-V[k+1].
    Matrices are sparse columns. Returns (birth, death_exclusive, multiplicity).
    A bar ending at len(dimensions) is still alive at the last observed step.
    This limit/colimit computation processes the whole diagram at once and is
    meant as a reference for small inputs; use ZigzagSession for large ones.
    """
    if len(dimensions) > max_states:
        raise ValueError('reference zigzag state limit')
    return tuple(tuple(x) for x in n.map_barcode(list(dimensions),list(matrices),
                 list(directions),fill_limit,max_total_dimension))
