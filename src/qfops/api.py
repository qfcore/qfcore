"""Convenience API for the shared native induced-map consumer (F2 only)."""
from __future__ import annotations
from dataclasses import dataclass
from qfcore import QFMap
from . import _ops as op


def freeze_matrices(matrices):
    """Each matrix is a tuple of sparse columns; entries are row indices."""
    return tuple(tuple(tuple(map(int,col)) for col in matrix) for matrix in matrices)

@dataclass(frozen=True)
class HomologyMapResult:
    """Coordinates over F2. Cycles are indexed in each degree's chain basis.

    matrices[d][j] lists nonzero row indices of column j of H_d(f).
    kernel_basis[d] uses source H coordinates; cokernel_basis[d] uses target H
    coordinates. source_cycles / target_cycles convert these back to chains.
    """
    source_betti: tuple[int, ...]
    target_betti: tuple[int, ...]
    ranks: tuple[int, ...]
    matrices: tuple
    kernel_basis: tuple
    cokernel_basis: tuple
    source_cycles: tuple
    target_cycles: tuple

    @property
    def kernel_dimensions(self):
        return tuple(len(b) for b in self.kernel_basis)

    @property
    def cokernel_dimensions(self):
        return tuple(len(b) for b in self.cokernel_basis)


def induced_homology_map(morphism: QFMap, *, fill_limit: int = 10_000_000,
                         verify: bool = True) -> HomologyMapResult:
    """Compute a new map, with representative cycles and kernel/cokernel bases.

    This convenience call rebuilds both homology bases each time.
    """
    if not isinstance(morphism,QFMap):raise TypeError('expected QFMap')
    if isinstance(fill_limit,bool) or not isinstance(fill_limit,int) or fill_limit<1:
        raise ValueError('fill_limit must be a positive integer')
    a,b=morphism.domain.chain_complex(),morphism.codomain.chain_complex()
    images=op.qf_images(morphism.domain._handle,morphism.codomain._handle,morphism.image.tolist())
    if verify and not op.chain_map_valid(a._handle,b._handle,images,fill_limit):
        raise ValueError('morphism does not linearize to a chain map')
    source,target=op.Homology(a._handle,fill_limit),op.Homology(b._handle,fill_limit)
    matrices=op.induced(source,target,images)
    invariant=op.map_invariants(matrices,target.betti(),fill_limit)
    return HomologyMapResult(tuple(source.betti()),tuple(target.betti()),
        tuple(int(v[0]) for v in invariant),freeze_matrices(matrices),
        freeze_matrices([v[1] for v in invariant]),freeze_matrices([v[2] for v in invariant]),
        freeze_matrices(source.cycles()),freeze_matrices(target.cycles()))
