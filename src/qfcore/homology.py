"""Optional consumers of QFTree: signed incidences and unfiltered F2 homology.

No topology is reconstructed from these outputs. The object is valid and can be
saved/traversed/further factored without importing this module.
"""
from __future__ import annotations
from . import _native, _integer, _f2, ChainComplex, BoundaryMatrix
from .qftree import QFTree


def _require_tree(q):
    if not isinstance(q, QFTree):
        raise TypeError('expected QFTree')


def chain_complex(q: QFTree, *, relative=False, homology_coeff_field=2) -> ChainComplex:
    """Absolute C_*(Q;F2) or relative C_*(Q,D;F2), D=component points."""
    _require_tree(q); _f2(homology_coeff_field)
    return ChainComplex(_native.qft_chain(q._handle, bool(relative)))


def betti_numbers(q: QFTree, *, relative=False, homology_coeff_field=2) -> list[int]:
    return chain_complex(q, relative=relative, homology_coeff_field=homology_coeff_field).betti_numbers()


def boundary_matrix(q: QFTree, dimension: int, *, relative=False) -> BoundaryMatrix:
    _require_tree(q)
    return BoundaryMatrix(_native.qft_boundary(q._handle, _integer(dimension, 'dimension'), bool(relative)))


def signed_boundary(q: QFTree, cell_id: int, *, relative=False) -> dict[int, int]:
    """Integer cellular incidence coefficients of one cell.

    Uses the ordered local facet occurrences with signs (-1)^i. Constant
    higher-dimensional facets contribute zero; endpoints stay in absolute C_0.
    This gives boundary coefficients only; there is no integral homology solver.
    """
    _require_tree(q)
    targets, coefficients = _native.qft_incidence(q._handle, q._id(cell_id), bool(relative))
    return dict(zip(targets, coefficients))
