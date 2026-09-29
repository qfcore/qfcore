"""Editable cell complexes (EditableQF) and the algorithms that use them.

EditableQF is saved as .qfe: polygon attachments and stable IDs do not fit
the source-labelled .qft format used by qfcore.QFTree.
"""
from .api import EditableQF, CohomologyRing, CellMap, glue_points, ZigzagSession, map_zigzag
from .orbits import OrbitSpace, orbit_space
from qfcore._version import __version__
