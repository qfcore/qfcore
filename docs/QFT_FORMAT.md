# QFTREE04 binary format, schema 1

A `.qft` file stores one QF-tree: a source-labelled quotient of a flag complex
by componentwise collapse. All integers are fixed-width little-endian; signed
integers use two's complement. The file consists only of the header, integer
vectors and a checksum described below. The implementation is in
`src/qfcore/qftree.hpp`, namespace `qfio`.

Each cell of the quotient is either a *genuine* cell, which comes from a simplex
of the original complex, or a *component point*, which replaces one connected
component of the collapsed subcomplex.

## Header (32 bytes)

| Offset | Type | Meaning |
|---:|---|---|
| 0 | 8 bytes | ASCII `QFTREE04` |
| 8 | uint32 | Schema version, currently 1 |
| 12 | uint32 | Feature flags: bit 0 word index, bit 1 source archive |
| 16 | uint64 | Number of simplices in the original source |
| 24 | int32 | Dimension of the original complex; -1 if it is empty |
| 28 | uint32 | Number of distinguished component points |

Files with any other feature bit set are rejected. The dimension at offset 24
is that of the original complex and can differ from the dimension of the
quotient.

## Vectors, in this order

Every vector is stored as a uint64 element count followed by its fixed-width
elements, with no alignment padding.

| Vector | Type | Length / content |
|---|---|---|
| dims | int32 | Dimension of each quotient cell |
| source_ids | int64 | Original simplex index; -1 for a component point |
| supports | uint32 | Concatenated ordered original supports of genuine cells |
| support_off | int64 | Cell count + 1 offsets; empty support for component points |
| facets | uint32 | Ordered target IDs, one for each local facet occurrence |
| facet_off | int64 | Cell count + 1 offsets; empty for all zero-cells |
| component_vertices | uint32 | Concatenated sorted original vertices per component |
| component_off | int64 | Component count + 1 offsets |

If feature bit 1 is set, append:

| Vector | Type | Content |
|---|---|---|
| original_verts | uint32 | Concatenated source simplex supports |
| original_off | int64 | Source simplex count + 1 offsets |
| original_mask | uint8 | One 0/1 membership flag in cumulative A per source simplex |

With the source archive, the complete original pair (K, A) can be
reconstructed. Without it, the file still describes the quotient completely,
including the original supports of the surviving cells, but the simplices that
were collapsed are lost.

## Footer

Append the uint64 FNV-1a hash of every preceding byte. Offset basis is
14695981039346656037, multiplier 1099511628211, arithmetic modulo 2^64. The footer
itself is not hashed. No trailing bytes are permitted.

The checksum detects accidental corruption. It is not a cryptographic hash:
anyone who edits a file can recompute it. The reader also validates lengths,
IDs, dimensions, supports, constant faces and the codimension-two face
identities, but it has not been hardened against deliberately malformed input.
Do not load `.qft` files from untrusted sources.

## Reconstruction and determinism

Dimension ranges, images of the original vertices and the word trie are
rebuilt from the stored topology on load. The word-index flag decides whether
the trie is rebuilt, unless `QFTree.load` overrides it. Genuine cells with the
same word go into the same trie bucket and are not merged. Cell IDs, including
those of component points, are the same after loading. The in-memory layout of
the trie is not part of the file format.

Saving, loading and saving again with the same options produces identical
bytes. The format does not store filtration values, chain complexes or ranks,
`QFMap` objects, permutation data or point coordinates. An F₂ chain
object cannot be converted into a QF-tree, because the attaching data is lost
when a complex is reduced to its chain complex.

## Reader limits and writing

By default `QFTree.load` accepts at most 10,000,000 cells and 2 GiB of file
data; both limits are parameters. The cell limit also applies to the source
archive. Cell IDs are uint32. Peak memory during loading can exceed the file
size, because validation and trie reconstruction allocate extra storage.

`QFTree.save` writes a temporary file in the target directory and atomically
renames it. The parent directory must exist. The file is not fsynced, so a
power failure right after saving can still lose it. The C++ function
`qf::save_qftree` writes directly to the given path and assumes that the table
is valid; tables created from Python always are, since they cannot be modified
from Python.
