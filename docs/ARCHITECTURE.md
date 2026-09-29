# Repository layout

## Packages

The `qfcore` distribution installs three import packages:

| Package | Contents | Native extension |
| --- | --- | --- |
| `qfcore` | `FlagComplex`, `QFTree`, `QFMap`, local quotients, `.qft` files, `visualize_dag` | `qfcore._native` |
| `qfops` | F₂ chain algebra and induced homology maps for QF-trees | `qfops._ops` |
| `qfnext` | `EditableQF`, `CellMap`, gluing, cup products, π₁ presentations, zigzag, orbit spaces, `.qfe` files | `qfnext._native` |

The only runtime dependency is NumPy. The sdist contains the C++ sources and the
GUDHI headers they include; the wheels contain the Python modules, the compiled
extensions and the license files.

## Tests, tools and data

- `tests/` contains the test suite. `tests/data/input_sphere.qft` is a small
  binary fixture for the `.qft` reader.
- `tools/build_provenance.py` writes and checks the build-input manifest
  `src/qfcore/_build_inputs.sha256`.
- `tools/check_distributions.py` checks the file lists, metadata and platform
  tags of built sdists and wheels.
- `tools/check_installation.py` runs a few basic operations on the installed
  package with psutil, pandas and matplotlib blocked from import.
- `ci/install_boost_headers.sh` installs a pinned Boost release for the wheel
  builds.

Compiled binaries, build outputs and files written by the examples are ignored
by Git.
