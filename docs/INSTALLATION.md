# Installation

## From PyPI

```bash
python -m pip install qfcore
```

Wheels are built for CPython 3.10–3.14 on Linux (x86_64 and aarch64, manylinux)
and macOS (Intel and Apple Silicon). Windows, musl Linux, free-threaded CPython
and PyPy are not supported release targets; on Windows, use WSL2. On other
platforms pip builds from the sdist, as described below.

## From source

Requirements: CPython 3.10 or newer, a C++17 compiler, the Python development
headers and the Boost headers (Boost 1.71 or newer).

- Debian/Ubuntu: `sudo apt-get install build-essential python3-dev libboost-dev`
- Fedora: `sudo dnf install gcc-c++ python3-devel boost-devel`
- macOS: `xcode-select --install` and `brew install boost`

Then, from a checkout:

```bash
python -m pip install .              # regular install
python -m pip install -e ".[test]"   # development install with pytest
python tools/check_installation.py   # optional sanity check
```

The build looks for `boost/version.hpp` in `BOOST_INCLUDE`, `BOOST_INCLUDEDIR`,
`BOOST_ROOT`, `CPLUS_INCLUDE_PATH`, `CPATH`, `$CONDA_PREFIX/include`, and the
usual system and Homebrew prefixes. Set `BOOST_INCLUDE` if Boost is installed
somewhere else.

pip installs the build dependencies, setuptools and pybind11, in an isolated
environment. NumPy is needed at run time only. To build with
`--no-build-isolation`, install setuptools and pybind11 first. On an offline
machine that has the pybind11 headers but not the Python package, set
`PYBIND11_INCLUDE` to the header directory.

Compiled extensions are specific to the Python version and architecture they
were built for.

## Optional dependencies

- The Python `gudhi` package is used only when you ask for it:
  `FlagComplex.from_graph(..., use_gudhi=True)`,
  `FlagComplex.from_points(..., use_gudhi=True)` and
  `FlagComplex.to_simplex_tree()`. The native homology and zigzag code uses the
  bundled GUDHI headers and does not need it.
- Graphviz (the `dot` executable) is needed only to render images with
  `visualize_dag(..., render=True)`. Writing DOT files works without it.
- SciPy is needed only for `BoundaryMatrix.to_scipy()`.

## Local wheels

A wheel built locally with `python -m build --wheel` on Linux is tagged
`linux_x86_64` (or similar). It works on the machine that built it but is not a
portable manylinux wheel. Wheels for PyPI are built and repaired by the release
workflow with cibuildwheel; see `RELEASING.md`.
