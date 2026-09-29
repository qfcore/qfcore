"""Shared native-build configuration. This module imports no qfcore runtime code."""
from pathlib import Path
import ast
import os
import sys

ROOT = Path(__file__).resolve().parent


def package_version():
    tree = ast.parse((ROOT / "src/qfcore/_version.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("Missing qfcore version")


def pybind_include():
    try:
        import pybind11
        return pybind11.get_include()
    except ImportError:
        value = os.environ.get("PYBIND11_INCLUDE")
        if value and (Path(value) / "pybind11/pybind11.h").is_file():
            return value
        raise RuntimeError("Install pybind11, or set PYBIND11_INCLUDE to its header directory") from None


def boost_include_dirs():
    """Find Boost headers only while compiling, not while generating metadata."""
    if os.environ.get("QF_SKIP_BOOST_CHECK") == "1":
        return []
    candidates = [os.environ.get("BOOST_INCLUDE"), os.environ.get("BOOST_INCLUDEDIR")]
    if os.environ.get("BOOST_ROOT"):
        candidates += [str(Path(os.environ["BOOST_ROOT"]) / "include"), os.environ["BOOST_ROOT"]]
    for variable in ("CPLUS_INCLUDE_PATH", "CPATH"):
        candidates += os.environ.get(variable, "").split(os.pathsep)
    if os.environ.get("CONDA_PREFIX"):
        candidates.append(str(Path(os.environ["CONDA_PREFIX"]) / "include"))
    candidates += ["/opt/homebrew/include", "/usr/local/include", "/usr/include"]
    for directory in filter(None, candidates):
        header = Path(directory) / "boost/version.hpp"
        if header.is_file():
            return [] if directory == "/usr/include" else [directory]
    raise RuntimeError(
        "Boost headers were not found. Install libboost-dev (Debian/Ubuntu), "
        "boost-devel (Fedora), or boost (Homebrew), or set BOOST_INCLUDE to the "
        "directory containing boost/version.hpp. Binary wheels do not need Boost installed."
    )


def compile_args():
    return ["/O2", "/std:c++17", "/EHsc"] if sys.platform == "win32" else ["-O3", "-std=c++17", "-DNDEBUG"]
