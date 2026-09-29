"""Installation checks: versions, dependencies, provenance, licenses, examples."""
from pathlib import Path
import ast
import subprocess
import sys
from importlib.metadata import version
import qfcore
import qfnext
import qfops

ROOT = Path(__file__).resolve().parents[1]


def test_version_consistency():
    from qfops import _ops
    from qfnext import _native
    assert qfcore.__version__ == qfnext.__version__ == qfops.__version__ == version("qfcore")
    assert _ops.version == _native.version == qfcore.__version__


def test_core_never_imports_undeclared_packages():
    forbidden = {"psutil", "pandas", "matplotlib"}
    for package in ("qfcore", "qfops", "qfnext"):
        for path in (ROOT / "src" / package).glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    assert not {n.name.split(".")[0] for n in node.names} & forbidden
                elif isinstance(node, ast.ImportFrom) and node.module:
                    assert node.module.split(".")[0] not in forbidden


def test_installed_core_with_undeclared_imports_blocked():
    subprocess.run([sys.executable, str(ROOT / "tools/check_installation.py")],
                   check=True, cwd=ROOT.parent, capture_output=True, text=True)


def test_core_build_provenance_is_current():
    subprocess.run([sys.executable, str(ROOT / "tools/build_provenance.py"), "--check"],
                   check=True, capture_output=True, text=True)


def test_licenses_and_build_sources_are_present():
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")
    for path in ("vendor/GUDHI_LICENSE", "licenses/pybind11-LICENSE.txt", "licenses/Boost-LICENSE_1_0.txt"):
        assert (ROOT / path).is_file()
    for name in (ROOT / "vendor/CORE_HEADERS.txt").read_text().splitlines():
        assert (ROOT / "vendor/gudhi_include" / name).is_file()


def test_quotient_example():
    subprocess.run([sys.executable, str(ROOT / "examples/quotient.py")],
                   check=True, cwd=ROOT.parent, capture_output=True, text=True)
