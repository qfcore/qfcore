"""Exercise the installed library with packages it does not depend on blocked from import."""
from __future__ import annotations
import argparse
import importlib.abc
import importlib.util
from importlib.metadata import version
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

UNDECLARED = {"psutil", "pandas", "matplotlib"}


class BlockUndeclared(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in UNDECLARED:
            raise ModuleNotFoundError("Undeclared dependency blocked: " + fullname)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-only", action="store_true", help="also require that these packages are not installed at all")
    args = parser.parse_args()
    if args.core_only:
        for name in sorted(UNDECLARED):
            if importlib.util.find_spec(name) is not None:
                raise RuntimeError(f"Not a core-only environment: {name} is installed")
    sys.meta_path.insert(0, BlockUndeclared())
    import qfcore
    import qfnext
    import qfops
    from qfnext import EditableQF, ZigzagSession
    from qfops.api import induced_homology_map
    assert qfcore.__version__ == qfnext.__version__ == qfops.__version__ == version("qfcore")
    assert qfnext._native.version == qfcore.__version__
    assert qfops._ops.version == qfcore.__version__
    assert qfcore.self_test(verbose=False)
    k = qfcore.FlagComplex.from_graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)], max_dim=1)
    a = k.induced_subcomplex([0, 1])
    q = k.quotient(a)
    assert q.betti_numbers() == [1, 1]
    assert k.local_quotient(a).compact().betti_numbers() == [1, 1]
    target, cell_map = q.collapse([], return_map=True)
    assert induced_homology_map(cell_map).ranks == (1, 1)
    with TemporaryDirectory(prefix="qf-install-") as directory:
        path = Path(directory) / "cycle.qft"
        q.save(path)
        assert qfcore.QFTree.load(path).betti_numbers() == [1, 1]
        editable = EditableQF()
        v = editable.add_vertex()
        x, y = editable.add_edge(v, v), editable.add_edge(v, v)
        editable.attach_disk([(x, 1), (y, 1), (x, -1), (y, -1)], basepoint=v)
        assert editable.homology().betti == (1, 2, 1)
        assert editable.cohomology().cup(1, 0, 1, 1)
        assert editable.fundamental_group()[0].relators
        editable.save(Path(directory) / "torus.qfe")
        assert EditableQF.load(Path(directory) / "torus.qfe").homology().betti == (1, 2, 1)
        circle = EditableQF()
        v = circle.add_vertex()
        edge = circle.add_edge(v, v)
        zigzag = ZigzagSession(circle)
        disk = zigzag.attach_disk([(edge, 1)], basepoint=v)
        zigzag.delete(disk)
        assert zigzag.barcode()
    assert not (UNDECLARED & set(sys.modules))
    print(f"qfcore {qfcore.__version__}: installed core API OK")


if __name__ == "__main__":
    main()
