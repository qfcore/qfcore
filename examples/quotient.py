"""Build and round-trip a quotient without writing into the repository."""
from pathlib import Path
from tempfile import TemporaryDirectory
from qfcore import FlagComplex, QFTree


def main():
    complex_ = FlagComplex.from_graph(4, [(0, 1), (1, 2), (2, 3), (3, 0)], max_dim=1)
    quotient = complex_.quotient(complex_.induced_subcomplex([0, 1]))
    with TemporaryDirectory(prefix="qf-example-") as directory:
        path = Path(directory) / "circle.qft"
        quotient.save(path)
        restored = QFTree.load(path)
        assert restored.betti_numbers() == [1, 1]
    print("Quotient Betti numbers:", quotient.betti_numbers())


if __name__ == "__main__":
    main()
