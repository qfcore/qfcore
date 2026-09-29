"""Torus vs. S1 v S1 v S2: same boundaries, different cup products and pi_1; then a zigzag.

Both spaces have one vertex v, two loops a, b and one 2-cell. In the torus the
2-cell is attached along the word a b a^-1 b^-1; in the wedge it is attached by
a constant map. All cellular boundaries agree, but the cup product and the
fundamental group tell the spaces apart.

Output goes to examples/output/: a .qfe file and a Graphviz drawing for each
space (SVG if Graphviz is installed, DOT otherwise), controls.csv and zigzag.csv.
"""
from pathlib import Path
import csv
import shutil

from qfnext import EditableQF, ZigzagSession


def model(commutator):
    q = EditableQF()
    v = q.add_vertex()
    a, b = q.add_edge(v, v), q.add_edge(v, v)
    word = [(a, 1), (b, 1), (a, -1), (b, -1)] if commutator else []
    disk = q.attach_disk(word, basepoint=v)
    return q, (v, a, b, disk)


def main():
    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)

    torus, _ = model(True)
    wedge, _ = model(False)
    assert [torus.boundary(i, signed=True) for i in torus.cell_ids()] == [
        wedge.boundary(i, signed=True) for i in wedge.cell_ids()
    ]

    rows = []
    for name, q in [("torus", torus), ("wedge", wedge)]:
        q.save(out / f"{name}.qfe")
        q = EditableQF.load(out / f"{name}.qfe")
        ring = q.cohomology()
        group = q.fundamental_group()[0]
        rows.append(dict(
            space=name,
            betti=str(ring.betti),
            cup_xy_basis_indices=str(ring.cup(1, 0, 1, 1)),
            generator_edges=str(group.edge_generators),
            relators=str(group.relators),
        ))
        q.visualize_dag(out / name, render=shutil.which("dot") is not None)

    with (out / "controls.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Start from a circle: fill it with a disc, remove the disc, fill it again.
    circle = EditableQF()
    v = circle.add_vertex()
    a = circle.add_edge(v, v)
    zigzag = ZigzagSession(circle)
    disk = zigzag.attach_disk([(a, 1)], basepoint=v)
    zigzag.delete(disk)
    zigzag.attach_disk([(a, 1)], basepoint=v)

    with (out / "zigzag.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("dimension", "birth", "death"))
        writer.writerows(zigzag.barcode())

    for row in rows:
        print(row)
    print("zigzag:", zigzag.barcode())


if __name__ == "__main__":
    main()
