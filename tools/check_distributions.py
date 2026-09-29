"""Check the contents, metadata, wheel RECORD and PyPI platform tags of built archives."""
from __future__ import annotations
import argparse
import ast
import base64
import csv
from email.parser import BytesParser
import hashlib
import io
from pathlib import Path, PurePosixPath
import tarfile
import zipfile

PACKAGES = {"qfcore", "qfnext", "qfops"}
FORBIDDEN = {"experiments", "runs", "results", "logs", "__pycache__", ".git", ".github", ".venv"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_archive(path, *, for_pypi=False):
    path = Path(path)
    wheel = path.suffix == ".whl"
    if wheel:
        with zipfile.ZipFile(path) as archive:
            files = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
    else:
        with tarfile.open(path, "r:gz") as archive:
            files = {}
            for member in archive.getmembers():
                require(member.isdir() or member.isfile(), f"Unexpected archive member: {member.name}")
                if member.isfile():
                    pieces = PurePosixPath(member.name).parts
                    require(len(pieces) > 1, "Source files need a single enclosing directory")
                    files["/".join(pieces[1:])] = archive.extractfile(member).read()
    for name in files:
        parts = PurePosixPath(name).parts
        require(".." not in parts and not name.startswith("/"), f"Unsafe archive path: {name}")
        require(not FORBIDDEN.intersection(parts), f"Generated or development content in release: {name}")
        require(not name.endswith((".pyc", ".pyo")), f"Bytecode in release: {name}")
        if not wheel:
            require(not name.endswith((".so", ".pyd", ".o")), f"Binary in source release: {name}")
    metadata_paths = [n for n in files if n.endswith(".dist-info/METADATA")] if wheel else ["PKG-INFO"]
    require(len(metadata_paths) == 1 and metadata_paths[0] in files, "Missing/ambiguous project metadata")
    metadata = BytesParser().parsebytes(files[metadata_paths[0]])
    require(metadata["Name"] == "qfcore", "Wrong distribution name")
    reqs = metadata.get_all("Requires-Dist", [])
    require(any(r.startswith("numpy") for r in reqs), "Missing NumPy requirement")
    require(not any(any(n in r.lower() for n in ("psutil", "matplotlib", "pandas")) for r in reqs), "Unexpected dependency in metadata")
    prefix = "" if wheel else "src/"
    tree = ast.parse(files[prefix + "qfcore/_version.py"].decode())
    expected = ast.literal_eval(tree.body[0].value)
    require(metadata["Version"] == expected, "Python/metadata versions disagree")
    require(prefix + "qfcore/_build_inputs.sha256" in files, "Missing core build provenance")
    require(any(n.endswith("GUDHI_LICENSE") for n in files), "Missing GUDHI license")
    if wheel:
        roots = {PurePosixPath(n).parts[0] for n in files}
        require({r for r in roots if not r.endswith(".dist-info")} == PACKAGES, f"Unexpected wheel packages: {roots}")
        require(not any(n.endswith((".cpp", ".hpp", ".h", ".qft")) for n in files), "Source/test data leaked into wheel")
        extensions = [n for n in files if n.endswith((".so", ".pyd"))]
        require(len(extensions) == 3, f"Expected three native extensions, got {extensions}")
        record_name = next(n for n in files if n.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(io.StringIO(files[record_name].decode())))
        require({r[0] for r in rows} == set(files), "RECORD inventory mismatch")
        for name, digest, size in rows:
            if name == record_name:
                continue
            require(digest.startswith("sha256="), f"Missing SHA256 RECORD entry: {name}")
            computed = base64.urlsafe_b64encode(hashlib.sha256(files[name]).digest()).rstrip(b"=").decode()
            require(digest == "sha256=" + computed and int(size) == len(files[name]), f"RECORD mismatch: {name}")
        wheel_info = BytesParser().parsebytes(files[next(n for n in files if n.endswith(".dist-info/WHEEL"))])
        tags = wheel_info.get_all("Tag", [])
        require(tags, "No wheel tags")
        if for_pypi:
            require(all(not t.rsplit("-", 1)[-1].startswith("linux_") for t in tags),
                    "Local linux_* wheel is not a portable PyPI wheel; build with cibuildwheel")
    else:
        require("tests/test_distribution.py" in files, "Core sdist lost its regression suite")
        allowed_headers = set(files["vendor/CORE_HEADERS.txt"].decode().splitlines())
        actual_headers = {n.removeprefix("vendor/gudhi_include/") for n in files if n.startswith("vendor/gudhi_include/")}
        require(allowed_headers == actual_headers, "Core GUDHI include manifest differs from sdist contents")
    return {"file": path.name, "version": expected, "files": len(files), "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="+")
    parser.add_argument("--for-pypi", action="store_true")
    args = parser.parse_args()
    for path in args.archives:
        result = inspect_archive(path, for_pypi=args.for_pypi)
        print(f"OK {result['file']}: {result['files']} files, {result['bytes']} bytes")


if __name__ == "__main__":
    main()
