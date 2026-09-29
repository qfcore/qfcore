# Vendored GUDHI headers

`gudhi_include/` contains unmodified headers from GUDHI 3.13.0 with their
copyright and license notices. The GUDHI license is in `GUDHI_LICENSE`. These
headers are needed to compile the extensions; the Python `gudhi` package is a
separate, optional dependency.

`CORE_HEADERS.txt` lists the headers included, directly or transitively, by the
three `qfcore` extensions, counting all conditional includes. This directory and
the sdist contain exactly these headers.

After changing the list, rebuild and test on all release platforms and run
`python tools/build_provenance.py`.
