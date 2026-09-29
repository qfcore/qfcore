# Contributing

Set up a development install and run the tests:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Changes to attaching maps, quotients, serialization or the algebra should come
with regression tests, preferably checked against a small independent
implementation. Topology storage must not call the algebraic code (homology,
cohomology, zigzag).

Before opening a pull request:

- Do not commit compiled extensions, build outputs or files written by the
  examples.
- If you changed native or Python sources, the build configuration or the
  vendored headers, run `python tools/build_provenance.py` and commit the
  updated `src/qfcore/_build_inputs.sha256`.
  `python tools/build_provenance.py --check` must pass.
- If you changed which GUDHI headers are included, update
  `vendor/CORE_HEADERS.txt` and keep the license notices.
