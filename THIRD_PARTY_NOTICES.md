# Third-party notices

qfcore is distributed under the MIT License (`LICENSE`). It includes or is built
with the following third-party code.

- **GUDHI 3.13.0**, MIT License (`vendor/GUDHI_LICENSE`). The headers in
  `vendor/gudhi_include/` are copied from GUDHI with their copyright notices;
  the notice in each header applies to that file. qfcore uses the GUDHI
  SimplexTree, Persistence Matrix and streaming zigzag implementations.
- **pybind11**, BSD 3-Clause License (`licenses/pybind11-LICENSE.txt`). pybind11
  is a build dependency. Its header-only code is compiled into the extension
  modules of the binary wheels.
- **Boost**, Boost Software License 1.0 (`licenses/Boost-LICENSE_1_0.txt`).
  The Boost headers are needed to build from source and are not included in
  the source distribution. Header-only Boost code is compiled into the
  extension modules of the binary wheels.

NumPy is a runtime dependency; it is installed separately and covered by its
own license.
