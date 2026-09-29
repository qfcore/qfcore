"""Build the three native extensions of qfcore."""
from pathlib import Path
import sys
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from _build_support import boost_include_dirs, compile_args, package_version, pybind_include

GUDHI = "vendor/gudhi_include"


class BuildExt(build_ext):
    def build_extensions(self):
        for extension in self.extensions:
            if GUDHI in extension.include_dirs:
                extension.include_dirs.extend(boost_include_dirs())
        super().build_extensions()


includes = [pybind_include(), GUDHI, "src/qfcore", "src/qfops", "src/qfnext"]
macros = [("QF_PACKAGE_VERSION", '"' + package_version() + '"')]
setup(
    cmdclass={"build_ext": BuildExt},
    ext_modules=[
        Extension("qfcore._native", ["src/qfcore/_native.cpp"], include_dirs=["src/qfcore"],
                  language="c++", extra_compile_args=compile_args()),
        Extension("qfops._ops", ["src/qfops/_ops.cpp"], include_dirs=includes,
                  language="c++", extra_compile_args=compile_args(), define_macros=macros),
        Extension("qfnext._native", ["src/qfnext/_native.cpp"], include_dirs=includes,
                  language="c++", extra_compile_args=compile_args(), define_macros=macros),
    ],
)
