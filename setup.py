"""setup.py

Copyright 2014-2017, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy

Options.docstrings = True
Options.annotate = False

ext_modules = [
    Extension("pyspike.cython.cython_add", ["pyspike/cython/cython_add.pyx"]),
    Extension("pyspike.cython.cython_get_tau", ["pyspike/cython/cython_get_tau.pyx"]),
    Extension("pyspike.cython.cython_profiles", ["pyspike/cython/cython_profiles.pyx"]),
    Extension(
        "pyspike.cython.cython_distances", ["pyspike/cython/cython_distances.pyx"]
    ),
    Extension(
        "pyspike.cython.cython_directionality",
        ["pyspike/cython/cython_directionality.pyx"],
    ),
    Extension(
        "pyspike.cython.cython_simulated_annealing",
        ["pyspike/cython/cython_simulated_annealing.pyx"],
    ),
]

setup(
    name="pyspike",
    ext_modules=cythonize(ext_modules),
    include_dirs=[numpy.get_include()],
)
