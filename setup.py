""" setup.py

Handles the compilation of pyx source files

run as:
python setup.py build_ext --inplace

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""
from distutils.core import setup
import numpy

try:
    from Cython.Build import cythonize
    setup(
        ext_modules=cythonize("pyspike/*.pyx"),
        include_dirs=[numpy.get_include()]
    )
except ImportError:
    print("Error: Cython is not installed! You will only be able to use the \
much slower Python backend in PySpike.")
