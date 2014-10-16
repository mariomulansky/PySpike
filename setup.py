""" setup.py

Handles the compilation of pyx source files

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("pyspike/*.pyx"),
    include_dirs=[numpy.get_include()]
)
