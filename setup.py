""" setup.py

Handles the compilation of pyx source files

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

"""
import os
from distutils.core import setup
from Cython.Build import cythonize

import numpy.distutils.intelccompiler

setup(
    ext_modules = cythonize("pyspike/*.pyx")
)
