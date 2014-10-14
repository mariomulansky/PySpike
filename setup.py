""" setup.py

Handles the compilation of pyx source files

Copyright 2014, Mario Mulansky <mario.mulansky@gmx.net>

"""
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("pyspike/*.pyx")
)
