""" setup.py

Copyright 2014-2017, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""
from setuptools import setup, find_packages
from distutils.extension import Extension

# using a PEP-518 compatible build means these packages will be installed before
# setup begins
import numpy
from Cython.Distutils import build_ext


ext_modules = [
    Extension("pyspike.cython.cython_add", ["pyspike/cython/cython_add.pyx"]),
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
cmdclass = {"build_ext": build_ext}

setup(
    name="pyspike",
    packages=find_packages(exclude=["doc"]),
    version="0.7.0",
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    description="A Python library for the numerical analysis of spike train similarity",
    author="Mario Mulansky",
    author_email="mario.mulansky@gmx.net",
    license="BSD",
    url="https://github.com/mariomulansky/PySpike",
    install_requires=["numpy"],
    keywords=["data analysis", "spike", "neuroscience"],  # arbitrary keywords
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    package_data={
        "pyspike": [
            "cython/cython_add.c",
            "cython/cython_profiles.c",
            "cython/cython_distances.c",
            "cython/cython_directionality.c",
            "cython/cython_simulated_annealing.c",
        ],
        "test": ["Spike_testdata.txt"],
    },
)
