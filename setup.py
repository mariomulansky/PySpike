""" setup.py

to compile cython files:
python setup.py build_ext --inplace


Copyright 2014-2017, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import os.path


class numpy_include(os.PathLike):
    """Defers import of numpy until install_requires is through"""
    def __str__(self):
        import numpy
        return numpy.get_include()

    def __fspath__(self):
        return str(self)


ext_modules = cythonize([
    Extension("pyspike.cython.cython_add",
              ["pyspike/cython/cython_add.pyx"]),
    Extension("pyspike.cython.cython_get_tau",
              ["pyspike/cython/cython_get_tau.pyx"]),
    Extension("pyspike.cython.cython_profiles",
              ["pyspike/cython/cython_profiles.pyx"]),
    Extension("pyspike.cython.cython_distances",
              ["pyspike/cython/cython_distances.pyx"]),
    Extension("pyspike.cython.cython_directionality",
              ["pyspike/cython/cython_directionality.pyx"]),
    Extension("pyspike.cython.cython_simulated_annealing",
              ["pyspike/cython/cython_simulated_annealing.pyx"])
])


setup(
    name='pyspike',
    packages=find_packages(exclude=['doc', 'test*']),
    version='0.8.0',
    ext_modules=ext_modules,
    include_dirs=[numpy_include()],
    description='A Python library for the numerical analysis of spike\
train similarity',
    author='Mario Mulansky',
    author_email='mario.mulansky@gmx.net',
    license='BSD',
    url='https://github.com/mariomulansky/PySpike',
    install_requires=['numpy'],
    keywords=['data analysis', 'spike', 'neuroscience'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    package_data={
        'test': ['Spike_testdata.txt']
    }
)