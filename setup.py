""" setup.py

to compile cython files:
python setup.py build_ext --inplace


Copyright 2014-2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""
from setuptools import setup, find_packages
from distutils.extension import Extension
import os.path
import numpy

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

if os.path.isfile("pyspike/cython/cython_add.c") and \
   os.path.isfile("pyspike/cython/cython_profiles.c") and \
   os.path.isfile("pyspike/cython/cython_distances.c"):
    use_c = True
else:
    use_c = False

cmdclass = {}
ext_modules = []

if use_cython:  # Cython is available, compile .pyx -> .c
    ext_modules += [
        Extension("pyspike.cython.cython_add",
                  ["pyspike/cython/cython_add.pyx"]),
        Extension("pyspike.cython.cython_profiles",
                  ["pyspike/cython/cython_profiles.pyx"]),
        Extension("pyspike.cython.cython_distances",
                  ["pyspike/cython/cython_distances.pyx"])
    ]
    cmdclass.update({'build_ext': build_ext})
elif use_c:  # c files are there, compile to binaries
    ext_modules += [
        Extension("pyspike.cython.cython_add",
                  ["pyspike/cython/cython_add.c"]),
        Extension("pyspike.cython.cython_profiles",
                  ["pyspike/cython/cython_profiles.c"]),
        Extension("pyspike.cython.cython_distances",
                  ["pyspike/cython/cython_distances.c"])
    ]
# neither cython nor c files available -> automatic fall-back to python backend

setup(
    name='pyspike',
    packages=find_packages(exclude=['doc']),
    version='0.4.0',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    description='A Python library for the numerical analysis of spike\
train similarity',
    author='Mario Mulansky',
    author_email='mario.mulanskygmx.net',
    license='BSD',
    url='https://github.com/mariomulansky/PySpike',
    install_requires=['numpy'],
    keywords=['data analysis', 'spike', 'neuroscience'],  # arbitrary keywords
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',

        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ]
)
