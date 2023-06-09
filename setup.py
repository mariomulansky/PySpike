""" setup.py

to compile cython files:
python setup.py build_ext --inplace


Copyright 2014-2017, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""
from setuptools import setup, find_packages
from distutils.extension import Extension
import os.path

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True


class numpy_include(object):
    """Defers import of numpy until install_requires is through"""
    def __str__(self):
        import numpy
        return numpy.get_include()


if os.path.isfile("pyspike/cython/cython_add.c") and \
   os.path.isfile("pyspike/cython/cython_get_tau.c") and \
   os.path.isfile("pyspike/cython/cython_profiles.c") and \
   os.path.isfile("pyspike/cython/cython_distances.c") and \
   os.path.isfile("pyspike/cython/cython_directionality.c") and \
   os.path.isfile("pyspike/cython/cython_simulated_annealing.c"):
    use_c = True
else:
    use_c = False

if not use_cython and not use_c:
    print('Cython not installed. Programs will be slow.')
    # Ans = input('Abort? (Y/N)\n')
    # if len(Ans)>0 and (Ans[0]=='Y' or Ans[0]=='y'):
    #     print("\nAborting\n")
    #     raise RuntimeError('User termination')

cmdclass = {}
ext_modules = []

if use_cython:  # Cython is available, compile .pyx -> .c
    ext_modules += [
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
    ]
    cmdclass.update({'build_ext': build_ext})
elif use_c:  # c files are there, compile to binaries
    ext_modules += [
        Extension("pyspike.cython.cython_add",
                  ["pyspike/cython/cython_add.c"]),
        Extension("pyspike.cython.cython_get_tau",
                  ["pyspike/cython/cython_get_tau.c"]),
        Extension("pyspike.cython.cython_profiles",
                  ["pyspike/cython/cython_profiles.c"]),
        Extension("pyspike.cython.cython_distances",
                  ["pyspike/cython/cython_distances.c"]),
        Extension("pyspike.cython.cython_directionality",
                  ["pyspike/cython/cython_directionality.c"]),
        Extension("pyspike.cython.cython_simulated_annealing",
                  ["pyspike/cython/cython_simulated_annealing.c"])
    ]
# neither cython nor c files available -> automatic fall-back to python backend

setup(
    name='pyspike',
    packages=find_packages(exclude=['doc']),
    version='0.8.0',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[numpy_include()],
    description='A Python library for the numerical analysis of spike\
train similarity',
    author='Mario Mulansky',
    author_email='mario.mulansky@gmx.net',
    license='BSD',
    url='https://github.com/mariomulansky/PySpike',
    install_requires=['numpy'],
    keywords=['data analysis', 'spike', 'neuroscience'],  # arbitrary keywords
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Information Analysis',

        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    package_data={
        'pyspike': ['cython/cython_add.c', 
                    'cython/cython_profiles.c',
                    'cython/cython_get_tau.c',
                    'cython/cython_distances.c',
                    'cython/cython_directionality.c',
                    'cython/cython_simulated_annealing.c'],
        'test': ['Spike_testdata.txt']
    }
)
