from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'irregular_lasso',
    ext_modules = cythonize('irregular_lasso.pyx'),
    include_dirs=[numpy.get_include()]
    )
