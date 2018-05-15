#!/usr/bin/python
# -*- coding: utf8 -*-
# Author: Shun Arahata
from distutils.core import setup
from Cython.Build import cythonize
import numpy

"""For compilation of irregular_lasso.pyx, Cython source code.

Usage:
    $ python setup.py build_ext --inplace
"""

setup(
    name = 'irregular_lasso',
    ext_modules = cythonize('irregular_lasso.pyx'),
    include_dirs=[numpy.get_include()]
    )
