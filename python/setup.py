#!/usr/bin/env python

from distutils.core import setup


packages = [
    'nt', 
    'nt.dataio', 
    'nt.dataio.field_reader', 
    'nt.dataio.file_reader', 
    'nt.modules',
    'nt.models',
    'nt.criterion',
    'nt.optimizer',
    'nt.datasets',
    'nt.trainer',
]

setup(name='Neural Text',
      version='0.1',
      description='Neural text processing library using pytorch.',
      author='Chris Kedzie',
      author_email='kedzie@cs.columbia.edu',
      packages=packages
     )
