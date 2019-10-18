#!/usr/bin/env python
from setuptools import setup

__version__ = '0.1'

setup(name = 'starFS',
      version = __version__,
      python_requires='>3.5.2',
      description = 'a data-driven approach to identifying the Star Forming Sequence',
      requires = ['numpy', 'scipy', 'pytest'],
      install_requires = ['numpy', 'scipy', 'pytest', 'sklearn'],
      provides = ['starfs'],
      packages = ['starfs']
      )
