#!/usr/bin/env python
# Veusz distutils setup script

# $Id$

import glob
from setuptools import setup

version = open('VERSION').read().strip()

descr = '''Veusz is a scientific plotting package, designed to create
publication-ready Postscript output. It features GUI, command-line,
and scripting interfaces. Graphs are constructed from "widgets",
allowing complex layouts to be designed. Veusz supports plotting
functions, data with errors, keys, labels, stacked plots,
multiple plots, and fitting data.'''

setup(name = 'veusz',
      version = version,
      description = 'A scientific plotting package',
      long_description = descr,
      author = 'Jeremy Sanders',
      author_email = 'jeremy@jeremysanders.net',
      url = 'http://home.gna.org/veusz/',
      license = 'GPL',
      classifiers = ['Programming Language :: Python',
                     'Development Status :: 4 - Beta',
                     'Environment :: X11 Applications :: Qt',
                     'Intended Audiance :: Advanced End Users',
                     'Licence :: OSI Approved :: '
                     'GNU General Public License (GPL)',
                     'Topic :: Scientific/Engineering :: Visualization'],
      package_dir = { 'veusz': '',
                      'veusz.dialogs': 'dialogs',
                      'veusz.document': 'document',
                      'veusz.setting': 'setting',
                      'veusz.utils': 'utils',
                      'veusz.widgets': 'widgets',
                      'veusz.windows': 'windows' },
      package_data = { 'veusz.windows': ['icons/*.png'],
                       'veusz': ['VERSION', 'images/*.png'] },
      packages = ['veusz',
                  'veusz.dialogs',
                  'veusz.document',
                  'veusz.setting',
                  'veusz.utils',
                  'veusz.widgets',
                  'veusz.windows']
      )
