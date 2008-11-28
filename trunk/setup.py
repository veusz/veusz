#!/usr/bin/env python
# Veusz distutils setup script
# see the file INSTALL for details on how to install Veusz

# $Id$

import sys
import os.path
import glob
import numpy

from distutils.core import setup, Extension
from distutils.command.install_data import install_data

# use py2exe if available
try:
    import py2exe
except ImportError:
    pass

version = open('VERSION').read().strip()

# Pete Shinner's distutils data file fix... from distutils-sig
#  data installer with improved intelligence over distutils
#  data files are copied into the project directory instead
#  of willy-nilly
class smart_install_data(install_data):   
    def run(self):
        # need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command('install')
        self.install_dir = getattr(install_cmd, 'install_lib')
        return install_data.run(self)

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
      cmdclass = { 'install_data': smart_install_data },
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
                      'veusz.helpers': 'helpers',
                      'veusz.setting': 'setting',
                      'veusz.utils': 'utils',
                      'veusz.widgets': 'widgets',
                      'veusz.windows': 'windows',
                      'veusz.tests': 'tests' },
      data_files = [ ('veusz', ['VERSION']),
                     ('veusz/dialogs', glob.glob('dialogs/*.ui')),
                     ('veusz/widgets/data', glob.glob('widgets/data/*.dat')),
                     ('veusz/windows/icons',
                      glob.glob('windows/icons/*.png') ) ],
      scripts = ['scripts/veusz', 'scripts/veusz_listen'],
      packages = ['veusz',
                  'veusz.dialogs',
                  'veusz.document',
                  'veusz.setting',
                  'veusz.utils',
                  'veusz.widgets',
                  'veusz.helpers',
                  'veusz.windows'],
      ext_modules = [ Extension('veusz.helpers._nc_cntr',
                                ['helpers/src/_nc_cntr.c'],
                                include_dirs=[numpy.get_include()]) ]
      )
