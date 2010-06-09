#!/usr/bin/env python

#    Copyright (C) 2008 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

# $Id$

"""
Veusz distutils setup script
see the file INSTALL for details on how to install Veusz
"""

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
                     'Intended Audience :: Science/Research',
                     'License :: OSI Approved :: '
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
                      'veusz.plugins': 'plugins',
                      'veusz.tests': 'tests' },
      data_files = [ ('veusz', ['VERSION']),
                     ('veusz/dialogs', glob.glob('dialogs/*.ui')),
                     ('veusz/widgets/data', glob.glob('widgets/data/*.dat')),
                     ('veusz/windows/icons',
                      glob.glob('windows/icons/*.png')+
                      glob.glob('windows/icons/*.svg')) ],
      scripts = ['scripts/veusz', 'scripts/veusz_listen'],
      packages = ['veusz',
                  'veusz.dialogs',
                  'veusz.document',
                  'veusz.setting',
                  'veusz.utils',
                  'veusz.widgets',
                  'veusz.helpers',
                  'veusz.windows',
                  'veusz.plugins',
                  ],
      ext_modules = [ Extension('veusz.helpers._nc_cntr',
                                ['helpers/src/_nc_cntr.c'],
                                include_dirs=[numpy.get_include()]) ]
      )
