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

"""
Veusz distutils setup script
see the file INSTALL for details on how to install Veusz
"""

import glob
import os.path
import sys
import numpy
from distutils.command.install_data import install_data
import pyqtdistutils

# try to get py2app if it exists
try:
    import py2app
    from setuptools import setup, Extension
    py2app = True
except ImportError:
    from distutils.core import setup, Extension
    py2app = False

# get version
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
publication-ready Postscript, PDF and SVG output. It features GUI,
command-line, and scripting interfaces. Graphs are constructed from
"widgets", allowing complex layouts to be designed. Veusz supports
plotting functions, data with errors, keys, labels, stacked plots,
multiple plots, and fitting data.'''

if py2app and sys.platform == 'darwin':
    # extra arguments for mac py2app to associate files
    plist = {
        'CFBundleName': 'Veusz',
        'CFBundleShortVersionString': version,
        'CFBundleIdentifier': 'org.python.veusz',
        'CFBundleDocumentTypes': [{
                'CFBundleTypeExtensions': ['vsz'],
                'CFBundleTypeName': 'Veusz document',
                'CFBundleTypeRole': 'Editor',
                }]
        }
    
    extraoptions = {
        'setup_requires': ['py2app'],
        'app': ['veusz_main.py'],
        'options': { 'py2app': {'argv_emulation': True,
                                'includes': ('veusz.helpers._nc_cntr',
                                             'veusz.helpers.qtloops'),
                                'plist': plist,
                                'iconfile': 'windows/icons/veusz.icns',
                                }
                     }
	}
else:
    # otherwise package scripts
    extraoptions = {
        'scripts': ['scripts/veusz', 'scripts/veusz_listen']
        }

def findData(dirname, extns):
    """Return tuple for directory name and list of file extensions for data."""
    files = []
    for extn in extns:
        files += glob.glob(os.path.join(dirname, '*.'+extn))
    files.sort()
    return ( os.path.join('veusz', dirname), files )

setup(name = 'veusz',
      version = version,
      description = 'A scientific plotting package',
      long_description = descr,
      author = 'Jeremy Sanders',
      author_email = 'jeremy@jeremysanders.net',
      url = 'http://home.gna.org/veusz/',
      license = 'GPL',
      classifiers = [ 'Programming Language :: Python',
                      'Development Status :: 5 - Production/Stable',
                      'Environment :: X11 Applications :: Qt',
                      'Intended Audience :: Science/Research',
                      'License :: OSI Approved :: '
                      'GNU General Public License (GPL)',
                      'Topic :: Scientific/Engineering :: Visualization' ],
      package_dir = { 'veusz': '',
                      'veusz.dialogs': 'dialogs',
                      'veusz.document': 'document',
                      'veusz.helpers': 'helpers',
                      'veusz.plugins': 'plugins',
                      'veusz.qtwidgets': 'qtwidgets',
                      'veusz.setting': 'setting',
                      'veusz.tests': 'tests',
                      'veusz.utils': 'utils',
                      'veusz.widgets': 'widgets',
                      'veusz.windows': 'windows',
                      },
      data_files = [ ('veusz', ['VERSION']),
                     findData('dialogs', ('ui',)),
                     findData('windows/icons', ('png', 'svg')),
                     findData('examples', ('vsz', 'py', 'csv', 'dat')),
                     ],
      packages = [ 'veusz',
                   'veusz.dialogs',
                   'veusz.document',
                   'veusz.helpers',
                   'veusz.plugins',
                   'veusz.qtwidgets',
                   'veusz.setting',
                   'veusz.utils',
                   'veusz.widgets',
                   'veusz.windows',
                   ],

      ext_modules = [
        # contour plotting library
        Extension('veusz.helpers._nc_cntr',
                  ['helpers/src/_nc_cntr.c'],
                  include_dirs=[numpy.get_include()]),

        # qt helper module
        Extension('veusz.helpers.qtloops',
                  ['helpers/src/qtloops.cpp',
                   'helpers/src/qtloops_helpers.cpp',
                   'helpers/src/polygonclip.cpp',
                   'helpers/src/polylineclip.cpp',
                   'helpers/src/beziers.cpp',
                   'helpers/src/beziers_qtwrap.cpp',
                   'helpers/src/recordpaintdevice.cpp',
                   'helpers/src/recordpaintengine.cpp',
                   'helpers/src/qtloops.sip'],
                  language="c++",
                  include_dirs=['/helpers/src',
                                numpy.get_include()],
                  ),
        ],
                                
      cmdclass = {'build_ext': pyqtdistutils.build_ext,
                  'install_data': smart_install_data },

      **extraoptions
      )
