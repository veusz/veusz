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
from distutils.command.install import install as orig_install
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

class install(orig_install):
    user_options = orig_install.user_options + [
        # tell veusz where to install its data files
        ('veusz-resource-dir=', None,
         'override veusz resource directory location'),
        ('disable-install-examples', None,
         'do not install examples files'),
        ]
    boolean_options = orig_install.boolean_options + [
        'disable-install-examples']

    def initialize_options(self):
        orig_install.initialize_options(self)
        self.veusz_resource_dir = None
        self.disable_install_examples = False

# Pete Shinner's distutils data file fix... from distutils-sig
#  data installer with improved intelligence over distutils
#  data files are copied into the project directory instead
#  of willy-nilly
class smart_install_data(install_data):   
    def run(self):
        install_cmd = self.get_finalized_command('install')
        if install_cmd.veusz_resource_dir:
            # override location with veusz-resource-dir option
            self.install_dir = install_cmd.veusz_resource_dir
        else:
            # change self.install_dir to the library dir + veusz by default
            self.install_dir = os.path.join(install_cmd.install_lib, 'veusz')

        # disable examples install if requested
        if install_cmd.disable_install_examples:
            self.data_files = [f for f in self.data_files
                               if f[0][-8:] != 'examples']

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
    return (dirname, files)

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
      data_files = [ ('', ['VERSION']),
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
        # mathml widget
        Extension('veusz.helpers.qtmml',
                  ['helpers/src/qtmml/qtmmlwidget.cpp',
                   'helpers/src/qtmml/qtmml.sip'],
                  language="c++",
                  include_dirs=['/helpers/src/qtmml'],
                  ),

        # device to record paint commands
        Extension('veusz.helpers.recordpaint',
                  ['helpers/src/recordpaint/recordpaintdevice.cpp',
                   'helpers/src/recordpaint/recordpaintengine.cpp',
                   'helpers/src/recordpaint/recordpaint.sip'],
                  language="c++",
                  include_dirs=['/helpers/src/recordpaint'],
                  ),

        # contour plotting library
        Extension('veusz.helpers._nc_cntr',
                  ['helpers/src/nc_cntr/_nc_cntr.c'],
                  include_dirs=[numpy.get_include()]),

        # qt helper module
        Extension('veusz.helpers.qtloops',
                  ['helpers/src/qtloops/qtloops.cpp',
                   'helpers/src/qtloops/qtloops_helpers.cpp',
                   'helpers/src/qtloops/polygonclip.cpp',
                   'helpers/src/qtloops/polylineclip.cpp',
                   'helpers/src/qtloops/beziers.cpp',
                   'helpers/src/qtloops/beziers_qtwrap.cpp',
                   'helpers/src/qtloops/numpyfuncs.cpp',
                   'helpers/src/qtloops/qtloops.sip'],
                  language="c++",
                  include_dirs=['/helpers/src/qtloops',
                                numpy.get_include()],
                  ),
        ],
                                
      cmdclass = {'build_ext': pyqtdistutils.build_ext,
                  'install_data': smart_install_data,
                  'install': install},

      **extraoptions
      )
