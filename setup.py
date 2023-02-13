#!/usr/bin/env python3

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
Veusz setuputils script
see the file INSTALL.md for details on how to install Veusz
"""

import glob
import os.path
import numpy

from setuptools import setup, Extension
from setuptools.command.install import install as orig_install

# code taken from distutils for installing data that was removed in
# setuptools
from install_data import install_data

# setuptools extension for building SIP/PyQt modules
from pyqt_setuptools import sip_build_ext

class install(orig_install):
    user_options = orig_install.user_options + [
        # tell veusz where to install its data files
        (
            'veusz-resource-dir=', None,
            'override veusz resource directory location'
        ),
        (
            'disable-install-examples', None,
            'do not install examples files'
        ),
    ]
    boolean_options = orig_install.boolean_options + [
        'disable-install-examples',
    ]

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
            self.data_files = [
                f for f in self.data_files if f[0][-8:] != 'examples'
            ]

        return install_data.run(self)

def findData(dirname, extns):
    """Return tuple for directory name and list of file extensions for data."""
    files = []
    for extn in extns:
        files += glob.glob(os.path.join(dirname, '*.'+extn))
    files.sort()
    return (dirname, files)

setup(
    data_files = [
        ('', ['VERSION', 'AUTHORS', 'ChangeLog', 'COPYING']),
        findData('ui', ('ui',)),
        findData('icons', ('png', 'svg')),
        findData('examples', ('vsz', 'py', 'csv', 'dat')),
    ],

    ext_modules = [
        # threed support
        Extension(
            'veusz.helpers.threed',
            [
                'src/threed/camera.cpp',
                'src/threed/mmaths.cpp',
                'src/threed/objects.cpp',
                'src/threed/scene.cpp',
                'src/threed/fragment.cpp',
                'src/threed/numpy_helpers.cpp',
                'src/threed/clipcontainer.cpp',
                'src/threed/bsp.cpp',
                'src/threed/twod.cpp',
                'src/threed/threed.sip'
            ],
            language="c++",
            include_dirs=[
                'src/threed', numpy.get_include()
            ],
        ),

        # mathml widget
        Extension(
            'veusz.helpers.qtmml',
            [
                'src/qtmml/qtmmlwidget.cpp',
                'src/qtmml/qtmml.sip'
            ],
            language="c++",
            include_dirs=['src/qtmml'],
        ),

        # device to record paint commands
        Extension(
            'veusz.helpers.recordpaint',
            [
                'src/recordpaint/recordpaintdevice.cpp',
                'src/recordpaint/recordpaintengine.cpp',
                'src/recordpaint/recordpaint.sip'
            ],
            language="c++",
            include_dirs=['src/recordpaint'],
        ),

        # contour plotting library
        Extension(
            'veusz.helpers._nc_cntr',
            [
                'src/nc_cntr/_nc_cntr.c'
            ],
            include_dirs=[numpy.get_include()]
        ),

        # qt helper module
        Extension(
            'veusz.helpers.qtloops',
            [
                'src/qtloops/qtloops.cpp',
                'src/qtloops/qtloops_helpers.cpp',
                'src/qtloops/polygonclip.cpp',
                'src/qtloops/polylineclip.cpp',
                'src/qtloops/beziers.cpp',
                'src/qtloops/beziers_qtwrap.cpp',
                'src/qtloops/numpyfuncs.cpp',
                'src/qtloops/qtloops.sip'
            ],
            language="c++",
            include_dirs=[
                'src/qtloops',
                numpy.get_include()
            ],
        ),
    ],

    # new command options
    cmdclass = {
        'build_ext': sip_build_ext,
        'install_data': smart_install_data,
        'install': install
    },
)
