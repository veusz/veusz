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

import numpy
from setuptools import setup, Extension

# setuptools extension for building SIP/PyQt modules
from pyqt_setuptools import sip_build_ext

setup(
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
    },
)
