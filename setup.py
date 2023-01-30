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

# version is stored in a file
with open('veusz/resources/VERSION') as verf:
    version = verf.read().strip()

descr = '''Veusz is a 2D and 3D scientific plotting package, designed to
create publication-ready PDF and SVG output. It features GUI,
command-line, and scripting interfaces. Graphs are constructed from
"widgets", allowing complex layouts to be designed. Veusz supports
plotting functions, data with errors, keys, labels, stacked plots,
multiple plots, and fitting data.'''

setup(
    name = 'veusz',
    version = version,
    description = 'A scientific plotting package',
    long_description = descr,
    author = 'Jeremy Sanders',
    author_email = 'jeremy@jeremysanders.net',
    url = 'https://veusz.github.io/',
    license = 'GPLv2+',
    classifiers = [
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Development Status :: 5 - Production/Stable',
        'Environment :: X11 Applications :: Qt',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: '
        'GNU General Public License v2 or later (GPLv2+)',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    python_requires='>=3.3',

    include_package_data=True,
    packages = [
        'veusz',
        'veusz.dataimport',
        'veusz.datasets',
        'veusz.dialogs',
        'veusz.document',
        'veusz.helpers',
        'veusz.plugins',
        'veusz.qtwidgets',
        'veusz.setting',
        'veusz.utils',
        'veusz.widgets',
        'veusz.windows',

        # not real modules, but setuptools needs them like this
        'veusz.resources',
        'veusz.resources.examples',
        'veusz.resources.icons',
        'veusz.resources.ui',
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
    },

    # requires these modules to install
    install_requires = [
        'numpy', 'PyQt5'
    ],

    # optional requirements
    extras_require = {
        "optional": [
            'astropy', 'pyemf3', 'sampy', 'iminuit', 'h5py'
        ]
    },

    # GUI entry points
    entry_points = {
        'gui_scripts' : [
            'veusz = veusz.veusz_main:run',
        ]
    },
)
