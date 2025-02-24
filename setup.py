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

from setuptools import setup
from setuptools.command.install import install as orig_install

# code taken from distutils for installing data that was removed in
# setuptools
from install_data import install_data


class install(orig_install):
    user_options = orig_install.user_options + [
        # tell veusz where to install its data files
        ("veusz-resource-dir=", None, "override veusz resource directory location"),
        ("disable-install-examples", None, "do not install examples files"),
    ]
    boolean_options = orig_install.boolean_options + [
        "disable-install-examples",
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
        install_cmd = self.get_finalized_command("install")
        if install_cmd.veusz_resource_dir:
            # override location with veusz-resource-dir option
            self.install_dir = install_cmd.veusz_resource_dir
        else:
            # change self.install_dir to the library dir + veusz by default
            self.install_dir = os.path.join(install_cmd.install_lib, "veusz")

        # disable examples install if requested
        if install_cmd.disable_install_examples:
            self.data_files = [f for f in self.data_files if f[0][-8:] != "examples"]

        return install_data.run(self)


def findData(dirname, extns):
    """Return tuple for directory name and list of file extensions for data."""
    files = []
    for extn in extns:
        files += glob.glob(os.path.join(dirname, "*." + extn))
    files.sort()
    return (dirname, files)


setup(
    data_files=[
        ("", ["VERSION", "AUTHORS", "ChangeLog", "COPYING"]),
        findData("ui", ("ui",)),
        findData("icons", ("png", "svg")),
        findData("examples", ("vsz", "py", "csv", "dat")),
    ],
    # new command options
    cmdclass={"install_data": smart_install_data, "install": install},
)
