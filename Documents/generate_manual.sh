#!/usr/bin/env bash
#    Copyright (C) 2009 Jeremy S. Sanders
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

# generate the manual and man pages

# DO NOT EDIT THE OUTPUT FILES!

set -ex

# Make the man pages
release=$(cat ../VERSION)
pod2man --release=${release} --center="Veusz"  veusz.pod > veusz.1
pod2man --release=${release} --center="Veusz"  veusz_listen.pod > veusz_listen.1

MANWIDTH=76 man ./veusz.1 > veusz.man.txt
MANWIDTH=76 man ./veusz_listen.1 > veusz_listen.man.txt

# Make the manual
pushd manual-source
make clean
make html
make latexpdf
popd
mkdir manual/pdf
mv manual/latex/veusz.pdf manual/pdf/
rm -rf manual/latex/ manual/doctrees/


