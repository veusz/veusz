#    Copyright (C) 2017 Jeremy S. Sanders
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

"""Store document colors."""

from __future__ import division
import re

from .. import qtall as qt4

# match name of color
plotcolor_re = re.compile(r'^plot([1-9][0-9]*)$')

# match extended color
extendedcolor_re = re.compile('^#[0-9A-Fa-f]{8}$')

def makeColor(name):
    """Make a new color, allowing extended hex format with extra two digits."""
    m = extendedcolor_re.match(name)
    if m:
        col = qt4.QColor(name[:7])
        col.setAlpha( int(name[7:], 16) )
        return col
    else:
        return qt4.QColor(name)

class Colors:
    """Document colors."""

    def __init__(self):
        self.colors = {
            # line and background colors
            'foreground': '#000000',
            'background': '#ffffff',
        }

        self.plotcolors = [
            # default colors (colorbrewer sets 1 and 2)
            '#e41a1c',
            '#377eb8',
            '#4daf4a',
            '#984ea3',
            '#ff7f00',
            '#ffff33',
            '#a65628',
            '#f781bf',
            '#999999',
            '#66c2a5',
            '#fc8d62',
            '#8da0cb',
            '#e78ac3',
            '#a6d854',
            '#ffd92f',
            '#e5c494',
            '#b3b3b3',
            ]

    def get(self, name):
        """Get QColor given name."""

        if name in self.colors:
            name = self.colors[name]

        # special colors plotXXX, where XXX is a number from 1
        m = plotcolor_re.match(name)
        if m:
            idx = int(m.group(1))
            return self.getIndex(idx)

        # standard colors
        return makeColor(name)

    def getIndex(self, idx):
        """Get color by index given."""
        try:
            wrapidx = (idx-1) % len(self.plotcolors)
        except ZeroDivisionError:
            return qt4.QColor()
        return makeColor(self.plotcolors[wrapidx])
