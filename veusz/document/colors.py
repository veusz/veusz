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
from ..compat import crange

# match name of color
themecolor_re = re.compile(r'^theme([1-9][0-9]*)$')

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

class Colors(qt4.QObject):
    """Document colors."""

    sigColorsModified = qt4.pyqtSignal()

    def __init__(self):
        qt4.QObject.__init__(self)

        self.defaultcolors = [
            'foreground',
            'background',
            'white',
            'black',
            'red',
            'green',
            'blue',
            'cyan',
            'magenta',
            'yellow',
            'grey',
            'darkred',
            'darkgreen',
            'darkblue',
            'darkcyan',
            'darkmagenta'
            ]

        self.colors = {
            # line and background colors
            'foreground': '#000000',
            'background': '#ffffff',
        }

        self.themecolors = [
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

        themecolors = ['theme%i' % (i+1) for i in crange(len(self.themecolors))]
        self.allcolors = self.defaultcolors + themecolors

        self.model = ColorModel(self, self)

    def get(self, name):
        """Get QColor given name."""

        if name in self.colors:
            name = self.colors[name]

        # special colors themeXXX, where XXX is a number from 1
        m = themecolor_re.match(name)
        if m:
            idx = int(m.group(1))
            return self.getIndex(idx)

        # standard colors
        return makeColor(name)

    def getIndex(self, idx):
        """Get color by index given."""
        try:
            wrapidx = (idx-1) % len(self.themecolors)
        except ZeroDivisionError:
            return qt4.QColor()
        return makeColor(self.themecolors[wrapidx])

class ColorModel(qt4.QAbstractListModel):
    """This is a Qt model to get access to the complete list of colors."""

    def __init__(self, parent, colors):
        qt4.QAbstractListModel.__init__(self, parent)
        self.colors = colors
        self.iconcache = {}

    def rowCount(self, index):
        if index.isValid():
            return 0
        return len(self.colors.allcolors)

    def data(self, index, role):

        row = index.row()
        if row<0 or row>=len(self.colors.allcolors):
            return None

        color = self.colors.allcolors[index.row()]

        if role == qt4.Qt.DisplayRole or role == qt4.Qt.EditRole:
            return color
        elif role == qt4.Qt.DecorationRole:
            # icons are cached using rgba as index
            rgba = self.colors.get(color).rgba()
            if rgba not in self.iconcache:
                pixmap = qt4.QPixmap(12, 12)
                pixmap.fill(self.colors.get(color))
                icon = qt4.QIcon(pixmap)
                self.iconcache[rgba] = icon
            return self.iconcache[rgba]

        return None

    def flags(self, index):
        if not index.isValid():
            return qt4.Qt.ItemIsEnabled
        return (
            qt4.QAbstractListModel.flags(self, index) | qt4.Qt.ItemIsEditable)

    def setData(self, index, value, role):
        if role == qt4.Qt.EditRole or role == qt4.Qt.DisplayRole:
            row = index.row()
            if row>=0 and row<len(self.colors.allcolors):
                self.colors.allcolors[row] = value
                self.dataChanged.emit(index, index)
            return True
        return False

    def insertRows(self, row, count, parent):
        if count<1 or row<0 or row>len(self.colors.allcolors):
            return False

        self.beginInsertRows(qt4.QModelIndex(), row, row+count-1)
        self.colors.allcolors = (
            self.colors.allcolors[:row] + ['']*count +
            self.colors.allcolors[row:])
        self.endInsertRows()
        return True

    def removeRows(self, row, count, parent):
        if count<=0 or row<0 or (row+count)>len(self.colors.allcolors):
            return False

        self.beginRemoveRows(qt4.QModelIndex(), row, row+count-1)
        self.colors.allcolors = (
            self.colors.allcolors[:row] +
            self.colors.allcolors[row+count:])
        self.endRemoveRows()
        return True
