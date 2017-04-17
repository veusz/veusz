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

# Default color themes
colorthemes = {

    # backward compatibility with old documents
    'black': [
        'black',
    ],

    # black and colorbrewer sets 1 and 2 (minus yellow, as it doesn't show)
    'default1': [
        '#000000',
        '#e41a1c',
        '#377eb8',
        '#4daf4a',
        '#984ea3',
        '#ff7f00',
        #'#ffff33',
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
    ],

    # colorbrewer set 1
    'colorbrewer1': [
        '#e41a1c',
        '#377eb8',
        '#4daf4a',
        '#984ea3',
        '#ff7f00',
        '#ffff33',
        '#a65628',
        '#f781bf',
        '#999999',
    ],

    # colorbrewer set 2
    'colorbrewer2': [
        '#66c2a5',
        '#fc8d62',
        '#8da0cb',
        '#e78ac3',
        '#a6d854',
        '#ffd92f',
        '#e5c494',
        '#b3b3b3',
    ],

}
# most up-to-date theme
colorthemes['default-latest'] = colorthemes['default1']

class Colors(qt4.QObject):
    """Document colors."""

    sigColorsModified = qt4.pyqtSignal()

    def __init__(self):
        qt4.QObject.__init__(self)

        self.defaultnames = [
            'auto',
            'foreground',
            'background',
            'transparent',
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

        # model for UI for choosing colors
        self.model = None

        # maximum index of colors in the current theme
        self.maxthemeidx = 1
        # current theme
        self.colortheme = 'black'
        # names of colors in theme
        self.themenames = []
        # setup default colors from theme
        self.wipe()

        # model for colors to use in qt widgets
        self.model = ColorModel(self, self)

    def wipe(self):
        self.colors = {
            # line and background colors
            'foreground': '#000000',
            'background': '#ffffff',
            'transparent': '#ffffff00',

            # this is a special color with a special (fake) value
            'auto': '#31323334',
        }
        # maximum color index used
        self.maxthemeidx = 1
        # colors defined by user in custom definition
        self.definednames = []
        # update the theme
        self.setColorTheme(self.colortheme)

    def addColor(self, color, val):
        """Add color to defined list."""
        self.colors[color] = val
        self.definednames.append(color)

        # keep track of maximum theme index
        m = themecolor_re.match(color)
        if m:
            self.maxthemeidx = max(self.maxthemeidx, int(m.group(1)))

    def setColorTheme(self, theme):
        """Set color theme to name given.
        """

        try:
            themecolors = colorthemes[theme]
        except KeyError:
            raise ValueError('Unknown color theme')
        self.colortheme = theme

        # delete old theme colors from dict
        defnset = set(self.definednames)
        for col in list(self.colors):
            if themecolor_re.match(col) and col not in defnset:
                del self.colors[col]

        # now add colors from theme (excluding defined colors)
        self.themenames = []
        for i, col in enumerate(themecolors):
            key = 'theme%i' % (i+1)
            if key not in defnset:
                self.colors[key] = col
                self.themenames.append(key)

        # keep track of maximum theme index
        self.maxthemeidx = len(themecolors)
        for color in self.definednames:
            m = themecolor_re.match(color)
            if m:
                self.maxthemeidx = max(self.maxthemeidx, int(m.group(1)))

        self.updateModel()

    def updateModel(self):
        """Update user color model. Call after using addColor."""
        if self.model is not None:
            self.model.updateColorList()

    def get(self, name):
        """Get QColor given name."""

        if name in self.colors:
            name = self.colors[name]

        # special colors themeXXX, where XXX is a number from 1
        # requires wrapping number according to the maximum definition
        m = themecolor_re.match(name)
        if m:
            idx = int(m.group(1))
            name = self.getIndex(idx)

        # standard colors
        return makeColor(name)

    def getIndex(self, idx):
        """Get name of color by index given."""
        try:
            # wrap index to maximum number of colors defined in theme
            wrapidx = (idx-1) % self.maxthemeidx + 1
            return self.colors['theme%i' % wrapidx]
        except (ZeroDivisionError, KeyError):
            return 'foreground'

class ColorModel(qt4.QAbstractListModel):
    """This is a Qt model to get access to the complete list of colors."""

    def __init__(self, parent, colors):
        qt4.QAbstractListModel.__init__(self, parent)
        self.colors = colors

        # cache of icons for colors indexed by rgba value
        self.iconcache = {}
        # list of extra colors added during operation by user
        self.xtranames = []

        # initialise list of colors
        self.colorlist = []
        self.updateColorList()

    def rowCount(self, index):
        if index.isValid():
            return 0
        return len(self.colorlist)

    def makeIcon(self, color):
        """Make icon for color in cache."""

        xw, yw = 16, 12
        qcolor = self.colors.get(color)
        if color.lower() in ('auto', 'transparent'):
            # make a checkerboard pattern for special colors
            image = qt4.QImage(xw, yw, qt4.QImage.Format_RGB32)
            if color.lower() == 'auto':
                cnames = ['orange', 'skyblue', 'green']
            else:
                cnames = ['lightgrey', 'darkgrey']
            cols = [qt4.QColor(c).rgba() for c in cnames]
            for x in crange(xw):
                for y in crange(yw):
                    idx = (x//4 + y//4) % len(cols)
                    image.setPixel(x, y, cols[idx])
            pixmap = qt4.QPixmap.fromImage(image)
        else:
            # solid color
            pixmap = qt4.QPixmap(xw, yw)
            pixmap.fill(qcolor)
        icon = qt4.QIcon(pixmap)
        self.iconcache[qcolor.rgba()] = icon

    def data(self, index, role):
        row = index.row()
        if row<0 or row>=len(self.colorlist):
            return None

        color = self.colorlist[index.row()]
        if role == qt4.Qt.DisplayRole or role == qt4.Qt.EditRole:
            return color
        elif role == qt4.Qt.DecorationRole:
            # icons are cached using rgba as index
            rgba = self.colors.get(color).rgba()
            if rgba not in self.iconcache:
                self.makeIcon(color)
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
            if row>=0 and row<len(self.colorlist):
                self.colorlist[row] = value

                # manually added colors for later
                if value not in self.xtranames and value[:5] != 'theme':
                    self.xtranames.append(value)

                self.dataChanged.emit(index, index)
            return True
        return False

    def insertRows(self, row, count, parent):
        if count<1 or row<0 or row>len(self.colorlist):
            return False

        self.beginInsertRows(qt4.QModelIndex(), row, row+count-1)
        self.colorlist = (
            self.colorlist[:row] + ['']*count +
            self.colorlist[row:])
        self.endInsertRows()
        return True

    def removeRows(self, row, count, parent):
        if count<=0 or row<0 or (row+count)>len(self.colorlist):
            return False

        self.beginRemoveRows(qt4.QModelIndex(), row, row+count-1)
        self.colorlist = (
            self.colorlist[:row] +
            self.colorlist[row+count:])
        self.endRemoveRows()
        return True

    def updateColorList(self):
        """Update internal set of colors with updated set from Colors."""

        curcols = self.colorlist
        oldset = set(curcols)

        # make list + set of new colors
        newcols = (
            self.colors.defaultnames + self.colors.definednames +
            self.colors.themenames + self.xtranames)
        newset = set(newcols)

        # prune any duplicates
        if len(newcols) > len(newset):
            seen = set()
            out = []
            for c in newcols:
                if c not in seen:
                    out.append(c)
                    seen.add(c)
            newcols = out

        # delete missing entries first
        i = 0
        while i < len(curcols):
            col = curcols[i]
            if col not in newset:
                self.beginRemoveRows(qt4.QModelIndex(), i, i)
                del curcols[i]
                self.endRemoveRows()
            else:
                i += 1

        # add new entries
        for i, ncol in enumerate(newcols):
            if i == len(curcols) or curcols[i] != ncol:
                # maybe swap
                if i<len(curcols) and ncol in oldset:
                    self.beginRemoveRows(qt4.QModelIndex(), i, i)
                    del curcols[i]
                    self.endRemoveRows()

                # simple add
                self.beginInsertRows(qt4.QModelIndex(), i, i)
                curcols.insert(i, ncol)
                self.endInsertRows()

        # delete any extra rows
        if len(newcols) < len(curcols):
            self.beginRemoveRows(
                qt4.QModelIndex(), len(newcols), len(curcols)-1)
            del curcols[len(newcols):]
            self.endRemoveRows()
