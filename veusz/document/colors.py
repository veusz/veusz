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

from .. import qtall as qt
from ..compat import crange

# match name of color
themecolor_re = re.compile(r'^theme([1-9][0-9]*)$')

# match extended color
extendedcolor_re = re.compile('^#[0-9A-Fa-f]{8}$')

def makeColor(name):
    """Make a new color, allowing extended hex format with extra two digits."""
    m = extendedcolor_re.match(name)
    if m:
        col = qt.QColor(name[:7])
        col.setAlpha( int(name[7:], 16) )
        return col
    else:
        return qt.QColor(name)

# Default color themes
colorthemes = {

    # backward compatibility with old documents
    'black': [
        'black',
    ],

    # black and colorbrewer sets 1 and 2 (minus yellow, as it doesn't show)
    'default1': [
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

    # rgb
    'rgb6': [
        '#ff0000',
        '#00ff00',
        '#0000ff',
        '#ffff00',
        '#00ffff',
        '#ff00ff',
    ],


    # maximum dissimilar colors
    # taken from http://stackoverflow.com/questions/33295120/how-to-generate-gif-256-colors-palette
    'max128': [
        "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
        ],

}
# most up-to-date theme
colorthemes['default-latest'] = colorthemes['default1']

class Colors(qt.QObject):
    """Document colors."""

    sigColorsModified = qt.pyqtSignal()

    def __init__(self):
        qt.QObject.__init__(self)

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

class ColorModel(qt.QAbstractListModel):
    """This is a Qt model to get access to the complete list of colors."""

    def __init__(self, parent, colors):
        qt.QAbstractListModel.__init__(self, parent)
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
            image = qt.QImage(xw, yw, qt.QImage.Format_RGB32)
            if color.lower() == 'auto':
                cnames = ['orange', 'skyblue', 'green']
            else:
                cnames = ['lightgrey', 'darkgrey']
            cols = [qt.QColor(c).rgba() for c in cnames]
            for x in crange(xw):
                for y in crange(yw):
                    idx = (x//4 + y//4) % len(cols)
                    image.setPixel(x, y, cols[idx])
            pixmap = qt.QPixmap.fromImage(image)
        else:
            # solid color
            pixmap = qt.QPixmap(xw, yw)
            pixmap.fill(qcolor)
        icon = qt.QIcon(pixmap)
        self.iconcache[qcolor.rgba()] = icon

    def data(self, index, role):
        row = index.row()
        if row<0 or row>=len(self.colorlist):
            return None

        color = self.colorlist[index.row()]
        if role == qt.Qt.DisplayRole or role == qt.Qt.EditRole:
            return color
        elif role == qt.Qt.DecorationRole:
            # icons are cached using rgba as index
            rgba = self.colors.get(color).rgba()
            if rgba not in self.iconcache:
                self.makeIcon(color)
            return self.iconcache[rgba]

        return None

    def flags(self, index):
        if not index.isValid():
            return qt.Qt.ItemIsEnabled
        return (
            qt.QAbstractListModel.flags(self, index) | qt.Qt.ItemIsEditable)

    def setData(self, index, value, role):
        if role == qt.Qt.EditRole or role == qt.Qt.DisplayRole:
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

        self.beginInsertRows(qt.QModelIndex(), row, row+count-1)
        self.colorlist = (
            self.colorlist[:row] + ['']*count +
            self.colorlist[row:])
        self.endInsertRows()
        return True

    def removeRows(self, row, count, parent):
        if count<=0 or row<0 or (row+count)>len(self.colorlist):
            return False

        self.beginRemoveRows(qt.QModelIndex(), row, row+count-1)
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
                self.beginRemoveRows(qt.QModelIndex(), i, i)
                del curcols[i]
                self.endRemoveRows()
            else:
                i += 1

        # add new entries
        for i, ncol in enumerate(newcols):
            if i == len(curcols) or curcols[i] != ncol:
                # maybe swap
                if i<len(curcols) and ncol in oldset:
                    self.beginRemoveRows(qt.QModelIndex(), i, i)
                    del curcols[i]
                    self.endRemoveRows()

                # simple add
                self.beginInsertRows(qt.QModelIndex(), i, i)
                curcols.insert(i, ncol)
                self.endInsertRows()

        # delete any extra rows
        if len(newcols) < len(curcols):
            self.beginRemoveRows(
                qt.QModelIndex(), len(newcols), len(curcols)-1)
            del curcols[len(newcols):]
            self.endRemoveRows()
