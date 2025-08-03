# utilfuncs.py
# utility functions

#    Copyright (C) 2003 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This file is part of Veusz.
#
#    Veusz is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    Veusz is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################

from collections import defaultdict
import codecs
import csv
import io
import math
import os
import os.path
import re
import string
import sys
import threading
import time

import numpy as N

from .. import qtall as qt

DEG2RAD = math.pi/180
RAD2DEG = 180/math.pi

class IgnoreException(Exception):
    """A special exception class to be ignored by the exception handler."""

class InvalidType(Exception):
    """Exception used when invalid values are used in settings."""

def _getVeuszDirectory():
    """Get resource and examples directories for Veusz."""

    if hasattr(sys, 'frozen'):
        # pyinstaller or similar things
        exedir = os.path.dirname(os.path.abspath(sys.executable))
        resdir = exedir
        for place in (
                os.path.join(exedir, '..', 'Resources'),
                os.path.join(exedir, '_internal'),
            ):
            if os.path.isfile(os.path.join(place, 'VERSION')):
                resdir = place

    else:
        # standard installation
        resdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        if os.path.exists( os.path.join(resdir, 'resources') ):
            # override data directory with symlink
            resdir = os.path.realpath( os.path.join(resdir, 'resources') )
        elif not os.path.exists(os.path.join(resdir, 'VERSION')):
            # running from the current direction
            resdir = os.path.join(resdir, '..')

    # override with VEUSZ_RESOURCE_DIR environment variable if necessary
    resdir = os.environ.get('VEUSZ_RESOURCE_DIR', resdir)

    # now get example directory (which may be a symlink)
    examplesdir = os.path.realpath( os.path.join(resdir, 'examples') )

    return resdir, examplesdir

# get resource and example directories
resourceDirectory, exampleDirectory = _getVeuszDirectory()

def getLicense():
    """Return license text."""
    try:
        with open(os.path.join(resourceDirectory, 'COPYING')) as f:
            text = f.read()
    except EnvironmentError:
        text = (
            'Could not open the license file.\n'
            'See license at http://www.gnu.org/licenses/gpl-2.0.html'
        )
    return text

id_re = re.compile('^[A-Za-z_][A-Za-z0-9_]*$')
def validPythonIdentifier(name):
    """Is this a valid python identifier?"""
    return id_re.match(name) is not None

def validateDatasetName(name):
    """Validate dataset name is okay.
    Dataset names can contain anything except back ticks!
    """
    return len(name.strip()) > 0 and name.find('`') == -1

def validateWidgetName(name):
    """Validate widget name is okay.
    Widget names are valid if no surrounding whitespace and do not contain /
    """
    return ( len(name) > 0 and name.strip() == name and name.find('/') == -1
             and name != '.' and name != '..' )

def cleanDatasetName(name):
    """Make string into a valid dataset name."""
    # replace backticks and get rid of whitespace at ends
    return name.replace('`', '_').strip()

def relpath(filename, dirname):
    """Make filename a relative filename relative to dirname."""

    # spit up paths into components
    filename = os.path.abspath(filename)
    fileparts = filename.split(os.path.sep)
    dirparts = os.path.abspath(dirname).split(os.path.sep)

    # if first non empty part is non equal, return original
    i = 0
    while not fileparts[i] and not dirparts[i]:
        i += 1
    if fileparts[i] != dirparts[i]:
        return filename

    # remove equal bits at start
    while fileparts and dirparts and fileparts[0] == dirparts[0]:
        fileparts.pop(0)
        dirparts.pop(0)

    # add on right number of .. to get back up
    fileparts = [os.path.pardir]*len([d for d in dirparts if d]) + fileparts

    # join parts back together
    return os.path.sep.join(fileparts)

def extendedColorFromQColor(col):
    """Make an extended color #RRGGBBAA or #RRGGBB string."""
    if col.alpha() == 255:
        return str(col.name())
    else:
        return '#%02x%02x%02x%02x' % (
            col.red(), col.green(), col.blue(), col.alpha())

def pixmapAsHtml(pix):
    """Get QPixmap as html image text."""
    ba = qt.QByteArray()
    buf = qt.QBuffer(ba)
    buf.open(qt.QIODevice.OpenModeFlag.WriteOnly)
    pix.toImage().save(buf, "PNG")
    b64 = bytes(buf.data().toBase64()).decode('ascii')
    return '<img src="data:image/png;base64,%s">' % b64

def pythonise(text):
    """Turn an expression of the form 'A b c d' into 'A(b,c,d)'.

    This is for 'pythonising' commands from a command-line interface
    to make it easier for users. We also have to take account of quotes
    and backslashes.
    """

    out = ''
    insingle = False    # in a single quote section
    indouble = False    # in a double quote section
    firstnonws = False  # have we reached first non WS char
    firstpart = True    # have we appended the first part of the expr
    lastbslash = False  # was the last character a back-slash

    # iterate over characters
    for c in text:

        # keep leading WS
        if c in string.whitespace and not firstnonws:
            out += c
            continue
        firstnonws = True

        # this character isn't escaped
        if not lastbslash:

            # quoted section
            if c == "'":
                insingle = not insingle
            elif c == '"':
                indouble = not indouble

            elif c == '\\':
                lastbslash = True
                continue

            # spacing between parts
            if c == ' ' and not insingle and not indouble:
                if firstpart:
                    out += '('
                    firstpart = False
                else:
                    out += ','
            else:
                out += c
        else:
            out += '\\' + c
            lastbslash = False

    # we're still in the first part
    if firstpart:
        out += '('

    # we can add a right bracket
    if not insingle and not indouble:
        out += ')'

    return out

def validLinePoints(x, y):
    """Take x and y points and split into sets of points which
    don't have invalid points.
    This is a generator.
    """
    xvalid = N.logical_not( N.isfinite(x) ).nonzero()[0]
    yvalid = N.logical_not( N.isfinite(y) ).nonzero()[0]
    invalid = N.concatenate((xvalid, yvalid))
    invalid.sort()
    last = 0
    for valid in invalid:
        if valid > last:
            yield x[last:valid], y[last:valid]
        last = valid + 1
    if last < x.shape[0]-1:
        yield x[last:], y[last:]

class NonBlockingReaderThread(threading.Thread):
    """A class to read blocking file objects and return the result.

    Usage:
     r = ReadThread(myfile)
     r.start()
     while True:
      newdata, done = r.getNewData()
      print newdata
      if done: break

    This is used mainly because windows doesn't properly support
    non-blocking pipes as files.

    If exiteof is True, then exit capturing when we can capture no
    more data.
    """

    def __init__(self, fileobject, exiteof=True):
        """Create the thread object."""
        threading.Thread.__init__(self)
        self.fileobject = fileobject
        self.lock = threading.Lock()
        self.data = ''
        self.done = False
        self.exiteof = exiteof

    def getNewData(self):
        """Get any data waiting to be read, and whether
        the reading is finished.

        Returns (data, done)
        """
        self.lock.acquire()
        data = self.data
        done = self.done
        self.data = ''
        self.lock.release()
        if isinstance(data, Exception):
            # if the reader errored somewhere
            raise data
        else:
            return data, done

    def run(self):
        """Do the reading from the file object."""

        while True:
            try:
                data = self.fileobject.readline()
            except Exception as e:
                # error in reading
                self.lock.acquire()
                self.data = e
                self.lock.release()
                break

            # no more data: end of file
            if len(data) == 0:
                if self.exiteof:
                    self.lock.acquire()
                    self.done = True
                    self.lock.release()
                    break
                else:
                    time.sleep(0.1)
            else:
                self.lock.acquire()
                self.data += data
                self.lock.release()

# standard python encodings
encodings = [
    'ascii', 'big5hkscs', 'big5', 'charmap', 'cp037', 'cp424',
    'cp437', 'cp500', 'cp737', 'cp775', 'cp850', 'cp852', 'cp855',
    'cp856', 'cp857', 'cp860', 'cp861', 'cp862', 'cp863', 'cp864',
    'cp865', 'cp866', 'cp869', 'cp874', 'cp875', 'cp932', 'cp949',
    'cp950', 'cp1006', 'cp1026', 'cp1140','cp1250', 'cp1251', 'cp1252',
    'cp1253', 'cp1254', 'cp1255', 'cp1256', 'cp1257', 'cp1258',
    'euc_jis_2004', 'euc_jisx0213', 'euc_jp', 'euc_kr', 'gb18030', 'gb2312',
    'gbk', 'hp_roman8', 'hz', 'iso2022_jp_1', 'iso2022_jp_2004',
    'iso2022_jp_2', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_jp',
    'iso2022_kr', 'iso8859_10', 'iso8859_11', 'iso8859_13', 'iso8859_14',
    'iso8859_15', 'iso8859_16', 'iso8859_1', 'iso8859_2', 'iso8859_3',
    'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7', 'iso8859_8',
    'iso8859_9', 'johab', 'koi8_r', 'koi8_u', 'latin_1', 'mac_arabic',
    'mac_centeuro', 'mac_croatian', 'mac_cyrillic', 'mac_farsi', 'mac_greek',
    'mac_iceland', 'mac_latin2', 'mac_romanian', 'mac_roman', 'mac_turkish',
    'ptcp154', 'shift_jis_2004', 'shift_jis', 'shift_jisx0213',
    'tis_620','utf_16_be', 'utf_16_le', 'utf_16',
    'utf_32_be', 'utf_32_le', 'utf_32', 'utf_7', 'utf_8', 'utf_8_sig'
]

def openEncoding(filename, encoding, mode='r'):
    """Convenience function for opening file with encoding given.

    If filename == '{clipboard}', then load the data from the clipboard
    instead.
    """

    # Windows programs sometimes use a BOM at the start of files,
    # causing problems reading a file. We force the below encoding to
    # skip the BOM.
    if encoding.lower() in ('utf_8', 'utf-8'):
        encoding = 'utf_8_sig'

    if filename == '{clipboard}':
        text = qt.QApplication.clipboard().text()
        return io.StringIO(text)
    else:
        return io.open(filename, mode, encoding=encoding, errors='ignore')

# The following two classes are adapted from the Python documentation
# they are modified to turn off encoding errors

class _UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8

    Needed for python2
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f, errors='ignore')
    def __iter__(self):
        return self
    def next(self):
        line = self.reader.next()
        return line.encode("utf-8")

class _UTF8Decoder:
    """
    Python2 iterator than decodes lists of utf-8 encoded strings
    """
    def __init__(self, iterator):
        self.iterator = iterator
    def __iter__(self):
        return self
    def next(self):
        line = self.iterator.next()
        return [str(x, "utf-8") for x in line]

def get_unicode_csv_reader(filename, dialect=csv.excel,
                           encoding='utf-8', **kwds):
    """Return an iterator to iterate over CSV file with encoding given."""

    f = openEncoding(filename, encoding)
    reader = csv.reader(f, dialect=dialect, **kwds)
    return reader

# End python doc classes

def populateCombo(combo, items):
    """Populate the combo with the list of items given.

    This also makes sure the currently entered text persists,
    or if currenttext is set, use this
    """

    # existing setting
    currenttext = combo.currentText()

    # add to list if not included
    if currenttext not in items:
        items = items + [currenttext]

    # put in new entries
    for i, val in enumerate(items):
        if i >= combo.count():
            combo.addItem(val)
        else:
            if combo.itemText(i) != val:
                combo.insertItem(i, val)

    # remove any extra items
    while combo.count() > len(items):
        combo.removeItem( combo.count()-1 )

    # get index for current value
    index = combo.findText(currenttext)
    combo.setCurrentIndex(index)

def positionFloatingPopup(popup, widget):
    """Position a popped up window (popup) to side and below widget given."""
    pos = widget.parentWidget().mapToGlobal( widget.pos() )
    size = widget.screen().geometry()

    # recalculates out position so that size is correct below
    popup.adjustSize()

    # is there room to put this widget besides the widget?
    if pos.y() + popup.height() + 1 < size.height():
        # put below
        y = pos.y() + 1
    else:
        # put above
        y = pos.y() - popup.height() - 1

    # is there room to the left for us?
    if ( (pos.x() + widget.width() + popup.width() < size.width()) or
         (pos.x() + widget.width() < size.width()/2) ):
        # put left justified with widget
        x = pos.x() + widget.width()
    else:
        # put extending to left
        x = pos.x() - popup.width() - 1

    popup.move(x, y)
    popup.setFocus()

# based on http://stackoverflow.com/questions/10607841/algorithm-for-topological-sorting-if-cycles-exist
def topological_sort(dependency_pairs):
    """Given a list of pairs, perform a topological sort.
    That means, each item has something which needs to be done first.
    topsort( [(1,2), (3,4), (5,6), (1,3), (1,5), (1,6), (2,5)] )
    returns  [1, 2, 3, 5, 4, 6], []
    for ordered and cyclic items
    """

    num_heads = defaultdict(int)   # num arrows pointing in
    tails = defaultdict(list)      # list of arrows going out
    for h, t in dependency_pairs:
        num_heads[t] += 1
        tails[h].append(t)

    ordered = [h for h in tails if h not in num_heads]
    i = 0
    while i < len(ordered):
        h = ordered[i]
        for t in tails[h]:
            num_heads[t] -= 1
            if not num_heads[t]:
                ordered.append(t)
        i += 1
    cyclic = [n for n, heads in num_heads.items() if heads]
    return ordered, cyclic

def isiternostr(i):
    """Is this iterator, but not a string?"""
    return hasattr(i, '__iter__') and not isinstance(i, str)

def nextfloat(fin):
    """Return (approximately) next float value (for f>0)."""
    d = 2**-52
    split = N.frexp(fin)
    while True:
        fout = N.ldexp(split[0] + d, split[1])
        if fin != fout:
            return fout
        d *= 2

def round2delt(fin1, fin2):
    """Take two float values. Return value rounded to number
    of decimal places where they differ."""

    if not N.isfinite(fin1) or not N.isfinite(fin2):
        return fin1

    # round up to next value to avoid 0.999999...
    f1 = nextfloat(abs(fin1))
    f2 = nextfloat(abs(fin2))

    maxlog = int( max(N.log10(f1), N.log10(f2)) + 1 )
    # note: out2 unused, but useful for debugging
    if maxlog < 0:
        out1 = out2 = '0.' + '0'*(-1-maxlog)
    else:
        out1 = out2 = ''

    for i in range(maxlog,-200,-1):
        p = 10**i
        d1, d2 = int(f1/p), int(f2/p)
        f1 -= int(d1)*p
        f2 -= int(d2)*p

        c1 = chr(d1 + 48) # 48 == '0'
        c2 = chr(d2 + 48)
        out1 += c1
        out2 += c2

        if c1 != c2 and p < abs(fin1): # at least 1 sig fig
            if i > 0:
                # add missing zeros
                out1 += '0'*i
                out2 += '0'*i
            break

        if i == 0:
            out1 += '.'
            out2 += '.'

    # convert back to float for output
    fout = float(out1)
    return fout if fin1 > 0 else -fout

def checkOrder(inv):
    """Check order of inv
    Returns: +1: ascending order
             -1: descending order
              0: other, or non finite."""
    v = N.array(inv)
    if not N.all(N.isfinite(v)):
        return 0
    delta = v[1:] - v[:-1]
    if N.all(delta > 0):
        return 1
    if N.all(delta < 0):
        return -1
    return 0

def checkAscending(v):
    """Check list of values is finite and ascending."""
    v = N.array(v)
    if not N.all( N.isfinite(v) ):
        return False
    return N.all( (v[1:] - v[:-1]) > 0 )

def rrepr(val):
    """Reproducible repr.

    The idea is to make a repr which won't change. We sort dict and
    set entries."""

    if isinstance(val, dict):
        l = [ "%s: %s" % (rrepr(k), rrepr(val[k]))
              for k in sorted(val) ]
        return "{%s}" % ", ".join(l)
    elif isinstance(val, set):
        l = [rrepr(v) for v in sorted(val)]
        return "set([%s])" % ", ".join(l)
    elif isinstance(val, list):
        l = [rrepr(v) for v in val]
        return "[%s]" % ", ".join(l)
    else:
        return repr(val)

def escapeHDFDataName(name):
    """Return escaped dataset name for saving in HDF5 files.
    This is because names cannot include / characters in HDF5
    """
    name = name.replace('`', '`BT')
    name = name.replace('/', '`SL')
    return name.encode('utf-8')

def unescapeHDFDataName(name):
    """Return original name after being escaped."""
    name = name.replace('`SL', '/')
    name = name.replace('`BT', '`')
    return name

def allNotNone(*items):
    """Are all the items not None."""
    return not any((x is None for x in items))

def anyNone(*items):
    """Are any items None."""
    return any((x is None for x in items))

def findOnPath(cmd):
    """Find a command on the system path, or None if does not exist."""
    path = os.getenv('PATH', os.path.defpath)
    pathparts = path.split(os.path.pathsep)
    for dirname in pathparts:
        dirname = dirname.strip('"')
        cmdtry = os.path.join(dirname, cmd)
        if os.path.isfile(cmdtry) and os.access(cmdtry, os.X_OK):
            return cmdtry
    return None

def listIndex(inlist, item):
    """Return index of item in list or -1 if not available."""
    try:
        return inlist.index(item)
    except ValueError:
        return -1

class Struct:
    """Simple structure-like class."""

    def __init__(self, **args):
        self.__dict__.update(args)

    def __repr__(self):
        return '<%s>' % str(
            ', '.join(
                '%s:%s' % (k, repr(getattr(self, k)))
                for k in sorted(self.__dict__)
            )
        )

class SvgWidgetFixedAspect(qt.QWidget):
    """Draw an SVG file with the aspect ratio fixed to the original."""

    def __init__(self, filename, *args):
        qt.QWidget.__init__(self, *args)
        self.renderer = qt.QSvgRenderer(filename, self)
        self.defwidth = self.renderer.defaultSize().width()
        self.defheight = self.renderer.defaultSize().height()

    def sizeHint(self):
        return qt.QSize(self.defwidth, self.defheight)

    def paintEvent(self, evt):
        painter = qt.QPainter(self)

        w = self.width()
        h = self.height()

        ratio_x = w / self.defwidth
        ratio_y = h / self.defheight
        if ratio_x < ratio_y:
            outw = w
            outh = w*self.defheight//self.defwidth
            left = 0
            top = (h-outh)//2
        else:
            outw = h*self.defwidth//self.defheight
            outh = h
            left = (w-outw)//2
            top = 0

        self.renderer.render(painter, qt.QRectF(left, top, outw, outh))

class OverrideCursor:
    """A context manager to handle changing the mouse cursor temporarily."""

    def __init__(self, cursor=qt.Qt.CursorShape.WaitCursor):
        self.cursor = cursor

    def __enter__(self):
        qt.QApplication.setOverrideCursor(qt.QCursor(self.cursor))

    def __exit__(self, typ, value, traceback):
        qt.QApplication.restoreOverrideCursor()

class DisabledIconEngine(qt.QIconEngine):
    """Icon engine which draws icons in a disabled state."""

    def __init__(self, icon):
        qt.QIconEngine.__init__(self)
        self.icon = icon

    def actualSize(self, size, mode, state):
        return size

    def clone(self):
        return GreyIconEngine(self.icon)

    def iconName(self):
        return ''

    def key(self):
        return str(id(self))

    def paint(self, painter, rect, mode, state):
        self.icon.paint(painter, rect, qt.Qt.AlignmentFlag.AlignCenter, qt.QIcon.Mode.Disabled, state)

    def pixmap(self, size, mode, state):
        return self.icon.pixmap(size, qt.QIcon.Mode.Disabled, state)


def dumpPalette(pal):
    """Dump a palette to a string."""

    cg = qt.QPalette.ColorGroup
    cr = qt.QPalette.ColorRole
    lines = []
    for role in (
            cr.Window,
            cr.WindowText,
            cr.Base,
            cr.AlternateBase,
            cr.ToolTipBase,
            cr.ToolTipText,
            cr.ToolTipText,
            cr.PlaceholderText,
            cr.Text,
            cr.Button,
            cr.ButtonText,
            cr.BrightText,
            cr.Light,
            cr.Midlight,
            cr.Dark,
            cr.Mid,
            cr.Shadow,
            cr.Highlight,
            cr.HighlightedText,
            cr.Link,
            cr.LinkVisited,
            ):
        v = []
        for group in cg.Disabled, cg.Active, cg.Inactive:
            v.append( pal.color(group, role).name() )

        lines.append(f'  ({role.value:2}, "{v[0]}", "{v[1]}", "{v[2]}"),\n')

    out = '[\n' + ''.join(lines) + ']\n'
    return out

def getPalette(name):
    """Get a dumped palette."""

    # these are dumped using the dumpPalette function above
    palettes = {
        # dumped from KDE on plasma version 5.27.12
        'breeze-light': [
            (10, "#fcfcfc", "#eff0f1", "#eff0f1"),
            ( 0, "#7e7e7e", "#232629", "#232629"),
            ( 9, "#fcfcfc", "#ffffff", "#ffffff"),
            (16, "#f7f7f7", "#f7f7f7", "#f7f7f7"),
            (18, "#f7f7f7", "#f7f7f7", "#f7f7f7"),
            (19, "#232629", "#232629", "#232629"),
            (19, "#232629", "#232629", "#232629"),
            (20, "#000000", "#000000", "#000000"),
            ( 6, "#7e7e7e", "#232629", "#232629"),
            ( 1, "#fcfcfc", "#fcfcfc", "#fcfcfc"),
            ( 8, "#7e7e7e", "#232629", "#232629"),
            ( 7, "#ffffff", "#ffffff", "#ffffff"),
            ( 2, "#ffffff", "#ffffff", "#ffffff"),
            ( 3, "#ffffff", "#ffffff", "#ffffff"),
            ( 4, "#7e7e7e", "#7e7e7e", "#7e7e7e"),
            ( 5, "#a8a8a8", "#a8a8a8", "#a8a8a8"),
            (11, "#b1b1b1", "#767676", "#767676"),
            (12, "#a8a8a8", "#3daee9", "#3daee9"),
            (13, "#ffffff", "#ffffff", "#ffffff"),
            (14, "#2980b9", "#2980b9", "#2980b9"),
            (15, "#9b59b6", "#9b59b6", "#9b59b6"),
        ],
        # dumped from KDE on plasma version 5.27.12
        'breeze-dark': [
            (10, "#31363b", "#2a2e32", "#2a2e32"),
            ( 0, "#626c76", "#fcfcfc", "#fcfcfc"),
            ( 9, "#31363b", "#1b1e20", "#1b1e20"),
            (16, "#232629", "#232629", "#232629"),
            (18, "#31363b", "#31363b", "#31363b"),
            (19, "#fcfcfc", "#fcfcfc", "#fcfcfc"),
            (19, "#fcfcfc", "#fcfcfc", "#fcfcfc"),
            (20, "#f0f0f0", "#f0f0f0", "#f0f0f0"),
            ( 6, "#626c76", "#fcfcfc", "#fcfcfc"),
            ( 1, "#31363b", "#31363b", "#31363b"),
            ( 8, "#626c76", "#fcfcfc", "#fcfcfc"),
            ( 7, "#ffffff", "#4b4b4b", "#4b4b4b"),
            ( 2, "#181b1d", "#181b1d", "#181b1d"),
            ( 3, "#25292c", "#25292c", "#25292c"),
            ( 4, "#626c76", "#626c76", "#626c76"),
            ( 5, "#41484e", "#41484e", "#41484e"),
            (11, "#252525", "#191919", "#191919"),
            (12, "#41484e", "#3daee9", "#3daee9"),
            (13, "#25292c", "#fcfcfc", "#fcfcfc"),
            (14, "#1d99f3", "#1d99f3", "#1d99f3"),
            (15, "#9b59b6", "#9b59b6", "#9b59b6"),
        ],
    }

    pal = qt.QPalette()

    if name not in palettes:
        return None

    for role, disabled, active, inactive in palettes[name]:
        role = pal.ColorRole(role)
        pal.setColor(pal.ColorGroup.Disabled, role, qt.QColor(disabled))
        pal.setColor(pal.ColorGroup.Active, role, qt.QColor(active))
        pal.setColor(pal.ColorGroup.Inactive, role, qt.QColor(inactive))
    return pal
