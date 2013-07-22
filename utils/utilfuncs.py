# utilfuncs.py
# utility functions

#    Copyright (C) 2003 Jeremy S. Sanders
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
###############################################################################

from __future__ import division
import sys
import string
import re
import os.path
import threading
import codecs
import csv
import StringIO
import locale
from collections import defaultdict

import veusz.qtall as qt4
import numpy as N

class IgnoreException(Exception):
    """A special exception class to be ignored by the exception handler."""

def _getVeuszDirectory():
    """Get resource and examples directories for Veusz."""

    if hasattr(sys, 'frozen'):
        # for pyinstaller/py2app compatability
        resdir = os.path.dirname(os.path.abspath(sys.executable))
        if sys.platform == 'darwin':
            # special case for py2app
            resdir = os.path.join(resdir, '..', 'Resources')
    else:
        # standard installation
        resdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # override data directory with symlink
    if os.path.exists( os.path.join(resdir, 'resources') ):
        resdir = os.path.realpath( os.path.join(resdir, 'resources') )

    # override with VEUSZ_RESOURCE_DIR environment variable if necessary
    resdir = os.environ.get('VEUSZ_RESOURCE_DIR', resdir)

    # now get example directory (which may be a symlink)
    examplesdir = os.path.realpath( os.path.join(resdir, 'examples') )

    return resdir, examplesdir

# get resource and example directories
veuszDirectory, exampleDirectory = _getVeuszDirectory()

def getLicense():
    """Return license text."""
    try:
        f = open(os.path.join(veuszDirectory, 'COPYING'), 'rU')
        text = f.read()
        f.close()
    except EnvironmentError:
        text = ('Could not open the license file.\n'
                'See license at http://www.gnu.org/licenses/gpl-2.0.html')
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

# handle and extended #RRGGBBAA color
extendedcolor_re = re.compile('^#[0-9A-Fa-f]{8}$')
def extendedColorToQColor(s):
    if extendedcolor_re.match(s):
        col = qt4.QColor(s[:-2])
        col.setAlpha( int(s[-2:], 16) )
        return col
    else:
        return qt4.QColor(s)

def extendedColorFromQColor(col):
    """Make an extended color #RRGGBBAA or #RRGGBB string."""
    if col.alpha() == 255:
        return str(col.name())
    else:
        return '#%02x%02x%02x%02x' % (col.red(), col.green(), col.blue(),
                                      col.alpha())

def pixmapAsHtml(pix):
    """Get QPixmap as html image text."""
    ba = qt4.QByteArray()
    buf = qt4.QBuffer(ba)
    buf.open(qt4.QIODevice.WriteOnly)
    pix.toImage().save(buf, "PNG")
    b64 = str(buf.data().toBase64())
    return '<img src="data:image/png;base64,%s">' % b64

def BoundCaller(function, *params):
    """Wrap a function with its initial arguments."""
    def wrapped(*args):
        function( *(params+args) )
    return wrapped

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
    """

    def __init__(self, fileobject):
        """Create the thread object."""
        threading.Thread.__init__(self)
        self.fileobject = fileobject
        self.lock = threading.Lock()
        self.data = ''
        self.done = False

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
            except Exception, e:
                # error in reading
                self.lock.acquire()
                self.data = e
                self.lock.release()
                break

            # no more data: end of file
            if len(data) == 0:
                self.lock.acquire()
                self.done = True
                self.lock.release()
                break

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
    if filename == '{clipboard}':
        text = unicode(qt4.QApplication.clipboard().text())
        return StringIO.StringIO(text)
    else:
        return codecs.open(filename, mode, encoding, 'ignore')

# The following two classes are adapted from the Python documentation
# they are modified to turn off encoding errors

class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f, errors='ignore')

    def __iter__(self):
        return self

    def next(self):
        line = self.reader.next()
        return line.encode("utf-8")

class UnicodeCSVReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, filename, dialect=csv.excel, encoding='utf-8', **kwds):

        if filename != '{clipboard}':
            # recode the opened file as utf-8
            f = UTF8Recoder(open(filename), encoding)
        else:
            # take the unicode clipboard and just put into utf-8 format
            s = unicode(qt4.QApplication.clipboard().text())
            s = s.encode('utf-8')
            f = StringIO.StringIO(s)

        # the actual csv reader based on the file above
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, 'utf-8') for s in row]

    def __iter__(self):
        return self

# End python doc classes

def populateCombo(combo, items):
    """Populate the combo with the list of items given.

    This also makes sure the currently entered text persists,
    or if currenttext is set, use this
    """

    # existing setting
    currenttext = unicode(combo.currentText())

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
    desktop = qt4.QApplication.desktop()

    # recalculates out position so that size is correct below
    popup.adjustSize()

    # is there room to put this widget besides the widget?
    if pos.y() + popup.height() + 1 < desktop.height():
        # put below
        y = pos.y() + 1
    else:
        # put above
        y = pos.y() - popup.height() - 1

    # is there room to the left for us?
    if ( (pos.x() + widget.width() + popup.width() < desktop.width()) or
         (pos.x() + widget.width() < desktop.width()/2) ):
        # put left justified with widget
        x = pos.x() + widget.width()
    else:
        # put extending to left
        x = pos.x() - popup.width() - 1

    popup.move(x, y)
    popup.setFocus()

def unique(inlist):
    """Get unique entries in list."""

    inlist = list(inlist)
    inlist.sort()

    class X: pass
    last = X()
    out = []
    for x in inlist:
        if x != last:
            out.append(x)
            last = x
    return out

def decodeDefault(s):
    """Decode the string using current locale.
    Used for decoding exceptions."""
    return s.decode(locale.getdefaultlocale()[1])

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
    cyclic = [n for n, heads in num_heads.iteritems() if heads]
    return ordered, cyclic
