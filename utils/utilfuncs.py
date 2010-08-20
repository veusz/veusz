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

# $Id$

import sys
import string
import weakref
import re
import os.path
import threading
import dates
import codecs
import csv
import math

import veusz.qtall as qt4

class IgnoreException(Exception):
    """A special exception class to be ignored by the exception handler."""

def _getVeuszDirectory():
    """Get installed directory to find files relative to this one."""

    if hasattr(sys, 'frozen'):
        # for py2exe compatability
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        # standard installation
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

veuszDirectory = _getVeuszDirectory()

id_re = re.compile('^[A-Za-z_][A-Za-z0-9_]*$')
def validPythonIdentifier(name):
    """Is this a valid python identifier?"""
    return id_re.match(name) is not None

def validateDatasetName(name):
    """Validate dataset name is okay.
    Dataset names can contain anything except back ticks!
    """
    return len(name) > 0 and name.find('`') == -1

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

class WeakBoundMethod:
    """A weak reference to a bound method.

    Based on code by Frederic Jolliton
    See http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/81253
    """
    
    def __init__(self, f):
        self.f = f.im_func
        self.c = weakref.ref(f.im_self)

    def isEqual(self, f):
        """Is the bound method pointed to the same as this one?"""
        return f.im_func == self.f and f.im_self == self.c

    def __call__(self , *arg):
        if self.c() is None:
            raise ValueError, 'Method called on dead object'
        self.f(self.c(), *arg)

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

_formaterror = 'FormatError'

def formatSciNotation(num, formatargs=''):
    """Format number into form X \times 10^{Y}.
    This function trims trailing zeros and decimal point unless a formatting
    argument is supplied

    This is similar to the %e format string
    formatargs is the standard argument in a format string to control the
    number of decimal places, etc."""

    # create an initial formatting string
    if formatargs:
        format = '%' + formatargs + 'e'
    else:
        format = '%.10e'

    # try to format the number
    # this may be user-supplied data, so don't crash hard by returning
    # useless output
    try:
        text = format % num
    except:
        return _formaterror

    # split around the exponent
    leader, exponent = text.split('e')

    # strip off trailing decimal point and zeros if no format args
    if not formatargs:
        leader = '%.10g' % float(leader)

    # trim off leading 1
    if leader == '1' and not formatargs:
        leader = ''
    else:
        # the unicode string is a small space, multiply and small space
        leader += u'\u00d7'

    return '%s10^{%i}' % (leader, int(exponent))

def formatGeneral(num, fmtarg):
    """General formatting which switches from normal to scientic
    notation."""
    
    a = abs(num)
    # manually choose when to switch from normal to scientific
    # as the default isn't very good
    if a >= 1e4 or (a < 1e-2 and a > 1e-110):
        return formatSciNotation(num, fmtarg)
    else:
        if fmtarg:
            f = '%' + fmtarg + 'g'
        else:
            f = '%.10g'

        try:
            return f % num
        except:
            return _formaterror

engsuffixes = ( 'y', 'z', 'a', 'f', 'p', 'n',
                u'\u03bc', 'm', '', 'k', 'M', 'G',
                'T', 'P', 'E', 'Z', 'Y' )

def formatEngineering(num, fmtarg):
    """Engineering suffix format notation using SI suffixes."""

    if num != 0.:
        logindex = math.log10( abs(num) ) / 3.

        # for numbers < 1 round down suffix
        if logindex < 0. and (int(logindex)-logindex) > 1e-6:
            logindex -= 1

        # make sure we don't go out of bounds
        logindex = min( max(logindex, -8),
                        len(engsuffixes) - 9 )

        suffix = engsuffixes[ int(logindex) + 8 ]
        val = num / 10**( int(logindex) *3)
    else:
        suffix = ''
        val = num

    text = ('%' + fmtarg + 'g%s') % (val, suffix)
    return text

_formatRE = re.compile(r'%([^A-Za-z]*)(VDVS|VD.|V.|[A-Za-z])')

def formatNumber(num, format):
    """ Format a number in different ways.

    format is a standard C format string, with some additions:
     %Ve    scientific notation X \times 10^{Y}
     %Vg    switches from normal notation to scientific outside 10^-2 to 10^4
     %VE    engineering suffix option

     %VDx   date formatting, where x is one of the arguments in 
            http://docs.python.org/lib/module-time.html in the function
            strftime
    """

    while True:
        # repeatedly try to do string format
        m = _formatRE.search(format)
        if not m:
            break

        # argument and type of formatting
        farg, ftype = m.groups()

        # special veusz formatting
        if ftype[:1] == 'V':
            # special veusz formatting
            if ftype == 'Ve':
                out = formatSciNotation(num, farg)
            elif ftype == 'Vg':
                out = formatGeneral(num, farg)
            elif ftype == 'VE':
                out = formatEngineering(num, farg)
            elif ftype[:2] == 'VD':
                d = dates.floatToDateTime(num)
                # date formatting (seconds since start of epoch)
                if ftype[:4] == 'VDVS':
                    # special seconds operator
                    out = ('%'+ftype[4:]+'g') % (d.second+d.microsecond*1e-6)
                else:
                    # use date formatting
                    try:
                        out = d.strftime(str('%'+ftype[2:]))
                    except ValueError:
                        out = _formaterror
            else:
                out = _formaterror

            # replace hyphen with true - and small space
            out = out.replace('-', u'\u2212')

        else:
            # standard C formatting
            try:
                out = ('%' + farg + ftype) % num
            except:
                out = _formaterror

        format = format[:m.start()] + out + format[m.end():]

    return format

# This is Tim Peter's <tim_one@msn.com> topological sort
# see http://www.python.org/tim_one/000332.html
# adapted to use later python features

def topsort(pairlist):
    """Given a list of pairs, perform a topological sort.
    That means, each item has something which needs to be done first.
    
    topsort( [(1,2), (3,4), (5,6), (1,3), (1,5), (1,6), (2,5)] )
    returns [1, 2, 3, 5, 4, 6]
    """
    
    numpreds = {}   # elt -> # of predecessors
    successors = {} # elt -> list of successors
    for first, second in pairlist:
        # make sure every elt is a key in numpreds
        if not numpreds.has_key( first ):
            numpreds[first] = 0

        if not numpreds.has_key( second ):
            numpreds[second] = 0

        # since first &lt; second, second gains a pred ...
        numpreds[second] += 1

        # ... and first gains a succ
        if successors.has_key( first ):
            successors[first].append( second )
        else:
            successors[first] = [second]

    # suck up everything without a predecessor
    answer = [key for key, item in numpreds.iteritems()
              if item == 0]

    # for everything in answer, knock down the pred count on
    # its successors; note that answer grows *in* the loop

    for x in answer:
        del numpreds[x]
        if successors.has_key( x ):
            for y in successors[x]:
                numpreds[y] -= 1
                if numpreds[y] == 0:
                    answer.append( y )
            # following del; isn't needed; just makes
            # CycleError details easier to grasp
            # del successors[x]

    # assert catches cycle errors
    assert not numpreds
    
    return answer

class _NoneSoFar:
    pass
_NoneSoFar = _NoneSoFar()

def lazy(func, resultclass):
    """A decorator to allow lazy evaluation of functions.
    The products of this function is a lazy version of the function
    given.

    func is the function to evaluate
    resultclass is the class this function returns."""
    
    class __proxy__:
        def __init__(self, args, kw):
            self.__func = func
            self.__args = args
            self.__kw = kw
            self.__result = _NoneSoFar
            for (k, v) in resultclass.__dict__.items():
                setattr(self, k, self.__promise__(v))
        
        def __promise__(self, func):
            def __wrapper__(*args, **kw):
                if self.__result is _NoneSoFar:
                    self.__result = self.__func(*self.__args, **self.__kw)
                return func(self.__result, *args, **kw)
            return __wrapper__
        
    def __wrapper__(*args, **kw):
        return __proxy__(args, kw)
            
    return __wrapper__

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
    """Convenience function for opening file with encoding given."""
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

    def __init__(self, f, dialect=csv.excel, encoding='utf-8', **kwds):
        f = UTF8Recoder(f, encoding)
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
