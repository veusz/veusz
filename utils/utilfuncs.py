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
import time
import os.path
import threading
import dates

def _getVeuszDirectory():
    """Get installed directory to find files relative to this one."""

    if hasattr(sys, 'frozen'):
        # for py2exe compatability
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        # standard installation
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

veuszDirectory = _getVeuszDirectory()

def reverse(data):
    """For iterating over a sequence in reverse."""
    for index in xrange(len(data)-1, -1, -1):
        yield data[index]

dsname_re = re.compile('^[A-Za-z][A-Za-z0-9_]*$')
def validateDatasetName(name):
    """Validate dataset name is okay."""
    return dsname_re.match(name) is not None

def validateWidgetName(name):
    """Validate widget name is okay."""
    return dsname_re.match(name) is not None

def escapeDatasetName(name):
    """Make string into a valid dataset name."""
    # replace invalid characters
    out = re.sub('[^0-9A-Za-z]', '_', name)
    # add underscores for leading numbers
    if re.match('^[0-9]', out):
        return '_' + out
    else:
        return out

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
        leader = leader + r' \times '

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

_formatRE = re.compile(r'%([^A-Za-z]*)(VDVS|VD.|V.|[A-Za-z])')

def formatNumber(num, format):
    """ Format a number in different ways.

    format is a standard C format string, with some additions:
     %Ve    scientific notation X \times 10^{Y}
     %Vg    switches from normal notation to scientific outside 10^-2 to 10^4

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
            elif ftype[:2] == 'VD':
                d = dates.floatToDateTime(num)
                # date formatting (seconds since start of epoch)
                if ftype[:4] == 'VDVS':
                    # special seconds operator
                    out = ('%'+ftype[4:]+'g') % (d.second+d.microsecond*1e-6)
                else:
                    # use date formatting
                    try:
                        out = d.strftime('%'+ftype[2:])
                    except ValueError:
                        out = _formaterror
            else:
                out = _formaterror

        else:
            # standard C formatting
            try:
                out = ('%' + farg + ftype) % num
            except:
                out = _formaterror

        format = format[:m.start()] + out + format[m.end():]

    return format

def clipper(xpts, ypts, bounds):
    """ Clip points that are safe to remove.

    Takes points in xpts, ypts.
    If any are clippable by the bounds and they lie between clipped points
    then those points are remove.

       -1,-1 | 0,-1  | 1,-1
       ---------------------
       -1,0  | 0, 0  | 1, 0
       ---------------------
       -1,1  | 0, 1  | 1, 1
       
    Data are returned in an array in the form (x1,y1, x2, y2...)
    """

    x1, y1, x2, y2 = bounds

    clipx = []
    clipy = []

    # find out whether points are clippable
    for x, y in zip(xpts, ypts):

        # is clippable?
        xclip = 0
        yclip = 0

        if   x<x1: xclip = -1
        elif x>x2: xclip = 1
        if   y<y1: yclip = -1
        elif y>y2: yclip = 1

        clipx.append(xclip)
        clipy.append(yclip)

    outx = []
    outy = []

    # now go through and collect the points we need...
    nopts = len(xpts)
    for i in xrange(nopts):
        cx = clipx[i]
        cy = clipy[i]

        # unclipped
        if cx == 0 and cy == 0:
            outx.append( xpts[i] )
            outy.append( ypts[i] )
            
        else:
            dx = abs( clipx[i+1] - cx )
            dy = abs( clipy[i+1] - cy )

            # set if it may be true we may see this pt
            visible = True
            if (dx == 0 and dy == 0) or \
               (dx == 0 and cx != 0) or \
               (dy == 0 and cy != 0):
                visible = False

##     lastclip = False
##     firstpass = True
##     for x,y in zip(xpts,ypts):


##         if xclip == 0 and yclip == 0:
##             if lastclip:
##                 pts.append(lastclippedx)
##                 pts.append(lastclippedy)
##                 lastclip = False
##             pts.append(x)
##             pts.append(y)
##         else:
##             if not lastclip:
##                 pts.append(x)
##                 pts.append(y)
##                 lastclip = True
##                 oldxclip = xclip
                
##             else:

##             deltax = abs( xclip - oldxclip )
##             deltay = abs( yclip - oldyclip )


##             if (deltax != 0 and deltay == 0) or \
##                (deltax == 0 and deltay != 0):

##         if x<x1 or x>x2 or y<y1 or y>y2:
##             lastx = x
##             lasty = y
##             if not lastclip and not firstpass:
##                 pts.append(x)
##                 pts.append(y)
##             lastclip = True
##         else:
##             # non-clippable:
##             # put back the last clipped point
##             if lastclip:
##                 if len(pts) < 2 or pts[-2] != lastx or pts[-1] != lasty:
##                     pts.append(lastx)
##                     pts.append(lasty)
##                 lastclip = False
##             # add the point
##             pts.append(x)
##             pts.append(y)
##         firstpass = False

##     return pts

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
