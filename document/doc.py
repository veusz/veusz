# document.py
# A module to handle documents

#    Copyright (C) 2004 Jeremy S. Sanders
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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

import os
import os.path
import time
import numarray
import random
import string

import qt

import widgets
import utils

def _cnvt_numarray(a):
    """Convert to a numarray if possible (doing copy)."""
    if a == None:
        return None
    elif type(a) != type(numarray.arange(1)):
        return numarray.array(a, type=numarray.Float64)
    else:
        return a.astype(numarray.Float64)

class Dataset:
    '''Represents a dataset.'''

    def __init__(self, data = None, serr = None, nerr = None, perr = None):
        '''Initialse storage.'''
        self.data = _cnvt_numarray(data)
        self.serr = _cnvt_numarray(serr)
        self.nerr = _cnvt_numarray(nerr)
        self.perr = _cnvt_numarray(perr)

    def hasErrors(self):
        '''Whether errors on dataset'''
        return self.serr != None or self.nerr != None or self.perr != None

    def getPointRanges(self):
        '''Get range of coordinates for each point in the form
        (minima, maxima).'''

        minvals = self.data.copy()
        maxvals = self.data.copy()

        if self.serr != None:
            minvals -= self.serr
            maxvals += self.serr

        if self.nerr != None:
            minvals += self.nerr

        if self.perr != None:
            maxvals += self.perr

        return (minvals, maxvals)

    def getRange(self):
        '''Get total range of coordinates.'''
        minvals, maxvals = self.getPointRanges()
        return ( numarray.minimum.reduce(minvals),
                 numarray.maximum.reduce(maxvals) )

    def empty(self):
        '''Is the data defined?'''
        return self.data == None or len(self.data) == 0

    # TODO implement mathematical operations on this type

    def saveToFile(self, file, name):
        '''Save data to file.'''

        # build up descriptor
        datasets = [self.data]
        descriptor = name
        if self.serr != None:
            descriptor += ',+-'
            datasets.append(self.serr)
        if self.perr != None:
            descriptor += ',+'
            datasets.append(self.perr)
        if self.nerr != None:
            descriptor += ',-'
            datasets.append(self.nerr)

        text = "ImportString('%s','''\n" % descriptor

        # write line line-by-line
        for line in zip( *datasets ):
            l = ''
            for i in line:
                l += '%e ' % i
            text += l[:-1] + '\n'

        text += "''')\n"
        file.write(text)

class Document( qt.QObject ):
    """Document class for holding the graph data.

    Emits: sigModified when the document has been modified
           sigWiped when document is wiped
    """

    def __init__(self):
        """Initialise the document."""
        qt.QObject.__init__( self )
        self.wipe()

    def wipe(self):
        """Wipe out any stored data."""

        self.data = {}
        self.basewidget = widgets.Root(None)
        self.basewidget.setDocument(self)
        self.setModified()
        self.emit( qt.PYSIGNAL("sigWiped"), () )

    def setData(self, name, dataset):
        """Set data to val, with symmetric or negative and positive errors."""
        self.data[name] = dataset
        self.setModified()

    def getBaseWidget(self):
        """Return the base widget."""
        return self.basewidget

    def getData(self, name):
        """Get data with name"""
        return self.data[name]

    def hasData(self, name):
        """Whether dataset is defined."""
        return name in self.data

    def setModified(self, ismodified=True):
        """Set the modified flag on the data, and inform views."""
        self.modified = ismodified
        self.emit( qt.PYSIGNAL("sigModified"), ( ismodified, ) )

    def isModified(self):
        """Return whether modified flag set."""
        return self.modified
    
    def getSize(self):
        """Get the size of the main plot widget."""
        s = self.basewidget.settings
        return (s.width, s.height)

    def printTo(self, printer, pages, scaling = 1.):
        """Print onto printing device."""

        painter = qt.QPainter()
        painter.begin( printer )

        painter.veusz_scaling = scaling

        # work out how many pixels correspond to the given size
        width, height = utils.cnvtDists(self.getSize(), painter)
        children = self.basewidget.getChildren()

        # This all assumes that only pages can go into the root widget
        i = 0
        no = len(pages)

        for p in pages:
            c = children[p]
            c.draw( (0, 0, width, height), painter )

            # start new pages between each page
            if i < no-1:
                printer.newPage()
            i += 1

        painter.end()

    def getNumberPages(self):
        """Return the number of pages in the document."""
        return len(self.basewidget.getChildren())

    def saveToFile(self, file):
        """Save the text representing a document to a file."""

        file.write('# Veusz saved document (version %s)\n' % utils.version())
        file.write('# User: %s\n' % os.getlogin() )
        file.write('# Date: %s\n\n' % time.strftime(
            "%a, %d %b %Y %H:%M:%S +0000", time.gmtime()) )
        
        for name, dataset in self.data.items():
            dataset.saveToFile(file, name)
        file.write(self.getBaseWidget().getSaveText())
        
        self.setModified(False)

    def export(self, filename, color=True):
        """Export the figure to the filename."""

        ext = os.path.splitext(filename)[1]

        if ext == '.eps':
            # write eps file
            p = qt.QPrinter(qt.QPrinter.HighResolution)
            p.setOutputToFile(True)
            p.setOutputFileName(filename)
            p.setColorMode( (qt.QPrinter.GrayScale, qt.QPrinter.Color)[color] )
            p.newPage()
            self.printTo( p )

        elif ext == '.png':
            # write png file
            # unfortunately we need to pass QPrinter the name of an eps
            # file: no secure way we can produce the file. FIXME INSECURE

            dir = os.path.dirname(os.path.abspath(filename))
            while 1:
                digits = string.digits + string.ascii_letters
                rndstr = ''
                for i in xrange(40):
                    rndstr += random.choice(digits)
                tmpfilename = "%s/tmp_%s.eps" % (dir, rndstr)
                try:
                    os.stat(tmpfilename)
                except OSError:
                    break
            
            # write eps file
            p = qt.QPrinter(qt.QPrinter.HighResolution)
            p.setOutputToFile(True)
            p.setOutputFileName(tmpfilename)
            p.setColorMode( (qt.QPrinter.GrayScale, qt.QPrinter.Color)[color] )
            p.newPage()
            self.printTo( p )

            # now use ghostscript to convert the file into the relevent type
            cmdline = ( 'gs -sDEVICE=pngalpha -dEPSCrop -dBATCH -dNOPAUSE'
                        ' -sOutputFile="%s" "%s"' % (filename, tmpfilename) )
            stdin, stdout, stderr = os.popen3(cmdline)
            stdin.close()

            # if anything goes to stderr, then report it
            text = stderr.read().strip()
            if len(text) != 0:
                raise RuntimeError, text

            os.unlink(tmpfilename)

        else:
            raise RuntimeError, "File type '%s' not supported" % ext
        
