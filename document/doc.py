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
import itertools

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

class LinkedFile:
    '''Instead of reading data from a string, data can be read from
    a "linked file". This means the same document can be reloaded, and
    the data would be reread from the file.

    This class is used to store a link filename with the descriptor
    '''

    def __init__(self, filename, descriptor):
        '''Set up the linked file with the descriptor given.'''
        self.filename = filename
        self.descriptor = descriptor

    def saveToFile(self, file):
        '''Save the link to the document file.'''

        file.write('ImportFile(%s, %s, linked=True)\n' %
                   (repr(self.filename), repr(self.descriptor)))

class Dataset:
    '''Represents a dataset.'''

    def __init__(self, data = None, serr = None, nerr = None, perr = None,
                 linked = None):
        '''Initialise storage.'''
        self.document = None
        self.data = _cnvt_numarray(data)
        self.serr = self.nerr = self.perr = None

        # adding data*0 ensures types of errors are the same
        if self.data != None:
            if serr != None:
                self.serr = serr + self.data*0.
            if nerr != None:
                self.nerr = nerr + self.data*0.
            if perr != None:
                self.perr = perr + self.data*0.

        self.linked = linked

        # check the sizes of things match up
        s = self.data.shape
        for i in (self.serr, self.nerr, self.perr):
            assert i == None or i.shape == s

    def duplicate(self):
        """Return new dataset based on this one."""
        return Dataset(self.data, self.serr, self.nerr, self.perr, None)

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

    def changeValues(self, type, vals):
        """Change the requested part of the dataset to vals.

        type == vals | serr | perr | nerr
        """
        if type == 'vals':
            self.data = vals
        elif type == 'serr':
            self.serr = vals
        elif type == 'nerr':
            self.nerr = vals
        elif type == 'perr':
            self.perr = vals
        else:
            raise ValueError, 'type does not contain an allowed value'

        # just a check...
        s = self.data.shape
        for i in (self.serr, self.nerr, self.perr):
            assert i == None or i.shape == s

        self.document.setModified(True)

    # TODO implement mathematical operations on this type

    def saveLinksToFile(self, file, savedlinks):
        '''Save the link to the file, if this dataset is linked.

        savedlinks is a dict containing any linked files which have
        already been written
        '''

        # links should only be saved once
        if self.linked != None and self.linked not in savedlinks:
            savedlinks[self.linked] = True
            self.linked.saveToFile(file)

    def saveToFile(self, file, name):
        '''Save data to file.
        '''

        # return if there is a link
        if self.linked != None:
            return

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

        file.write( "ImportString(%s,'''\n" % repr(descriptor) )

        # write line line-by-line
        format = '%e ' * len(datasets)
        format = format[:-1] + '\n'
        for line in itertools.izip( *datasets ):
            file.write( format % line )

        file.write( "''')\n" )

def _recursiveGet(root, name, typename, outlist, maxlevels):
    """Add those widgets in root with name and type to outlist.

    If name or typename are None, then ignore the criterion.
    maxlevels is the maximum number of levels to check
    """

    if maxlevels != 0:

        # if levels is not zero, add the children of this root
        newmaxlevels = maxlevels - 1
        for w in root.children:
            if ( (w.name == name or name == None) and
                 (w.typename == typename or typename == None) ):
                outlist.append(w)

            _recursiveGet(w, name, typename, outlist, newmaxlevels)

class Document( qt.QObject ):
    """Document class for holding the graph data.

    Emits: sigModified when the document has been modified
           sigWiped when document is wiped
    """

    def __init__(self):
        """Initialise the document."""
        qt.QObject.__init__( self )

        self.changeset = 0
        self.wipe()

    def wipe(self):
        """Wipe out any stored data."""
        self.data = {}
        self.basewidget = widgets.Root(None)
        self.basewidget.document = self
        self.setModified(False)
        self.emit( qt.PYSIGNAL("sigWiped"), () )

    def isBlank(self):
        """Does the document contain widgets and no data"""
        return len(self.basewidget.children) == 0 and len(self.data) == 0

    def setData(self, name, dataset):
        """Set data to val, with symmetric or negative and positive errors."""
        self.data[name] = dataset
        dataset.document = self
        self.setModified()

    def deleteDataset(self, name):
        """Remove the selected dataset."""
        del self.data[name]
        self.setModified()

    def renameDataset(self, oldname, newname):
        """Rename the dataset."""
        d = self.data[oldname]
        del self.data[oldname]
        self.data[newname] = d

        self.setModified()

    def duplicateDataset(self, name, newname):
        """Duplicate the dataset to the newname."""

        if newname in self.data:
            raise ValueError, "Dataset %s already exists" % newname

        self.data[newname] = self.data[name].duplicate()
        self.setModified()

    def unlinkDataset(self, name):
        """Remove any links to file from the dataset."""
        self.data[name].linked = None
        self.setModified()

    def getData(self, name):
        """Get data with name"""
        return self.data[name]

    def hasData(self, name):
        """Whether dataset is defined."""
        return name in self.data

    def setModified(self, ismodified=True):
        """Set the modified flag on the data, and inform views."""

        # useful for tracking back modifications
        # import traceback
        # traceback.print_stack()

        self.modified = ismodified
        self.changeset += 1

        self.emit( qt.PYSIGNAL("sigModified"), ( ismodified, ) )

    def isModified(self):
        """Return whether modified flag set."""
        return self.modified
    
    def printTo(self, printer, pages, scaling = 1.):
        """Print onto printing device."""

        painter = qt.QPainter()
        painter.begin( printer )

        painter.veusz_scaling = scaling

        # work out how many pixels correspond to the given size
        width, height = self.basewidget.getSize(painter)
        children = self.basewidget.children

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
        return len(self.basewidget.children)

    def saveToFile(self, file):
        """Save the text representing a document to a file."""

        file.write('# Veusz saved document (version %s)\n' % utils.version())
        try:
            file.write('# User: %s\n' % os.environ['LOGNAME'] )
        except KeyError:
            pass
        file.write('# Date: %s\n\n' % time.strftime(
            "%a, %d %b %Y %H:%M:%S +0000", time.gmtime()) )

        # save those datasets which are linked
        # we do this first in case the datasets are overridden below
        savedlinks = {}
        for name, dataset in self.data.items():
            dataset.saveLinksToFile(file, savedlinks)

        # save the remaining datasets
        for name, dataset in self.data.items():
            dataset.saveToFile(file, name)

        # save the actual tree structure
        file.write(self.basewidget.getSaveText())
        
        self.setModified(False)

    def export(self, filename, pagenumber, color=True):
        """Export the figure to the filename."""

        ext = os.path.splitext(filename)[1]

        if ext == '.eps':
            # write eps file
            p = qt.QPrinter(qt.QPrinter.HighResolution)
            p.setCreator('Veusz %s' % utils.version())
            p.setOutputToFile(True)
            p.setOutputFileName(filename)
            p.setColorMode( (qt.QPrinter.GrayScale, qt.QPrinter.Color)[color] )
            p.setCreator('Veusz %s' % utils.version())
            p.newPage()
            self.printTo( p, [pagenumber] )

        elif ext == '.png':
            # write png file
            # unfortunately we need to pass QPrinter the name of an eps
            # file: no secure way we can produce the file. FIXME INSECURE

            fdir = os.path.dirname(os.path.abspath(filename))
            while 1:
                digits = string.digits + string.ascii_letters
                rndstr = ''
                for i in xrange(40):
                    rndstr += random.choice(digits)
                tmpfilename = "%s/tmp_%s.eps" % (fdir, rndstr)
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
            self.printTo( p, [pagenumber] )

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


    def propagateSettings(self, setting, widgetname=None,
                          root=None, maxlevels=-1):

        """Take the setting given, and propagate it to other widgets,
        according to the parameters here.
        
        If widgetname is given then only propagate it to widgets with
        the name given.

        widgets are located from the widget given (root if not set)
        """

        # locate widget with the setting (building up path)
        path = []
        widget = setting
        while not isinstance(widget, widgets.Widget):
            path.insert(0, widget.name)
            widget = widget.parent

        # remove the name of the main settings of the widget
        path = path[1:]

        # default is root widget
        if root == None:
            root = self.basewidget

        # get a list of matching widgets
        widgetlist = []
        _recursiveGet(root, widgetname, widget.typename, widgetlist,
                      maxlevels)

        val = setting.get()
        # set the settings for the widgets
        for w in widgetlist:
            # lookup the setting
            s = w.settings
            for i in path:
                s = s.get(i)

            # set the setting
            s.set(val)
            
    def resolve(self, fromwidget, where):
        """Resolve graph relative to the widget fromwidget

        Allows unix-style specifiers, e.g. /graph1/x
        Returns widget
        """

        parts = where.split('/')

        if where[:1] == '/':
            # relative to base directory
            obj = self.basewidget
        else:
            # relative to here
            obj = fromwidget

        # iterate over parts in string
        for p in parts:
            if p == '..':
                # relative to parent object
                p = obj.parent
                if p == None:
                    raise ValueError, "Base graph has no parent"
                obj = p
            elif p == '.' or len(p) == 0:
                # relative to here
                pass
            else:
                # child specified
                obj = obj.getChild( p )
                if obj == None:
                    raise ValueError, "Child '%s' does not exist" % p

        # return widget
        return obj

