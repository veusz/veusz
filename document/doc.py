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

"""A class to represent Veusz documents, with dataset classes."""

import os
import os.path
import time
import random
import string

import veusz.qtall as qt4

import widgetfactory
import simpleread
import datasets

import veusz.utils as utils
import veusz.setting as setting

class Document( qt4.QObject ):
    """Document class for holding the graph data.

    Emits: sigModified when the document has been modified
           sigWiped when document is wiped
    """

    def __init__(self):
        """Initialise the document."""
        qt4.QObject.__init__( self )

        self.changeset = 0
        self.clearHistory()
        self.wipe()

    def applyOperation(self, operation):
        """Apply operation to the document.
        
        Operations represent atomic actions which can be done to the document
        and undone.
        """
        
        retn = operation.do(self)

        if self.historybatch:
            # in batch mode, create an OperationMultiple for all changes
            self.historybatch[-1].addOperation(operation)
        else:
            # standard mode
            self.historyundo.append(operation)
        self.historyredo = []

        self.setModified()
        return retn

    def clearHistory(self):
        """Clear any history."""
        
        self.historybatch = []
        self.historyundo = []
        self.historyredo = []
        
    def batchHistory(self, batch):
        """Enable/disable batch history mode.
        
        In this mode further operations are added to the OperationMultiple specified,
        untile batchHistory is called with None.
        
        The objects are pushed into a list and popped off
        
        This allows multiple operations to be batched up for simple undo.
        """
        if batch:
            self.historybatch.append(batch)
        else:
            self.historybatch.pop()
        
    def undoOperation(self):
        """Undo the previous operation."""
        
        operation = self.historyundo.pop()
        operation.undo(self)
        self.historyredo.append(operation)
        self.setModified()
        
    def canUndo(self):
        """Returns True if previous operation can be removed."""
        return len(self.historyundo) != 0

    def redoOperation(self):
        """Redo undone operations."""
        
        operation = self.historyredo.pop()
        operation.do(self)
        self.historyundo.append(operation)
        self.setModified()

    def canRedo(self):
        """Returns True if previous operation can be redone."""
        return len(self.historyredo) != 0
        
    def resolveFullWidgetPath(self, path):
        """Translate the widget path given into the widget."""
        
        widget = self.basewidget
        for p in [i for i in path.split('/') if i != '']:
            for child in widget.children:
                if p == child.name:
                    widget = child
                    break
            else:
                # break wasn't called
                assert False
        return widget
        
    def resolveFullSettingPath(self, path):
        """Translate setting path into setting object."""
        
        # find appropriate widget
        widget = self.basewidget
        parts = [i for i in path.split('/') if i != '']
        while len(parts) > 0:
            for child in widget.children:
                if parts[0] == child.name:
                    widget = child
                    del parts[0]
                    break
            else:
                # no child with name
                break
            
        # get Setting object
        s = widget.settings
        while isinstance(s, setting.Settings) and parts[0] in s.setdict:
            s = s.get(parts[0])
            del parts[0]
            
        assert isinstance(s, setting.Setting)
        return s
            
    def wipe(self):
        """Wipe out any stored data."""
        self.data = {}
        self.basewidget = widgetfactory.thefactory.makeWidget(
            'document', None, None)
        self.basewidget.document = self
        self.setModified(False)
        self.emit( qt4.SIGNAL("sigWiped") )

    def isBlank(self):
        """Does the document contain widgets and no data"""
        return len(self.basewidget.children) == 0 and len(self.data) == 0

    def setData(self, name, dataset):
        """Set data to val, with symmetric or negative and positive errors."""
        self.data[name] = dataset
        dataset.document = self
        self.setModified()

    def reloadLinkedDatasets(self):
        """Reload linked datasets from their files.

        Returns a tuple of
        - List of datasets read
        - Dict of tuples containing dataset names and number of errors
        """

        # build up a list of linked files
        links = {}
        for ds in self.data.itervalues():
            if ds.linked:
                links[ ds.linked ] = True

        read = []
        errors = {}

        # load in the files, merging the vars read and errors
        if links:
            for l in links.iterkeys():
                nread, nerrors = l.reloadLinks(self)
                read += nread
                errors.update(nerrors)
            self.setModified()

        read.sort()
        return (read, errors)

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

        self.emit( qt4.SIGNAL("sigModified"), ismodified )

    def isModified(self):
        """Return whether modified flag set."""
        return self.modified
    
    def printTo(self, printer, pages, scaling = 1., dpi = None,
                antialias = False):
        """Print onto printing device."""

        painter = Painter()
        painter.veusz_scaling = scaling
        if dpi is not None:
            painter.veusz_pixperpt = dpi / 72.
        
        painter.begin( printer )
        if antialias:
            painter.setRenderHint(qt4.QPainter.Antialiasing)

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

    def paintTo(self, painter, page, scaling = 1., dpi = None):
        """Paint page specified to the painter."""
        
        painter.veusz_scaling = scaling
        if dpi is not None:
            painter.veusz_pixperpt = dpi / 72.
        width, height = self.basewidget.getSize(painter)
        self.basewidget.children[page].draw( (0, 0, width, height), painter)

    def getNumberPages(self):
        """Return the number of pages in the document."""
        return len(self.basewidget.children)

    def _writeFileHeader(self, file, type):
        """Write a header to a saved file of type."""
        file.write('# Veusz %s (version %s)\n' % (type, utils.version()))
        try:
            file.write('# User: %s\n' % os.environ['LOGNAME'] )
        except KeyError:
            pass
        file.write('# Date: %s\n\n' % time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()) )
        
    def saveToFile(self, file):
        """Save the text representing a document to a file."""

        self._writeFileHeader(file, 'saved document')
        
        # save those datasets which are linked
        # we do this first in case the datasets are overridden below
        savedlinks = {}
        for name, dataset in self.data.items():
            dataset.saveLinksToSavedDoc(file, savedlinks)

        # save the remaining datasets
        for name, dataset in self.data.items():
            dataset.saveToFile(file, name)

        # save the actual tree structure
        file.write(self.basewidget.getSaveText())
        
        self.setModified(False)

    def exportStyleSheet(self, file):
        """Export the StyleSheet to a file."""

        self._writeFileHeader(file, 'exported stylesheet')
        stylesheet = self.basewidget.settings.StyleSheet

        file.write( stylesheet.saveText(True, rootname='') )
        
    def export(self, filename, pagenumber, color=True):
        """Export the figure to the filename."""

        ext = os.path.splitext(filename)[1]

        if ext == '.eps' or ext == '.pdf':
            # write eps file
            p = qt4.QPrinter()
            if ext == '.pdf':
                p.setOutputFormat(qt4.QPrinter.PdfFormat)
            p.setOutputFileName(filename)
            p.setColorMode( (qt4.QPrinter.GrayScale, qt4.QPrinter.Color)[color] )
            p.setCreator('Veusz %s' % utils.version())
            p.newPage()
            self.printTo( p, [pagenumber] )

        elif ext == '.png':
            # write png file
            # unfortunately we need to pass QPrinter the name of an eps
            # file: no secure way we can produce the file. FIXME INSECURE

            # FIXME: doesn't work in Windows

            fdir = os.path.dirname(os.path.abspath(filename))
            if not os.path.exists(fdir):
                raise RuntimeError, 'Directory "%s" does not exist' % fdir

            digits = string.digits + string.ascii_letters
            while True:
                rndstr = ''.join( [random.choice(digits) for i in xrange(20)] )
                tmpfilename = os.path.join(fdir, "tmp_%s.eps" % rndstr)
                try:
                    os.stat(tmpfilename)
                except OSError:
                    break
            
            # write eps file
            p = qt4.QPrinter(qt4.QPrinter.HighResolution)
            p.setOutputToFile(True)
            p.setOutputFileName(tmpfilename)
            p.setColorMode( (qt4.QPrinter.GrayScale, qt4.QPrinter.Color)[color] )
            p.newPage()
            self.printTo( p, [pagenumber] )

            # now use ghostscript to convert the file into the relevent type
            cmdline = ( 'gs -sDEVICE=pngalpha -dEPSCrop -dBATCH -dNOPAUSE'
                        ' -sOutputFile="%s" "%s"' % (filename, tmpfilename) )
            stdin, stdout, stderr = os.popen3(cmdline)
            stdin.close()

            # if anything goes to stderr, then report it
            text = stderr.read().strip()
            os.unlink(tmpfilename)
            if len(text) != 0:
                raise RuntimeError, text

        else:
            raise RuntimeError, "File type '%s' not supported" % ext
            
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
                if p is None:
                    raise ValueError, "Base graph has no parent"
                obj = p
            elif p == '.' or len(p) == 0:
                # relative to here
                pass
            else:
                # child specified
                obj = obj.getChild( p )
                if obj is None:
                    raise ValueError, "Child '%s' does not exist" % p

        # return widget
        return obj

class Painter(qt4.QPainter):
    """A painter which allows the program to know which widget it is
    currently drawing."""
    
    def __init__(self, *args):
        qt4.QPainter.__init__(self, *args)

        self.veusz_scaling = 1.

    def beginPaintingWidget(self, widget, bounds):
        """Keep track of the widget currently being painted."""
        pass

    def endPaintingWidget(self):
        """Widget is now finished."""
        pass
    
