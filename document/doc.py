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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

# $Id$

"""A class to represent Veusz documents, with dataset classes."""

import os.path
import time
import random
import math
import re
from collections import defaultdict

import numpy as N

import veusz.qtall as qt4

import widgetfactory

import veusz.utils as utils
import veusz.setting as setting

try:
    import emf_export
    hasemf = True
except ImportError:
    hasemf = False

import svg_export

# python identifier
identifier_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
# for splitting
identifier_split_re = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

# python module
module_re = re.compile(r'^[A-Za-z_\.]+$')

# function(arg1, arg2...) for custom functions
# not quite correct as doesn't check for commas in correct places
function_re = re.compile(r'''
^([A-Za-z_][A-Za-z0-9_]*)[ ]*  # identifier
\((                            # begin args
(?: [ ]* ,? [ ]* [A-Za-z_][A-Za-z0-9_]* )* # args
)\)$                           # endargs''', re.VERBOSE)

def getSuitableParent(widgettype, initialwidget):
    """Find the nearest relevant parent for the widgettype given."""

    # find the parent to add the child to, we go up the tree looking
    # for possible parents
    parent = initialwidget
    wc = widgetfactory.thefactory.getWidgetClass(widgettype)
    while parent is not None and not wc.willAllowParent(parent):
        parent = parent.parent
    return parent
            
class Document( qt4.QObject ):
    """Document class for holding the graph data.

    Emits: sigModified when the document has been modified
           sigWiped when document is wiped
    """

    def __init__(self):
        """Initialise the document."""
        qt4.QObject.__init__( self )

        self.changeset = 0          # increased when the document changes
        self.suspendupdates = False # if True then do not notify listeners of updates
        self.clearHistory()
        self.wipe()

        # directories to examine when importing
        self.importpath = []

        # store custom functions and constants
        # consists of tuples of (name, type, value)
        # type is constant or function
        self.customs = []
        self.updateEvalContext()

    def suspendUpdates(self):
        """Holds sending update messages. This speeds up modification of the document."""
        assert not self.suspendupdates
        self.suspendchangeset = self.changeset
        self.suspendupdates = True

    def makeDefaultDoc(self):
        """Add default widgets to create document."""
        page = widgetfactory.thefactory.makeWidget('page', self.basewidget)
        graph = widgetfactory.thefactory.makeWidget('graph', page)
        self.setModified()
        self.setModified(False)
        self.changeset = 0

    def enableUpdates(self):
        """Reenables document updates."""
        assert self.suspendupdates
        self.suspendupdates = False
        if self.suspendchangeset != self.changeset:
            self.setModified()

    def log(self, message):
        """Log a message - this is emitted as a signal."""
        self.emit( qt4.SIGNAL("sigLog"), message )

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
            self.historyundo = self.historyundo[-9:] + [operation]
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
        return self.changeset == 0

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

        duplicate = self.data[name].duplicate()
        duplicate.document = self
        self.data[newname] = duplicate
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

        if not self.suspendupdates:
            self.emit( qt4.SIGNAL("sigModified"), ismodified )

    def isModified(self):
        """Return whether modified flag set."""
        return self.modified
    
    def printTo(self, printer, pages, scaling = 1., dpi = None,
                antialias = False):
        """Print onto printing device."""

        painter = Painter(scaling=scaling, dpi=dpi)
       
        painter.begin( printer )
        painter.setRenderHint(qt4.QPainter.Antialiasing,
                              antialias)
        painter.setRenderHint(qt4.QPainter.TextAntialiasing,
                              antialias)

        # This all assumes that only pages can go into the root widget
        num = len(pages)
        for count, page in enumerate(pages):
            self.basewidget.draw(painter, page)

            # start new pages between each page
            if count < num-1:
                printer.newPage()

        painter.end()

    def paintTo(self, painter, page, scaling = 1., dpi = None):
        """Paint page specified to the painter."""
        
        painter.veusz_scaling = scaling
        if dpi is not None:
            painter.veusz_pixperpt = dpi / 72.
        self.basewidget.draw(painter, page)

    def getNumberPages(self):
        """Return the number of pages in the document."""
        return len(self.basewidget.children)

    def getPage(self, pagenumber):
        """Return widget for page."""
        return self.basewidget.children[pagenumber]

    def _writeFileHeader(self, fileobj, type):
        """Write a header to a saved file of type."""

        fileobj.write('# Veusz %s (version %s)\n' % (type, utils.version()))
        try:
            fileobj.write('# User: %s\n' % os.environ['LOGNAME'] )
        except KeyError:
            pass
        fileobj.write('# Date: %s\n\n' %
                      time.strftime("%a, %d %b %Y %H:%M:%S +0000",
                                    time.gmtime()) )
        
    def saveCustomDefinitions(self, fileobj):
        """Save custom constants and functions."""

        for vals in self.customs:
            fileobj.write('AddCustom(%s, %s, %s)\n' %
                          tuple([repr(x) for x in vals]))

    def saveCustomFile(self, fileobj):
        """Export the custom settings to a file."""

        self._writeFileHeader(fileobj, 'custom definitions')
        self.saveCustomDefinitions(fileobj)

    def saveToFile(self, fileobj):
        """Save the text representing a document to a file."""

        self._writeFileHeader(fileobj, 'saved document')
        
        # add file directory to import path if we know it
        reldirname = None
        if getattr(fileobj, 'name', False):
            reldirname = os.path.dirname( os.path.abspath(fileobj.name) )
            fileobj.write('AddImportPath(%s)\n' % repr(reldirname))

        # add custom definitions
        self.saveCustomDefinitions(fileobj)

        # save those datasets which are linked
        # we do this first in case the datasets are overridden below
        savedlinks = {}
        for name, dataset in sorted(self.data.items()):
            dataset.saveLinksToSavedDoc(fileobj, savedlinks,
                                        relpath=reldirname)

        # save the remaining datasets
        for name, dataset in sorted(self.data.items()):
            dataset.saveToFile(fileobj, name)

        # save the actual tree structure
        fileobj.write(self.basewidget.getSaveText())
        
        self.setModified(False)

    def exportStyleSheet(self, fileobj):
        """Export the StyleSheet to a file."""

        self._writeFileHeader(fileobj, 'exported stylesheet')
        stylesheet = self.basewidget.settings.StyleSheet

        fileobj.write( stylesheet.saveText(True, rootname='') )

    def _exportBitmap(self, filename, pagenumber, dpi=100, antialias=True,
                      quality=85):
        """Export the pagenumber to the requested bitmap filename."""

        # firstly have to convert dpi to image size
        # have to use a temporary bitmap first
        tmp = qt4.QPixmap(1, 1)
        tmppainter = Painter(tmp)
        realdpi = tmppainter.device().logicalDpiY()
        width, height = self.basewidget.getSize(tmppainter)
        scaling = dpi/float(realdpi)
        tmppainter.end()
        del tmp, tmppainter

        # work out format
        format = os.path.splitext(filename)[1]
        if not format:
            format = '.png'
        # str is required as unicode not supported
        format = str(format[1:].lower())

        # create real output image
        pixmap = qt4.QPixmap(int(width*scaling), int(height*scaling))
        if format == 'png':
            # allows transparent
            pixmap.fill( qt4.Qt.transparent )
        else:
            # fill white
            pixmap.fill()        

        # paint to the image
        painter = Painter(pixmap)
        painter.setRenderHint(qt4.QPainter.Antialiasing,
                              antialias)
        painter.setRenderHint(qt4.QPainter.TextAntialiasing,
                              antialias)
        self.paintTo(painter, pagenumber, scaling=scaling, dpi=realdpi)
        painter.end()

        # write image to disk
        writer = qt4.QImageWriter()
        writer.setFormat(qt4.QByteArray(format))
        writer.setFileName(filename)
        writer.setQuality(quality)
        writer.write( pixmap.toImage() )
        
    def _exportPS(self, filename, page, color=True):
        """Postscript or eps format."""

        ext = os.path.splitext(filename)[1]

        printer = qt4.QPrinter()
        printer.setFullPage(True)

        # set printer parameters
        printer.setColorMode( (qt4.QPrinter.GrayScale, qt4.QPrinter.Color)[color] )

        if ext == '.pdf':
            f = qt4.QPrinter.PdfFormat
        else:
            try:
                f = qt4.QPrinter.PostScriptFormat
            except AttributeError:
                # < qt4.2 bah
                f = qt4.QPrinter.NativeFormat
        printer.setOutputFormat(f)
        printer.setOutputFileName(filename)
        printer.setCreator('Veusz %s' % utils.version())

        # draw the page
        printer.newPage()
        painter = Painter(printer)
        width, height = self.basewidget.getSize(painter)
        self.basewidget.draw(painter, page)
        painter.end()

        # fixup eps/pdf file - yuck HACK! - hope qt gets fixed
        # this makes the bounding box correct
        if ext == '.eps' or ext == '.pdf':
            # copy eps to a temporary file
            tmpfile = "%s.tmp.%i" % (filename, random.randint(0,1000000))
            fout = open(tmpfile, 'wb')
            fin = open(filename, 'rb')

            if ext == '.eps':
                # adjust bounding box
                for line in fin:
                    if line[:14] == '%%BoundingBox:':
                        # replace bounding box line by calculated one
                        parts = line.split()
                        widthfactor = float(parts[3]) / printer.width()
                        origheight = float(parts[4])
                        line = "%s %i %i %i %i\n" % (
                            parts[0], 0,
                            int(math.floor(origheight-widthfactor*height)),
                            int(math.ceil(widthfactor*width)),
                            int(math.ceil(origheight)) )
                    fout.write(line)

            elif ext == '.pdf':
                # change pdf bounding box and correct pdf index
                text = fin.read()
                text = utils.scalePDFMediaBox(text,
                                              printer.width(),
                                              width, height)
                text = utils.fixupPDFIndices(text)
                fout.write(text)

            fout.close()
            fin.close()
            os.remove(filename)
            os.rename(tmpfile, filename)

    def _exportSVG(self, filename, page):
        """Export document as SVG"""

        if qt4.PYQT_VERSION >= 0x40600:
            # custom paint devices don't work in old PyQt versions
            pixmap = qt4.QPixmap(1,1)
            dpi=90.
            painter = Painter(pixmap, scaling=1., dpi=dpi)
            width, height = self.basewidget.getSize(painter)
            painter.end()

            f = open(filename, 'w')
            paintdev = svg_export.SVGPaintDevice(f, width/dpi, height/dpi)
            painter = Painter(paintdev)
            self.basewidget.draw(painter, page)
            painter.end()
            f.close()

        else:
            # use built-in svg generation, which doesn't work very well
            # (no clipping, font size problems)
            import PyQt4.QtSvg

            # we have to make a temporary painter first to get the document size
            # this is because setSize needs to come before begin
            temprend =  PyQt4.QtSvg.QSvgGenerator()
            temprend.setFileName(filename)
            p = Painter(temprend)
            width, height = self.basewidget.getSize(p)
            p.end()

            # actually paint the image
            rend = PyQt4.QtSvg.QSvgGenerator()
            rend.setFileName(filename)
            rend.setSize( qt4.QSize(int(width), int(height)) )
            painter = Painter(rend)
            self.basewidget.draw(painter, page)
            painter.end()

    def _exportPIC(self, filename, page):
        """Export document as SVG"""

        pic = qt4.QPicture()
        painter = Painter(pic)
        self.basewidget.draw( painter, page )
        painter.end()
        pic.save(filename)

    def _exportEMF(self, filename, page):
        """Export document as EMF."""
        pixmap = qt4.QPixmap(1,1)
        dpi=75.
        painter = Painter(pixmap, scaling=1., dpi=dpi)
        width, height = self.basewidget.getSize(painter)
        painter.end()

        paintdev = emf_export.EMFPaintDevice(width/dpi, height/dpi,
                                             dpi=dpi)
        painter = Painter(paintdev)
        self.basewidget.draw( painter, page )
        painter.end()
        paintdev.paintEngine().saveFile(filename)

    def export(self, filename, pagenumber, color=True, dpi=100,
               antialias=True, quality=85):
        """Export the figure to the filename."""

        ext = os.path.splitext(filename)[1]

        if ext in ('.eps', '.pdf'):
            self._exportPS(filename, pagenumber, color=color)

        elif ext in ('.png', '.jpg', '.jpeg', '.bmp'):
            self._exportBitmap(filename, pagenumber, dpi=dpi,
                               antialias=antialias, quality=quality)

        elif ext == '.svg':
            self._exportSVG(filename, pagenumber)

        elif ext == '.pic':
            self._exportPIC(filename, pagenumber)

        elif ext == '.emf' and hasemf:
            self._exportEMF(filename, pagenumber)

        else:
            raise RuntimeError, "File type '%s' not supported" % ext
            
    def getExportFormats(self):
        """Get list of export formats:
        [
        (['ext'], 'Filename type'),
        ...
        ]
        """
        formats = [(["eps"], "Encapsulated Postscript"),
                   (["png"], "Portable Network Graphics"),
                   (["jpg"], "Jpeg bitmap"),
                   (["bmp"], "Windows bitmap"),
                   (["pdf"], "Portable Document Format"),
                   (["svg"], "Scalable Vector Graphics"),
                   #(["pic"], "QT Pic format"),
                   ]
        if hasemf:
            formats.append( (["emf"], "Windows Enhanced Metafile") )
        return formats

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

    def _processSafeImports(self, module, symbols):
        """Check what symbols are safe to import."""

        # empty list
        if not symbols:
            return symbols

        # do import anyway
        if setting.transient_settings['unsafe_mode']:
            return symbols

        # two-pass to ask user whether they want to import symbol
        for thepass in xrange(2):
            # remembered during session
            a = 'import_allowed'
            if a not in setting.transient_settings:
                setting.transient_settings[a] = defaultdict(set)
            allowed = setting.transient_settings[a][module]

            # not allowed during session
            a = 'import_notallowed'
            if a not in setting.transient_settings:
                setting.transient_settings[a] = defaultdict(set)
            notallowed = setting.transient_settings[a][module]

            # remembered in setting file
            a = 'import_allowed'
            if a not in setting.settingdb:
                setting.settingdb[a] = {}
            if module not in setting.settingdb[a]:
                setting.settingdb[a][module] = {}
            allowed_always = setting.settingdb[a][module]

            # collect up
            toimport = []
            possibleimport = []
            for symbol in symbols:
                if symbol in allowed or symbol in allowed_always:
                    toimport.append(symbol)
                elif symbol not in notallowed:
                    possibleimport.append(symbol)

            # nothing to do, so leave
            if not possibleimport:
                break

            # only ask the user the first time
            if thepass == 0:
                self.emit(qt4.SIGNAL('check_allowed_imports'), module,
                          possibleimport)

        return toimport

    def _updateEvalContextImport(self, module, val):
        """Add an import statement to the eval function context."""
        if module_re.match(module):
            # work out what is safe to import
            symbols = identifier_split_re.findall(val)
            toimport = self._processSafeImports(module, symbols)
            if toimport:
                defn = 'from %s import %s' % (module,
                                              ', '.join(toimport))
                try:
                    exec defn in self.eval_context
                except Exception:
                    self.log("Failed to import '%s' from "
                             "module '%s'" % (', '.join(toimport),
                                              module))
                    return

            delta = set(symbols)-set(toimport)
            if delta:
                self.log("Did not import '%s' from module '%s'" %
                         (', '.join(list(delta)), module))

        else:
            self.log( "Invalid module name '%s'" % module )

    def _updateEvalContextFuncOrConst(self, ctype, name, val):
        """Update a function or constant in eval function context."""

        if ctype == 'constant':
            if not identifier_re.match(name):
                self.log( "Invalid constant name '%s'" % name )
                return
            defn = val
        elif ctype == 'function':
            m = function_re.match(name)
            if not m:
                self.log( "Invalid function specification '%s'" % name )
                return
            name = funcname = m.group(1)
            args = m.group(2)
            defn = 'lambda %s: %s' % (args, val)

        # evaluate, but we ignore any unsafe commands or exceptions
        checked = utils.checkCode(defn)
        if checked is not None:
            self.log( "Expression '%s' failed safe code test" %
                      defn )
            return
        try:
            self.eval_context[name] = eval(defn, self.eval_context)
        except Exception, e:
            self.log( "Error evaluating '%s': '%s'" %
                      (name, unicode(e)) )

    def updateEvalContext(self):
        """To be called after custom constants or functions are changed.
        This sets up a safe environment where things can be evaluated
        """
        
        self.eval_context = c = {}

        # add numpy things
        # we try to avoid various bits and pieces for safety
        for name, val in N.__dict__.iteritems():
            if ( (callable(val) or type(val)==float) and
                 name not in __builtins__ and
                 name[:1] != '_' and name[-1:] != '_' ):
                c[name] = val
        
        # safe functions
        c['os_path_join'] = os.path.join
        c['os_path_dirname'] = os.path.dirname
        c['veusz_markercodes'] = tuple(utils.MarkerCodes)

        # custom definitions
        for ctype, name, val in self.customs:
            name = name.strip()
            val = val.strip()
            if ctype == 'constant' or ctype == 'function':
                self._updateEvalContextFuncOrConst(ctype, name, val)
            elif ctype == 'import':
                self._updateEvalContextImport(name, val)
            else:
                raise ValueError, 'Invalid custom type'

class Painter(qt4.QPainter):
    """A painter which allows the program to know which widget it is
    currently drawing."""
    
    def __init__(self, *args, **argsv):
        qt4.QPainter.__init__(self, *args)

        self.veusz_scaling = argsv.get('scaling', 1.)
        if 'dpi' in argsv and argsv['dpi'] is not None:
            self.veusz_pixperpt = argsv['dpi'] / 72.

    def beginPaintingWidget(self, widget, bounds):
        """Keep track of the widget currently being painted."""
        pass

    def endPaintingWidget(self):
        """Widget is now finished."""
        pass
    
