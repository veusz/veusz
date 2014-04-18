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

"""A class to represent Veusz documents, with dataset classes."""

from __future__ import division
import codecs
import os.path
import re
import traceback
import datetime
from collections import defaultdict

import numpy as N

try:
    import h5py
except ImportError:
    h5py = None

from ..compat import crange, citems, cvalues, cstr, cexec, CStringIO
from .. import qtall as qt4

from . import widgetfactory
from . import datasets
from . import painthelper

from .. import utils
from .. import setting

def _(text, disambiguation=None, context="Document"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

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
(?: [ ]* ,? [ ]* [A-Za-z_][A-Za-z0-9_]* )*     # named args
(?: [ ]* ,? [ ]* \*[A-Za-z_][A-Za-z0-9_]* )?   # *args
(?: [ ]* ,? [ ]* \*\*[A-Za-z_][A-Za-z0-9_]* )? # **kwargs
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
    """

    pluginsloaded = False

    # this is emitted when the document is modified
    signalModified = qt4.pyqtSignal(int)
    # emited to log a message
    sigLog = qt4.pyqtSignal(cstr)
    # emitted when document wiped
    sigWiped = qt4.pyqtSignal()
    # to ask whether the import is allowed (module name and symbol list)
    sigAllowedImports = qt4.pyqtSignal(cstr, list)

    def __init__(self):
        """Initialise the document."""
        qt4.QObject.__init__( self )

        if not Document.pluginsloaded:
            Document.loadPlugins()
            Document.pluginsloaded = True

        # change tracking of document as a whole
        self.changeset = 0            # increased when the document changes

        # map tags to dataset names
        self.datasettags = defaultdict(list)

        # if set, do not notify listeners of updates
        # wait under enableUpdates
        self.suspendupdates = []

        # default document locale
        self.locale = qt4.QLocale()

        self.clearHistory()
        self.wipe()

        # directories to examine when importing
        self.importpath = []

        # store custom functions and constants
        # consists of tuples of (name, type, value)
        # type is constant or function
        # we use this format to preserve evaluation order
        self.customs = []
        self.updateEvalContext()

        # copy default colormaps
        self.colormaps = dict(utils.defaultcolormaps)

        # copies of validated compiled expressions
        self.exprcompiled = {}
        self.exprfailed = set()
        self.exprfailedchangeset = -1

    def wipe(self):
        """Wipe out any stored data."""
        self.data = {}
        self.basewidget = widgetfactory.thefactory.makeWidget(
            'document', None, None)
        self.basewidget.document = self
        self.setModified(False)
        self.sigWiped.emit()

    def clearHistory(self):
        """Clear any history."""
        self.historybatch = []
        self.historyundo = []
        self.historyredo = []
        
    def suspendUpdates(self):
        """Holds sending update messages.
        This speeds up modification of the document and prevents the document
        from being updated on the screen."""
        self.suspendupdates.append(self.changeset)

    def enableUpdates(self):
        """Reenables document updates."""
        changeset = self.suspendupdates.pop()
        if len(self.suspendupdates) == 0 and changeset != self.changeset:
            # bump this up as some watchers might ignore this otherwise
            self.changeset += 1
            self.setModified()

    def makeDefaultDoc(self):
        """Add default widgets to create document."""
        page = widgetfactory.thefactory.makeWidget('page', self.basewidget)
        widgetfactory.thefactory.makeWidget('graph', page)
        self.setModified()
        self.setModified(False)
        self.changeset = 0

    def log(self, message):
        """Log a message - this is emitted as a signal."""
        self.sigLog.emit(message)

    def applyOperation(self, operation):
        """Apply operation to the document.
        
        Operations represent atomic actions which can be done to the document
        and undone.

        Updates are suspended during the operation.
        """

        self.suspendUpdates()
        try:
            retn = operation.do(self)
            self.changeset += 1
        except:
            self.enableUpdates()
            raise
        self.enableUpdates()

        if self.historybatch:
            # in batch mode, create an OperationMultiple for all changes
            self.historybatch[-1].addOperation(operation)
        else:
            # standard mode
            self.historyundo = self.historyundo[-9:] + [operation]
        self.historyredo = []

        return retn

    def batchHistory(self, batch):
        """Enable/disable batch history mode.
        
        In this mode further operations are added to the OperationMultiple specified,
        until batchHistory is called with None.
        
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
        self.suspendUpdates()
        try:
            operation.undo(self)
            self.changeset += 1
        except:
            self.enableUpdates()
            raise
        self.enableUpdates()
        self.historyredo.append(operation)
        
    def canUndo(self):
        """Returns True if previous operation can be removed."""
        return len(self.historyundo) != 0

    def redoOperation(self):
        """Redo undone operations."""
        operation = self.historyredo.pop()
        return self.applyOperation(operation)

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

    def isBlank(self):
        """Does the document contain widgets and no data"""
        return self.changeset == 0

    def setData(self, name, dataset):
        """Set data to val, with symmetric or negative and positive errors."""
        self.data[name] = dataset
        dataset.document = self
        
        # update the change tracking
        self.setModified()
    
    def deleteData(self, name):
        """Remove a dataset"""
        if name in self.data:
            del self.data[name]
            self.setModified()

    def modifiedData(self, dataset):
        """The named dataset was modified"""
        for name, ds in citems(self.data):
            if ds is dataset:
                self.setModified()

    def getLinkedFiles(self, filenames=None):
        """Get a list of LinkedFile objects used by the document.
        if filenames is a set, only get the objects with filenames given
        """
        links = set()
        for ds in cvalues(self.data):
            if ds.linked and (filenames is None or
                              ds.linked.filename in filenames):
                links.add(ds.linked)
        return list(links)

    def reloadLinkedDatasets(self, filenames=None):
        """Reload linked datasets from their files.
        If filenames is a set(), only reload from these filenames

        Returns a tuple of
        - List of datasets read
        - Dict of tuples containing dataset names and number of errors
        """

        links = self.getLinkedFiles(filenames=filenames)

        read = []
        errors = {}

        # load in the files, merging the vars read and errors
        if links:
            for lf in links:
                nread, nerrors = lf.reloadLinks(self)
                read += nread
                errors.update(nerrors)
            self.setModified()

        read.sort()
        return (read, errors)

    def datasetName(self, dataset):
        """Find name for given dataset, raising ValueError if missing."""
        for name, ds in citems(self.data):
            if ds is dataset:
                return name
        raise ValueError("Cannot find dataset")

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

        if len(self.suspendupdates) == 0:
            self.signalModified.emit(ismodified)

    def isModified(self):
        """Return whether modified flag set."""
        return self.modified

    @classmethod
    def loadPlugins(kls, pluginlist=None):
        """Load plugins and catch exceptions."""
        if pluginlist is None:
            pluginlist = setting.settingdb.get('plugins', [])

        for plugin in pluginlist:
            try:
                cexec(compile(open(plugin).read(), plugin, 'exec'), dict())
            except Exception:
                err = _('Error loading plugin %s\n\n%s') % (
                    plugin, traceback.format_exc())
                qt4.QMessageBox.critical(None, _("Error loading plugin"), err)

    def printTo(self, printer, pages, scaling = 1., dpi = None,
                antialias = False):
        """Print onto printing device."""

        dpi = (printer.logicalDpiX(), printer.logicalDpiY())
        painter = painthelper.DirectPainter(printer)
        if antialias:
            painter.setRenderHint(qt4.QPainter.Antialiasing, True)
            painter.setRenderHint(qt4.QPainter.TextAntialiasing, True)
   
        with painter:
            # This all assumes that only pages can go into the root widget
            for count, page in enumerate(pages):
                painter.save()
                size = self.pageSize(page, dpi=dpi)
                helper = painthelper.PaintHelper(size, dpi=dpi, directpaint=painter)
                self.paintTo(helper, page)
                painter.restore()

                # start new pages between each page
                if count < len(pages)-1:
                    printer.newPage()

    def paintTo(self, painthelper, page):
        """Paint page specified to the paint helper."""
        self.basewidget.draw(painthelper, page)

    def getNumberPages(self):
        """Return the number of pages in the document."""
        return len(self.basewidget.children)

    def getPage(self, pagenumber):
        """Return widget for page."""
        return self.basewidget.children[pagenumber]

    def datasetTags(self):
        """Get list of all tags in datasets."""
        tags = set()
        for dataset in cvalues(self.data):
            tags.update(dataset.tags)
        return sorted(tags)

    def _writeFileHeader(self, fileobj, type):
        """Write a header to a saved file of type."""

        fileobj.write('# Veusz %s (version %s)\n' % (type, utils.version()))
        fileobj.write('# Saved at %s\n\n' %
                      datetime.datetime.utcnow().isoformat())

    def saveCustomDefinitions(self, fileobj):
        """Save custom constants and functions."""

        for vals in self.customs:
            fileobj.write('AddCustom(%s, %s, %s)\n' %
                          tuple([repr(x) for x in vals]))

    def saveDatasetTags(self, fileobj):
        """Write dataset tags to output file"""

        # get a list of all tags and which datasets have them
        bytag = defaultdict(list)
        for name, dataset in sorted(citems(self.data)):
            for t in dataset.tags:
                bytag[t].append(name)

        # write out tags
        for tag, val in sorted(citems(bytag)):
            fileobj.write('TagDatasets(%s, %s)\n' %
                          (repr(tag), repr(val)))

    def saveCustomFile(self, fileobj):
        """Export the custom settings to a file."""

        self._writeFileHeader(fileobj, 'custom definitions')
        self.saveCustomDefinitions(fileobj)

    def saveToFile(self, fileobj):
        """Save the text representing a document to a file.

        The ordering can be important, as some things override
        previous steps:

         - Tagging doesn't work if the dataset isn't
           already defined.
         - Loading from files may bring in new datasets which
           override defined datasets, so save links first
        """

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
        for name, dataset in sorted(citems(self.data)):
            dataset.saveLinksToSavedDoc(fileobj, savedlinks,
                                        relpath=reldirname)

        # save the remaining datasets
        for name, dataset in sorted(citems(self.data)):
            dataset.saveToFile(fileobj, name)

        # save tags of datasets
        self.saveDatasetTags(fileobj)

        # save the actual tree structure
        fileobj.write(self.basewidget.getSaveText())
        
        self.setModified(False)

    def saveToHDF5File(self, fileobj):
        """Save to HDF5 (h5py) output file given."""

        # groups in output hdf5
        vszgrp = fileobj.create_group('Veusz')
        vszgrp.attrs['vsz_version'] = utils.version()
        vszgrp.attrs['vsz_saved_at'] = datetime.datetime.utcnow().isoformat()
        vszgrp.attrs['vsz_format'] = 1  # version number (currently unused)
        datagrp = vszgrp.create_group('Data')
        docgrp = vszgrp.create_group('Document')

        textstream = CStringIO()

        self._writeFileHeader(textstream, 'saved document')

        # add file directory to import path if we know it
        reldirname = None
        if getattr(fileobj, 'filename', False):
            reldirname = os.path.dirname( os.path.abspath(fileobj.filename) )
            textstream.write('AddImportPath(%s)\n' % repr(reldirname))

        # add custom definitions
        self.saveCustomDefinitions(textstream)

        # save those datasets which are linked
        # we do this first in case the datasets are overridden below
        savedlinks = {}
        for name, dataset in sorted(citems(self.data)):
            dataset.saveLinksToSavedDoc(textstream, savedlinks,
                                        relpath=reldirname)

        # save the remaining datasets
        for name, dataset in sorted(citems(self.data)):
            dataset.saveToFile(textstream, name, mode='hdf5', hdfgroup=datagrp)

        # handle tagging
        # get a list of all tags and which datasets have them
        bytag = defaultdict(list)
        for name, dataset in sorted(citems(self.data)):
            for t in dataset.tags:
                bytag[t].append(name)

        # write out tags as datasets
        tagsgrp = docgrp.create_group('Tags')
        for tag, dsnames in sorted(citems(bytag)):
            tagsgrp[tag] = [v.encode('utf-8') for v in sorted(dsnames)]

        # save the actual tree structure
        textstream.write(self.basewidget.getSaveText())

        # create single dataset contains document
        docgrp['document'] = [ textstream.getvalue().encode('utf-8') ]

        self.setModified(False)

    def save(self, filename, mode='vsz'):
        """Save to output file.

        mode is 'vsz' or 'hdf5'
        """
        if mode == 'vsz':
            with codecs.open(filename, 'w', 'utf-8') as f:
                self.saveToFile(f)
        elif mode == 'hdf5':
            if h5py is None:
                raise RuntimeError('Missing h5py module')
            with h5py.File(filename, 'w') as f:
                self.saveToHDF5File(f)
        else:
            raise RuntimeError('Invalid save mode')

    def load(self, filename, mode='vsz', callbackunsafe=None):
        """Load document from file.

        mode is 'vsz' or 'hdf5'
        """
        from . import loader
        loader.loadDocument(self, filename, mode=mode,
                            callbackunsafe=callbackunsafe)

    def exportStyleSheet(self, fileobj):
        """Export the StyleSheet to a file."""

        self._writeFileHeader(fileobj, 'exported stylesheet')
        stylesheet = self.basewidget.settings.StyleSheet

        fileobj.write( stylesheet.saveText(True, rootname='') )

    def _pagedocsize(self, widget, dpi, scaling, integer):
        """Helper for page or doc size."""
        if dpi is None:
            p = qt4.QPixmap(1, 1)
            dpi = (p.logicalDpiX(), p.logicalDpiY())
        helper = painthelper.PaintHelper( (1,1), dpi=dpi, scaling=scaling )
        w = widget.settings.get('width').convert(helper)
        h = widget.settings.get('height').convert(helper)
        if integer:
            return int(w), int(h)
        else:
            return w, h        

    def pageSize(self, pagenum, dpi=None, scaling=1., integer=True):
        """Get the size of a particular page in pixels.

        If dpi is None, use the default Qt screen dpi
        Use dpi if given."""

        page = self.basewidget.getPage(pagenum)
        if page is None:
            return self.docSize(dpi=dpi, scaling=scaling, integer=integer)
        return self._pagedocsize(
            page, dpi=dpi, scaling=scaling, integer=integer)

    def docSize(self, dpi=None, scaling=1., integer=True):
        """Get size for document."""
        return self._pagedocsize(
            self.basewidget,
            dpi=dpi, scaling=scaling, integer=integer)

    def resolveItem(self, fromwidget, where):
        """Resolve item relative to fromwidget.
        Returns a widget, setting or settings as appropriate.
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
                p = obj.parent
                if p is None:
                    raise ValueError("Base graph has no parent")
                obj = p
            elif p == '.' or len(p) == 0:
                pass
            else:
                if obj.isWidget():
                    child = obj.getChild(p)
                    if child is not None:
                        obj = child
                    else:
                        if p in obj.settings:
                            obj = obj.settings[p]
                        else:
                            raise ValueError("Widget has no child %s" % p)
                else:
                    if isinstance(obj, setting.Settings):
                        try:
                            obj = obj.get(p)
                        except KeyError:
                            raise ValueError("Settings has no child %s" % p)
                    else:
                        raise ValueError("Item has no children")

        # return widget
        return obj

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
                    raise ValueError("Base graph has no parent")
                obj = p
            elif p == '.' or len(p) == 0:
                # relative to here
                pass
            else:
                # child specified
                obj = obj.getChild( p )
                if obj is None:
                    raise ValueError("Child '%s' does not exist" % p)

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
        for thepass in crange(2):
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
                self.sigAllowedImports.emit(module, possibleimport)

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
                    cexec(defn, self.eval_context)
                except Exception:
                    self.log(_("Failed to import '%s' from "
                               "module '%s'") % (', '.join(toimport),
                                                 module))
                    return

            delta = set(symbols)-set(toimport)
            if delta:
                self.log(_("Did not import '%s' from module '%s'") %
                         (', '.join(list(delta)), module))

        else:
            self.log( _("Invalid module name '%s'") % module )

    def validateProcessColormap(self, colormap):
        """Validate and process a colormap value.

        Returns a list of B,G,R,alpha tuples or raises ValueError if a problem."""

        try:
            if len(colormap) < 2:
                raise ValueError( _("Need at least two entries in colormap") )
        except TypeError:
            raise ValueError( _("Invalid type for colormap") )

        out = []
        for entry in colormap:
            for v in entry:
                try:
                    v - 0
                except TypeError:
                    raise ValueError(
                        _("Colormap entries should be numerical") )
                if v < 0 or v > 255:
                    raise ValueError(
                        _("Colormap entries should be between 0 and 255") )

            if len(entry) == 3:
                out.append( (int(entry[2]), int(entry[1]), int(entry[0]),
                             255) )
            elif len(entry) == 4:
                out.append( (int(entry[2]), int(entry[1]), int(entry[0]),
                             int(entry[3])) )
            else:
                raise ValueError( _("Each colormap entry consists of R,G,B "
                                    "and optionally alpha values") )

        return tuple(out)

    def _updateEvalContextColormap(self, name, val):
        """Add a colormap entry."""

        try:
            cmap = self.validateProcessColormap(val)
        except ValueError as e:
            self.log( cstr(e) )
        else:
            self.colormaps[ cstr(name) ] = cmap

    def _updateEvalContextFuncOrConst(self, ctype, name, val):
        """Update a function or constant in eval function context."""

        if ctype == 'constant':
            if not identifier_re.match(name):
                self.log( _("Invalid constant name '%s'") % name )
                return
            defn = val
        elif ctype == 'function':
            m = function_re.match(name)
            if not m:
                self.log( _("Invalid function specification '%s'") % name )
                return
            name = m.group(1)
            args = m.group(2)
            defn = 'lambda %s: %s' % (args, val)

        # evaluate, but we ignore any unsafe commands or exceptions
        comp = self.compileCheckedExpression(defn)
        if comp is None:
            return
        try:
            self.eval_context[name] = eval(comp, self.eval_context)
        except Exception as e:
            self.log( _("Error evaluating '%s': '%s'") %
                      (name, cstr(e)) )

    def compileCheckedExpression(self, expr, origexpr=None):
        """Compile expression and check for errors.

        origexpr is an expression to show in error messages. This is
        used if replacements have been done, etc.
        """

        try:
            return self.exprcompiled[expr]
        except KeyError:
            pass

        # track failed compilations, so we only print them once
        if self.exprfailedchangeset != self.changeset:
            self.exprfailedchangeset = self.changeset
            self.exprfailed.clear()
        elif expr in self.exprfailed:
            return None

        if origexpr is None:
            origexpr = expr

        try:
            checked = utils.compileChecked(
                expr,
                ignoresecurity=setting.transient_settings['unsafe_mode'])
        except utils.SafeEvalException as e:
            self.log(
                _("Unsafe expression '%s': %s") % (origexpr, cstr(e)))
            self.exprfailed.add(expr)
            return None
        except Exception as e:
            self.log(
                _("Error in expression '%s': %s") % (origexpr, cstr(e)))
            return None
        else:
            self.exprcompiled[expr] = checked
            return checked

    def updateEvalContext(self):
        """To be called after custom constants or functions are changed.
        This sets up a safe environment where things can be evaluated
        """
        
        self.eval_context = c = {}

        # add numpy things
        # we try to avoid various bits and pieces for safety
        for name, val in citems(N.__dict__):
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
            if ctype == 'constant' or ctype == 'function':
                self._updateEvalContextFuncOrConst(ctype, name, val.strip())
            elif ctype == 'import':
                self._updateEvalContextImport(name, val)
            elif ctype == 'colormap':
                self._updateEvalContextColormap(name, val)
            else:
                raise ValueError('Invalid custom type')

    def customDict(self):
        """Return a dictionary mapping custom names to (idx, type, value)."""
        retn = {}
        for i, (ctype, name, val) in enumerate(self.customs):
            retn[name] = (i, ctype, val)
        return retn

    def evalDatasetExpression(self, expr, part='data', datatype='numeric',
                              dimensions=1,
                              _cache={}, _cache_changeset=[None]):
        """Return dataset after evaluating a dataset expression.
        part is 'data', 'serr', 'perr' or 'nerr' - these are the
        dataset parts which are evaluated by the expression

        None is returned on error
        """

        key = (expr, part, datatype, dimensions)
        if _cache_changeset[0] != self.changeset:
            _cache_changeset[0] = self.changeset
            _cache.clear()
        elif key in _cache:
            return _cache[key]

        _cache[key] = ds = datasets.evalDatasetExpression(
            self, expr, part=part, datatype=datatype, dimensions=dimensions)
        return ds

    def valsToDataset(self, vals, datatype, dimensions):
        """Return a dataset given a numpy array of values."""

        if datatype == 'numeric':
            try:
                nvals = N.array(vals, dtype=N.float64)

                if nvals.ndim == dimensions:
                    if nvals.ndim == 1:
                        return datasets.Dataset(data=nvals)
                    elif nvals.ndim == 2:
                        return datasets.Dataset2D(nvals)
            except ValueError:
                pass

        elif datatype == 'text':
            try:
                return datasets.DatasetText([cstr(x) for x in vals])
            except ValueError:
                pass

        raise RuntimeError('Invalid array')

    def walkNodes(self, tocall, root=None,
                  nodetypes=('widget', 'setting', 'settings'),
                  _path=None):
        """Walk the widget/settings/setting nodes in the document.
        For each one call tocall(path, node).
        nodetypes is tuple of possible node types
        """

        if root is None:
            root = self.basewidget
        if _path is None:
            _path = root.path

        if root.nodetype in nodetypes:
            tocall(_path, root)

        if root.nodetype == 'widget':
            # get rid of // at start of path
            if _path == '/':
                _path = ''

            # do the widget's children
            for w in root.children:
                self.walkNodes(tocall, root=w, nodetypes=nodetypes,
                               _path = _path + '/' + w.name)
            # then do the widget's settings
            self.walkNodes(tocall, root=root.settings,
                           nodetypes=nodetypes, _path=_path)
        elif root.nodetype == 'settings':
            # do the settings of the settings
            for name, s in sorted(citems(root.setdict)):
                self.walkNodes(tocall, root=s, nodetypes=nodetypes,
                               _path = _path + '/' + s.name)
        # elif root.nodetype == 'setting': pass

    def getColormap(self, name, invert):
        """Get colormap with name given (returning grey if does not exist)."""
        cmap = self.colormaps.get(name, self.colormaps['grey'])
        if invert:
            if cmap[0][0] >= 0:
                return cmap[::-1]
            else:
                # ignore marker at beginning for stepped maps
                return tuple([cmap[0]] + list(cmap[-1:0:-1]))
        return cmap

    def formatValsWithDatatypeToText(self, vals, displaydatatype):
        """Given a set of values, datatype, return a list of strings
        corresponding to these data."""

        if displaydatatype == 'text':
            return vals
        elif displaydatatype == 'numeric':
            return [ utils.formatNumber(val, '%Vg', locale=self.locale)
                     for val in vals ]
        elif displaydatatype == 'date':
            return [ utils.dateFloatToString(val) for val in vals ]
        else:
            raise RuntimeError('Invalid data type')
