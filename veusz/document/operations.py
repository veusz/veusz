#    Copyright (C) 2006 Jeremy S. Sanders
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

"""Represents atomic operations to take place on a document which can be undone.
Rather than the document modified directly, this interface should be used.

Operations should be passed to the document to be enacted with applyOperation

Each operation provides do(document) and undo(document) methods.
Operations store paths to objects to be modified rather than object references
because some operations cannot restore references (e.g. add object)
"""

from __future__ import division
import os.path
from itertools import izip

import numpy as N

from . import datasets
from . import widgetfactory
from . import simpleread
from . import readcsv
from . import linked

from .. import utils
from .. import plugins
from .. import qtall as qt4

def _(text, disambiguation=None, context="Operations"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

###############################################################################
# Setting operations

class OperationSettingSet(object):
    """Set a variable to a value."""

    descr = _('change setting')
    
    def __init__(self, setting, value):
        """Set the setting to value.
        Setting may be a widget path
        """
        
        if isinstance(setting, basestring):
            self.settingpath = setting
        else:
            self.settingpath = setting.path
        self.value = value
        
    def do(self, document):
        """Apply setting variable."""
        setting = document.resolveFullSettingPath(self.settingpath)
        if setting.isReference():
            self.oldvalue = setting.getReference()
        else:
            self.oldvalue = setting.get()
        setting.set(self.value)
        
    def undo(self, document):
        """Return old value back..."""
        setting = document.resolveFullSettingPath(self.settingpath)
        setting.set(self.oldvalue)

class OperationSettingPropagate(object):
    """Propagate setting to other widgets."""
    
    descr = _('propagate setting')
    
    def __init__(self, setting, widgetname = None, root = None,
                 maxlevels = -1):

        """Take the setting given, and propagate it to other widgets,
        according to the parameters here.
        
        If widgetname is given then only propagate it to widgets with
        the name given.

        widgets are located from the widget given (root if not set)
        
        Up to maxlevels levels of widgets are changed (<0 means infinite)
        """

        self.val = setting.val
        self.widgetname = widgetname
        if root:
            self.rootpath = root.path
        else:
            self.rootpath = None
        self.maxlevels = maxlevels

        # work out path of setting relative to widget
        path = []
        s = setting
        while not s.isWidget():
            path.insert(0, s.name)
            s = s.parent
        self.setpath = path[1:]
        self.widgettype = s.typename
        
    def do(self, document):
        """Apply the setting propagation."""
        # default is root widget
        if not self.rootpath:
            root = document.basewidget
        else:
            root = document.resolveFullWidgetPath(self.rootpath)
            
        # get a list of matching widgets
        widgetlist = []
        self._recursiveGet(root, self.widgetname, self.widgettype, widgetlist,
                           self.maxlevels)

        self.restorevals = {}
        # set the settings for the widgets
        for w in widgetlist:
            # lookup the setting
            s = w.settings
            for i in self.setpath:
                s = s.get(i)

            self.restorevals[s.path] = s.val
            s.set(self.val)
          
    def undo(self, document):
        """Undo all those changes."""
        
        for setpath, setval in self.restorevals.iteritems():
            setting = document.resolveFullSettingPath(setpath)
            setting.set(setval)

    def _recursiveGet(root, name, typename, outlist, maxlevels):
        """Add those widgets in root with name and type to outlist.
    
        If name or typename are None, then ignore the criterion.
        maxlevels is the maximum number of levels to check
        """
    
        if maxlevels != 0:
    
            # if levels is not zero, add the children of this root
            newmaxlevels = maxlevels - 1
            for w in root.children:
                if ( (w.name == name or name is None) and
                     (w.typename == typename or typename is None) ):
                    outlist.append(w)
    
                OperationSettingPropagate._recursiveGet(w, name, typename,
                                                        outlist, newmaxlevels)

    _recursiveGet = staticmethod(_recursiveGet)

###############################################################################
# Widget operations
        
class OperationWidgetRename(object):
    """Rename widget."""
    
    descr = _('rename')
    
    def __init__(self, widget, newname):
        """Rename the widget to newname."""
        
        self.widgetpath = widget.path
        self.newname = newname
        
    def do(self, document):
        """Rename widget."""
        
        widget = document.resolveFullWidgetPath(self.widgetpath)
        self.oldname = widget.name
        widget.rename(self.newname)
        self.newpath = widget.path
        
    def undo(self, document):
        """Undo rename."""
        
        widget = document.resolveFullWidgetPath(self.newpath)
        widget.rename(self.oldname)
        
class OperationWidgetDelete(object):
    """Delete widget."""
    
    descr = _('delete')
    
    def __init__(self, widget):
        """Delete the widget."""
        
        self.widgetpath = widget.path
        
    def do(self, document):
        """Delete widget."""
        
        self.oldwidget = document.resolveFullWidgetPath(self.widgetpath)
        oldparent = self.oldwidget.parent
        self.oldwidget.parent = None
        self.oldparentpath = oldparent.path
        self.oldindex = oldparent.children.index(self.oldwidget)
        oldparent.removeChild(self.oldwidget.name)
        
    def undo(self, document):
        """Restore deleted widget."""
        
        oldparent = document.resolveFullWidgetPath(self.oldparentpath)
        self.oldwidget.parent = oldparent
        oldparent.addChild(self.oldwidget, index=self.oldindex)

class OperationWidgetsDelete(object):
    """Delete mutliple widget."""
    
    descr = _('delete')
    
    def __init__(self, widgets):
        """Delete the widget."""
        self.widgetpaths = [w.path for w in widgets]
        
    def do(self, document):
        """Delete widget."""
        
        # ignore widgets which share ancestry
        # as deleting the parent deletes the child
        widgetpaths = list(self.widgetpaths)
        widgetpaths.sort( cmp=lambda a, b: len(a)-len(b) )
        i = 0
        while i < len(widgetpaths):
            wp = widgetpaths[i]
            for j in xrange(i):
                if wp[:len(widgetpaths[j])+1] == widgetpaths[j]+'/':
                    del widgetpaths[i]
                    break
            else:
                i += 1

        self.oldwidgets = []
        self.oldparentpaths = []
        self.oldindexes = []

        # delete each widget keeping track of details
        for path in widgetpaths:
            self.oldwidgets.append( document.resolveFullWidgetPath(path) )
            oldparent = self.oldwidgets[-1].parent
            self.oldparentpaths.append( oldparent.path )
            self.oldindexes.append( oldparent.children.index(self.oldwidgets[-1]) )
            oldparent.removeChild(self.oldwidgets[-1].name)

    def undo(self, document):
        """Restore deleted widget."""
        
        # put back widgets in reverse order so that indexes are corrent
        for i in xrange(len(self.oldwidgets)-1,-1,-1):
            oldparent = document.resolveFullWidgetPath(self.oldparentpaths[i])
            oldparent.addChild(self.oldwidgets[i], index=self.oldindexes[i])
        
class OperationWidgetMoveUpDown(object):
    """Move a widget up or down in the hierarchy."""

    descr = _('move')
    
    def __init__(self, widget, direction):
        """Move the widget specified up or down in the hierarchy.
        
        direction is -1 for 'up' or +1 for 'down'
        """
        
        self.widgetpath = widget.path
        self.direction = direction
    
    def do(self, document):
        """Move the widget."""
        
        widget = document.resolveFullWidgetPath(self.widgetpath)
        parent = widget.parent
        self.suceeded = parent.moveChild(widget, self.direction)
        self.newpath = widget.path
    
    def undo(self, document):
        """Move it back."""
        if self.suceeded:
            widget = document.resolveFullWidgetPath(self.newpath)
            parent = widget.parent
            parent.moveChild(widget, -self.direction)
            
class OperationWidgetMove(object):
    """Move a widget arbitrarily in the hierarchy."""

    descr = _('move')

    def __init__(self, oldchildpath, newparentpath, newindex):
        """Move widget with path oldchildpath to be a child of
        newparentpath and with index newindex."""
        self.oldchildpath = oldchildpath
        self.newparentpath = newparentpath
        self.newindex = newindex

    def do(self, document):
        """Move widget."""

        child = document.resolveFullWidgetPath(self.oldchildpath)
        oldparent = child.parent
        newparent = document.resolveFullWidgetPath(self.newparentpath)
        self.oldchildindex = oldparent.children.index(child)
        self.oldparentpath = oldparent.path
        self.oldname = None

        if self.newindex < 0:
            # convert negative index to normal index
            self.newindex = len(newparent.children)

        if oldparent is newparent:
            # moving within same parent
            self.movemode = 'sameparent'
            del oldparent.children[self.oldchildindex]
            if self.newindex > self.oldchildindex:
                self.newindex -= 1
            oldparent.children.insert(self.newindex, child)
        else:
            # moving to different parent
            self.movemode = 'differentparent'

            # remove from old parent
            del oldparent.children[self.oldchildindex]

            # current names of children
            childnames = newparent.childnames

            # record previous parent and position
            newparent.children.insert(self.newindex, child)
            child.parent = newparent

            # set a new name, if required
            if child.name in childnames:
                self.oldname = child.name
                child.name = child.chooseName()

        self.newchildpath = child.path

    def undo(self, document):
        """Undo move."""

        newparent = document.resolveFullWidgetPath(self.newparentpath)
        child = document.resolveFullWidgetPath(self.newchildpath)
        oldparent = document.resolveFullWidgetPath(self.oldparentpath)

        # remove from new parent
        del newparent.children[self.newindex]
        # restore parent
        oldparent.children.insert(self.oldchildindex, child)
        child.parent = oldparent

        # restore name
        if self.oldname is not None:
            child.name = self.oldname

class OperationWidgetAdd(object):
    """Add a widget of specified type to parent."""

    descr = _('add')
    
    def __init__(self, parent, type, autoadd=True, name=None,
                 index=-1, **defaultvals):
        """Add a widget of type given
        
        parent is the parent widget
        type is the type to add (string)
        autoadd adds children automatically for some widgets
        name is the (optional) name of the new widget
        index is position in parent to add the widget
        settings can be passed to the created widgets as optional arguments
        """
        
        self.parentpath = parent.path
        self.type = type
        self.autoadd = autoadd
        self.name = name
        self.index = index
        self.defaultvals = defaultvals
        
    def do(self, document):
        """Create the new widget.
        
        Returns the new widget
        """
        
        parent = document.resolveFullWidgetPath(self.parentpath)
        w = widgetfactory.thefactory.makeWidget(self.type, parent,
                                                autoadd=self.autoadd,
                                                name=self.name,
                                                index=self.index,
                                                **self.defaultvals)
        self.createdname = w.name
        return w
        
    def undo(self, document):
        """Remove the added widget."""
        
        parent = document.resolveFullWidgetPath(self.parentpath)
        parent.removeChild(self.createdname)

###############################################################################
# Dataset operations
    
class OperationDatasetSet(object):
    """Set a dataset to that specified."""
    
    descr = _('set dataset')
    
    def __init__(self, datasetname, dataset):
        self.datasetname = datasetname
        self.dataset = dataset
        
    def do(self, document):
        """Set dataset, backing up existing one."""
    
        if self.datasetname in document.data:
            self.olddata = document.data[self.datasetname]
        else:
            self.olddata = None
            
        document.setData(self.datasetname, self.dataset)

    def undo(self, document):
        """Undo the data setting."""
        
        document.deleteData(self.datasetname)
        if self.olddata is not None:
            document.setData(self.datasetname, self.olddata)
    
class OperationDatasetDelete(object):
    """Delete a dateset."""
    
    descr = _('delete dataset')
    
    def __init__(self, datasetname):
        self.datasetname = datasetname
    
    def do(self, document):
        """Remove dataset from document, but preserve for undo."""
        self.olddata = document.data[self.datasetname]
        document.deleteData(self.datasetname)
        
    def undo(self, document):
        """Put dataset back"""
        document.setData(self.datasetname, self.olddata)
    
class OperationDatasetRename(object):
    """Rename the dataset.
    
    Assumes newname doesn't already exist
    """
    
    descr = _('rename dataset')
    
    def __init__(self, oldname, newname):
        self.oldname = oldname
        self.newname = newname
    
    def do(self, document):
        """Rename dataset from oldname to newname."""
        
        document.renameDataset(self.oldname, self.newname)
        
    def undo(self, document):
        """Change name back."""
        
        document.renameDataset(self.newname, self.oldname)
        
class OperationDatasetDuplicate(object):
    """Duplicate a dataset.
    
    Assumes duplicate name doesn't already exist
    """
    
    descr = _('duplicate dataset')
    
    def __init__(self, origname, duplname):
        self.origname = origname
        self.duplname = duplname
        
    def do(self, document):
        """Make the duplicate"""
        self.olddata = document.data.get(self.duplname, None)

        dataset = document.data[self.origname]
        duplicate = dataset.returnCopy()
        document.setData(self.duplname, duplicate)
        
    def undo(self, document):
        """Delete the duplicate"""
        
        if self.olddata is None:
            document.deleteData(self.duplname)
        else:
            document.setData(self.duplname, self.olddata)
        
class OperationDatasetUnlinkFile(object):
    """Remove association between dataset and file."""
    descr = _('unlink dataset')
    
    def __init__(self, datasetname):
        self.datasetname = datasetname
        
    def do(self, document):
        dataset = document.data[self.datasetname]
        self.oldfilelink = dataset.linked
        dataset.linked = None
        
    def undo(self, document):
        dataset = document.data[self.datasetname]
        dataset.linked = self.oldfilelink

class OperationDatasetUnlinkRelation(object):
    """Remove association between dataset and another dataset.
    """
    
    descr = _('unlink dataset')
    
    def __init__(self, datasetname):
        self.datasetname = datasetname
        
    def do(self, document):
        dataset = document.data[self.datasetname]
        self.olddataset = dataset
        ds = dataset.returnCopy()
        document.setData(self.datasetname, ds)
        
    def undo(self, document):
        document.setData(self.datasetname, self.olddataset)
        
class OperationDatasetCreate(object):
    """Create dataset base class."""
    
    def __init__(self, datasetname):
        self.datasetname = datasetname
        self.parts = {}
        
    def setPart(self, part, val):
        self.parts[part] = val
        
    def do(self, document):
        """Record old dataset if it exists."""
        self.olddataset = document.data.get(self.datasetname, None)
        
    def undo(self, document):
        """Delete the created dataset."""
        document.deleteData(self.datasetname)
        if self.olddataset is not None:
            document.setData(self.datasetname, self.olddataset)
        
class OperationDatasetCreateRange(OperationDatasetCreate):
    """Create a dataset in a specfied range."""
    
    descr = _('create dataset from range')
    
    def __init__(self, datasetname, numsteps, parts, linked=False):
        """Create a dataset with numsteps values.
        
        parts is a dict containing keys 'data', 'serr', 'perr' and/or 'nerr'. The values
        are tuples with (start, stop) values for each range.
        """
        OperationDatasetCreate.__init__(self, datasetname)
        self.numsteps = numsteps
        self.parts = parts
        self.linked = linked
        
    def do(self, document):
        """Create dataset using range."""
        
        OperationDatasetCreate.do(self, document)
        data = self.parts['data']
        serr = self.parts.get('serr', None)
        perr = self.parts.get('perr', None)
        nerr = self.parts.get('nerr', None)
        
        ds = datasets.DatasetRange(self.numsteps, data, serr=serr,
                                   perr=perr, nerr=nerr)
        if not self.linked:
            # copy these values if we don't want to link
            ds = datasets.Dataset(data=ds.data, serr=ds.serr,
                                  perr=ds.perr, nerr=ds.nerr)

        document.setData(self.datasetname, ds)
        return ds
        
class CreateDatasetException(Exception):
    """Thrown by dataset creation routines."""
    pass
        
class OperationDatasetCreateParameteric(OperationDatasetCreate):
    """Create a dataset using expressions dependent on t."""
    
    descr = _('create parametric dataset')
    
    def __init__(self, datasetname, t0, t1, numsteps, parts, linked=False):
        """Create a parametric dataset.
        
        Variable t goes from t0 to t1 in numsteps.
        parts is a dict with keys 'data', 'serr', 'perr' and/or 'nerr'
        The values are expressions for evaluating."""
        
        OperationDatasetCreate.__init__(self, datasetname)
        self.numsteps = numsteps
        self.t0 = t0
        self.t1 = t1
        self.parts = parts
        self.linked = linked

    def do(self, document):
        """Create the dataset."""
        OperationDatasetCreate.do(self, document)

        p = self.parts.copy()
        p['parametric'] = (self.t0, self.t1, self.numsteps)
        ds = datasets.DatasetExpression(**p)
        ds.document = document

        if not self.linked:
            # copy these values if we don't want to link
            ds = datasets.Dataset(data=ds.data, serr=ds.serr,
                                  perr=ds.perr, nerr=ds.nerr)
        
        document.setData(self.datasetname, ds)
        return ds
        
class OperationDatasetCreateExpression(OperationDatasetCreate):
    descr = _('create dataset from expression')

    def __init__(self, datasetname, parts, link, parametric=None):
        """Create a dataset from existing dataset using expressions.
        
        parts is a dict with keys 'data', 'serr', 'perr' and/or 'nerr'
        The values are expressions for evaluating.
        
        If link is True, then the dataset is linked to the expressions
        Parametric is a tuple (min, max, numitems) if creating parametric
        datasets.
        """
        
        OperationDatasetCreate.__init__(self, datasetname)
        self.parts = parts
        self.link = link
        self.parametric = parametric

    def validateExpression(self, document):
        """Validate the expression is okay.

        Returns True if ok
        """

        p = self.parts.copy()
        p['parametric'] = self.parametric
        ds = datasets.DatasetExpression(**p)
        ds.document = document

        return ds.updateEvaluation()
        
    def do(self, document):
        """Create the dataset."""
        OperationDatasetCreate.do(self, document)

        p = self.parts.copy()
        p['parametric'] = self.parametric
        ds = datasets.DatasetExpression(**p)
        ds.document = document

        if not self.link:
            # copy these values if we don't want to link
            ds = datasets.Dataset(data=ds.data, serr=ds.serr,
                                  perr=ds.perr, nerr=ds.nerr)
        
        document.setData(self.datasetname, ds)
        return ds

class OperationDataset2DBase(object):
    """Operation as base for 2D dataset creation operations."""

    def __init__(self, name, link):
        """Setup operation."""
        self.datasetname = name
        self.link = link
    
    def validateExpression(self, document):
        """Validate expression is okay."""
        ds = self.makeDSClass()
        ds.document = document
        ds.evalDataset()
        if 0 in ds.data.shape:
            raise CreateDatasetException()

    def do(self, document):
        """Make new dataset."""
        # keep backup of old if exists
        self.olddataset = document.data.get(self.datasetname, None)

        # make new dataset
        ds = self.makeDSClass()
        ds.document = document
        if not self.link:
            # unlink if necessary
            ds = datasets.Dataset2D(ds.data, xrange=ds.xrange,
                                    yrange=ds.yrange)
        document.setData(self.datasetname, ds)
        return ds

    def undo(self, document):
        """Undo dataset creation."""
        document.deleteData(self.datasetname)
        if self.olddataset:
            document.setData(self.datasetname, self.olddataset)

class OperationDataset2DCreateExpressionXYZ(OperationDataset2DBase):
    descr = _('create 2D dataset from x, y and z expressions')
    
    def __init__(self, datasetname, xexpr, yexpr, zexpr, link):
        OperationDataset2DBase.__init__(self, datasetname, link)
        self.xexpr = xexpr
        self.yexpr = yexpr
        self.zexpr = zexpr

    def makeDSClass(self):
        return datasets.Dataset2DXYZExpression(
            self.xexpr, self.yexpr, self.zexpr)

class OperationDataset2DCreateExpression(OperationDataset2DBase):
    descr = _('create 2D dataset from expression')
    
    def __init__(self, datasetname, expr, link):
        OperationDataset2DBase.__init__(self, datasetname, link)
        self.expr = expr

    def makeDSClass(self):
        return datasets.Dataset2DExpression(self.expr)

class OperationDataset2DXYFunc(OperationDataset2DBase):
    descr = _('create 2D dataset from function of x and y')

    def __init__(self, datasetname, xstep, ystep, expr, link):
        """Create 2d dataset:

        xstep: tuple(xmin, xmax, step)
        ystep: tuple(ymin, ymax, step)
        expr: expression of x and y
        link: whether to link to this expression
        """
        OperationDataset2DBase.__init__(self, datasetname, link)
        self.xstep = xstep
        self.ystep = ystep
        self.expr = expr

    def makeDSClass(self):
        return datasets.Dataset2DXYFunc(self.xstep, self.ystep, self.expr)

class OperationDatasetUnlinkByFile(object):
    """Unlink all datasets associated with file."""

    descr = _('unlink datasets')

    def __init__(self, filename):
        """Unlink all datasets associated with filename."""
        self.filename = filename

    def do(self, document):
        """Remove links."""
        self.oldlinks = {}
        for name, ds in document.data.iteritems():
            if ds.linked is not None and ds.linked.filename == self.filename:
                self.oldlinks[name] = ds.linked
                ds.linked = None

    def undo(self, document):
        """Restore links."""
        for name, link in self.oldlinks.iteritems():
            try:
                document.data[name].linked = link
            except KeyError:
                pass

class OperationDatasetDeleteByFile(object):
    """Delete all datasets associated with file."""

    descr = _('delete datasets')

    def __init__(self, filename):
        """Delete all datasets associated with filename."""
        self.filename = filename

    def do(self, document):
        """Remove datasets."""
        self.olddatasets = {}
        for name, ds in document.data.items():
            if ds.linked is not None and ds.linked.filename == self.filename:
                self.olddatasets[name] = ds
                document.deleteData(name)

    def undo(self, document):
        """Restore datasets."""
        for name, ds in self.olddatasets.iteritems():
            document.setData(name, ds)

###############################################################################
# Import datasets

class OperationDataImportBase(object):
    """Default useful import class."""

    def __init__(self, params):
        self.params = params

        # list of returned datasets
        self.outdatasets = []
        # list of returned custom variables
        self.outcustoms = []
        # invalid conversions
        self.outinvalids = {}

    def doImport(self, document):
        """Do import, override this.
        Return list of names of datasets read
        """

    def addCustoms(self, document, consts):
        """Optionally, add the customs return by plugins to document."""

        if len(consts) > 0:
            self.oldconst = list(document.customs)
            cd = document.customDict()
            for item in consts:
                if item[1] in cd:
                    idx, ctype, val = cd[item[1]]
                    document.customs[idx] = item
                else:
                    document.customs.append(item)
            document.updateEvalContext()

    def do(self, document):
        """Do import."""

        # remember datasets in document for undo
        olddatasets = dict(document.data)
        self.oldconst = None

        # do actual import
        self.doImport(document)

        # only remember the parts we need
        self.olddatasets = [ (n, olddatasets.get(n)) for n in self.outdatasets ]

        # apply tags
        if self.params.tags:
            for n in self.outdatasets:
                document.data[n].tags.update(self.params.tags)

    def undo(self, document):
        """Undo import."""

        # put back old datasets
        for name, ds in self.olddatasets:
            if ds is None:
                document.deleteData(name)
            else:
                document.setData(name, ds)

        # for custom definitions
        if self.oldconst is not None:
            document.customs = self.oldconst
            document.updateEvalContext()

class OperationDataImport(OperationDataImportBase):
    """Import 1D data from text files."""

    descr = _('import data')

    def __init__(self, params):
        """Setup operation.
        """

        OperationDataImportBase.__init__(self, params)
        self.simpleread = simpleread.SimpleRead(params.descriptor)

    def doImport(self, document):
        """Import data.
        
        Returns a list of datasets which were imported.
        """

        p = self.params
        # open stream to import data from
        if p.filename is not None:
            stream = simpleread.FileStream(
                utils.openEncoding(p.filename, p.encoding))
        elif p.datastr is not None:
            stream = simpleread.StringStream(p.datastr)
        else:
            raise RuntimeError("No filename or string")

        # do the import
        self.simpleread.clearState()
        self.simpleread.readData(stream, useblocks=p.useblocks,
                                 ignoretext=p.ignoretext)

        # associate linked file
        LF = None
        if p.linked:
            assert p.filename
            LF = linked.LinkedFile(p)

        # actually set the data in the document
        self.outdatasets = self.simpleread.setInDocument(
            document, linkedfile=LF, prefix=p.prefix, suffix=p.suffix)
        self.outinvalids = self.simpleread.getInvalidConversions()

class OperationDataImportCSV(OperationDataImportBase):
    """Import data from a CSV file."""

    descr = _('import CSV data')

    def doImport(self, document):
        """Do the data import."""
        
        csvr = readcsv.ReadCSV(self.params)
        csvr.readData()

        LF = None
        if self.params.linked:
            LF = linked.LinkedFileCSV(self.params)
        
        # set the data
        self.outdatasets = csvr.setData(document, linkedfile=LF)

class OperationDataImport2D(OperationDataImportBase):
    """Import a 2D matrix from a file."""
    
    descr = _('import 2d data')

    def doImport(self, document):
        """Import data."""

        p = self.params

        # get stream
        if p.filename is not None:
            stream = simpleread.FileStream(
                utils.openEncoding(p.filename, p.encoding) )
        elif p.datastr is not None:
            stream = simpleread.StringStream(p.datastr)
        else:
            assert False

        # linked file
        LF = None
        if p.linked:
            assert p.filename
            LF = linked.LinkedFile2D(p)

        for name in p.datasetnames:
            sr = simpleread.SimpleRead2D(name, p)
            sr.readData(stream)
            self.outdatasets += sr.setInDocument(document, linkedfile=LF)

class OperationDataImportFITS(OperationDataImportBase):
    """Import 1d or 2d data from a fits file."""

    descr = _('import FITS file')
    
    def _import1d(self, hdu):
        """Import 1d data from hdu."""

        data = hdu.data
        datav = None
        symv = None
        posv = None
        negv = None

        # read the columns required
        p = self.params
        if p.datacol is not None:
            datav = data.field(p.datacol)
        if p.symerrcol is not None:
            symv = data.field(p.symerrcol)
        if p.poserrcol is not None:
            posv = data.field(p.poserrcol)
        if p.negerrcol is not None:
            negv = data.field(p.negerrcol)

        # actually create the dataset
        return datasets.Dataset(data=datav, serr=symv, perr=posv, nerr=negv)

    def _import1dimage(self, hdu):
        """Import 1d image data form hdu."""
        return datasets.Dataset(data=hdu.data)

    def _import2dimage(self, hdu):
        """Import 2d image data from hdu."""

        p = self.params
        if ( p.datacol is not None or p.symerrcol is not None
             or p.poserrcol is not None
             or p.negerrcol is not None ):
            print "Warning: ignoring columns as import 2D dataset"

        header = hdu.header
        data = hdu.data

        try:
            # try to read WCS for image, and work out x/yrange
            wcs = [header[i] for i in ('CRVAL1', 'CRPIX1', 'CDELT1',
                                       'CRVAL2', 'CRPIX2', 'CDELT2')]

            if p.wcsmode == 'pixel':
                # no coordinate system - just pixel values
                rangex = rangey = None
            elif p.wcsmode == 'pixel_wcs':
                rangex = (data.shape[1]-wcs[1], 0-wcs[1])
                rangey = (0-wcs[4], data.shape[0]-wcs[4])
            elif p.wcsmode == 'fraction':
                rangex = rangey = (0., 1.)
            else:
                # linear wcs mode (linear_wcs)
                rangex = ( (data.shape[1]-wcs[1])*wcs[2] + wcs[0],
                           (0-wcs[1])*wcs[2] + wcs[0])
                rangey = ( (0-wcs[4])*wcs[5] + wcs[3],
                           (data.shape[0]-wcs[4])*wcs[5] + wcs[3] )

                rangex = (rangex[1], rangex[0])

        except KeyError:
            # no / broken wcs
            rangex = rangey = None

        return datasets.Dataset2D(data, xrange=rangex, yrange=rangey)

    def doImport(self, document):
        """Do the import."""

        try:
            import pyfits
        except ImportError:
            raise RuntimeError( 'PyFITS is required to import '
                                  'data from FITS files')

        p = self.params
        f = pyfits.open( str(p.filename), 'readonly')
        hdu = f[p.hdu]

        try:
            # raise an exception if this isn't a table therefore is an image
            hdu.get_coldefs()
            ds = self._import1d(hdu)

        except AttributeError:
            naxis = hdu.header.get('NAXIS')
            if naxis == 1:
                ds = self._import1dimage(hdu)
            elif naxis == 2:
                ds = self._import2dimage(hdu)
            else:
                raise RuntimeError("Cannot import images with %i dimensions" % naxis)
        f.close()

        if p.linked:
            ds.linked = linked.LinkedFileFITS(self.params)
        if p.dsname in document.data:
            self.olddataset = document.data[p.dsname]
        else:
            self.olddataset = None
        document.setData(p.dsname.strip(), ds)
        self.outdatasets.append(p.dsname)

class OperationDataImportPlugin(OperationDataImportBase):
    """Import data using a plugin."""

    descr = _('import using plugin')

    def doImport(self, document):
        """Do import."""

        pluginnames = [p.name for p in plugins.importpluginregistry]
        plugin = plugins.importpluginregistry[
            pluginnames.index(self.params.plugin)]

        # if the plugin is a class, make an instance
        # the old API is for the plugin to be instances
        if isinstance(plugin, type):
            plugin = plugin()

        # strip out parameters for plugin itself
        p = self.params

        # stick back together the plugin parameter object
        plugparams = plugins.ImportPluginParams(
            p.filename, p.encoding,  p.pluginpars)
        results = plugin.doImport(plugparams)

        # make link for file
        LF = None
        if p.linked:
            LF = linked.LinkedFilePlugin(p)

        customs = []

        # convert results to real datasets
        names = []
        for d in results:
            if isinstance(d, plugins.Dataset1D):
                ds = datasets.Dataset(data=d.data, serr=d.serr, perr=d.perr,
                                      nerr=d.nerr)
            elif isinstance(d, plugins.Dataset2D):
                ds = datasets.Dataset2D(data=d.data, xrange=d.rangex,
                                        yrange=d.rangey)
            elif isinstance(d, plugins.DatasetText):
                ds = datasets.DatasetText(data=d.data)
            elif isinstance(d, plugins.DatasetDateTime):
                ds = datasets.DatasetDateTime(data=d.data)
            elif isinstance(d, plugins.Constant):
                customs.append( ['constant', d.name, d.val] )
                continue
            elif isinstance(d, plugins.Function):
                customs.append( ['function', d.name, d.val] )
                continue
            else:
                raise RuntimeError("Invalid data set in plugin results")

            # set any linking
            if linked:
                ds.linked = LF

            # construct name
            name = p.prefix + d.name + p.suffix

            # actually make dataset
            document.setData(name, ds)

            names.append(name)

        # add constants, functions to doc, if any
        self.addCustoms(document, customs)

        self.outdatasets = names
        self.outcustoms = list(customs)

class OperationDataCaptureSet(object):
    """An operation for setting the results from a SimpleRead into the
    docunment's data from a data capture.

    This is a bit primative, but it is not obvious how to isolate the capturing
    functionality elsewhere."""

    descr = _('data capture')

    def __init__(self, simplereadobject):
        """Takes a simpleread object containing the data to be set."""
        self.simplereadobject = simplereadobject

    def do(self, document):
        """Set the data in the document."""
        # before replacing data, get a backup of document's data
        databackup = dict(document.data)
        
        # set the data to the document and keep a list of what's changed
        self.nameschanged = self.simplereadobject.setInDocument(document)

        # keep a copy of datasets which have changed from backup
        self.olddata = {}
        for name in self.nameschanged:
            if name in databackup:
                self.olddata[name] = databackup[name]

    def undo(self, document):
        """Undo the results of the capture."""

        for name in self.nameschanged:
            if name in self.olddata:
                # replace datasets with what was there previously
                document.setData(name, self.olddata[name])
            else:
                # or delete datasets that weren't there before
                document.deleteData(name)

class OperationDataTag(object):
    """Add a tag to a list of datasets."""

    descr = _('add dataset tags')

    def __init__(self, tag, datasetnames):
        """Add tag to datasets listed."""
        self.tag = tag
        self.datasetnames = datasetnames

    def do(self, document):
        """Add new tags, if required."""
        self.removetags = []
        for name in self.datasetnames:
            existing = document.data[name].tags
            if self.tag not in existing:
                existing.add(self.tag)
                self.removetags.append(name)

    def undo(self, document):
        """Remove tags, if not previously present."""
        for name in self.removetags:
            document.data[name].tags.remove(self.tag)

class OperationDataUntag(object):
    """Add a tag to a list of datasets."""

    descr = _('remove dataset tags')

    def __init__(self, tag, datasetnames):
        """Remove tag to datasets listed."""
        self.tag = tag
        self.datasetnames = datasetnames

    def do(self, document):
        """Add new tags, if required."""
        for name in self.datasetnames:
            document.data[name].tags.remove(self.tag)

    def undo(self, document):
        """Remove tags, if not previously present."""
        for name in self.datasetnames:
            document.data[name].tags.add(self.tag)

###############################################################################
# Alter dataset

class OperationDatasetAddColumn(object):
    """Add a column to a dataset, blanked to zero."""

    descr = _('add dataset column')

    def __init__(self, datasetname, columnname):
        """Initialise column columnname in datasetname.
        
        columnname can be one of 'data', 'serr', 'perr' or 'nerr'
        """
        self.datasetname = datasetname
        self.columnname = columnname

    def do(self, document):
        """Zero the column."""
        ds = document.data[self.datasetname]
        datacol = ds.data
        try:
            setattr(ds, self.columnname,
                    N.zeros(datacol.shape, dtype='float64'))
        except AttributeError:
            raise RuntimeError("Invalid column name for dataset")
        document.setData(self.datasetname, ds)

    def undo(self, document):
        """Remove the column."""
        ds = document.data[self.datasetname]
        setattr(ds, self.columnname, None)
        document.setData(self.datasetname, ds)

class OperationDatasetSetVal(object):
    """Set a value in the dataset."""

    descr = _('change dataset value')
    
    def __init__(self, datasetname, columnname, row, val):
        """Set row in column columnname to val."""
        self.datasetname = datasetname
        self.columnname = columnname
        self.row = row
        self.val = val
        
    def do(self, document):
        """Set the value."""
        ds = document.data[self.datasetname]
        datacol = getattr(ds, self.columnname)
        self.oldval = datacol[self.row]
        datacol[self.row] = self.val
        ds.changeValues(self.columnname, datacol)

    def undo(self, document):
        """Restore the value."""
        ds = document.data[self.datasetname]
        datacol = getattr(ds, self.columnname)
        datacol[self.row] = self.oldval
        ds.changeValues(self.columnname, datacol)
    
class OperationDatasetSetVal2D(object):
    """Set a value in a 2D dataset."""

    descr = _('change 2D dataset value')

    def __init__(self, datasetname, row, col, val):
        """Set row in column columnname to val."""
        self.datasetname = datasetname
        self.row = row
        self.col = col
        self.val = val

    def do(self, document):
        """Set the value."""
        ds = document.data[self.datasetname]
        self.oldval = ds.data[self.row, self.col]
        ds.data[self.row, self.col] = self.val
        document.modifiedData(ds)

    def undo(self, document):
        """Restore the value."""
        ds = document.data[self.datasetname]
        ds.data[self.row, self.col] = self.oldval
        document.modifiedData(ds)

class OperationDatasetDeleteRow(object):
    """Delete a row or several in the dataset."""

    descr = _('delete dataset row')
    
    def __init__(self, datasetname, row, numrows=1):
        """Delete a row in a dataset."""
        self.datasetname = datasetname
        self.row = row
        self.numrows = numrows
        
    def do(self, document):
        """Set the value."""
        ds = document.data[self.datasetname]
        self.saveddata = ds.deleteRows(self.row, self.numrows)

    def undo(self, document):
        """Restore the value."""
        ds = document.data[self.datasetname]
        ds.insertRows(self.row, self.numrows, self.saveddata)

class OperationDatasetInsertRow(object):
    """Insert a row or several in the dataset."""

    descr = _('insert dataset row')
    
    def __init__(self, datasetname, row, numrows=1):
        """Delete a row in a dataset."""
        self.datasetname = datasetname
        self.row = row
        self.numrows = numrows
        
    def do(self, document):
        """Set the value."""
        ds = document.data[self.datasetname]
        ds.insertRows(self.row, self.numrows, {})

    def undo(self, document):
        """Restore the value."""
        ds = document.data[self.datasetname]
        ds.deleteRows(self.row, self.numrows)

###############################################################################
# Custom setting operations

class OperationSetCustom(object):
    """Set custom objects, such as constants."""

    descr = _('set a custom definition')

    def __init__(self, vals):
        """customtype is the type of custom object to set:
        eg functions, constants
        customval is a dict of the values."""

        self.customvals = list(vals)

    def do(self, document):
        """Set the custom object."""
        self.oldval = list(document.customs)
        document.customs = self.customvals
        document.updateEvalContext()
        
    def undo(self, document):
        """Restore custom object."""
        document.customs = self.oldval
        document.updateEvalContext()

###############################################################################
# Misc operations
        
class OperationMultiple(object):
    """Multiple operations batched into one."""
    
    def __init__(self, operations, descr='change'):
        """A batch operation made up of the operations in list.
        
        Optional argument descr gives a description of the combined operation
        """
        self.operations = operations
        if descr:
            self.descr = descr
        
    def addOperation(self, op):
        """Add an operation to the list of operations."""
        self.operations.append(op)
        
    def do(self, document):
        """Do the multiple operations."""
        for op in self.operations:
            op.do(document)
            
    def undo(self, document):
        """Undo the multiple operations."""
        
        # operations need to undone in reverse order
        for op in self.operations[::-1]:
            op.undo(document)

class OperationLoadStyleSheet(OperationMultiple):
    """An operation to load a stylesheet."""
    
    descr = _('load stylesheet')

    def __init__(self, filename):
        """Load stylesheet with filename."""
        OperationMultiple.__init__(self, [], descr=None)
        self.filename = os.path.abspath(filename)
        
    def do(self, document):
        """Do the import."""

        from . import commandinterpreter

        # get document to keep track of changes for undo/redo
        document.batchHistory(self)

        # fire up interpreter to read file
        interpreter = commandinterpreter.CommandInterpreter(document)
        try:
            interpreter.runFile( open(self.filename) )
        except:
            document.batchHistory(None)
            raise
        
class OperationLoadCustom(OperationLoadStyleSheet):
    descr = _('load custom definitions')

class OperationToolsPlugin(OperationMultiple):
    """An operation to represent what a tools plugin does."""
    
    def __init__(self, plugin, fields):
        """Use tools plugin, passing fields."""
        OperationMultiple.__init__(self, [], descr=None)
        self.plugin = plugin
        self.fields = fields
        self.descr = plugin.name
        
    def do(self, document):
        """Use the plugin."""

        from . import commandinterface

        # get document to keep track of changes for undo/redo
        document.batchHistory(self)

        # fire up interpreter to read file
        ifc = commandinterface.CommandInterface(document)
        try:
            self.plugin.apply(ifc, self.fields)
        except:
            document.batchHistory(None)
            raise
        document.batchHistory(None)

class OperationDatasetPlugin(object):
    """An operation to activate a dataset plugin."""
    
    def __init__(self, plugin, fields, datasetnames={}):
        """Use dataset plugin, passing fields."""
        self.plugin = plugin
        self.fields = fields
        self.descr = plugin.name
        self.names = datasetnames
        
    def do(self, document):
        """Use the plugin.
        """

        self.datasetnames = []
        self.olddata = {}

        manager = self.manager = plugins.DatasetPluginManager(
            self.plugin, document, self.fields)

        names = self.datasetnames = list(manager.datasetnames)

        # rename if requested
        for i in xrange(len(names)):
            if names[i] in self.names:
                names[i] = self.names[names[i]]

        # preserve old datasets
        for name in names:
            if name in document.data:
                self.olddata[name] = document.data[name]

        # add new datasets to document
        for name, ds in izip(names, manager.veuszdatasets):
            if name is not None:
                document.setData(name, ds)

        return names

    def validate(self):
        """Check that the plugin works the first time."""
        self.manager.update(raiseerrors=True)

    def undo(self, document):
        """Undo dataset plugin."""

        # delete datasets which were created
        for name in self.datasetnames:
            if name is not None:
                document.deleteData(name)

        # put back old datasets
        for name, ds in self.olddata.iteritems():
            document.setData(name, ds)
