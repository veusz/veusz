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

from __future__ import division, print_function
import os.path
import io

import numpy as N

from ..compat import czip, crange, citems, cbasestr
from . import widgetfactory

from .. import datasets
from .. import plugins
from .. import qtall as qt4

def _(text, disambiguation=None, context="Operations"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

###############################################################################
# Setting operations

class Operation(object):
    """Root class for operations."""

    descr = 'REPLACE THIS'

    def do(self, document):
        """Apply operation to document."""

    def undo(self, document):
        """Undo operation."""

class OperationSettingSet(Operation):
    """Set a variable to a value."""

    descr = _('change setting')

    def __init__(self, setting, value):
        """Set the setting to value.
        Setting may be a widget path
        """

        if isinstance(setting, cbasestr):
            self.settingpath = setting
        else:
            self.settingpath = setting.path
        self.value = value

    def do(self, document):
        """Apply setting variable."""
        setting = document.resolveSettingPath(None, self.settingpath)
        if setting.isReference():
            self.oldvalue = setting.getReference()
        else:
            self.oldvalue = setting.get()
        setting.set(self.value)

    def undo(self, document):
        """Return old value back..."""
        setting = document.resolveSettingPath(None, self.settingpath)
        setting.set(self.oldvalue)

class OperationSettingPropagate(Operation):
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
        while not s.iswidget:
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
            root = document.resolveWidgetPath(None, self.rootpath)

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

        for setpath, setval in citems(self.restorevals):
            setting = document.resolveSettingPath(None, setpath)
            setting.set(setval)

    @staticmethod
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

                OperationSettingPropagate._recursiveGet(
                    w, name, typename, outlist, newmaxlevels)

###############################################################################
# Widget operations

class OperationWidgetRename(Operation):
    """Rename widget."""

    descr = _('rename')

    def __init__(self, widget, newname):
        """Rename the widget to newname."""

        self.widgetpath = widget.path
        self.newname = newname

    def do(self, document):
        """Rename widget."""

        widget = document.resolveWidgetPath(None, self.widgetpath)
        self.oldname = widget.name
        widget.rename(self.newname)
        self.newpath = widget.path

    def undo(self, document):
        """Undo rename."""

        widget = document.resolveWidgetPath(None, self.newpath)
        widget.rename(self.oldname)

class OperationWidgetDelete(Operation):
    """Delete widget."""

    descr = _('delete')

    def __init__(self, widget):
        """Delete the widget."""

        self.widgetpath = widget.path

    def do(self, document):
        """Delete widget."""

        self.oldwidget = document.resolveWidgetPath(None, self.widgetpath)
        oldparent = self.oldwidget.parent
        self.oldwidget.parent = None
        self.oldparentpath = oldparent.path
        self.oldindex = oldparent.children.index(self.oldwidget)
        oldparent.removeChild(self.oldwidget.name)

    def undo(self, document):
        """Restore deleted widget."""

        oldparent = document.resolveWidgetPath(None, self.oldparentpath)
        self.oldwidget.parent = oldparent
        oldparent.addChild(self.oldwidget, index=self.oldindex)

class OperationWidgetsDelete(Operation):
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
        widgetpaths.sort(key=len)
        i = 0
        while i < len(widgetpaths):
            wp = widgetpaths[i]
            for j in crange(i):
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
            self.oldwidgets.append( document.resolveWidgetPath(None, path) )
            oldparent = self.oldwidgets[-1].parent
            self.oldparentpaths.append( oldparent.path )
            self.oldindexes.append( oldparent.children.index(self.oldwidgets[-1]) )
            oldparent.removeChild(self.oldwidgets[-1].name)

    def undo(self, document):
        """Restore deleted widget."""

        # put back widgets in reverse order so that indexes are corrent
        for i in crange(len(self.oldwidgets)-1,-1,-1):
            oldparent = document.resolveWidgetPath(None, self.oldparentpaths[i])
            oldparent.addChild(self.oldwidgets[i], index=self.oldindexes[i])

class OperationWidgetMoveUpDown(Operation):
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

        widget = document.resolveWidgetPath(None, self.widgetpath)
        parent = widget.parent
        self.suceeded = parent.moveChild(widget, self.direction)
        self.newpath = widget.path

    def undo(self, document):
        """Move it back."""
        if self.suceeded:
            widget = document.resolveWidgetPath(None, self.newpath)
            parent = widget.parent
            parent.moveChild(widget, -self.direction)

class OperationWidgetMove(Operation):
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

        child = document.resolveWidgetPath(None, self.oldchildpath)
        oldparent = child.parent
        newparent = document.resolveWidgetPath(None, self.newparentpath)
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

        newparent = document.resolveWidgetPath(None, self.newparentpath)
        child = document.resolveWidgetPath(None, self.newchildpath)
        oldparent = document.resolveWidgetPath(None, self.oldparentpath)

        # remove from new parent
        del newparent.children[self.newindex]
        # restore parent
        oldparent.children.insert(self.oldchildindex, child)
        child.parent = oldparent

        # restore name
        if self.oldname is not None:
            child.name = self.oldname

class OperationWidgetAdd(Operation):
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
        self.wtype = type
        self.autoadd = autoadd
        self.name = name
        self.index = index
        self.defaultvals = defaultvals

    def do(self, document):
        """Create the new widget.

        Returns the new widget
        """

        parent = document.resolveWidgetPath(None, self.parentpath)
        w = widgetfactory.thefactory.makeWidget(
            self.wtype, parent, document,
            autoadd=self.autoadd,
            name=self.name,
            index=self.index,
            **self.defaultvals)
        self.createdname = w.name
        return w

    def undo(self, document):
        """Remove the added widget."""

        parent = document.resolveWidgetPath(None, self.parentpath)
        parent.removeChild(self.createdname)

###############################################################################
# Dataset operations

class OperationDatasetSet(Operation):
    """Set a dataset to that specified."""

    descr = _('set dataset')

    def __init__(self, datasetname, dataset):
        self.datasetname = datasetname
        self.dataset = dataset

    def do(self, document):
        """Set dataset, backing up existing one."""

        self.olddata = document.data.get(self.datasetname)
        document.setData(self.datasetname, self.dataset)

    def undo(self, document):
        """Undo the data setting."""

        if self.olddata is None:
            document.deleteData(self.datasetname)
        else:
            document.setData(self.datasetname, self.olddata)

class OperationDatasetDelete(Operation):
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

class OperationDatasetRename(Operation):
    """Rename the dataset.

    Assumes newname doesn't already exist
    """

    descr = _('rename dataset')

    def __init__(self, oldname, newname):
        self.oldname = oldname
        self.newname = newname

    def do(self, document):
        """Rename dataset from oldname to newname."""
        ds = document.data[self.oldname]
        self.origname = self.origrename = None

        if ds.linked:
            p = ds.linked.params
            if p.renames is None:
                p.renames = {}

            # dataset might have been renamed before, so we have to
            # remove that entry and remember how to put it back
            origname = self.oldname
            for o, n in list(citems(p.renames)):
                if n == self.oldname:
                    origname = o
                    # store in case of undo
                    self.origrename = (o, n)
                    break
            p.renames[origname] = self.newname
            self.origname = origname

        document.renameDataset(self.oldname, self.newname)

    def undo(self, document):
        """Change name back."""

        ds = document.data[self.newname]
        if ds.linked:
            p = ds.linked.params
            del p.renames[self.origname]
            if self.origrename:
                p.renames[self.origrename[0]] = self.origrename[1]

        document.renameDataset(self.newname, self.oldname)

class OperationDatasetDuplicate(Operation):
    """Duplicate a dataset.

    Assumes duplicate name doesn't already exist
    """

    descr = _('duplicate dataset')

    def __init__(self, origname, duplname):
        self.origname = origname
        self.duplname = duplname

    def do(self, document):
        """Make the duplicate"""
        self.olddata = document.data.get(self.duplname)

        dataset = document.data[self.origname]
        duplicate = dataset.returnCopy()
        document.setData(self.duplname, duplicate)

    def undo(self, document):
        """Delete the duplicate"""

        if self.olddata is None:
            document.deleteData(self.duplname)
        else:
            document.setData(self.duplname, self.olddata)

class OperationDatasetUnlinkFile(Operation):
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

class OperationDatasetUnlinkRelation(Operation):
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

class OperationDatasetCreate(Operation):
    """Create dataset base class."""

    def __init__(self, datasetname):
        self.datasetname = datasetname
        self.parts = {}

    def setPart(self, part, val):
        self.parts[part] = val

    def do(self, document):
        """Record old dataset if it exists."""
        self.olddataset = document.data.get(self.datasetname)

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

class OperationDatasetsFilter(Operation):
    """Operation to filter datasets."""

    descr = _("filter datasets")

    def __init__(self, inexpr, indatasets,
                 prefix="", suffix="",
                 invert=False, replaceblanks=False):
        """Initialise operation:
        inexpr: input expression
        indatasets: list of dataset names
        prefix, suffix: output prefix/suffix
        invert: invert filter expression
        replaceblanks: replace output with blank/nan values.
        """

        if not prefix and not suffix:
            raise ValueError("Prefix and/or suffix must be given")
        self.inexpr = inexpr
        self.indatasets = indatasets
        self.prefix = prefix
        self.suffix = suffix
        self.invert = invert
        self.replaceblanks = replaceblanks

    def makeGen(self):
        """Return generator object."""
        return datasets.DatasetFilterGenerator(
            self.inexpr, self.indatasets,
            prefix=self.prefix, suffix=self.suffix,
            invert=self.invert, replaceblanks=self.replaceblanks)

    def check(self, doc):
        """Check the filter is ok.

        Return (ok, [list of errors])
        """

        log = self.makeGen().evaluateFilter(doc)
        if log:
            return (False, log)
        return (True, [])

    def do(self, doc):
        """Do the operation."""

        gen = self.makeGen()
        self.olddatasets = {}
        for name in self.indatasets:
            outname = self.prefix + name + self.suffix
            self.olddatasets[outname] = doc.data.get(outname)
            doc.setData(outname, datasets.DatasetFiltered(gen, name, doc))

    def undo(self, doc):
        """Undo operation."""

        for name, val in citems(self.olddatasets):
            if val is None:
                doc.deleteData(name)
            else:
                doc.setData(name, val)

class OperationDataset2DBase(Operation):
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
            ds = datasets.Dataset2D(ds.data,
                                    xrange=ds.xrange, yrange=ds.yrange,
                                    xedge=ds.xedge, yedge=ds.yedge,
                                    xcent=ds.xcent, ycent=ds.ycent)
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

class OperationDatasetUnlinkByFile(Operation):
    """Unlink all datasets associated with file."""

    descr = _('unlink datasets')

    def __init__(self, filename):
        """Unlink all datasets associated with filename."""
        self.filename = filename

    def do(self, document):
        """Remove links."""
        self.oldlinks = {}
        for name, ds in citems(document.data):
            if ds.linked is not None and ds.linked.filename == self.filename:
                self.oldlinks[name] = ds.linked
                ds.linked = None

    def undo(self, document):
        """Restore links."""
        for name, link in citems(self.oldlinks):
            try:
                document.data[name].linked = link
            except KeyError:
                pass

class OperationDatasetDeleteByFile(Operation):
    """Delete all datasets associated with file."""

    descr = _('delete datasets')

    def __init__(self, filename):
        """Delete all datasets associated with filename."""
        self.filename = filename

    def do(self, document):
        """Remove datasets."""
        self.olddatasets = {}
        for name, ds in list(document.data.items()):
            if ds.linked is not None and ds.linked.filename == self.filename:
                self.olddatasets[name] = ds
                document.deleteData(name)

    def undo(self, document):
        """Restore datasets."""
        for name, ds in citems(self.olddatasets):
            document.setData(name, ds)

###############################################################################
# Import datasets

class OperationDataTag(Operation):
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

class OperationDataUntag(Operation):
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

class OperationDatasetAddColumn(Operation):
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

class OperationDatasetSetVal(Operation):
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

class OperationDatasetSetVal2D(Operation):
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

class OperationDatasetDeleteRow(Operation):
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

class OperationDatasetInsertRow(Operation):
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

class OperationSetCustom(Operation):
    """Set custom objects, such as constants."""

    descr = _('set a custom definition')

    # translate ctype below into attribute of evaluate
    type_to_attr = {
        'definition': 'def_definitions',
        'function':   'def_definitions',
        'constant':   'def_definitions',
        'import':     'def_imports',
        'color':      'def_colors',
        'colormap':   'def_colormaps',
    }

    def __init__(self, ctype, vals):
        """Set custom values to be the list given.

        ctype is one of 'definition', 'function', 'constant',
        'import', 'color' or 'colormap'
        """
        self.ctype = ctype
        self.customvals = list(vals)

    def _getlist(self, document):
        return getattr(document.evaluate, self.type_to_attr[self.ctype])

    def do(self, document):
        """Set the custom object."""
        lst = self._getlist(document)
        self.oldval = list(lst)
        lst[:] = self.customvals
        document.evaluate.update()

    def undo(self, document):
        """Restore custom object."""
        self._getlist(document)[:] = self.oldval
        document.evaluate.update()

###############################################################################
# Misc operations

class OperationMultiple(Operation):
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
            interpreter.runFile( io.open(self.filename, 'rU',
                                         encoding='utf8') )
        except:
            document.batchHistory(None)
            raise
        document.batchHistory(None)

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

class OperationDatasetPlugin(Operation):
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
        for i in crange(len(names)):
            if names[i] in self.names:
                names[i] = self.names[names[i]]

        # preserve old datasets
        for name in names:
            if name in document.data:
                self.olddata[name] = document.data[name]

        # add new datasets to document
        for name, ds in czip(names, manager.veuszdatasets):
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
        for name, ds in citems(self.olddata):
            document.setData(name, ds)

class OperationDatasetHistogram(Operation):
    """Operation to make histogram from data."""

    descr = _("make histogram")

    def __init__(self, expr, outposns, outvalues,
                 binparams=None, binmanual=None, method='counts',
                 cumulative = 'none',
                 errors=False):
        """
        inexpr = input dataset expression
        outposns = name of dataset for bin positions
        outvalues = name of dataset for bin values
        binparams = None / (num, minval, maxval, islog)
        binmanual = None / [1,2,3,4,5]
        method = ('counts', 'density', or 'fractions')
        cumulative = ('none', 'smalltolarge', 'largetosmall')
        errors = True/False
        """

        self.expr = expr
        self.outposns = outposns
        self.outvalues = outvalues
        self.binparams = binparams
        self.binmanual = binmanual
        self.method = method
        self.cumulative = cumulative
        self.errors = errors

    def do(self, document):
        """Create histogram datasets."""

        gen = datasets.DatasetHistoGenerator(
            document, self.expr, binparams=self.binparams,
            binmanual=self.binmanual,
            method=self.method,
            cumulative=self.cumulative,
            errors=self.errors)

        self.oldposnsds = self.oldvaluesds = None

        if self.outvalues != '':
            self.oldvaluesds = document.data.get(self.outvalues, None)
            document.setData(self.outvalues, gen.generateValueDataset())

        if self.outposns != '':
            self.oldposnsds = document.data.get(self.outposns, None)
            document.setData(self.outposns, gen.generateBinDataset())

    def undo(self, document):
        """Undo creation of datasets."""

        if self.oldposnsds is not None:
            if self.outposns != '':
                document.setData(self.outposns, self.oldposnsds)
        else:
            document.deleteData(self.outposns)

        if self.oldvaluesds is not None:
            if self.outvalues != '':
                document.setData(self.outvalues, self.oldvaluesds)
        else:
            document.deleteData(self.outvalues)
