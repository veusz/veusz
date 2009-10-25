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

# $Id$

"""Represents atomic operations to take place on a document which can be undone.
Rather than the document modified directly, this interface should be used.

Operations should be passed to the document to be enacted with applyOperation

Each operation provides do(document) and undo(document) methods.
Operations store paths to objects to be modified rather than object references
because some operations cannot restore references (e.g. add object)
"""

import os.path

import numpy as N

import datasets
import widgetfactory
import simpleread
import commandinterpreter
import readcsv

import veusz.utils as utils
    
###############################################################################
# Setting operations

class OperationSettingSet(object):
    """Set a variable to a value."""

    descr = 'change setting'
    
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
    
    descr = 'propagate setting'
    
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
    
    descr = 'rename'
    
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
    
    descr = 'delete'
    
    def __init__(self, widget):
        """Delete the widget."""
        
        self.widgetpath = widget.path
        
    def do(self, document):
        """Delete widget."""
        
        self.oldwidget = document.resolveFullWidgetPath(self.widgetpath)
        oldparent = self.oldwidget.parent
        self.oldparentpath = oldparent.path
        self.oldindex = oldparent.children.index(self.oldwidget)
        oldparent.removeChild(self.oldwidget.name)
        
    def undo(self, document):
        """Restore deleted widget."""
        
        oldparent = document.resolveFullWidgetPath(self.oldparentpath)
        oldparent.addChild(self.oldwidget, index=self.oldindex)
                
class OperationWidgetMove(object):
    """Move a widget up or down in the hierarchy."""

    descr = 'move'
    
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
            
class OperationWidgetAdd(object):
    """Add a widget of specified type to parent."""

    descr = 'add'
    
    def __init__(self, parent, type, autoadd=True, name=None, **defaultvals):
        """Add a widget of type given
        
        parent is the parent widget
        type is the type to add (string)
        autoadd adds children automatically for some widgets
        name is the (optional) name of the new widget
        settings can be passed to the created widgets as optional arguments
        """
        
        self.parentpath = parent.path
        self.type = type
        self.autoadd = autoadd
        self.name = name
        self.defaultvals = defaultvals
        
    def do(self, document):
        """Create the new widget.
        
        Returns the new widget
        """
        
        parent = document.resolveFullWidgetPath(self.parentpath)
        w = widgetfactory.thefactory.makeWidget(self.type, parent,
                                                autoadd=self.autoadd,
                                                name=self.name,
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
    
    descr = 'set dataset'
    
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
        
        del document.data[self.datasetname]
        if self.olddata is not None:
            document.setData(self.datasetname, self.olddata)
    
class OperationDatasetDelete(object):
    """Delete a dateset."""
    
    descr = 'delete dataset'
    
    def __init__(self, datasetname):
        self.datasetname = datasetname
    
    def do(self, document):
        """Remove dataset from document, but preserve for undo."""
        self.olddata = document.data[self.datasetname]
        del document.data[self.datasetname]
        
    def undo(self, document):
        """Put dataset back"""
        document.setData(self.datasetname, self.olddata)
    
class OperationDatasetRename(object):
    """Rename the dataset.
    
    Assumes newname doesn't already exist
    """
    
    descr = 'rename dataset'
    
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
    
    descr = 'duplicate dataset'
    
    def __init__(self, origname, duplname):
        self.origname = origname
        self.duplname = duplname
        
    def do(self, document):
        """Make the duplicate"""
        document.duplicateDataset(self.origname, self.duplname)
        
    def undo(self, document):
        """Delete the duplicate"""
        
        del document.data[self.duplname]
        
class OperationDatasetUnlink(object):
    """Remove association between dataset and file, or dataset and
    another dataset.
    """
    
    descr = 'unlink dataset'
    
    def __init__(self, datasetname):
        self.datasetname = datasetname
        
    def do(self, document):
        dataset = document.data[self.datasetname]
        
        if isinstance(dataset, datasets.DatasetExpression):
            # if it's an expression, unlink from other dataset
            self.mode = 'expr'
            self.olddataset = dataset
            ds = datasets.Dataset(data=dataset.data, serr=dataset.serr,
                                  perr=dataset.perr, nerr=dataset.nerr)
            document.setData(self.datasetname, ds)
        else:
            # unlink from file
            self.mode = 'file'
            self.oldfilelink = dataset.linked
            dataset.linked = None
        
    def undo(self, document):
        if self.mode == 'file':
            dataset = document.data[self.datasetname]
            dataset.linked = self.oldfilelink
        elif self.mode == 'expr':
            document.setData(self.datasetname, self.olddataset)
        else:
            assert False
        
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
        del document.data[self.datasetname]
        if self.olddataset is not None:
            document.data[self.datasetname] = self.olddataset
        
class OperationDatasetCreateRange(OperationDatasetCreate):
    """Create a dataset in a specfied range."""
    
    descr = 'create dataset from range'
    
    def __init__(self, datasetname, numsteps, parts):
        """Create a dataset with numsteps values.
        
        parts is a dict containing keys 'data', 'serr', 'perr' and/or 'nerr'. The values
        are tuples with (start, stop) values for each range.
        """
        OperationDatasetCreate.__init__(self, datasetname)
        self.numsteps = numsteps
        self.parts = parts
        
    def do(self, document):
        """Create dataset using range."""
        
        OperationDatasetCreate.do(self, document)
        vals = {}
        for partname, therange in self.parts.iteritems():
            minval, maxval = therange
            if self.numsteps == 1:
                vals[partname] = N.array( [minval] )
            else:
                delta = (maxval - minval) / (self.numsteps-1)
                vals[partname] = N.arange(self.numsteps)*delta + minval

        ds = datasets.Dataset(**vals)
        document.setData(self.datasetname, ds)
        return ds
        
class CreateDatasetException(Exception):
    """Thrown by dataset creation routines."""
    pass
        
class OperationDatasetCreateParameteric(OperationDatasetCreate):
    """Create a dataset using expressions dependent on t."""
    
    descr = 'create parametric dataset'
    
    def __init__(self, datasetname, t0, t1, numsteps, parts):
        """Create a parametric dataset.
        
        Variable t goes from t0 to t1 in numsteps.
        parts is a dict with keys 'data', 'serr', 'perr' and/or 'nerr'
        The values are expressions for evaluating."""
        
        OperationDatasetCreate.__init__(self, datasetname)
        self.numsteps = numsteps
        self.t0 = t0
        self.t1 = t1
        self.parts = parts
        
    def do(self, document):
        """Create the dataset."""
        OperationDatasetCreate.do(self, document)

        deltat = (self.t1 - self.t0) / (self.numsteps-1)
        t = N.arange(self.numsteps)*deltat + self.t0
        
        # define environment to evaluate
        fnenviron = document.eval_context.copy()
        fnenviron['t'] = t

        # calculate for each of the dataset components
        vals = {}
        for key, expr in self.parts.iteritems():
            errors = utils.checkCode(expr, securityonly=True)
            if errors is not None:
                raise CreateDatasetException("Will not create dataset\n"
                                             "Unsafe code in expression '%s'\n" % expr)
                
            try:
                vals[key] = eval( expr, fnenviron ) + t*0.
            except Exception, e:
                raise CreateDatasetException("Error evaluating expession '%s'\n"
                                             "Error: '%s'" % (expr, str(e)) )

        ds = datasets.Dataset(**vals)
        document.setData(self.datasetname, ds)
        return ds
        
class OperationDatasetCreateExpression(OperationDatasetCreate):
    descr = 'create dataset from expression'

    def __init__(self, datasetname, parts, link):
        """Create a dataset from existing dataset using expressions.
        
        parts is a dict with keys 'data', 'serr', 'perr' and/or 'nerr'
        The values are expressions for evaluating.
        
        If link is True, then the dataset is linked to the expressions
        """
        
        OperationDatasetCreate.__init__(self, datasetname)
        self.parts = parts
        self.link = link

    def validateExpression(self, document):
        """Validate the expression is okay.
        A CreateDatasetException is raised if not
        """

        try:
            ds = datasets.DatasetExpression(**self.parts)
            ds.document = document
            
            # we force an evaluation of the dataset for the first time, to
            # check for errors in the expressions
            for i in self.parts.iterkeys():
                getattr(ds, i)
            
        except datasets.DatasetExpressionException, e:
            raise CreateDatasetException(str(e))
        
    def do(self, document):
        """Create the dataset."""
        OperationDatasetCreate.do(self, document)

        ds = datasets.DatasetExpression(**self.parts)
        ds.document = document

        if not self.link:
            # copy these values if we don't want to link
            ds = datasets.Dataset(data=ds.data, serr=ds.serr,
                                  perr=ds.perr, nerr=ds.nerr)
        
        document.setData(self.datasetname, ds)
        return ds

class OperationDataset2DCreateExpressionXYZ(object):
    descr = 'create 2D dataset from x, y and z expressions'
    
    def __init__(self, datasetname, xexpr, yexpr, zexpr, link):
        self.datasetname = datasetname
        self.xexpr = xexpr
        self.yexpr = yexpr
        self.zexpr = zexpr
        self.link = link

    def validateExpression(self, document):
        """Validate expression is okay."""
        try:
            ds = datasets.Dataset2DXYZExpression(self.xexpr, self.yexpr,
                                                 self.zexpr)
            ds.document = document
            temp = ds.data
            
        except datasets.DatasetExpressionException, e:
            raise CreateDatasetException(str(e))

    def do(self, document):
        # keep backup
        self.olddataset = document.data.get(self.datasetname, None)

        # make new dataset
        ds = datasets.Dataset2DXYZExpression(self.xexpr, self.yexpr,
                                             self.zexpr)
        ds.document = document
        if not self.link:
            # unlink if necessary
            ds = datasets.Dataset2D(ds.data, xrange=ds.xrange,
                                    yrange=ds.yrange)
        document.setData(self.datasetname, ds)
        return ds

    def undo(self, document):
        del document.data[self.datasetname]
        if self.olddataset:
            document.setData(self.datasetname, self.olddataset)
        
class OperationDataset2DXYFunc(object):
    descr = 'create 2D dataset from function of x and y'

    def __init__(self, datasetname, xstep, ystep, expr, link):
        """Create 2d dataset:

        xstep: tuple(xmin, xmax, step)
        ystep: tuple(ymin, ymax, step)
        expr: expression of x and y
        link: whether to link to this expression
        """
        self.datasetname = datasetname
        self.xstep = xstep
        self.ystep = ystep
        self.expr = expr
        self.link = link
        
    def do(self, document):
        # keep backup
        self.olddataset = document.data.get(self.datasetname, None)

        # make new dataset
        ds = datasets.Dataset2DXYFunc(self.xstep, self.ystep, self.expr)
        ds.document = document
        if not self.link:
            # unlink if necessary
            ds = datasets.Dataset2D(ds.data, xrange=ds.xrange,
                                    yrange=ds.yrange)
        document.setData(self.datasetname, ds)
        return ds

    def undo(self, document):
        del document.data[self.datasetname]
        if self.olddataset:
            document.setData(self.datasetname, self.olddataset)

###############################################################################
# Import datasets
        
class OperationDataImport(object):
    """Import 1D data from text files."""
    
    descr = 'import data'
    
    def __init__(self, descriptor, useblocks=False, linked=False,
                 filename=None, datastr=None,
                 prefix="", suffix="", ignoretext=False,
                 encoding="utf_8"):
        """Setup operation.
        
        descriptor is descriptor for import
        useblocks specifies whether blocks are used in the import
        if reading from a file, linked specfies whether linked
        filename is the filename if reading from a file
        datastr is a string to read from if reading from a string

        prefix and suffix are strings to add before and after dataset names

        encoding is file encoding character set

        filename and datastr are exclusive
        """
        
        self.simpleread = simpleread.SimpleRead(descriptor)
        self.descriptor = descriptor
        self.useblocks = useblocks
        self.linked = linked
        self.filename = filename
        self.datastr = datastr
        self.prefix = prefix
        self.suffix = suffix
        self.ignoretext = ignoretext
        self.encoding = encoding
        
    def do(self, document):
        """Import data.
        
        Returns a list of datasets which were imported.
        """
        
        # open stream to import data from
        if self.filename is not None:
            stream = simpleread.FileStream(
                utils.openEncoding(self.filename, self.encoding))
        elif self.datastr is not None:
            stream = simpleread.StringStream(self.datastr)
        else:
            assert False
        
        # do the import
        self.simpleread.clearState()
        self.simpleread.readData(stream, useblocks=self.useblocks,
                                 ignoretext=self.ignoretext)
        
        # associate file
        if self.linked:
            assert self.filename is not None
            LF = datasets.LinkedFile(self.filename, self.descriptor,
                                     useblocks=self.useblocks,
                                     prefix=self.prefix, suffix=self.suffix,
                                     ignoretext=self.ignoretext,
                                     encoding=self.encoding)
        else:
            LF = None

        # backup datasets in document for undo
        # this has possible space issues!
        self.olddatasets = dict(document.data)
        
        # actually set the data in the document
        names = self.simpleread.setInDocument(document, linkedfile=LF,
                                              prefix=self.prefix,
                                              suffix=self.suffix)
        return names
        
    def undo(self, document):
        """Undo import."""
        
        # restore old datasets
        document.data = self.olddatasets

class OperationDataImportCSV(object):
    """Import data from a CSV file."""

    descr = 'import CSV data'

    def __init__(self, filename, readrows=False,
                 delimiter=',', textdelimiter='"',
                 encoding='utf_8',
                 prefix='', suffix='',
                 linked=False):
        """Import CSV data from filename

        If readrows, then read in rows rather than columns.
        Prefix is appended to each dataset name.
        Data are linked to file if linked is True.
        """

        self.filename = filename
        self.readrows = readrows
        self.delimiter = delimiter
        self.textdelimiter = textdelimiter
        self.encoding = encoding
        self.prefix = prefix
        self.suffix = suffix
        self.linked = linked

    def do(self, document):
        """Do the data import."""
        
        csvr = readcsv.ReadCSV(self.filename, readrows=self.readrows,
                               delimiter=self.delimiter,
                               textdelimiter=self.textdelimiter,
                               encoding=self.encoding,
                               prefix=self.prefix, suffix=self.suffix)
        csvr.readData()

        if self.linked:
            LF = datasets.LinkedCSVFile(self.filename, readrows=self.readrows,
                                        delimiter=self.delimiter,
                                        textdelimiter=self.textdelimiter,
                                        encoding=self.encoding,
                                        prefix=self.prefix, suffix=self.suffix)
        else:
            LF = None

        # backup datasets in document for undo
        # this has possible space issues!
        self.olddatasets = dict(document.data)
        
        # set the data
        names = csvr.setData(document, linkedfile=LF)
        return names

    def undo(self, document):
        """Undo import."""
        
        # restore old datasets
        document.data = self.olddatasets
        
class OperationDataImport2D(object):
    """Import a 2D matrix from a file."""
    
    descr = 'import 2d data'

    def __init__(self, datasets,
                 filename=None, datastr=None,
                 xrange=None, yrange=None,
                 invertrows=None, invertcols=None, transpose=None,
                 prefix="", suffix="", encoding='utf_8',
                 linked=False):
        """Import two-dimensional data from a file.
        filename is the name of the file to read,
        or datastr is the string to read from
        datasets is a list of datasets to read from the file, or a single
        dataset name

        xrange is a tuple containing the range of data in x coordinates
        yrange is a tuple containing the range of data in y coordinates
        if invertrows=True, then rows are inverted when read
        if invertcols=True, then cols are inverted when read
        if transpose=True, then rows and columns are swapped

        prefix and suffix are strings to add before and after dataset names

        encoding is character encoding

        if linked=True then the dataset is linked to the file
        """

        self.datasets = datasets
        self.filename = filename
        self.datastr = datastr
        self.xrange = xrange
        self.yrange = yrange
        self.invertrows = invertrows
        self.invertcols = invertcols
        self.transpose = transpose
        self.prefix = prefix
        self.suffix = suffix
        self.linked = linked
        self.encoding = encoding

    def do(self, document):
        """Import data
        
        Returns list of datasets read."""
        
        if self.filename is not None:
            stream = simpleread.FileStream(
                utils.openEncoding(self.filename, self.encoding) )
        elif self.datastr is not None:
            stream = simpleread.StringStream(self.datastr)
        else:
            assert False
        
        if self.linked:
            assert self.filename
            LF = datasets.Linked2DFile(self.filename, self.datasets)
            LF.xrange = self.xrange
            LF.yrange = self.yrange
            LF.invertrows = self.invertrows
            LF.invertcols = self.invertcols
            LF.transpose = self.transpose
            LF.prefix = self.prefix
            LF.suffix = self.suffix
            LF.encoding = self.encoding
        else:
            LF = None

        # backup datasets in document for undo
        self.olddatasets = dict(document.data)
            
        readds = []
        for name in self.datasets:
            sr = simpleread.SimpleRead2D(name)
            if self.xrange is not None:
                sr.xrange = self.xrange
            if self.yrange is not None:
                sr.yrange = self.yrange
            if self.invertrows is not None:
                sr.invertrows = self.invertrows
            if self.invertcols is not None:
                sr.invertcols = self.invertcols
            if self.transpose is not None:
                sr.transpose = self.transpose

            sr.readData(stream)
            readds += sr.setInDocument(document, linkedfile=LF,
                                       prefix=self.prefix,
                                       suffix=self.suffix)
        return readds

    def undo(self, document):
        """Undo import."""
        
        # restore old datasets
        document.data = self.olddatasets
    
class OperationDataImportFITS(object):
    """Import 1d or 2d data from a fits file."""

    descr = 'import FITS file'
    
    def __init__(self, dsname, filename, hdu,
                 datacol = None, symerrcol = None,
                 poserrcol = None, negerrcol = None,
                 linked = False):
        """Import data from FITS file.

        dsname is the name of the dataset
        filename is name of the fits file to open
        hdu is the number/name of the hdu to access

        if the hdu is a table, datacol, symerrcol, poserrcol and negerrcol
        specify the columns containing the data, symmetric error,
        positive and negative errors.

        linked specfies that the dataset is linked to the file
        """

        self.dsname = dsname
        self.filename = filename
        self.hdu = hdu
        self.datacol = datacol
        self.symerrcol = symerrcol
        self.poserrcol = poserrcol
        self.negerrcol = negerrcol
        self.linked = linked
        
    def _import1d(self, hdu):
        """Import 1d data from hdu."""

        data = hdu.data
        datav = None
        symv = None
        posv = None
        negv = None

        # read the columns required
        if self.datacol is not None:
            datav = data.field(self.datacol)
        if self.symerrcol is not None:
            symv = data.field(self.symerrcol)
        if self.poserrcol is not None:
            posv = data.field(self.poserrcol)
        if self.negerrcol is not None:
            negv = data.field(self.negerrcol)

        # actually create the dataset
        return datasets.Dataset(data=datav, serr=symv, perr=posv, nerr=negv)

    def _import2d(self, hdu):
        """Import 2d data from hdu."""
    
        if ( self.datacol is not None or self.symerrcol is not None or self.poserrcol is not None or
             self.negerrcol is not None ):
            print "Warning: ignoring columns as import 2D dataset"

        header = hdu.header
        data = hdu.data

        try:
            # try to read WCS for image, and work out x/yrange
            wcs = [header[i] for i in ('CRVAL1', 'CRPIX1', 'CDELT1',
                                       'CRVAL2', 'CRPIX2', 'CDELT2')]

            rangex = ( (data.shape[1]-wcs[1])*wcs[2] + wcs[0],
                       (0-wcs[1])*wcs[2] + wcs[0])
            rangey = ( (0-wcs[4])*wcs[5] + wcs[3],
                       (data.shape[0]-wcs[4])*wcs[5] + wcs[3] )

            rangex = (rangex[1], rangex[0])

        except KeyError:
            # no / broken wcs
            rangex = None
            rangey = None

        return datasets.Dataset2D(data, xrange=rangex, yrange=rangey)

    def do(self, document):
        """Do the import."""

        try:
            import pyfits
        except ImportError:
            raise RuntimeError, ( 'PyFITS is required to import '
                                  'data from FITS files' )

        f = pyfits.open( str(self.filename), 'readonly')
        hdu = f[self.hdu]
        data = hdu.data

        try:
            # raise an exception if this isn't a table therefore is an image
            hdu.get_coldefs()
            ds = self._import1d(hdu)

        except AttributeError:
            ds = self._import2d(hdu)

        f.close()
            
        if self.linked:
            ds.linked = datasets.LinkedFITSFile(self.dsname, self.filename, self.hdu,
                                                [self.datacol, self.symerrcol,
                                                 self.poserrcol, self.negerrcol])

        if self.dsname in document.data:
            self.olddataset = document.data[self.dsname]
        else:
            self.olddataset = None
        document.setData(self.dsname, ds)

    def undo(self, document):
        """Undo the import."""
        
        if self.dsname in document.data:
            del document.data[self.dsname]
            
        if self.olddataset is not None:
            document.setData(self.dsname, self.olddataset)
        
class OperationDataCaptureSet(object):
    """An operation for setting the results from a SimpleRead into the
    docunment's data from a data capture.

    This is a bit primative, but it is not obvious how to isolate the capturing
    functionality elsewhere."""

    descr = 'data capture'

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
                document.data[name] = self.olddata[name]
            else:
                # or delete datasets that weren't there before
                del document.data[name]

###############################################################################
# Alter dataset

class OperationDatasetAddColumn(object):
    """Add a column to a dataset, blanked to zero."""
    
    descr = 'add dataset column'
    
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
        setattr(ds, self.columnname, N.zeros(datacol.shape, dtype='float64'))
        document.setData(self.datasetname, ds)
        
    def undo(self, document):
        """Remove the column."""
        ds = document.data[self.datasetname]
        setattr(ds, self.columnname, None)
        document.setData(self.datasetname, ds)
        
class OperationDatasetSetVal(object):
    """Set a value in the dataset."""

    descr = 'change dataset value'
    
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
    
class OperationDatasetDeleteRow(object):
    """Delete a row or several in the dataset."""

    descr = 'delete dataset row'
    
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

    descr = 'insert dataset row'
    
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
# Misc operations
        
class OperationMultiple(object):
    """Multiple operations batched into one."""
    
    def __init__(self, operations, descr='change'):
        """A batch operation made up of the operations in list.
        
        Optional argument descr gives a description of the combined operation
        """
        self.operations = operations
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
        for op in utils.reverse(self.operations):
            op.undo(document)

class OperationImportStyleSheet(OperationMultiple):
    """An operation to import a stylesheet."""
    
    def __init__(self, filename):
        """Import stylesheet with filename."""
        OperationMultiple.__init__(self, [], descr='import stylesheet')
        self.filename = os.path.abspath(filename)
        
    def do(self, document):
        """Do the import."""

        # get document to keep track of changes for undo/redo
        document.batchHistory(self)

        # fire up interpreter to read file
        interpreter = commandinterpreter.CommandInterpreter(document)
        e = None
        try:
            interpreter.runFile( open(self.filename) )
        except Exception, e:
            pass
        document.batchHistory(None)
        if e:
            raise e
        
