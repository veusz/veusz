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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
###############################################################################

# $Id$

"""Represents atomic operations to take place on a document which can be undone.
Rather than the document modified directly, this interface should be used.

Operations should be passed to the document to be enacted with applyOperation

Each operation provides do(document) and undo(document) methods.
Operations store paths to objects to be modified rather than object references
because some operations cannot restore references (e.g. add object)
"""

# FIXME:
# need operations for the following:
#  create dataset
#  set dataset?
#  paste widget

import utils
import doc
import datasets
import widgets
    
######################################################
# Setting operations

class OperationSettingSet:
    """Set a variable to a value."""

    descr = 'change setting'
    
    def __init__(self, setting, value):
        """Set the setting to value."""
        
        self.settingpath = setting.path
        self.value = value
        
    def do(self, document):
        """Apply setting variable."""
        
        setting = document.resolveFullSettingPath(self.settingpath)
        self.oldvalue = setting.get()
        setting.set(self.value)
        
    def undo(self, document):
        """Return old value back..."""
        
        setting = document.resolveFullSettingPath(self.settingpath)
        setting.set(self.oldvalue)
        

######################################################
# Widget operations
        
class OperationWidgetRename:
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
        
class OperationWidgetDelete:
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
                
class OperationWidgetMove:
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
            
class OperationWidgetAdd:
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
        w = widgets.thefactory.makeWidget(self.type, parent,
                                          autoadd=self.autoadd,
                                          name=self.name,
                                          **self.defaultvals)
        self.createdname = w.name
        return w
        
    def undo(self, document):
        """Remove the added widget."""
        
        parent = document.resolveFullWidgetPath(self.parentpath)
        parent.removeChild(self.createdname)

######################################################
# Dataset operations

class OperationDataImport:
    """Import a dataset."""
    
    descr = 'import dataset'
    
class OperationDataValSet:
    """Set a value manually in a dataset."""

    descr = 'change dataset value'
    
class OperationDatasetDelete:
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
        document.data[self.datasetname] = self.olddata
    
class OperationDatasetRename:
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
        
class OperationDatasetDuplicate:
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
        
class OperationDatasetUnlink:
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
        
class OperationDataCreate:
    """Create dataset base class."""
    
    def __init__(self, datasetname):
        self.datasetname = datasetname
        self.parts = {}
        
    def setPart(self, part, val):
        self.parts[part] = val
        
    def do(self, document):
        pass
        
    def undo(self, document):
        """Delete the created dataset."""
        del document.data[self.datasetname]
        
class OperationDataCreateRange(OperationDataCreate):
    """Create a dataset in a specfied range."""
    
    descr = 'create dataset from range'
    
    def __init__(self, datasetname, numsteps):
        """Create a dataset with numsteps values."""
        OperationDataCreate.__init__(self, datasetname)
        self.numsteps = numsteps
        
    def setPart(self, part, minval, maxval):
        """Set the part to begin at minval and end at maxval."""
        OperationDataCreate.setPart(self, part, (minval, maxval))
        
    def do(self, document):
        """Create dataset using range."""
        
        vals = {}
        for partname, range in self.parts.iteritems():
            minval, maxval = range
            delta = (maxval - minval) / (self.numsteps-1)
            vals[partname] = N.fromfunction( lambda x: minval+x*delta,
                                            (self.numsteps,) )
            
        
class OperationDataCreate:
    """Create a dataset."""
    
    def __init__(self, datasetname):
        """Setup create operation
        """
        self.datasetname = datasetname
        self.mode = None
        self.parts = {}
        
    def setModeRange(self, numsteps):
        """Use a range with number of steps."""
        self.mode = 'range'
        self.numsteps = numsteps
        
    def setPartRange(self, part, startval, stopval):
        """Specify the range to create.
        
        part can be one of 'data', 'serr', 'nerr', 'perr'
        """
        self.parts[part] = (startval, stopval)
        
    def setModeParametric(self, t0, t1, numsteps):
        """Use a parameteric mode where t goes from t0 to t1 in numsteps."""
        self.mode = 'parametric'
        self.t0 = t0
        self.t1 = t1
        self.numsteps = numsteps
    
    def setPartParametric(self, part, expr):
        """Set the part to be the expression expr."""
        self.part[part] = expr
        
    def setModeExpression(self, linked):
        """Each of the parts are expressions.
        
        If linked is True then the dataset is linked to the expressions."""
        self.mode = 'expression'
        self.linked = linked
        
    def setPartExpression(self, part, expr):
        """Set the part to the expression expr."""
        self.part[part] = expr
        
######################################################
# Misc operations
        
class OperationMultiple:
    """Multiple operations batched into one."""
    
    def __init__(self, operations, descr='change'):
        """A batch operation made up of the operations in list.
        
        Optional argument descr gives a description of the combined operation
        """
        self.operations = operations
        self.descr = descr
        
    def do(self, document):
        """Do the multiple operations."""
        for op in self.operations:
            op.do(document)
            
    def undo(self, document):
        """Undo the multiple operations."""
        
        # operations need to undone in reverse order
        for op in utils.reverse(self.operations):
            op.undo(document)
            
