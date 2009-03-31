# data editting dialog

#    Copyright (C) 2005 Jeremy S. Sanders
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

"""Module for implementing dialog box for viewing/editing data."""

import re
import itertools
import os.path

import numpy as N
import veusz.qtall as qt4

import veusz.setting as setting
import veusz.document as document
import veusz.utils as utils
import importdialog
import datacreate

class DatasetTableModel1D(qt4.QAbstractTableModel):
    """Provides access to editing and viewing of datasets."""

    def __init__(self, parent, document, datasetname):
        qt4.QAbstractTableModel.__init__(self, parent)

        self.document = document
        self.dsname = datasetname

    def rowCount(self, parent):
        """Return number of rows."""

        try:
            return len(self.document.data[self.dsname].data)
        except (KeyError, AttributeError):
            return 0
        
    def columnCount(self, parent):
        ds = self.document.data[self.dsname]
        return len( ds.column_descriptions )

    def data(self, index, role):
        if role == qt4.Qt.DisplayRole:
            # select correct part of dataset
            ds = self.document.data[self.dsname]
            data = getattr(ds, ds.columns[index.column()])

            if data is not None:
                return qt4.QVariant( data[index.row()] )

        # return nothing otherwise
        return qt4.QVariant()

    def headerData(self, section, orientation, role):
        """Return headers at top."""

        if role == qt4.Qt.DisplayRole:
            if orientation == qt4.Qt.Horizontal:
                ds = self.document.data[self.dsname]
                return qt4.QVariant( ds.column_descriptions[section] )
            else:
                # return row numbers
                return qt4.QVariant(section+1)

        return qt4.QVariant()
        
    def flags(self, index):
        """Update flags to say that items are editable."""
        
        if not index.isValid():
            return qt4.Qt.ItemIsEnabled
        else:
            return qt4.QAbstractTableModel.flags(self, index) | qt4.Qt.ItemIsEditable

    def setData(self, index, value, role):
        """Called to set the data."""

        if index.isValid() and role == qt4.Qt.EditRole:
            row = index.row()
            column = index.column()
            ds = self.document.data[self.dsname]
            data = getattr(ds, ds.columns[index.column()])

            # add new column if necessary
            if data is None:
                self.document.applyOperation( document.OperationDatasetAddColumn(self.dsname,
                                                                                 ds.columns[column]) )


            # update if conversion okay
            try:
                val = ds.convertToDataItem( value.toString() )
            except ValueError:
                return False
            
            op = document.OperationDatasetSetVal(self.dsname,
                                                 ds.columns[column],
                                                 row, val)
            self.document.applyOperation(op)
            return True

        else:
            return False

class DatasetTableModel2D(qt4.QAbstractTableModel):
    """A 2D dataset model."""

    def __init__(self, parent, document, datasetname):
        qt4.QAbstractTableModel.__init__(self, parent)

        self.document = document
        self.dsname = datasetname

    def rowCount(self, parent):
        ds = self.document.data[self.dsname].data
        return ds.shape[0]

    def columnCount(self, parent):
        ds = self.document.data[self.dsname].data
        return ds.shape[1]

    def data(self, index, role):
        if role == qt4.Qt.DisplayRole:
            # get data (note y is reversed, sigh)
            ds = self.document.data[self.dsname].data
            num = ds[ds.shape[0]-index.row()-1, index.column()]
            return qt4.QVariant(num)

        return qt4.QVariant()

    def headerData(self, section, orientation, role):
        """Return headers at top."""

        if role == qt4.Qt.DisplayRole:
            ds = self.document.data[self.dsname]

            # return a number for the top left of the cell
            if orientation == qt4.Qt.Horizontal:
                r = ds.xrange
                num = ds.data.shape[1]
            else:
                r = ds.yrange
                r = (r[1], r[0]) # swap (as y reversed)
                num = ds.data.shape[0]
            val = (r[1]-r[0])/num*(section+0.5)+r[0]
            return qt4.QVariant( '%g' % val )

        return qt4.QVariant()

class DatasetListModel(qt4.QAbstractListModel):
    """A model to allow the list of datasets to be viewed."""

    def __init__(self, parent, document):
        qt4.QAbstractListModel.__init__(self, parent)
        self.document = document

        self.connect(document, qt4.SIGNAL('sigModified'),
                     self.slotDocumentModified)

        # initial variable state
        self._changeset = -1

    def _getDSList(self):
        """A cached copy of a list of datasets, which updates if doc changes."""
        if self._changeset != self.document.changeset:
            self._changeset = self.document.changeset
            self._realDSList = self.document.data.keys()
            self._realDSList.sort()
        return self._realDSList
    datasets = property(_getDSList)

    def slotDocumentModified(self):
        """Called when document modified."""
        self.emit( qt4.SIGNAL('layoutChanged()') )

    def rowCount(self, parent):
        return len(self.datasets)

    def datasetName(self, index):
        return self.datasets[index.row()]

    def data(self, index, role):
        if role == qt4.Qt.DisplayRole:
            return qt4.QVariant(self.datasets[index.row()])

        # return nothing otherwise
        return qt4.QVariant()

    def flags(self, index):
        """Return flags for items."""
        if not index.isValid():
            return qt4.Qt.ItemIsEnabled
        
        return qt4.QAbstractListModel.flags(self, index) | qt4.Qt.ItemIsEditable

    def setData(self, index, value, role):
        """Called to rename a dataset."""

        if index.isValid() and role == qt4.Qt.EditRole:
            name = self.datasetName(index)
            newname = unicode( value.toString() )
            if not re.match(r'^[A-za-z][^ +-,]+$', newname):
                return False

            self.datasets[index.row()] = newname
            self.emit(qt4.SIGNAL('dataChanged(const QModelIndex &, const QModelIndex &'), index, index)

            self.document.applyOperation(document.OperationDatasetRename(name, newname))
            return True

        return False
    
class DataEditDialog(qt4.QDialog):
    """Dialog for editing and rearranging data sets."""
    
    def __init__(self, parent, document, *args):

        # load up UI
        qt4.QDialog.__init__(self, parent, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'dataedit.ui'),
                   self)
        self.document = document

        # set up dataset list
        self.dslistmodel = DatasetListModel(self, document)
        self.datasetlistview.setModel(self.dslistmodel)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        # document changes
        self.connect(document, qt4.SIGNAL('sigModified'),
                     self.slotDocumentModified)

        # receive change in selection
        self.connect(self.datasetlistview.selectionModel(),
                     qt4.SIGNAL('selectionChanged(const QItemSelection &, const QItemSelection &)'),
                     self.slotDatasetSelected)

        # select first item (phew)
        if self.dslistmodel.rowCount(None) > 0:
            self.datasetlistview.selectionModel().select(
                self.dslistmodel.createIndex(0, 0),
                qt4.QItemSelectionModel.Select)

        # connect buttons
        self.connect(self.deletebutton, qt4.SIGNAL('clicked()'),
                     self.slotDatasetDelete)
        self.connect(self.unlinkbutton, qt4.SIGNAL('clicked()'),
                     self.slotDatasetUnlink)
        self.connect(self.duplicatebutton, qt4.SIGNAL('clicked()'),
                     self.slotDatasetDuplicate)
        self.connect(self.importbutton, qt4.SIGNAL('clicked()'),
                     self.slotDatasetImport)
        self.connect(self.createbutton, qt4.SIGNAL('clicked()'),
                     self.slotDatasetCreate)

    def slotDatasetSelected(self, current, deselected):
        """Called when a new dataset is selected."""

        # FIXME: Make readonly models readonly!!
        index = current.indexes()[0]
        name = self.dslistmodel.datasetName(index)
        ds = self.document.data[name]

        if ds.dimensions == 1:
            model = DatasetTableModel1D(self, self.document, name)
        elif ds.dimensions == 2:
            model = DatasetTableModel2D(self, self.document, name)

        self.datatableview.setModel(model)
            
        self.setUnlinkState()

    def setUnlinkState(self):
        """Enable the unlink button correctly."""
        # get dataset
        dsname = self.getSelectedDataset()

        try:
            ds = self.document.data[dsname]
        except KeyError:
            return

        # linked dataset
        readonly = False
        if ds.linked is None:
            fn = 'None'
            unlink = False
        else:
            fn = ds.linked.filename
            unlink = True
        text = 'Linked file: %s' % fn
        
        if isinstance(ds, document.DatasetExpression):
            # for datasets linked by expressions
            items = ['Linked expression dataset:']
            for label, part in itertools.izip(ds.column_descriptions,
                                              ds.columns):
                if ds.expr[part]:
                    items.append('%s: %s' % (label, ds.expr[part]))
            text = '\n'.join(items)
            unlink = True
            readonly = True
            
        self.unlinkbutton.setEnabled(unlink)
        self.linkedlabel.setText(text)

    def slotDocumentModified(self):
        """Set unlink status when document modified."""
        self.setUnlinkState()

    def getSelectedDataset(self):
        """Return the selected dataset."""
        selitems = self.datasetlistview.selectionModel().selection().indexes()
        if len(selitems) != 0:
            return self.dslistmodel.datasetName(selitems[0])
        else:
            return None
        
    def slotDatasetDelete(self):
        """Delete selected dataset."""

        datasetname = self.getSelectedDataset()
        if datasetname is not None:
            self.document.applyOperation(document.OperationDatasetDelete(datasetname))
        print self.getSelectedDataset()
        

    def slotDatasetUnlink(self):
        """Allow user to remove link to file or other datasets."""

        datasetname = self.getSelectedDataset()
        if datasetname is not None:
            self.document.applyOperation( document.OperationDatasetUnlink(datasetname) )

    def slotDatasetDuplicate(self):
        """Duplicate selected dataset."""
        
        datasetname = self.getSelectedDataset()
        if datasetname is not None:
            # generate new name for dataset
            newname = datasetname + '_copy'
            index = 2
            while newname in self.document.data:
                newname = '%s_copy_%i' % (datasetname, index)
                index += 1

            self.document.applyOperation(document.OperationDatasetDuplicate(datasetname, newname))

    def slotDatasetImport(self):
        """Show import dialog."""
        self.parent().slotDataImport()

    def slotDatasetCreate(self):
        """Show dataset creation dialog."""
        self.parent().slotDataCreate()
