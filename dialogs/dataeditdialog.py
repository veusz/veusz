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

"""Module for implementing dialog box for viewing/editing data."""

import bisect

import veusz.qtall as qt4

import veusz.document as document
import veusz.utils as utils
from veusz.qtwidgets.datasetbrowser import DatasetBrowser
from veuszdialog import VeuszDialog

# register function to dataset class to edit dataset
recreate_register = {}

class DatasetTableModel1D(qt4.QAbstractTableModel):
    """Provides access to editing and viewing of datasets."""

    def __init__(self, parent, document, datasetname):
        qt4.QAbstractTableModel.__init__(self, parent)

        self.document = document
        self.dsname = datasetname
        self.connect(document, qt4.SIGNAL('sigModified'),
                     self.slotDocumentModified)

    def rowCount(self, parent):
        """Return number of rows."""
        try:
            return len(self.document.data[self.dsname].data)+1
        except (KeyError, AttributeError):
            return 0
        
    def slotDocumentModified(self):
        """Called when document modified."""
        self.emit( qt4.SIGNAL('layoutChanged()') )

    def columnCount(self, parent):
        """Return number of columns."""
        try:
            ds = self.document.data[self.dsname]
        except KeyError:
            return 0
        return len( ds.column_descriptions )

    def data(self, index, role):
        """Return data associated with column given."""
        # get dataset
        ds = self.document.data[self.dsname]
        if ds is not None:
            # select correct part of dataset
            data = getattr(ds, ds.columns[index.column()])
        if ds is not None and data is not None and role == qt4.Qt.DisplayRole:
            # blank row at end of data
            if index.row() == len(data):
                return qt4.QVariant()

            # convert data to QVariant
            d = data[index.row()]
            return ds.uiDataItemToQVariant(d)

        # empty entry
        return qt4.QVariant()

    def headerData(self, section, orientation, role):
        """Return headers at top."""

        try:
            ds = self.document.data[self.dsname]
        except KeyError:
            return qt4.QVariant()

        if role == qt4.Qt.DisplayRole:
            if orientation == qt4.Qt.Horizontal:
                # column names
                return qt4.QVariant( ds.column_descriptions[section] )
            else:
                if section == len(ds.data):
                    return "+"
                # return row numbers
                return qt4.QVariant(section+1)

        return qt4.QVariant()
        
    def flags(self, index):
        """Update flags to say that items are editable."""
        
        if not index.isValid():
            return qt4.Qt.ItemIsEnabled
        else:
            return qt4.QAbstractTableModel.flags(self, index) | qt4.Qt.ItemIsEditable

    def removeRows(self, row, count):
        """Remove rows."""
        self.document.applyOperation(
            document.OperationDatasetDeleteRow(self.dsname, row, count))

    def insertRows(self, row, count):
        """Remove rows."""
        self.document.applyOperation(
            document.OperationDatasetInsertRow(self.dsname, row, count))

    def setData(self, index, value, role):
        """Called to set the data."""

        if not index.isValid() or role != qt4.Qt.EditRole:
            return False

        row = index.row()
        column = index.column()
        ds = self.document.data[self.dsname]
        data = getattr(ds, ds.columns[index.column()])

        # add new column if necessary
        ops = document.OperationMultiple([], descr='add value')
        if data is None:
            ops.addOperation(
                document.OperationDatasetAddColumn(self.dsname,
                                                   ds.columns[column]))

        # add a row if necessary
        if row == len(ds.data):
            ops.addOperation(
                document.OperationDatasetInsertRow(self.dsname, row, 1))

        # update if conversion okay
        try:
            val = ds.uiConvertToDataItem( value.toString() )
        except ValueError:
            return False

        ops.addOperation(
            document.OperationDatasetSetVal(self.dsname,
                                            ds.columns[column],
                                            row, val))
        self.document.applyOperation(ops)
        return True

class DatasetTableModel2D(qt4.QAbstractTableModel):
    """A 2D dataset model."""

    def __init__(self, parent, document, datasetname):
        qt4.QAbstractTableModel.__init__(self, parent)

        self.document = document
        self.dsname = datasetname

    def rowCount(self, parent):
        ds = self.document.data[self.dsname].data
        if ds is not None:
            return ds.shape[0]
        else:
            return 0

    def columnCount(self, parent):
        ds = self.document.data[self.dsname].data
        if ds is not None:
            return ds.shape[1]
        else:
            return 0

    def data(self, index, role):
        if role == qt4.Qt.DisplayRole:
            # get data (note y is reversed, sigh)
            ds = self.document.data[self.dsname].data
            if ds is not None:
                num = ds[ds.shape[0]-index.row()-1, index.column()]
                return qt4.QVariant( float(num) )

        return qt4.QVariant()

    def headerData(self, section, orientation, role):
        """Return headers at top."""

        if role == qt4.Qt.DisplayRole:
            ds = self.document.data[self.dsname]

            if ds is not None:
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
    
class DataEditDialog(VeuszDialog):
    """Dialog for editing and rearranging data sets."""
    
    def __init__(self, parent, document):
        VeuszDialog.__init__(self, parent, 'dataedit.ui')
        self.document = document

        # set up dataset list
        self.dsbrowser = DatasetBrowser(document, parent, parent)
        self.splitter.insertWidget(0, self.dsbrowser)

        # actions for data table
        for text, slot in (
            ('Copy', self.slotCopy),
            ('Delete row', self.slotDeleteRow),
            ('Insert row', self.slotInsertRow),
            ):
            act = qt4.QAction(text, self)
            self.connect(act, qt4.SIGNAL('triggered()'), slot)
            self.datatableview.addAction(act)
        self.datatableview.setContextMenuPolicy( qt4.Qt.ActionsContextMenu )

        # layout edit dialog improvement
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 4)

        # don't want text to look editable or special
        self.linkedlabel.setFrameShape(qt4.QFrame.NoFrame)
        self.linkedlabel.viewport().setBackgroundRole(qt4.QPalette.Window)

        # document changes
        self.connect(document, qt4.SIGNAL('sigModified'),
                     self.slotDocumentModified)

        # select first item, if any or initialise if none
        if len(self.document.data) > 0:
            self.selectDataset( sorted(self.document.data.keys())[0] )

        #if self.dslistmodel.rowCount() > 0:
        #    self.datasetlistview.selectionModel().select(
        #        self.dslistmodel.createIndex(0, 0),
        #        qt4.QItemSelectionModel.Select)
        #else:
        #    self.slotDatasetSelected(None, None)

        self.connect(self.dsbrowser.navtree, qt4.SIGNAL("selecteditem"),
                     self.slotDatasetSelected)

        # connect buttons
        for btn, slot in ( (self.deletebutton, self.slotDatasetDelete),
                           (self.unlinkbutton, self.slotDatasetUnlink),
                           (self.duplicatebutton, self.slotDatasetDuplicate),
                           (self.importbutton, self.slotDatasetImport),
                           (self.createbutton, self.slotDatasetCreate),
                           (self.editbutton, self.slotDatasetEdit),
                           ):
            self.connect(btn, qt4.SIGNAL('clicked()'), slot)

        # menu for new button
        self.newmenu = qt4.QMenu()
        for text, slot in ( ('Numerical dataset', self.slotNewNumericalDataset),
                            ('Text dataset', self.slotNewTextDataset),
                            ('Date/time dataset', self.slotNewDateDataset) ):
            a = self.newmenu.addAction(text)
            self.connect(a, qt4.SIGNAL('triggered()'), slot)
        self.newbutton.setMenu(self.newmenu)

    def slotDatasetSelected(self, name):
        """Called when a new dataset is selected."""

        # FIXME: Make readonly models readonly!!
        model = None
        if name:
            # get selected dataset
            ds = self.document.data[name]

            # make model for dataset
            if ds.dimensions == 1:
                model = DatasetTableModel1D(self, self.document, name)
            elif ds.dimensions == 2:
                model = DatasetTableModel2D(self, self.document, name)

        # disable context menu if no menu
        for a in self.datatableview.actions():
            a.setEnabled(model is not None)

        self.datatableview.setModel(model)    
        self.setUnlinkState()

    def setUnlinkState(self):
        """Enable the unlink button correctly."""
        # get dataset
        dsname = self.getSelectedDataset()
        try:
            ds = self.document.data[dsname]
            unlink = ds.canUnlink()
            linkinfo = ds.linkedInformation()
        except KeyError:
            ds = None
            unlink = False
            linkinfo = ""

        self.editbutton.setVisible(type(ds) in recreate_register)
        self.unlinkbutton.setEnabled(unlink)
        self.linkedlabel.setText(linkinfo)
        self.deletebutton.setEnabled(ds is not None)
        self.duplicatebutton.setEnabled(ds is not None)

    def slotDocumentModified(self):
        """Set unlink status when document modified."""
        self.setUnlinkState()

    def getSelectedDataset(self):
        """Return the selected dataset."""
        return self.dsbrowser.navtree.getSelectedDataset()

    def selectDataset(self, dsname):
        """Select dataset with name given."""
        self.dsbrowser.navtree.selectDataset(dsname)
        self.slotDatasetSelected(dsname)

    def slotDatasetDelete(self):
        """Delete selected dataset."""
        self.document.applyOperation(
            document.OperationDatasetDelete(self.getSelectedDataset()) )

    def slotDatasetUnlink(self):
        """Allow user to remove link to file or other datasets."""
        datasetname = self.getSelectedDataset()
        d = self.document.data[datasetname]
        if d.linked is not None:
            op = document.OperationDatasetUnlinkFile(datasetname)
        else:
            op = document.OperationDatasetUnlinkRelation(datasetname)
        self.document.applyOperation(op)

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

            self.document.applyOperation(
                document.OperationDatasetDuplicate(datasetname, newname))

    def slotDatasetImport(self):
        """Show import dialog."""
        self.mainwindow.slotDataImport()

    def slotDatasetCreate(self):
        """Show dataset creation dialog."""
        self.mainwindow.slotDataCreate()

    def slotDatasetEdit(self):
        """Reload dataset into dataset creation dialog."""
        dsname = self.getSelectedDataset()
        if dsname:
            dataset = self.document.data[dsname]
            recreate_register[type(dataset)](self.mainwindow, self.document,
                                             dataset, dsname)

    def slotCopy(self):
        """Copy text from selection."""
        # get list of selected rows and columns
        selmodel = self.datatableview.selectionModel()
        model = self.datatableview.model()
        indices = []
        for index in selmodel.selectedIndexes():
            indices.append( (index.row(), index.column()) )
        indices.sort()

        # build up text stream for copying to clipboard
        lines = []
        rowitems = []
        lastrow = -1
        for row, column in indices:
            if row != lastrow:
                if rowitems:
                    # items are tab separated
                    lines.append( '\t'.join(rowitems) )
                    rowitems = []
                lastrow = row
            rowitems.append( unicode(
                model.createIndex(row, column).data().toString()) )
        if rowitems:
            lines.append( '\t'.join(rowitems) )
        lines.append('')  # blank line at end
        lines = '\n'.join(lines)

        # put text on clipboard
        qt4.QApplication.clipboard().setText(lines)

    def slotDeleteRow(self):
        """Delete the current row."""
        self.datatableview.model().removeRows(
            self.datatableview.currentIndex().row(), 1)

    def slotInsertRow(self):
        """Insert a new row."""
        self.datatableview.model().insertRows(
            self.datatableview.currentIndex().row(), 1)

    def slotNewNumericalDataset(self):
        """Add new value dataset."""
        self.newDataset( document.Dataset(data=[0.]) )

    def slotNewTextDataset(self):
        """Add new text dataset."""
        self.newDataset( document.DatasetText(data=['']) )

    def slotNewDateDataset(self):
        """Add new date dataset."""
        self.newDataset( document.DatasetDateTime(data=[]) )

    def newDataset(self, ds):
        """Add new dataset to document."""
        # get a name for dataset
        name = 'new dataset'
        if name in self.document.data:
            count = 1
            while name in self.document.data:
                name = 'new dataset %i' % count
                count += 1

        # add new dataset
        self.document.applyOperation(
            document.OperationDatasetSet(name, ds))
