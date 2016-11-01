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

from __future__ import division

import numpy as N

from ..compat import cstr
from .. import qtall as qt4
from .. import document
from .. import datasets
from .. import setting
from ..qtwidgets.datasetbrowser import DatasetBrowser
from .veuszdialog import VeuszDialog, recreate_register

def _(text, disambiguation=None, context="DataEditDialog"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class DatasetTableModel1D(qt4.QAbstractTableModel):
    """Provides access to editing and viewing of datasets."""

    def __init__(self, parent, document, datasetname):
        qt4.QAbstractTableModel.__init__(self, parent)

        self.document = document
        self.dsname = datasetname

        document.signalModified.connect(self.slotDocumentModified)

    def rowCount(self, parent):
        """Return number of rows."""
        if parent.isValid():
            # docs say we should return zero
            return 0

        try:
            return len(self.document.data[self.dsname].data)+1
        except (KeyError, AttributeError):
            return 0

    def slotDocumentModified(self):
        """Called when document modified."""
        self.layoutChanged.emit()

    def columnCount(self, parent):
        """Return number of columns."""

        if parent.isValid():
            return 0
        try:
            ds = self.document.data[self.dsname]
        except KeyError:
            return 0
        return len( ds.column_descriptions )

    def data(self, index, role):
        """Return data for index."""
        # get dataset
        ds = self.document.data[self.dsname]
        if ds is not None:
            # select correct part of dataset
            data = getattr(ds, ds.columns[index.column()])
        if ds is not None and data is not None and role in (
            qt4.Qt.DisplayRole, qt4.Qt.EditRole):
            # blank row at end of data
            if index.row() == len(data):
                return None

            # convert data to data
            d = data[index.row()]
            return ds.uiDataItemToData(d)

        # empty entry
        return None

    def headerData(self, section, orientation, role):
        """Return row numbers or column names."""

        try:
            ds = self.document.data[self.dsname]
        except KeyError:
            return None

        if role == qt4.Qt.DisplayRole:
            if orientation == qt4.Qt.Horizontal:
                # column names
                return ds.column_descriptions[section]
            else:
                if section == len(ds.data):
                    return "+"
                # return row numbers
                return section+1

        return None

    def flags(self, index):
        """Update flags to say that items are editable."""
        if index.isValid():
            f = qt4.QAbstractTableModel.flags(self, index)
            ds = self.document.data.get(self.dsname)
            if ds is not None and ds.editable:
                f |= qt4.Qt.ItemIsEditable
            return f
        return qt4.Qt.ItemIsEnabled

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
        ops = document.OperationMultiple([], descr=_('set value'))
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
            val = ds.uiConvertToDataItem(value)
        except ValueError:
            return False

        ops.addOperation(
            document.OperationDatasetSetVal(self.dsname,
                                            ds.columns[column],
                                            row, val))
        try:
            self.document.applyOperation(ops)
        except RuntimeError:
            return False
        return True

class DatasetTableModelMulti(qt4.QAbstractTableModel):
    """Edit multiple datasets simultaneously with a spreadsheet-like style."""

    def __init__(self, parent, document, datasetnames):
        qt4.QAbstractTableModel.__init__(self, parent)

        self.document = document
        self.dsnames = datasetnames
        document.signalModified.connect(self.slotDocumentModified)

        self.changeset = -1
        self.rows = 0

    def updateCounts(self):
        """Count rows and columns."""

        self.changeset = self.document.changeset

        rows = 0
        rowcounts = self.rowcounts = []
        colcounts = self.colcounts = []
        colattrs = self.colattrs = []

        for dsidx, name in enumerate(self.dsnames):
            if name not in self.document.data:
                continue
            dataset = self.document.data[name]
            if (not hasattr(dataset, 'data') or
                not hasattr(dataset, 'columns') or
                dataset.dimensions != 1):
                continue

            r = len(dataset.data)+1
            rowcounts.append(r)
            rows = max(rows, r)

            attr = []
            for colidx, col in enumerate(dataset.columns):
                data = getattr(dataset, col)
                if data is not None:
                    attr.append( (name, col, dsidx, colidx) )
            colcounts.append( len(attr) )
            colattrs += attr

        self.rows = rows

    def rowCount(self, parent):
        if parent.isValid():
            return 0
        if self.changeset != self.document.changeset:
            self.updateCounts()
        return self.rows

    def columnCount(self, parent):
        if parent.isValid():
            return 0
        if self.changeset != self.document.changeset:
            self.updateCounts()
        return len(self.colattrs)

    def slotDocumentModified(self):
        self.updateCounts()
        self.layoutChanged.emit()

    def data(self, index, role):
        """Return data for index."""

        dsname, colname, dsidx, colidx = self.colattrs[index.column()]
        ds = self.document.data[dsname]
        data = getattr(ds, colname)

        if role == qt4.Qt.DisplayRole:
            if index.row() < self.rowcounts[dsidx]-1:
                # convert data to Data
                d = data[index.row()]
                return ds.uiDataItemToData(d)

        # empty entry
        return None

    def headerData(self, section, orientation, role):
        """Return row numbers or column names."""

        if role == qt4.Qt.DisplayRole:
            if orientation == qt4.Qt.Horizontal:
                # column names
                dsname, colname, dsidx, colidx = self.colattrs[section]
                ds = self.document.data[dsname]
                descr = ds.column_descriptions[colidx]
                header = dsname + '\n' + descr
                return header
            else:
                # return row numbers
                if section == self.rows-1:
                    return "+"
                return section+1

        return None

    def flags(self, index):
        """Update flags to say that items are editable."""
        if index.isValid():
            f = qt4.QAbstractTableModel.flags(self, index)
            dsname, colname, dsidx, colidx = self.colattrs[index.column()]
            ds = self.document.data.get(dsname)
            if ds is not None and ds.editable:
                f |= qt4.Qt.ItemIsEditable
            return f
        return qt4.Qt.ItemIsEnabled

    def setData(self, index, value, role):
        """Validate and set data in dataset."""

        if not index.isValid() or role != qt4.Qt.EditRole:
            return False

        row = index.row()
        column = index.column()
        dsname, colname, dsidx, colidx = self.colattrs[column]
        ds = self.document.data[dsname]

        ops = document.OperationMultiple([], descr=_('set value'))
        if row >= self.rowcounts[dsidx]-1:
            # add number of rows required to add new value below
            ops.addOperation(
                document.OperationDatasetInsertRow(
                    dsname, self.rowcounts[dsidx]-1,
                    row+1-self.rowcounts[dsidx]+1))

        # convert text to value
        try:
            val = ds.uiConvertToDataItem(value)
        except ValueError:
            return False

        ops.addOperation(
            document.OperationDatasetSetVal(dsname, colname, row, val))

        try:
            self.document.applyOperation(ops)
            return True
        except RuntimeError:
            return False

    def insertRows(self, row, count):
        ops = []
        for i, name in enumerate(self.dsnames):
            if self.rowcounts[i]-1 >= row:
                ops.append(
                    document.OperationDatasetInsertRow(name, row, count))
        self.document.applyOperation(
            document.OperationMultiple(ops, _('insert row(s)')))

    def removeRows(self, row, count):
        ops = []
        for i, name in enumerate(self.dsnames):
            if self.rowcounts[i]-1 >= row:
                ops.append(
                    document.OperationDatasetDeleteRow(name, row, count))
        self.document.applyOperation(
            document.OperationMultiple(ops, _('delete row(s)')))

class DatasetTableModel2D(qt4.QAbstractTableModel):
    """A 2D dataset model."""

    def __init__(self, parent, document, datasetname):
        qt4.QAbstractTableModel.__init__(self, parent)

        self.document = document
        self.dsname = datasetname
        self.updatePixelCoords()
        document.signalModified.connect(self.slotDocumentModified)

    def updatePixelCoords(self):
        """Get coordinates at edge of grid."""
        self.xedge = self.yedge = self.xcent = self.ycent = []
        ds = self.document.data.get(self.dsname)
        if ds and ds.dimensions==2:
            self.xcent, self.ycent = ds.getPixelCentres()
            self.xedge, self.yedge = ds.getPixelEdges()

    def rowCount(self, parent):
        if parent.isValid():
            return 0
        try:
            data = self.document.data[self.dsname].data
        except KeyError:
            return 0
        if data is not None and data.ndim==2:
            return data.shape[0]
        else:
            return 0

    def columnCount(self, parent):
        if parent.isValid():
            return 0
        try:
            data = self.document.data[self.dsname].data
        except KeyError:
            return 0
        if data is not None and data.ndim==2:
            return data.shape[1]
        else:
            return 0

    def data(self, index, role):
        if role == qt4.Qt.DisplayRole:
            # get data (note y is reversed, sigh)
            try:
                data = self.document.data[self.dsname].data
            except KeyError:
                return None
            if data is not None and data.ndim==2:
                try:
                    num = data[data.shape[0]-index.row()-1, index.column()]
                    return float(num)
                except IndexError:
                    pass

        return None

    def headerData(self, section, orientation, role):
        """Return headers at top."""

        ds = self.document.data.get(self.dsname)
        if ds.dimensions != 2:
            return None

        xaxis = orientation == qt4.Qt.Horizontal

        # note: y coordinates are upside down (high y is at top)
        if ds is not None and role == qt4.Qt.DisplayRole:
            v = self.xcent[section] if xaxis else self.ycent[
                len(self.ycent)-section-1]
            return '%i (%s)' % (
                len(self.ycent)-section, setting.ui_floattostring(v, maxdp=4))

        elif ds is not None and role == qt4.Qt.ToolTipRole:
            v1 = self.xedge[section] if xaxis else self.yedge[
                len(self.yedge)-section-2]
            v2 = self.xedge[section+1] if xaxis else self.yedge[
                len(self.yedge)-section-1]
            return u'%s\u2013%s' % (setting.ui_floattostring(v1),
                                    setting.ui_floattostring(v2))

        return None

    def flags(self, index):
        """Update flags to say that items are editable."""
        if not index.isValid():
            return qt4.Qt.ItemIsEnabled
        else:
            f = qt4.QAbstractTableModel.flags(self, index)
            ds = self.document.data.get(self.dsname)
            if ds is not None and ds.editable:
                f |= qt4.Qt.ItemIsEditable
            return f

    def slotDocumentModified(self):
        """Called when document modified."""
        self.updatePixelCoords()
        self.layoutChanged.emit()

    def setData(self, index, value, role):
        """Called to set the data."""

        if not index.isValid() or role != qt4.Qt.EditRole:
            return False

        ds = self.document.data[self.dsname]
        row = ds.data.shape[0]-index.row()-1
        col = index.column()

        # update if conversion okay
        try:
            val = ds.uiConvertToDataItem(value)
        except ValueError:
            return False

        op = document.OperationDatasetSetVal2D(
            self.dsname, row, col, val)
        self.document.applyOperation(op)
        return True

class DatasetTableModelND(qt4.QAbstractTableModel):
    """An ND dataset model."""

    def __init__(self, parent, document, datasetname):
        qt4.QAbstractTableModel.__init__(self, parent)

        self.document = document
        self.dsname = datasetname
        document.signalModified.connect(self.slotDocumentModified)

    def rowCount(self, parent):
        if parent.isValid():
            return 0
        try:
            data = self.document.data[self.dsname].data
        except KeyError:
            return 0
        return 0 if data is None else data.size

    def columnCount(self, parent):
        if parent.isValid():
            return 0
        try:
            data = self.document.data[self.dsname].data
        except KeyError:
            return 0
        return 1 if data is not None else 0

    def data(self, index, role):
        """Items in array."""
        if role == qt4.Qt.DisplayRole:
            try:
                data = self.document.data[self.dsname].data
            except KeyError:
                return None
            if data is not None:
                try:
                    num = N.ravel(data)[index.row()]
                    return float(num)
                except IndexError:
                    pass
        return None

    def headerData(self, section, orientation, role):
        """Return headers at top."""

        ds = self.document.data.get(self.dsname)
        if ds is None:
            return None

        if ds is not None and role == qt4.Qt.DisplayRole:
            if orientation == qt4.Qt.Horizontal:
                return _('Value')
            else:
                idx = N.unravel_index(section, ds.data.shape)
                txt = ','.join( [str(v+1) for v in idx] )
                return txt
        return None

    def slotDocumentModified(self):
        """Called when document modified."""
        self.layoutChanged.emit()

class ViewDelegate(qt4.QStyledItemDelegate):
    """Delegate for fixing double editing.
    Normal editing uses double spin box, which is inappropriate
    """

    def createEditor(self, parent, option, index):
        if type(index.data()) is float:
            return qt4.QLineEdit(parent)
        else:
            return qt4.QStyledItemDelegate.createEditor(
                self, parent, option, index)

    def setEditorData(self, editor, index):
        """Override setData to use correct formatting."""
        if type(index.data()) is float:
            txt = setting.ui_floattostring(index.data())
            editor.setText(txt)
        else:
            qt4.QStyledItemDelegate.setEditorData(self, editor, index)

class DataEditDialog(VeuszDialog):
    """Dialog for editing and rearranging data sets."""
    
    def __init__(self, parent, document):
        VeuszDialog.__init__(self, parent, 'dataedit.ui')
        self.document = document

        # set up dataset list
        self.dsbrowser = DatasetBrowser(document, parent, parent)
        self.dsbrowser.setToolTip(
            _('Select multiple datasets to edit simultaneously'))
        self.splitter.insertWidget(0, self.dsbrowser)

        self.deligate = ViewDelegate()
        self.datatableview.setItemDelegate(self.deligate)

        # actions for data table
        for text, slot in (
            (_('Copy'), self.slotCopy),
            (_('Delete row'), self.slotDeleteRow),
            (_('Insert row'), self.slotInsertRow),
            ):
            act = qt4.QAction(text, self)
            act.triggered.connect(slot)
            self.datatableview.addAction(act)
        self.datatableview.setContextMenuPolicy( qt4.Qt.ActionsContextMenu )

        # layout edit dialog improvement
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 4)

        # don't want text to look editable or special
        self.linkedlabel.setFrameShape(qt4.QFrame.NoFrame)
        self.linkedlabel.viewport().setBackgroundRole(qt4.QPalette.Window)

        # document changes
        document.signalModified.connect(self.slotDocumentModified)

        # select first item, if any or initialise if none
        if len(self.document.data) > 0:
            self.selectDataset( sorted(self.document.data)[0] )
        else:
            self.slotDatasetsSelected([])

        self.dsbrowser.navtree.selecteddatasets.connect(
            self.slotDatasetsSelected)

        # connect buttons
        for btn, slot in ( (self.deletebutton, self.slotDatasetDelete),
                           (self.unlinkbutton, self.slotDatasetUnlink),
                           (self.duplicatebutton, self.slotDatasetDuplicate),
                           (self.importbutton, self.slotDatasetImport),
                           (self.createbutton, self.slotDatasetCreate),
                           (self.editbutton, self.slotDatasetEdit),
                           ):
            btn.clicked.connect(slot)

        # menu for new button
        self.newmenu = qt4.QMenu()
        for text, slot in ( (_('Numerical dataset'), self.slotNewNumericalDataset),
                            (_('Text dataset'), self.slotNewTextDataset),
                            (_('Date/time dataset'), self.slotNewDateDataset) ):
            a = self.newmenu.addAction(text)
            a.triggered.connect(slot)
        self.newbutton.setMenu(self.newmenu)

    def slotDatasetsSelected(self, names):
        """Called when a new dataset is selected."""

        # FIXME: Make readonly models readonly!!
        model = None
        if len(names) == 1:
            # get selected dataset
            ds = self.document.data[names[0]]

            # make model for dataset
            if ds.dimensions == 1:
                model = DatasetTableModel1D(self, self.document, names[0])
            elif ds.dimensions == 2:
                model = DatasetTableModel2D(self, self.document, names[0])
            elif ds.dimensions == -1:
                model = DatasetTableModelND(self, self.document, names[0])
        elif len(names) > 1:
            model = DatasetTableModelMulti(self, self.document, names)

        # disable context menu if no menu
        for a in self.datatableview.actions():
            a.setEnabled(model is not None)

        self.datatableview.setModel(model)    
        self.setUnlinkState()

    def setUnlinkState(self):
        """Enable the unlink button correctly."""

        linkinfo = []
        canunlink = []
        canedit = []
        names = self.dsbrowser.navtree.getSelectedDatasets()
        for name in names:
            ds = self.document.data[name]
            canunlink.append(ds.canUnlink())
            if len(names) > 1:
                linkinfo.append(name)
            linkinfo.append(ds.linkedInformation())
            canedit.append(type(ds) in recreate_register)

        self.editbutton.setVisible(any(canedit))
        self.unlinkbutton.setEnabled(any(canunlink))
        self.linkedlabel.setText('\n'.join(linkinfo))
        self.deletebutton.setEnabled(bool(names))
        self.duplicatebutton.setEnabled(bool(names))

    def slotDocumentModified(self):
        """Set unlink status when document modified."""
        self.setUnlinkState()

    def selectDataset(self, dsname):
        """Select dataset with name given."""
        self.dsbrowser.navtree.selectDataset(dsname)
        self.slotDatasetsSelected([dsname])

    def slotDatasetDelete(self):
        """Delete selected dataset."""
        dsnames = self.dsbrowser.navtree.getSelectedDatasets()
        self.document.applyOperation(
            document.OperationMultiple(
                [document.OperationDatasetDelete(n) for n in dsnames],
                descr=_('delete dataset(s)')))

    def slotDatasetUnlink(self):
        """Allow user to remove link to file or other datasets."""
        ops = []
        for name in self.dsbrowser.navtree.getSelectedDatasets():
            d = self.document.data[name]
            if d.linked is not None:
                ops.append(document.OperationDatasetUnlinkFile(name))
            elif d.canUnlink():
                ops.append(document.OperationDatasetUnlinkRelation(name))
        if ops:
            self.document.applyOperation(
                document.OperationMultiple(ops, _('unlink dataset(s)')))

    def slotDatasetDuplicate(self):
        """Duplicate selected datasets."""
        ops = []
        for name in self.dsbrowser.navtree.getSelectedDatasets():
            # generate new name for dataset
            newname = name + '_copy'
            index = 2
            while newname in self.document.data:
                newname = '%s_copy_%i' % (name, index)
                index += 1
            ops.append(
                document.OperationDatasetDuplicate(name, newname))
        if ops:
            self.document.applyOperation(
                document.OperationMultiple(ops, _('duplicate dataset(s)')))

    def slotDatasetImport(self):
        """Show import dialog."""
        self.mainwindow.slotDataImport()

    def slotDatasetCreate(self):
        """Show dataset creation dialog."""
        self.mainwindow.slotDataCreate()

    def slotDatasetEdit(self):
        """Reload dataset into dataset creation dialog."""
        for name in self.dsbrowser.navtree.getSelectedDatasets():
            dataset = self.document.data[name]
            try:
                recreate_register[type(dataset)](self.mainwindow, self.document,
                                                 dataset, name)
            except KeyError:
                pass

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
            rowitems.append(
                cstr(model.createIndex(row, column).data()) )
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
        self.newDataset( datasets.Dataset(data=[0.]) )

    def slotNewTextDataset(self):
        """Add new text dataset."""
        self.newDataset( datasets.DatasetText(data=['']) )

    def slotNewDateDataset(self):
        """Add new date dataset."""
        self.newDataset( datasets.DatasetDateTime(data=[]) )

    def newDataset(self, ds):
        """Add new dataset to document."""
        # get a name for dataset
        name = _('new dataset')
        if name in self.document.data:
            count = 1
            while name in self.document.data:
                name = _('new dataset %i') % count
                count += 1

        # add new dataset
        self.document.applyOperation(
            document.OperationDatasetSet(name, ds))

        self.dsbrowser.selectDataset(name)
