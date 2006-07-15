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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

"""Module for implementing dialog box for viewing/editing data."""

import re
import itertools
import os.path

import numarray as N
import veusz.qtall as qt4

import veusz.setting as setting
import veusz.document as document
import importdialog

class _DatasetNameValidator(qt4.QValidator):
    """A validator to check for dataset names.

    Disallows existing names, " ", "+", "-" or ",", or zero-length
    """

    def __init__(self, document, parent):
        qt4.QValidator.__init__(self, parent)
        self.document = document
        self.dsre = re.compile('^[^ +-,]*$')

    def validate(self, input, pos):
        name = unicode(input)

        if name in self.document.data or len(name) == 0:
            return (qt4.QValidator.Intermediate, pos)
        elif self.dsre.match(name):
            return (qt4.QValidator.Acceptable, pos)
        else:
            return (qt4.QValidator.Invalid, pos)

class _DatasetNameDialog(qt4.QDialog):
    """A dialog for return a new dataset name.

    Input is checked using _DatasetNameValidator
    """

    def __init__(self, caption, prompt, document, oldname, *args):
        """Initialise the dialog.

        caption is the dialog's caption
        prompt is the prompt to show
        document is the document to check dataset names against
        oldname is an existing dataset name to show initially
        other arguments are passed to the QDialog __init__
        """
        
        qt4.QDialog.__init__(self, *args)
        self.setCaption(caption)

        spacing = self.fontMetrics().height() / 2

        # everything controlled with vbox
        formlayout = qt4.QVBoxLayout(self, spacing, spacing)
        spacer = qt4.QSpacerItem(spacing, spacing, qt4.QSizePolicy.Minimum,
                                qt4.QSizePolicy.Expanding)
        formlayout.addItem(spacer)

        # label at top
        l = qt4.QLabel(prompt, self)
        formlayout.addWidget(l)

        # edit box here (validated for dataset names)
        self.lineedit = qt4.QLineEdit(oldname, self)
        self.lineedit.setValidator( _DatasetNameValidator(document, self) )
        self.connect( self.lineedit, qt4.SIGNAL('returnPressed()'),
                      self.slotOK )
        formlayout.addWidget(self.lineedit)

        # buttons at  bottom of form
        buttonlayout = qt4.QHBoxLayout(None, 0, spacing)

        spacer = qt4.QSpacerItem(0, 0, qt4.QSizePolicy.Expanding,
                                qt4.QSizePolicy.Minimum)
        buttonlayout.addItem(spacer)

        okbutton = qt4.QPushButton("&OK", self)
        self.connect(okbutton, qt4.SIGNAL('clicked()'),
                     self.slotOK)
        buttonlayout.addWidget(okbutton)

        cancelbutton = qt4.QPushButton("&Cancel", self)
        self.connect(cancelbutton, qt4.SIGNAL('clicked()'),
                     self.reject)
        buttonlayout.addWidget(cancelbutton)

        formlayout.addLayout(buttonlayout)

        spacer = qt4.QSpacerItem(spacing, spacing, qt4.QSizePolicy.Minimum,
                                qt4.QSizePolicy.Expanding)
        formlayout.addItem(spacer)

    def slotOK(self):
        """Check the validator, and close if okay."""

        if self.lineedit.hasAcceptableInput():
            self.accept()
        else:
            qt4.QMessageBox("Veusz",
                           "Invalid dataset name '%s'" % self.getName(),
                           qt4.QMessageBox.Warning,
                           qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                           qt4.QMessageBox.NoButton,
                           qt4.QMessageBox.NoButton,
                           self).exec_loop()

    def getName(self):
        """Return the name entered."""
        return unicode(self.lineedit.text())

class _DSException(RuntimeError):
    """A class to handle errors while trying to create datasets."""
    pass

class DatasetNewDialog(qt4.QDialog):
    """New dataset dialog."""

    def __init__(self, document, parent):
        qt4.QDialog.__init__(self, parent, 'DataCreateDialog', False,
                            qt4.Qt.WDestructiveClose)
        self.document = document

        self.setCaption("Create dataset - Veusz")

        spacing = self.fontMetrics().height() / 2
        vboxlayout = qt4.QVBoxLayout(self, spacing, spacing)

        # change the name of the dataset
        hbox = qt4.QHBox(self)
        hbox.setSpacing(spacing)
        l = qt4.QLabel("&Name", hbox)
        self.nameedit = qt4.QLineEdit("", hbox)
        self.nameedit.setValidator( _DatasetNameValidator(document, self) )
        l.setBuddy(self.nameedit)
        vboxlayout.addWidget(hbox)

        # choose the method of creating data
        methgrp = qt4.QVButtonGroup("Method of creating new data", self)
        self.connect(methgrp, qt4.SIGNAL('clicked(int)'), self.slotRadioPressed)
        vboxlayout.addWidget(methgrp)

        # if we want to specify a value or range
        valds = qt4.QRadioButton("&Value or range", methgrp)

        hbox = qt4.QHBox(methgrp)
        hbox.setSpacing(spacing)
        qt4.QLabel('Number of steps', hbox)
        self.valsteps = qt4.QLineEdit('100', hbox)
        val = qt4.QIntValidator(self)
        val.setBottom(2)
        self.valsteps.setValidator(val)

        # create use parametric form
        exprds = qt4.QRadioButton("&Parametric form", methgrp)

        hbox = qt4.QHBox(methgrp)
        hbox.setSpacing(spacing)
        qt4.QLabel('t =', hbox)
        self.parastart = qt4.QLineEdit('0', hbox)
        self.parastart.setValidator( qt4.QDoubleValidator(self) )
        qt4.QLabel('to', hbox)
        self.paraend = qt4.QLineEdit('1', hbox)
        self.paraend.setValidator( qt4.QDoubleValidator(self) )
        qt4.QLabel('in', hbox)
        self.parasteps = qt4.QLineEdit('100', hbox)
        val = qt4.QIntValidator(self)
        val.setBottom(2)
        self.parasteps.setValidator(val)
        qt4.QLabel('steps (inclusive)', hbox)

        # use existing datasets to create an expression
        exprds = qt4.QRadioButton("&Expression from existing datasets", methgrp)

        # enter values for dataset
        dsgrp = qt4.QVButtonGroup("Dataset", self)
        vboxlayout.addWidget(dsgrp)

        self.valuelabel = qt4.QLabel('', dsgrp)
        v = self.valsettings = qt4.QWidget(dsgrp)
        grdlayout = qt4.QGridLayout(v, -1, -1, spacing)
        self.dsedits = {}
        for num, l in itertools.izip( itertools.count(),
                                      [('data', 'V&alue'),
                                       ('serr', '&Symmetric error'),
                                       ('perr', 'P&ositive error'),
                                       ('nerr', 'Ne&gative error')]):
            name, caption = l
            l = qt4.QLabel(caption, v)
            grdlayout.addWidget(l, num, 0)
            e = qt4.QLineEdit('', v)
            l.setBuddy(e)
            grdlayout.addWidget(e, num, 1)
            self.dsedits[name] = e

        # below text boxes...
        self.linkbutton = qt4.QCheckBox('Keep this dataset &linked to these expressions',
                                       dsgrp)
            
        # buttons
        w = qt4.QWidget(self)
        vboxlayout.addWidget(w)
        l = qt4.QHBoxLayout(w, 0, spacing)

        self.statuslabel = qt4.QLabel('', w)
        l.addWidget(self.statuslabel)
        l.addItem( qt4.QSpacerItem(1, 1, qt4.QSizePolicy.Expanding,
                                  qt4.QSizePolicy.Minimum) )
        b = qt4.QPushButton('C&reate', w)
        l.addWidget(b)
        self.connect(b, qt4.SIGNAL('clicked()'), self.slotCreate )
        b = qt4.QPushButton('&Close', w)
        l.addWidget(b)
        self.connect(b, qt4.SIGNAL('clicked()'), self.slotClose )
        
        # initially check the first option (value/range)
        valds.setChecked(True)
        self.slotRadioPressed(0)

    def slotClose(self):
        """Close the dialog."""
        self.close()

    def slotRadioPressed(self, item):
        """If a radio button is pressed."""

        self.selectedmethod = item

        # enable correct edit boxes
        # use item number to look up correct widgets
        for id, wid in [ (0, self.valsteps),
                         (1, self.parastart),
                         (1, self.paraend),
                         (1, self.parasteps) ]:
            wid.setEnabled( id == item )

        # set a help label
        self.valuelabel.setText(
            ['Enter constant values here, leave blank if appropriate, '
             'or enter an inclusive range, e.g. 1:10',
             'Enter expressions as a function of t, or leave blank',
             'Enter expressions as a function of existing datasets']
            [item]
            )

        # only allow links if an expression is used
        self.linkbutton.setEnabled(item == 2)

    def slotCreate(self):
        """Actually create the dataset."""

        try:
            name = unicode( self.nameedit.text() )
            if not self.nameedit.hasAcceptableInput():
                raise _DSException("Invalid dataset name '%s'" % name)

            # call appropriate function for option
            fn = [self.createFromRange, self.createParametric,
                  self.createFromExpression][self.selectedmethod]

            # make a new dataset from the returned data
            fn(name)

            self.statuslabel.setText("Created dataset '%s'" % name)

        except (document.CreateDatasetException, _DSException), e:
            # all bad roads lead here - take exception string and tell user
            self.statuslabel.setText("Creation failed")
            qt4.QMessageBox("Veusz",
                           str(e),
                           qt4.QMessageBox.Warning,
                           qt4.QMessageBox.Ok | qt4.QMessageBox.Default,
                           qt4.QMessageBox.NoButton,
                           qt4.QMessageBox.NoButton,
                           self).exec_loop()

    def createFromRange(self, name):
        """Make dataset from a range or constant.
        name is the name of the dataset
        
        Raises _DSException if error
        """

        # check whether number of steps is valid
        if not self.valsteps.hasAcceptableInput():
            raise _DSException("Number of steps is invalid")
        numsteps = int( unicode(self.valsteps.text()) )

        # go over each of the ranges / values
        vals = {}
        for key, cntrl in self.dsedits.iteritems():
            text = unicode( cntrl.text() ).strip()

            if not text:
                continue
                
            if text.find(':') != -1:
                # an actual range
                parts = text.split(':')
                
                if len(parts) != 2:
                    raise _DSException("Incorrect range format, use form 1:10")
                try:
                    minval, maxval = float(parts[0]), float(parts[1])
                except ValueError:
                    raise _DSException("Invalid number in range")

            else:
                try:
                    minval = float(text)
                except ValueError:
                    raise _DSException("Invalid number")
                maxval = minval
                
            vals[key] = (minval, maxval)
            
        op = document.OperationDatasetCreateRange(name, numsteps, vals)
        self.document.applyOperation(op)

    def createParametric(self, name):
        """Use a parametric form to create the dataset.

        Raises _DSException if error
        """

        # check whether number and range of steps is valid
        if not self.parastart.hasAcceptableInput():
            raise _DSException("Starting value invalid")
        t0 = float( unicode(self.parastart.text()) )
        if not self.paraend.hasAcceptableInput():
            raise _DSException("Ending value invalid")
        t1 = float( unicode(self.paraend.text()) )
        if not self.parasteps.hasAcceptableInput():
            raise _DSException("Invalid number of steps")
        numsteps = int( unicode(self.parasteps.text()) )

        # get expressions
        vals = {}
        for key, cntrl in self.dsedits.iteritems():
            text = unicode( cntrl.text() ).strip()
            if text:
                vals[key] = text
           
        op = document.OperationDatasetCreateParameteric(name, t0, t1, numsteps,
                                                        vals)
        self.document.applyOperation(op)
      
    def createFromExpression(self, name):
        """Create a dataset based on the expressions given."""

        # get expression for each part of the dataset
        vals = {}
        for key, cntrl in self.dsedits.iteritems():
            text = unicode( cntrl.text() ).strip()
            if text:
                vals[key] = text

        link = self.linkbutton.isChecked()
        op = document.OperationDatasetCreateExpression(name, vals, link)
        op.validateExpression(self.document)
        self.document.applyOperation(op)

class _DataEditTable(qt4.QWidget):
    """A QTable for displaying data from datasets.

    The table draws data itself, and notifies the document of any changes
    to the data.
    """
    
    # TODO: Add operation to delete columns
    # TODO: Add cut/paste operations

    colnames = ('data', 'serr', 'perr', 'nerr')
    
    def __init__(self, parent, document):
        """Initialise the table with the given parent."""

        qttable.QTable.__init__(self, 0, 4, parent)
        self.setSizePolicy(qt4.QSizePolicy.Expanding, qt4.QSizePolicy.Expanding)
        self.setLeftMargin(self.fontMetrics().width("W999999W"))
        self.setSelectionMode(qttable.QTable.NoSelection)

        # set the headers to appropriate values
        for num, text in itertools.izip( itertools.count(),
                                         ['Value', 'Symmetric error',
                                          'Positive error',
                                          'Negative error'] ):
            self.horizontalHeader().setLabel(num, text)

        # data the table shows
        self.coldata = [None, None, None, None]

        # the edit control, if any
        self.edit = None
        
        # keep track of document
        self.document = document
        
    def setDataset(self, datasetname):
        """Show the given dataset in the widget."""

        ds = self.document.data[datasetname]

        # FIXME: Multidimensional datasets not handled properly
        if ds.dimensions != 1:
            return

        self.datasetname = datasetname
        self.coldata = [ds.data, ds.serr, ds.perr, ds.nerr]
        self.setNumRows( len(self.coldata[0]) )

    # all these do nothing, as we're not using TableItems
    def resizeData(self, i):
        return
    def item(self, r, c):
        return None
    def setItem(self, r, c, i):
        return
    def clearCell(self, r, c):
        return
    def insertWidget(self, r, c, w):
        return
    def clearCellWidget(self, r, c):
        return

    def text(self, r, c):
        """Return the text for the specified cell."""
        ds = self.coldata[c]
        if ds == None:
            return ''
        else:
            return str(ds[r])

    def paintCell(self, painter, r, c, cr, selected, colgroup = None):
        """Draw the given cell (r,c) in rectangle cr, selected if it is
        selected, and in the colour colgroup."""

        if colgroup == None:
            colgroup = self.colorGroup()

        # draw background and grid
        qttable.QTable.paintCell(self, painter, r, c, cr, selected, colgroup)

        # change font to appropriate colour
        if selected:
            painter.setPen( qt4.QPen(colgroup.highlightedText()) )
        else:
            painter.setPen( qt4.QPen(colgroup.text()) )

        # draw text for cell
        painter.drawText(0, 0, cr.width(), cr.height(),
                         qt4.Qt.AlignRight | qt4.Qt.AlignVCenter,
                         self.text(r, c))

    def createEditor(self, r, c, initfromcell):
        """Called if the user starts editing a cell."""
        self.edit = qt4.QLineEdit(self.viewport())
        v = qt4.QDoubleValidator(self.edit)
        self.edit.setValidator(v)
        if initfromcell:
            self.edit.setText(self.text(r, c))
        return self.edit

    def cellWidget(self, r, c):
        """Return widget for cell."""
        if r == self.currEditRow() and c == self.currEditCol():
            return self.edit
        return None

    def beginEdit(self, r, c, replace):
        """Check whether it is valid to edit the column."""
        # check whether there is data in the column
        if self.coldata[c] == None:
            if self.isReadOnly():
                return None
            mb = qt4.QMessageBox("Veusz",
                                "This column has no data. Initialise with "
                                "zero and continue?",
                                qt4.QMessageBox.Warning,
                                qt4.QMessageBox.Yes|qt4.QMessageBox.Default,
                                qt4.QMessageBox.No,
                                qt4.QMessageBox.NoButton,
                                self)
            mb.setButtonText(qt4.QMessageBox.Yes, "&Initialise")
            mb.setButtonText(qt4.QMessageBox.No, "&Cancel")
            if mb.exec_loop() != qt4.QMessageBox.Yes:
                return None
            self.document.applyOperation( document.OperationDatasetAddColumn(self.datasetname, self.colnames[c]) )
            ds = self.document.data[self.datasetname]
            self.coldata[c] = getattr(ds, self.colnames[c])
                
        return qttable.QTable.beginEdit(self, r, c, replace)
    
    def endEdit(self, r, c, accept, replace):
        """User finished editing."""
        qttable.QTable.endEdit(self, r, c, accept, replace)
        if self.edit != None:
            e = self.edit
            self.edit = None
            e.deleteLater()

    def setCellContentFromEditor(self, r, c):
        """Use the edit control to set the value in the dataset."""

        if self.edit != None:
            nums = self.coldata[c]
            assert nums != None
            try:
                val = float( unicode(self.edit.text()) )
            except ValueError:
                # floating point conversion error
                self.edit.setText(self.text(r, c))
                qt4.QMessageBox("Veusz",
                               "Invalid number",
                               qt4.QMessageBox.Warning,
                               qt4.QMessageBox.Ok,
                               qt4.QMessageBox.NoButton,
                               qt4.QMessageBox.NoButton,
                               self).exec_loop()
                return

            op = document.OperationDatasetSetVal(self.datasetname,
                                                 self.colnames[c],
                                                 r, val)
            self.document.applyOperation(op)

class DatasetTableModel(qt4.QAbstractTableModel):
    """Provides access to editing and viewing of datasets."""

    colnames = ('data', 'serr', 'perr', 'nerr')
    
    def __init__(self, parent, document, datasetname):
        qt4.QAbstractTableModel.__init__(self, parent)

        self.document = document
        self.dsname = datasetname

    def rowCount(self, parent):
        """Return number of rows."""
        return len(self.document.data[self.dsname].data)
        
    def columnCount(self, parent):
        return 4

    def data(self, index, role):
        if role == qt4.Qt.DisplayRole:
            # select correct part of dataset
            ds = self.document.data[self.dsname]
            data = getattr(ds, self.colnames[index.column()])

            if data != None:
                return qt4.QVariant( data[index.row()] )

        # return nothing otherwise
        return qt4.QVariant()

    def headerData(self, section, orientation, role):
        """Return headers at top."""

        if role == qt4.Qt.DisplayRole:
            if orientation == qt4.Qt.Horizontal:
                val = ['Data', 'Sym. errors', 'Pos. errors',
                       'Neg. errors'][section]
                return qt4.QVariant(val)
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
            data = getattr(ds, self.colnames[index.column()])

            # add new column if necessary
            if data == None:
                self.document.applyOperation( document.OperationDatasetAddColumn(self.dsname, self.colnames[column]) )


            # update if conversion okay
            f, ok = value.toDouble()
            if ok:
                op = document.OperationDatasetSetVal(self.dsname,
                                                     self.colnames[column],
                                                     row, f)
                self.document.applyOperation(op)
                return True

        return False

class DatasetListModel(qt4.QAbstractListModel):
    """A model to allow the list of datasets to be viewed."""

    def __init__(self, parent, document):
        qt4.QAbstractListModel.__init__(self, parent)
        self.document = document

        self.connect(document, qt4.SIGNAL('sigModified'),
                     self.slotDocumentModified)
        self.updateList()

    def updateList(self):
        """Update internal list."""
        self.datasets = list( self.document.data.keys() )
        self.datasets.sort()

    def slotDocumentModified(self):
        """Called when document modified."""
        self.updateList()
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
    
class DataEditDialog2(qt4.QDialog):
    """Dialog for editing and rearranging data sets."""
    
    def __init__(self, parent, document, *args):

        # load up UI
        qt4.QDialog.__init__(self, parent, *args)
        qt4.loadUi(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'dataedit.ui'),
                   self)
        self.document = document

        # set up dataset list
        self.dslistmodel = DatasetListModel(self, document)
        self.datasetlistview.setModel(self.dslistmodel)

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

    def slotDatasetSelected(self, current, previous):
        """Called when a new dataset is selected."""

        # FIXME: Make readonly models readonly!!
        index = current.indexes()[0]
        name = self.dslistmodel.datasetName(index)
        self.datatableview.setModel( DatasetTableModel(self, self.document,
                                                       name) )

        self.setUnlinkState()

    def setUnlinkState(self):
        """Enable the unlink button correctly."""
        # get dataset
        dsname = self.getSelectedDataset()
        ds = self.document.data[dsname]

        # linked dataset
        readonly = False
        if ds.linked == None:
            fn = 'None'
            unlink = False
        else:
            fn = ds.linked.filename
            unlink = True
        text = 'Linked file: %s' % fn
        
        if isinstance(ds, document.DatasetExpression):
            # for datasets linked by expressions
            items = ['Linked expression dataset:']
            for label, part in ( ('Values', 'data'),
                                 ('Symmetric errors', 'serr'),
                                 ('Positive errors', 'perr'),
                                 ('Negative errors', 'nerr') ):
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
