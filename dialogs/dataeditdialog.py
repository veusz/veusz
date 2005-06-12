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

import numarray as N
import qt
import qttable

import setting
import document
import importdialog

class _DatasetNameValidator(qt.QValidator):
    """A validator to check for dataset names.

    Disallows existing names, " ", "+", "-" or ",", or zero-length
    """

    def __init__(self, document, parent):
        qt.QValidator.__init__(self, parent)
        self.document = document
        self.dsre = re.compile('^[^ +-,]*$')

    def validate(self, input, pos):
        name = unicode(input)

        if name in self.document.data or len(name) == 0:
            return (qt.QValidator.Intermediate, pos)
        elif self.dsre.match(name):
            return (qt.QValidator.Acceptable, pos)
        else:
            return (qt.QValidator.Invalid, pos)

class _DatasetNameDialog(qt.QDialog):
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
        
        qt.QDialog.__init__(self, *args)
        self.setCaption(caption)

        spacing = self.fontMetrics().height() / 2

        # everything controlled with vbox
        formlayout = qt.QVBoxLayout(self, spacing, spacing)
        spacer = qt.QSpacerItem(spacing, spacing, qt.QSizePolicy.Minimum,
                                qt.QSizePolicy.Expanding)
        formlayout.addItem(spacer)

        # label at top
        l = qt.QLabel(prompt, self)
        formlayout.addWidget(l)

        # edit box here (validated for dataset names)
        self.lineedit = qt.QLineEdit(oldname, self)
        self.lineedit.setValidator( _DatasetNameValidator(document, self) )
        self.connect( self.lineedit, qt.SIGNAL('returnPressed()'),
                      self.slotOK )
        formlayout.addWidget(self.lineedit)

        # buttons at  bottom of form
        buttonlayout = qt.QHBoxLayout(None, 0, spacing)

        spacer = qt.QSpacerItem(0, 0, qt.QSizePolicy.Expanding,
                                qt.QSizePolicy.Minimum)
        buttonlayout.addItem(spacer)

        okbutton = qt.QPushButton("&OK", self)
        self.connect(okbutton, qt.SIGNAL('clicked()'),
                     self.slotOK)
        buttonlayout.addWidget(okbutton)

        cancelbutton = qt.QPushButton("&Cancel", self)
        self.connect(cancelbutton, qt.SIGNAL('clicked()'),
                     self.reject)
        buttonlayout.addWidget(cancelbutton)

        formlayout.addLayout(buttonlayout)

        spacer = qt.QSpacerItem(spacing, spacing, qt.QSizePolicy.Minimum,
                                qt.QSizePolicy.Expanding)
        formlayout.addItem(spacer)

    def slotOK(self):
        """Check the validator, and close if okay."""

        if self.lineedit.hasAcceptableInput():
            self.accept()
        else:
            qt.QMessageBox("Veusz",
                           "Invalid dataset name '%s'" % self.getName(),
                           qt.QMessageBox.Warning,
                           qt.QMessageBox.Ok | qt.QMessageBox.Default,
                           qt.QMessageBox.NoButton,
                           qt.QMessageBox.NoButton,
                           self).exec_loop()

    def getName(self):
        """Return the name entered."""
        return unicode(self.lineedit.text())

class _DSException(RuntimeError):
    """A class to handle errors while trying to create datasets."""
    pass

class DatasetNewDialog(qt.QDialog):
    """New dataset dialog."""

    def __init__(self, document, parent):
        qt.QDialog.__init__(self, parent, 'DataCreateDialog', False,
                            qt.Qt.WDestructiveClose)
        self.document = document

        self.setCaption("Create dataset - Veusz")

        spacing = self.fontMetrics().height() / 2
        vboxlayout = qt.QVBoxLayout(self, spacing, spacing)

        # change the name of the dataset
        hbox = qt.QHBox(self)
        hbox.setSpacing(spacing)
        l = qt.QLabel("&Name", hbox)
        self.nameedit = qt.QLineEdit("", hbox)
        self.nameedit.setValidator( _DatasetNameValidator(document, self) )
        l.setBuddy(self.nameedit)
        vboxlayout.addWidget(hbox)

        # choose the method of creating data
        methgrp = qt.QVButtonGroup("Method of creating new data", self)
        self.connect(methgrp, qt.SIGNAL('clicked(int)'), self.slotRadioPressed)
        vboxlayout.addWidget(methgrp)

        # if we want to specify a value or range
        valds = qt.QRadioButton("&Value or range", methgrp)

        hbox = qt.QHBox(methgrp)
        hbox.setSpacing(spacing)
        qt.QLabel('Number of steps', hbox)
        self.valsteps = qt.QLineEdit('100', hbox)
        val = qt.QIntValidator(self)
        val.setBottom(2)
        self.valsteps.setValidator(val)

        # create use parametric form
        exprds = qt.QRadioButton("&Parametric form", methgrp)

        hbox = qt.QHBox(methgrp)
        hbox.setSpacing(spacing)
        qt.QLabel('t =', hbox)
        self.parastart = qt.QLineEdit('0', hbox)
        self.parastart.setValidator( qt.QDoubleValidator(self) )
        qt.QLabel('to', hbox)
        self.paraend = qt.QLineEdit('1', hbox)
        self.paraend.setValidator( qt.QDoubleValidator(self) )
        qt.QLabel('in', hbox)
        self.parasteps = qt.QLineEdit('100', hbox)
        val = qt.QIntValidator(self)
        val.setBottom(2)
        self.parasteps.setValidator(val)
        qt.QLabel('steps (inclusive)', hbox)

        # use existing datasets to create an expression
        exprds = qt.QRadioButton("&Expression from existing datasets", methgrp)

        # enter values for dataset
        dsgrp = qt.QVButtonGroup("Dataset", self)
        vboxlayout.addWidget(dsgrp)

        self.valuelabel = qt.QLabel('', dsgrp)
        v = self.valsettings = qt.QWidget(dsgrp)
        grdlayout = qt.QGridLayout(v, -1, -1, spacing)
        self.dsedits = {}
        for num, l in itertools.izip( itertools.count(),
                                      [('data', 'V&alue'),
                                       ('serr', '&Symmetric error'),
                                       ('perr', 'P&ositive error'),
                                       ('nerr', 'Ne&gative error')]):
            name, caption = l
            l = qt.QLabel(caption, v)
            grdlayout.addWidget(l, num, 0)
            e = qt.QLineEdit('', v)
            l.setBuddy(e)
            grdlayout.addWidget(e, num, 1)
            self.dsedits[name] = e
            
        # buttons
        w = qt.QWidget(self)
        vboxlayout.addWidget(w)
        l = qt.QHBoxLayout(w, 0, spacing)

        self.statuslabel = qt.QLabel('', w)
        l.addWidget(self.statuslabel)
        l.addItem( qt.QSpacerItem(1, 1, qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum) )
        b = qt.QPushButton('C&reate', w)
        l.addWidget(b)
        self.connect(b, qt.SIGNAL('clicked()'), self.slotCreate )
        b = qt.QPushButton('&Close', w)
        l.addWidget(b)
        self.connect(b, qt.SIGNAL('clicked()'), self.slotClose )
        
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

    def slotCreate(self):
        """Actually create the dataset."""

        try:
            name = unicode( self.nameedit.text() )
            if not self.nameedit.hasAcceptableInput():
                raise _DSException("Invalid dataset name '%s'" % name)

            # call appropriate function for option
            fn = [self.createFromRange, self.createParametric,
                  self.createFromExpression][self.selectedmethod]
            vals = fn()

            # make a new dataset from the returned data
            ds = document.Dataset(**vals)
            self.document.setData(name, ds)

            self.statuslabel.setText("Created dataset '%s'" % name)

        except _DSException, e:
            # all bad roads lead here - take exception string and tell user
            self.statuslabel.setText("Creation failed")
            qt.QMessageBox("Veusz",
                           str(e),
                           qt.QMessageBox.Warning,
                           qt.QMessageBox.Ok | qt.QMessageBox.Default,
                           qt.QMessageBox.NoButton,
                           qt.QMessageBox.NoButton,
                           self).exec_loop()

    def createFromRange(self):
        """Make dataset from a range or constant.

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
                # blank text
                vals[key] = None

            elif text.find(':') != -1:
                # a range a:b
                parts = text.split(':')
                if len(parts) != 2:
                    raise _DSException("Incorrect range format")
                try:
                    minval, maxval = float(parts[0]), float(parts[1])
                except ValueError:
                    raise _DSException("Incorrect number in range")

                # hard to use arange to do this as endpoints inaccurate
                delta = (maxval - minval) / (numsteps-1)
                vals[key] = N.fromfunction( lambda x: minval+x*delta,
                                            (numsteps,) )

            else:
                # a value (expand to size of numsteps)
                try:
                    vals[key] = float(text) + N.zeros( (numsteps,),
                                                       type=N.Float64 )

                except ValueError:
                    raise _DSException("Invalid floating point value")

        return vals

    def createParametric(self):
        """Use a parametric form to create the dataset.

        Raises _DSException if error
        """

        # check whether number and range of steps is valid
        if not self.parastart.hasAcceptableInput():
            raise _DSException("Starting value invalid")
        start = float( unicode(self.parastart.text()) )
        if not self.paraend.hasAcceptableInput():
            raise _DSException("Ending value invalid")
        end = float( unicode(self.paraend.text()) )
        if not self.parasteps.hasAcceptableInput():
            raise _DSException("Invalid number of steps")
        steps = int( unicode(self.parasteps.text()) )

        # use these values to calculate the array t
        delta = (end-start) / (steps-1)
        t =  N.fromfunction( lambda x: start+x*delta,
                             (steps,) )

        # define environment to evaluate
        fnenviron = globals().copy()
        exec 'from numarray import *' in fnenviron
        fnenviron['t'] = t

        # calculate for each of the dataset components
        vals = {}
        for key, cntrl in self.dsedits.iteritems():
            text = unicode( cntrl.text() ).strip()

            if not text:
                # blank text
                vals[key] = None
            else:
                # HACK to ensure the result is a numarray
                expr = '(%s) + t*0.' % text
                try:
                    vals[key] = eval( expr, fnenviron )
                except Exception, e:
                    raise _DSException("Error evaluating expession '%s'\n"
                                       "Error: '%s'" % (text, str(e)) )

        return vals

    def createFromExpression(self):

        # define environment to evaluate expressions
        fnenviron = globals().copy()
        exec 'from numarray import *' in fnenviron

        # calculate for each of the dataset components
        vals = {}
        for key, cntrl in self.dsedits.iteritems():
            text = unicode( cntrl.text() ).strip()

            if not text:
                # blank text
                vals[key] = None
            else:
                # populate environment with parts of dataset
                # this assumes that dataset has members data, serr, nerr, perr
                env = fnenviron.copy()
                for name, ds in self.document.data.iteritems():
                    env[name] = ds.__dict__[key]

                # evaluate the expression
                try:
                    vals[key] = eval(text, env)
                except Exception, e:
                    raise _DSException("Error evaluating expession '%s'\n"
                                       "Error: '%s'" % (text, str(e)) )

        # check length of values is the same
        last = None
        for val in vals.itervalues():
            if val != None:
                try:
                    l = len(val)
                except TypeError:
                    l = -100
                if last != None and last != l:
                    raise _DSException("Expressions for dataset parts do not "
                                       "yield results of the same length")
                last = len(val)

        return vals

class _DataEditTable(qttable.QTable):
    """A QTable for displaying data from datasets.

    The table draws data itself, and notifies the document of any changes
    to the data.
    """

    def __init__(self, parent):
        """Initialise the table with the given parent."""

        qttable.QTable.__init__(self, 0, 4, parent)
        self.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Expanding)
        self.setLeftMargin(self.fontMetrics().width("W999999W"))
        self.setSelectionMode(qttable.QTable.NoSelection)

        # set the headers to appropriate values
        for num, text in itertools.izip( itertools.count(),
                                         ['Value', 'Symmetric error',
                                          'Positive error',
                                          'Negative error'] ):
            self.horizontalHeader().setLabel(num, text)

        # data the table shows
        self.coldata = (None, None, None, None)

        # the edit control, if any
        self.edit = None

    def setDataset(self, ds):
        """Show the given dataset in the widget."""
        self.dataset = ds
        self.coldata = (ds.data, ds.serr, ds.perr, ds.nerr)
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
            painter.setPen( qt.QPen(colgroup.highlightedText()) )
        else:
            painter.setPen( qt.QPen(colgroup.text()) )

        # draw text for cell
        painter.drawText(0, 0, cr.width(), cr.height(),
                         qt.Qt.AlignRight | qt.Qt.AlignVCenter,
                         self.text(r, c))

    def createEditor(self, r, c, initfromcell):
        """Called if the user starts editing a cell."""
        self.edit = qt.QLineEdit(self.viewport())
        v = qt.QDoubleValidator(self.edit)
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
            mb = qt.QMessageBox("Veusz",
                                "This column has no data. Initialise with "
                                "zero and continue?",
                                qt.QMessageBox.Warning,
                                qt.QMessageBox.Yes|qt.QMessageBox.Default,
                                qt.QMessageBox.No,
                                qt.QMessageBox.NoButton,
                                self)
            mb.setButtonText(qt.QMessageBox.Yes, "&Initialise")
            mb.setButtonText(qt.QMessageBox.No, "&Cancel")
            if mb.exec_loop() != qt.QMessageBox.Yes:
                return None

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
            if nums == None:
                nums = N.zeros(self.coldata[0].shape, N.Float64)
            try:
                nums[r] = float( unicode(self.edit.text()) )
            except ValueError:
                # floating point conversion error
                self.edit.setText(self.text(r, c))
                qt.QMessageBox("Veusz",
                               "Invalid number",
                               qt.QMessageBox.Warning,
                               qt.QMessageBox.Ok,
                               qt.QMessageBox.NoButton,
                               qt.QMessageBox.NoButton,
                               self).exec_loop()
                return

            # modify the value in the dataset
            self.dataset.changeValues(['vals', 'serr', 'perr', 'nerr'][c],
                                      nums)

class DataEditDialog(qt.QDialog):
    """Data editting dialog."""

    def __init__(self, parent, document):
        """Initialise dialog."""

        qt.QDialog.__init__(self, parent, 'DataEditDialog', False,
                            qt.Qt.WDestructiveClose)
        self.parent = parent
        self.setCaption('Edit data - Veusz')
        self.document = document
        self.connect(document, qt.PYSIGNAL('sigModified'),
                     self.slotDocumentModified)

        spacing = self.fontMetrics().height() / 2
        self.layout = qt.QVBoxLayout(self, spacing)

        # list of datasets on left of table
        datasplitter = qt.QSplitter(self)
        self.layout.addWidget(datasplitter)

        self.dslistbox = qt.QListBox(datasplitter)
        self.connect( self.dslistbox, qt.SIGNAL('highlighted(const QString&)'),
                      self.slotDatasetHighlighted )

        # initialise table
        vbox = qt.QVBox(datasplitter)
        vbox.setSpacing(spacing)
        self.dstable = _DataEditTable(vbox)
        
        # if dataset is linked, show filename
        w = qt.QWidget(vbox)
        w.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Minimum)
        l = qt.QHBoxLayout(w, 0, spacing)

        self.linklabel = qt.QLabel('', w)
        l.addSpacing(spacing)
        l.addWidget(self.linklabel)
        l.addItem( qt.QSpacerItem(1, 1, qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum) )
        self.linkbutton = qt.QPushButton('Unlink...', w)
        l.addWidget(self.linkbutton)
        self.connect(self.linkbutton, qt.SIGNAL('clicked()'),
                     self.slotDatasetUnlink)

        # operation buttons
        buttons = [ ('&Delete', self.slotDatasetDelete),
                    ('&Rename...', self.slotDatasetRename),
                    ('D&uplicate...', self.slotDatasetDuplicate),
                    (None, None),
                    ('Crea&te...', self.slotDatasetNew),
                    ('&Import...', self.slotDatasetImport),
                    (None, None),
                    ('&Close', self.slotClose) ]

        w = qt.QWidget(self)
        self.layout.addWidget(w)
        w.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        l = qt.QHBoxLayout(w, 0, spacing)
        l.addItem( qt.QSpacerItem(1, 1, qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum) )

        for name, slot in buttons:
            if name == None:
                l.addSpacing(spacing)
            else:
                b = qt.QPushButton(name, w)
                l.addWidget(b)
                self.connect(b, qt.SIGNAL('clicked()'), slot)

        # populate initially
        self.slotDocumentModified()

    def sizeHint(self):
        """Returns recommended size of dialog."""
        return qt.QSize(600, 400)

    def closeEvent(self, evt):
        """Called when the window closes."""

        # store the current geometry in the settings database
        geometry = ( self.x(), self.y(), self.width(), self.height() )
        setting.settingdb['geometry_dataeditdialog'] = geometry

        qt.QDialog.closeEvent(self, evt)

    def showEvent(self, evt):
        """Restoring window geometry if possible."""

        # if we can restore the geometry, do so
        if 'geometry_dataeditdialog' in setting.settingdb:
            geometry =  setting.settingdb['geometry_dataeditdialog']
            self.resize( qt.QSize(geometry[2], geometry[3]) )
            self.move( qt.QPoint(geometry[0], geometry[1]) )

        qt.QDialog.showEvent(self, evt)

    def slotDocumentModified(self):
        '''Called when the dialog needs to be modified.'''

        # update dataset list
        datasets = self.document.data.keys()
        datasets.sort()

        # get current item (to reselect later)
        item = self.dslistbox.selectedItem()
        if item != None:
            name = unicode(item.text())
        else:
            name = None

        self.dslistbox.clear()
        self.dslistbox.insertStrList( datasets )

        # reselect old item
        item = None
        if name != None:
            item = self.dslistbox.findItem(name, qt.Qt.ExactMatch)
            if item != None:
                self.dslistbox.setCurrentItem(item)
        if name == None or item == None:
            # select first item
            if self.dslistbox.numRows() != 0:
                self.dslistbox.setCurrentItem(0)

    def slotClose(self):
        """Close the dialog."""
        self.close()

    def slotDatasetHighlighted(self, name):
        """Dataset highlighted in list box."""

        # convert to python string
        name = unicode(name)

        # update the table
        ds = self.document.data[name]
        self.dstable.setDataset(ds)

        # linked dataset
        if ds.linked == None:
            l = 'None'
        else:
            l = ds.linked.filename
        self.linkbutton.setEnabled( ds.linked != None)
        self.linklabel.setText('Linked file: %s' % l)

    def slotDatasetDelete(self):
        """Delete selected dataset."""

        item = self.dslistbox.selectedItem()
        if item != None:
            name = unicode(item.text())
            self.document.deleteDataset(name)

    def slotDatasetRename(self):
        """Rename selected dataset."""

        item = self.dslistbox.selectedItem()
        if item != None:
            name = unicode( item.text() )
            rn = _DatasetNameDialog("Rename dataset",
                                    "Enter a new name for dataset '%s'" % name,
                                    self.document, name, self)
            if rn.exec_loop() == qt.QDialog.Accepted:
                newname = rn.getName()
                self.document.renameDataset(name, newname)
                    
    def slotDatasetDuplicate(self):
        """Duplicate selected dataset."""
        
        item = self.dslistbox.selectedItem()
        if item != None:
            name = unicode( item.text() )
            dds = _DatasetNameDialog("Duplicate dataset",
                                     "Enter the duplicate's name for "
                                     "dataset '%s'" % name,
                                     self.document, name, self)
            if dds.exec_loop() == qt.QDialog.Accepted:
                newname = dds.getName()
                self.document.duplicateDataset(name, newname)

    def slotDatasetNew(self):
        """Create datasets from scratch."""

        nds = DatasetNewDialog(self.document, self.parent)
        nds.show()
        
    def slotDatasetImport(self):
        """Import data from a file."""

        ids = importdialog.ImportDialog(self.parent, self.document)
        ids.show()
        
    def slotDatasetUnlink(self):
        """Allow user to remove link to file."""

        item = self.dslistbox.selectedItem()
        if item != None:
            # check with the user first, as this is drastic
            name = unicode( item.text() )
            mb = qt.QMessageBox("Veusz",
                                "Unlink dataset '%s'?" % name,
                                qt.QMessageBox.Information,
                                qt.QMessageBox.Ok | qt.QMessageBox.Default,
                                qt.QMessageBox.Cancel,
                                qt.QMessageBox.NoButton,
                                self)
            mb.setButtonText(qt.QMessageBox.Ok, "&Unlink")
            mb.setButtonText(qt.QMessageBox.Cancel, "&Cancel")

            # if they want to carry on
            if mb.exec_loop() == qt.QMessageBox.Ok:
                self.document.unlinkDataset(name)
                # update display
                self.dslistbox.setCurrentItem( self.dslistbox.currentItem() )

