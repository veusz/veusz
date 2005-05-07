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

import qt
import qttable

import setting

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
        self.connect(okbutton, qt.SIGNAL('pressed()'),
                     self.slotOK)
        buttonlayout.addWidget(okbutton)

        cancelbutton = qt.QPushButton("&Cancel", self)
        self.connect(cancelbutton, qt.SIGNAL('pressed()'),
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

class DataEditDialog(qt.QDialog):
    """Data editting dialog."""

    def __init__(self, parent, document):
        """Initialise dialog."""

        qt.QDialog.__init__(self, parent, 'DataEditDialog', False)
        self.setCaption('Edit data - Veusz')
        self.document = document
        self.connect(document, qt.PYSIGNAL('sigModified'),
                     self.slotDocumentModified)

        spacing = self.fontMetrics().height() / 2
        self.layout = qt.QVBoxLayout(self, spacing)

        # list of datasets on left of table
        datasplitter = qt.QSplitter(self)
        self.layout.addWidget( datasplitter )

        self.dslistbox = qt.QListBox(datasplitter)
        self.connect( self.dslistbox, qt.SIGNAL('highlighted(const QString&)'),
                      self.slotDatasetHighlighted )

        # initialise table
        tab = self.dstable = qttable.QTable(datasplitter)
        tab.setSizePolicy(qt.QSizePolicy.Expanding,  qt.QSizePolicy.Expanding)
        tab.setReadOnly(True)
        tab.setNumCols(4)
        for num, text in zip( range(4),
                              ['Value', 'Symmetric error', 'Positive error',
                               'Negative error'] ):
            tab.horizontalHeader().setLabel(num, text)

        # operation buttons
        opbox = qt.QHBox(self)
        opbox.setSpacing(spacing)
        opbox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        self.layout.addWidget(opbox)

        for name, slot in [ ('&Delete', self.slotDatasetDelete),
                            ('&Rename...', self.slotDatasetRename),
                            ('D&uplicate...', self.slotDatasetDuplicate),
                            ('&New...', self.slotDatasetNew) ]:
            b = qt.QPushButton(name, opbox)
            self.connect(b, qt.SIGNAL('pressed()'), slot)
            
        # buttons
        bhbox = qt.QHBox(self)
        bhbox.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        self.layout.addWidget(bhbox)
        bhbox.setSpacing(spacing)
        
        closebutton = qt.QPushButton("&Close", bhbox)
        self.connect( closebutton, qt.SIGNAL('pressed()'),
                      self.slotClose )

        # populate initially
        self.slotDocumentModified()

    def sizeHint(self):
        """Returns recommended size of dialog."""
        return qt.QSize(600, 400)

    def closeEvent(self, evt):
        """Called when the window closes."""

        # store the current geometry in the settings database
        geometry = ( self.x(), self.y(), self.width(), self.height() )
        setting.settingdb.database['geometry_dataeditdialog'] = geometry

        qt.QDialog.closeEvent(self, evt)

    def showEvent(self, evt):
        """Restoring window geometry if possible."""

        # if we can restore the geometry, do so
        if 'geometry_dataeditdialog' in setting.settingdb.database:
            geometry =  setting.settingdb.database['geometry_dataeditdialog']
            self.resize( qt.QSize(geometry[2], geometry[3]) )
            self.move( qt.QPoint(geometry[0], geometry[1]) )

        qt.QDialog.showEvent(self, evt)

    def slotDocumentModified(self):
        '''Called when the dialog needs to be modified.'''

        # update dataset list
        datasets = self.document.data.keys()
        datasets.sort()
        self.dslistbox.clear()
        self.dslistbox.insertStrList( datasets )

    def slotClose(self):
        """Close the dialog."""

        self.close(True)

    def slotDatasetHighlighted(self, name):
        """Dataset highlighted in list box."""

        # convert to python string
        name = unicode(name)

        # update the table
        ds = self.document.data[name]
        norows = len(ds.data)
        t = self.dstable
        t.setUpdatesEnabled(False)
        t.setNumRows(norows)

        datasets = (ds.data, ds.serr, ds.perr, ds.nerr)
        for array, col in zip(datasets, range(len(datasets))):
            if array == None:
                for i in range(norows):
                    t.setText(i, col, '')
            else:
                for i, v in zip(range(norows), array):
                    t.setText(i, col, str(v))

        t.setUpdatesEnabled(True)

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
        pass

