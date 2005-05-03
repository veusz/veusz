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
        tab.setReadOnly(True)
        tab.setNumCols(4)
        for num, text in zip( range(4),
                              ['Value', 'Symmetric error', 'Positive error',
                               'Negative error'] ):
            tab.horizontalHeader().setLabel(num, text)

        # operation buttons
        opbox = qt.QHBox(self)
        opbox.setSpacing(spacing)
        self.layout.addWidget( opbox )

        for name, slot in [ ('&Delete', self.slotDatasetDelete),
                            ('&Rename...', self.slotDatasetRename),
                            ('D&uplicate...', self.slotDatasetDuplicate),
                            ('&New...', self.slotDatasetNew) ]:
            b = qt.QPushButton(name, opbox)
            self.connect(b, qt.SIGNAL('pressed()'), slot)
            
        # buttons
        bhbox = qt.QHBox(self)
        self.layout.addWidget( bhbox )
        bhbox.setSpacing(spacing)
        
        closebutton = qt.QPushButton("&Close", bhbox)
        self.connect( closebutton, qt.SIGNAL('clicked()'),
                      self.slotClose )

        # populate initially
        self.slotDocumentModified()

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
        """Delete button pressed."""

        item = self.dslistbox.selectedItem()
        if item != None:
            name = unicode(item.text())
            self.document.deleteDataset(name)

    def _checkDatasetName(self, name):
        """Is the name given valid?"""

        return not re.search('[ ,+-]', name)

    def slotDatasetRename(self):
        """Rename selected dataset."""

        item = self.dslistbox.selectedItem()
        if item != None:
            name = unicode(item.text())
            newname, okay = qt.QInputDialog.getText(
                'Rename dataset',
                'Enter a new name for the dataset "%s"' % name)

            newname = unicode(newname).strip()
            if okay and newname:
                if self._checkDatasetName(newname):
                    self.document.renameDataset(name, newname)

    def slotDatasetDuplicate(self):
        pass

    def slotDatasetNew(self):
        pass

