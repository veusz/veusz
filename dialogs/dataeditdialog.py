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

import qt
import qttable

class DataEditDialog(qt.QDialog):
    """Data editting dialog."""

    def __init__(self, parent, document):
        """Initialise dialog."""

        qt.QDialog.__init__(self, parent, 'DataEditDialog', False)
        self.setCaption('Edit data')
        self.document = document
        self.connect(document, qt.PYSIGNAL('sigModified'),
                     self.slotDocumentModified)

        spacing = self.fontMetrics().height() / 2

        datahbox = qt.QHBox(self)
        datahbox.setSpacing(spacing)

        self.dslistbox = qt.QListBox(datahbox)
        self.connect( self.dslistbox, qt.SIGNAL('highlighted(const QString&)'),
                      self.slotDatasetHighlighted )

        self.dstable = qttable.QTable(datahbox)

        # initialise table
        self.dstable.setNumCols(4)
        
        # buttons
        bhbox = qt.QHBox(self)
        bhbox.setSpacing(spacing)
        
        closebutton = qt.QPushButton("&Close", bhbox)
        self.connect( closebutton, qt.SIGNAL('clicked()'),
                      self.slotClose )

        self.layout = qt.QVBoxLayout(self, spacing)
        self.layout.addWidget( datahbox )
        self.layout.addWidget( bhbox )

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
        name = str(name)

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
