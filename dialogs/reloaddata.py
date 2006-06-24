# data reload dialog

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

"""Dialog for reloading linked data."""

import veusz.qtall as qt

import veusz.document as document

class ReloadData(qt.QDialog):
    """Dialog for reloading linked datasets."""

    def __init__(self, parent, document):
        """Initialise the dialog."""

        qt.QDialog.__init__(self, parent, 'DataReloadDialog', False)
        self.setCaption('Reload linked data - Veusz')

        self.document = document

        spacing = self.fontMetrics().height() / 2

        # layout for dialog
        self.layout = qt.QVBoxLayout(self, spacing)

        self.outputedit = qt.QTextEdit(self)
        self.layout.addWidget( self.outputedit )
        self.outputedit.setTextFormat(qt.Qt.PlainText)
        self.outputedit.setReadOnly(True)
        self.outputedit.setWordWrap(qt.QTextEdit.NoWrap)
        f = qt.QFont('Fixed')
        self.outputedit.setFont(f)

        # buttons
        w = qt.QWidget(self)
        self.layout.addWidget(w)
        w.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        l = qt.QHBoxLayout(w, 0, spacing)
        l.addItem( qt.QSpacerItem(1, 1, qt.QSizePolicy.Expanding,
                                  qt.QSizePolicy.Minimum) )
        
        b = qt.QPushButton("&Close", w)
        b.setDefault(True)
        l.addWidget(b)
        self.connect(b, qt.SIGNAL('clicked()'), self.slotClose)

        # actually reload the data (and show the user)
        self.reloadData()

    def sizeHint(self):
        """Returns recommended size of dialog."""
        return qt.QSize(600, 400)

    def slotClose(self):
        """Close the dialog."""

        self.close(True)

    def reloadData(self):
        """Reload linked data. Show the user what was done."""

        text = ''
        try:
            # try to reload the datasets
            datasets, errors = self.document.reloadLinkedDatasets()

            # show errors in read data
            for var, count in errors.items():
                if count != 0:
                    text += ( '%i conversions failed for dataset "%s"\n' %
                              (count, var) )

            # show successes
            for var in datasets:
                text += 'Imported dataset "%s"\n' % var

        except IOError, e:
            text = 'Error reading file:\n' + unicode(e)
        except document.DescriptorError:
            text = 'Could not interpret descriptor. Reload failed.'

        if text == '':
            text = 'Nothing to do. No linked datasets.'

        self.outputedit.setText(text)
        
