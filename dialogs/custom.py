#    Copyright (C) 2009 Jeremy S. Sanders
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

import os.path

import veusz.qtall as qt4
import veusz.utils as utils

class CustomDialog(qt4.QDialog):
    """Class to load help for standard veusz import."""
    def __init__(self, parent, document):
        qt4.QDialog.__init__(self, parent, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'custom.ui'),
                   self)
        self.document = document

        # connect different types of radio
        self.connect(self.functionsRadio, qt4.SIGNAL('clicked()'),
                     self.slotFunctionsClicked)
        self.connect(self.constantsRadio, qt4.SIGNAL('clicked()'),
                     self.slotConstantsClicked)
        # setup radio buttons
        self.functionsRadio.click()

        # connect buttons to slots
        self.connect(self.addButton, qt4.SIGNAL('clicked()'), self.slotAdd)
        self.connect(self.removeButton, qt4.SIGNAL('clicked()'),
                     self.slotRemove)
        self.connect(self.saveButton, qt4.SIGNAL('clicked()'), self.slotSave)
        self.connect(self.loadButton, qt4.SIGNAL('clicked()'), self.slotLoad)

    def slotFunctionsClicked(self):
        """Functions definitions radio clicked."""

        print "functions"

    def slotConstantsClicked(self):
        """Constant definitions radio clicked."""
        print "constants"

    def slotAdd(self):
        """Add an entry."""
        pass
    
    def slotRemove(self):
        """Remove an entry."""
        pass

    def slotSave(self):
        """Save entries."""
        pass

    def slotLoad(self):
        """Load entries."""
        pass
