#    Copyright (C) 2007 Jeremy S. Sanders
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

"""Dataset creation dialog for 2d data."""

import os.path
import veusz.qtall as qt4
import veusz.utils as utils
import veusz.document as document

class DataCreate2DDialog(qt4.QDialog):

    def __init__(self, parent, document, *args):
        """Initialise dialog with document."""

        qt4.QDialog.__init__(self, parent, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'datacreate2d.ui'),
                   self)
        self.document = document

        self.connect( self.createbutton, qt4.SIGNAL('clicked()'),
                      self.createButtonClickedSlot )

    def createButtonClickedSlot(self):
        """Create button pressed."""
        
        pass
