# about dialog box
# aboutdialog.py

#    Copyright (C) 2006 Jeremy S. Sanders
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

"""About dialog module."""

import os.path

import veusz.qtall as qt4
import veusz.utils as utils
import veusz.windows.action as action

class AboutDialog(qt4.QDialog):
    """About dialog."""

    def __init__(self, *args):
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'about.ui'),
                   self)

        # draw logo in dialog
        self.logolabel.setPixmap( action.getPixmap('logo.png') )

        # add version to copyright text
        copyrighttext = unicode(self.copyrightlabel.text())
        copyrighttext = copyrighttext % {'version': utils.version()}
        self.copyrightlabel.setText(copyrighttext)
        
