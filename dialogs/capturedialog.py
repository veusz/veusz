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

"""Veusz data capture dialog."""

import os.path

import veusz.qtall as qt4
import veusz.utils as utils

class CaptureDialog(qt4.QDialog):
    """Capture dialog."""

    def __init__(self, document, *args):
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'capture.ui'),
                   self)

        self.document = document

        # validate edit controls
        validator = qt4.QIntValidator(1, 65535, self)
        self.portEdit.setValidator(validator)
        validator = qt4.QIntValidator(1, 1000000000, self)
        self.numPtsStopEdit.setValidator(validator)
        self.timeStopEdit.setValidator(validator)

        # get notification of change of capture method
        self.methodBG = qt4.QButtonGroup(self)
        self.methodBG.addButton( self.captureFileButton, 0 )
        self.methodBG.addButton( self.captureInternetButton, 1 )
        self.methodBG.addButton( self.captureProgramButton, 2 )
        self.connect(self.methodBG, qt4.SIGNAL('buttonClicked(int)'),
                     self.slotMethodChanged)
        self.slotMethodChanged(0)

        # get notification of change of stop method
        self.stopBG = qt4.QButtonGroup(self)
        self.stopBG.addButton( self.clickingStopButton, 0 )
        self.stopBG.addButton( self.numPtsStopButton, 1 )
        self.stopBG.addButton( self.timeStopButton, 2 )
        self.connect(self.stopBG, qt4.SIGNAL('buttonClicked(int)'),
                     self.slotStopChanged)
        self.slotStopChanged(0)

    def slotMethodChanged(self, buttonid):
        """Enable/disable correct controls in methodBG."""
        # enable correct buttons
        fc = buttonid==0
        self.filenameEdit.setEnabled(fc)
        self.filenameButton.setEnabled(fc)

        ic = buttonid==1
        self.hostEdit.setEnabled(ic)
        self.portEdit.setEnabled(ic)

        xc = buttonid==2
        self.commandLineEdit.setEnabled(xc)

    def slotStopChanged(self, buttonid):
        """Enable/disable correct controls in stopBG."""

        ns = buttonid == 1
        self.numPtsStopEdit.setEnabled(ns)

        ts = buttonid == 2
        self.timeStopEdit.setEnabled(ts)
