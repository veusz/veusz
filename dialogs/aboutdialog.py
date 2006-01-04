# about dialog box
# aboutdialog.py

#    Copyright (C) 2004 Jeremy S. Sanders
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

"""Module defines about dialog for Veusz."""

import os.path

import qt
import utils

_abouttext=u"""Veusz %s   http://home.gna.org/veusz/
Veusz is Copyright \u00a9 2003-2006 Jeremy Sanders <jeremy@jeremysanders.net>

Veusz comes with ABSOLUTELY NO WARRANTY. Veusz is Free Software
and you are entitled to distribute it under the conditions of the GPL.
See the file COPYING for details.""" % \
utils.version()

_logolocation='%s/../images/logo.png' % os.path.dirname(__file__)

class AboutDialog(qt.QDialog):
    """About dialog class for Veusz."""

    def __init__(self, parent):
        """Initialise dialog."""

        qt.QDialog.__init__(self, parent, 'AboutDialog', True)
        self.setCaption( 'About Veusz' )

        # label with logo (white background)
        self.logo = qt.QPixmap( _logolocation )
        self.logolabel = qt.QLabel( self )
        self.logolabel.setPixmap( self.logo )
        self.logolabel.setAlignment( qt.Qt.AlignHCenter | qt.Qt.AlignVCenter )
        self.logolabel.setFrameStyle( qt.QFrame.Panel | qt.QFrame.Raised )
        self.logolabel.setBackgroundMode( qt.Qt.PaletteBase )

        # about text (see above)
        self.text = qt.QLabel( _abouttext, self )
        self.text.setAlignment( qt.Qt.AlignHCenter | qt.Qt.AlignVCenter )

        # vertical layout
        h = self.fontMetrics().height()
        self.layout = qt.QVBoxLayout( self, h )
        self.layout.addWidget( self.logolabel )
        self.layout.addWidget( self.text )

        # put OK button to right of dialog (is this correct)
        # OK button shouldn't expand to fill the space available
        self.buttonbox = qt.QHBoxLayout( self.layout )
        self.spacer = qt.QSpacerItem(0, 0, qt.QSizePolicy.Expanding,
                                     qt.QSizePolicy.Minimum)
        self.buttonbox.addItem( self.spacer )
        self.okbutton = qt.QPushButton( "OK", self )
        self.buttonbox.addWidget( self.okbutton )

        # close dialog on ok being pressed
        qt.QObject.connect( self.okbutton, qt.SIGNAL('clicked()'),
                            self, qt.SLOT( 'accept()' ) )
