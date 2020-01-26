# -*- coding: utf-8 -*-
#    Copyright (C) 2011 Jeremy S. Sanders
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
###############################################################################

from __future__ import division
from .. import qtall as qt
from .. import utils

class LineEditWithClear(qt.QLineEdit):
    """This is a line edit widget which supplies a clear button
    to delete the text if it is clicked.

    Adapted from:
    http://labs.qt.nokia.com/2007/06/06/lineedit-with-a-clear-button/
    """

    def __init__(self, *args):
        """Initialise the line edit."""
        qt.QLineEdit.__init__(self, *args)

        # the clear button itself, with no padding
        self.clearbutton = cb = qt.QToolButton(self)
        cb.setIcon( utils.getIcon('kde-edit-delete') )
        cb.setCursor(qt.Qt.ArrowCursor)
        cb.setStyleSheet('QToolButton { border: none; padding: 0px; }')
        cb.setToolTip("Clear text")
        cb.hide()

        cb.clicked.connect(self.clear)

        # button should appear if there is text
        self.textChanged.connect(self.updateCloseButton)

        # positioning of the button
        fw = self.style().pixelMetric(qt.QStyle.PM_DefaultFrameWidth)
        self.setStyleSheet("QLineEdit { padding-right: %ipx; } " %
                           (cb.sizeHint().width() + fw + 1))
        msz = self.minimumSizeHint()
        mx =  cb.sizeHint().height()+ fw*2 + 2
        self.setMinimumSize( max(msz.width(), mx), max(msz.height(), mx) )

    def resizeEvent(self, evt):
        """Move button if widget resized."""
        sz = self.clearbutton.sizeHint()
        fw = self.style().pixelMetric(qt.QStyle.PM_DefaultFrameWidth)
        r = self.rect()
        self.clearbutton.move( r.right() - fw - sz.width(),
                               (r.bottom() + 1 - sz.height())//2 )

    def updateCloseButton(self, text):
        """Button should only appear if there is text."""
        self.clearbutton.setVisible(text != '')
