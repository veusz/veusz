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
from .. import qtall as qt4
from ..qtwidgets.datasetbrowser import DatasetBrowser

def _(text, disambiguation=None, context="DataNavigator"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class DataNavigatorWindow(qt4.QDockWidget):
    """A dock window containing a dataset browsing widget."""

    def __init__(self, thedocument, mainwin, *args):
        qt4.QDockWidget.__init__(self, *args)
        self.setWindowTitle(_("Data - Veusz"))
        self.setObjectName("veuszdatawindow")

        self.nav = DatasetBrowser(thedocument, mainwin, self)
        self.setWidget(self.nav)
