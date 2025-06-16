#    Copyright (C) 2011 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This file is part of Veusz.
#
#    Veusz is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    Veusz is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################

from .. import qtall as qt
from ..qtwidgets.datasetbrowser import DatasetBrowser

def _(text, disambiguation=None, context="DataNavigator"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class DataNavigatorWindow(qt.QDockWidget):
    """A dock window containing a dataset browsing widget."""

    def __init__(self, thedocument, mainwin, *args):
        qt.QDockWidget.__init__(self, *args)
        self.setWindowTitle(_("Data - Veusz"))
        self.setObjectName("veuszdatawindow")

        self.nav = DatasetBrowser(thedocument, mainwin, self)
        self.setWidget(self.nav)
