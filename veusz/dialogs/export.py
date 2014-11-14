#    Copyright (C) 2014 Jeremy S. Sanders
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

from __future__ import division
from .. import qtall as qt4
from .. import setting
from .. import utils
from .veuszdialog import VeuszDialog

class ExportDialog(VeuszDialog):
    """Export dialog."""

    def __init__(self, mainwindow):
        """Setup dialog."""
        VeuszDialog.__init__(self, mainwindow, 'export.ui', modal=True)

    def accept(self):
        qt4.QDialog.accept(self)
