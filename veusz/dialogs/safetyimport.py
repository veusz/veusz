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

"""Ask user whether to import symbols."""

from __future__ import division
from .. import qtall as qt
from .. import setting

def _(text, disambiguation=None, context="SafetyImportDialog"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class SafetyImportDialog(qt.QMessageBox):

    def __init__(self, parent, module, names):
        """Initialise dialog.

        parent is parent widget
        module is module to import symbols from
        names is a list of names to import."""

        qt.QMessageBox.__init__(self, parent)

        self.names = names
        self.module = module

        self.setIcon(qt.QMessageBox.Warning)
        self.setWindowTitle(_("Allow Python import?"))
        self.setText(_("The document has requested that the symbol(s):\n"
                       " %s\nbe loaded from Python module '%s'.\n\n"
                       "This could be unsafe if the document comes from "
                       "an untrusted source.") % (
                ', '.join(names), module))
        self.allow = self.addButton(_("Allow"), qt.QMessageBox.YesRole)
        self.allow.setToolTip(_("Allow use of symbol in module during session"))
        self.allowalways = self.addButton(
            _("Allow always"), qt.QMessageBox.YesRole)
        self.allowalways.setToolTip(_("Always allow use of symbol in module"))
        self.notallow = self.addButton(
            _("Do not allow"), qt.QMessageBox.NoRole)
        self.notallow.setToolTip(_("Do allow use of symbol in module in session"))

    def exec_(self):
        """Execute dialog."""

        # when loading the document the busy cursor is on, this gets
        # rid of it for a while
        qt.qApp.setOverrideCursor(qt.QCursor(qt.Qt.ArrowCursor))
        qt.QMessageBox.exec_(self)
        qt.qApp.restoreOverrideCursor()

        b = self.clickedButton()

        # update lists of variables in settings depending on chosen button
        if b is self.allow:
            a = setting.transient_settings['import_allowed'][self.module]
            a |= set(self.names)
        elif b is self.allowalways:
            a = setting.settingdb['import_allowed'][self.module]
            a.update( [(x, True) for x in self.names] )
        elif b is self.notallow:
            a = setting.transient_settings['import_notallowed'][self.module]
            a |= set(self.names)
