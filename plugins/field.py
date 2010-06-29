#    Copyright (C) 2010 Jeremy S. Sanders
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

"""Data entry fields for plugins."""

import veusz.qtall as qt4

class Field(object):
    """A class to represent an input field on the dialog or command line."""
    def __init__(self, name, descr=None, default=None):
        """name: name of field
        descr: description to show to user
        default: default value."""
        self.name = name
        if descr:
            self.descr = descr
        else:
            self.descr = name
        self.default = default

    def makeControl(self):
        """Create a set of controls for field."""
        return None

    def getControlResults(self, cntrls):
        """Get result from created contrls."""
        return None

class FieldCheck(Field):
    """A check box on the dialog."""

    def makeControl(self):
        l = qt4.QLabel(self.descr)
        c = qt4.QCheckBox()
        if self.default:
            c.setChecked(True)
        return (l, c)

    def getControlResults(self, cntrls):
        return cntrls[1].isChecked()

class FieldText(Field):
    """Text entry on the dialog."""

    def makeControl(self):
        l = qt4.QLabel(self.descr)
        e = qt4.QLineEdit()
        if self.default:
            e.setText(self.default)
        return (l, e)

    def getControlResults(self, cntrls):
        return unicode( cntrls[1].text() )

class FieldFloat(Field):
    """Enter a floating point number."""

    def makeControl(self):
        l = qt4.QLabel(self.descr)
        e = qt4.QLineEdit()
        e.setValidator( qt4.QDoubleValidator(e) )
        if self.default is not None:
            e.setText( str(self.default) )
        return (l, e)

    def getControlResults(self, cntrls):
        try:
            return float( cntrls[1].text() )
        except:
            return None

class FieldInt(Field):
    """Enter an integer number."""

    def makeControl(self):
        l = qt4.QLabel(self.descr)
        e = qt4.QSpinBox()
        if self.default is not None:
            e.setValue( self.default )
        return (l, e)

    def getControlResults(self, cntrls):
        try:
            return cntrls[1].value()
        except:
            return None

class FieldCombo(Field):
    """Drop-down combobox on dialog."""
    def __init__(self, name, descr=None, default=None, items=(),
                 editable=True):
        """name: name of field
        descr: description to show to user
        default: default value
        items: items in drop-down box
        editable: whether user can enter their own value."""
        Field.__init__(self, name, descr=descr, default=default)
        self.items = items
        self.editable = editable

    def makeControl(self):
        l = qt4.QLabel(self.descr)
        c = qt4.QComboBox()
        c.addItems(self.items)
        c.setEditable(bool(self.editable))

        if self.default:
            if self.editable:
                c.setEditText(self.default)
            else:
                c.setCurrentIndex(c.findText(self.default))

        return (l, c)

    def getControlResults(self, cntrls):
        return unicode( cntrls[1].currentText() )
