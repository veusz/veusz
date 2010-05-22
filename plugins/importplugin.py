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

"""Import plugin base class and helpers."""

import veusz.utils as utils

# add an instance of your class to this list to get it registered
importpluginregistry = []

class ImportPluginParams(object):
    """Parameters to plugin are passed in this object."""
    def __init__(self, filename, encoding, field_results):
        self.filename = filename
        self.encoding = encoding
        self.field_results = field_results

    def openFileWithEncoding(self):
        """Helper to open filename but respecting encoding."""
        return utils.openEncoding(self.filename, self.encoding)

class ImportPluginException(RuntimeError):
    """An exception to return errors about importing or previewing data."""

class ImportField(object):
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

class ImportFieldCheck(ImportField):
    """A check box on the dialog."""

class ImportFieldText(ImportField):
    """Text entry on the dialog."""

class ImportFieldCombo(ImportField):
    """Drop-down combobox on dialog."""
    def __init__(self, name, descr=None, default=None, items=(),
                 editable=True):
        """name: name of field
        descr: description to show to user
        default: default value
        items: items in drop-down box
        editable: whether user can enter their own value."""
        ImportField.__init__(self, name, descr=descr, default=default)
        self.items = items
        self.editable = editable

class ImportPlugin(object):
    """Define a plugin to read data in a particular format."""

    name = 'Import plugin'
    author = ''
    description = ''

    # a list of ImportField objects to display
    fields = []

    def getPreview(self, params):
        """Get data to show in a text box to show a preview.
        params is a ImportPluginParams object.
        Returns (text, okaytoimport)
        """
        return '', False

    def doImport(self, params):
        """Actually import data
        params is a ImportPluginParams object."""
        pass

#################################################################

class ImportPluginExample(ImportPlugin):
    name = 'Example plugin'
    author = 'Jeremy Sanders'
    description = 'Reads a list of numbers in a text file'

    def getPreview(self, params):
        f = params.openFileWithEncoding()
        return f.read(4096), True

importpluginregistry.append(ImportPluginExample())
