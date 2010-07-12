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

from field import Field as ImportField
from field import FieldBool as ImportFieldCheck
from field import FieldText as ImportFieldText
from field import FieldFloat as ImportFieldFloat
from field import FieldInt as ImportFieldInt
from field import FieldCombo as ImportFieldCombo

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

class ImportDataset1D(object):
    """Return 1D dataset."""
    def __init__(self, name, data=None, serr=None, perr=None, nerr=None):
        """1D dataset
        name: name of dataset
        data: data in dataset: list of floats or numpy 1D array
        serr: (optional) symmetric errors on data: list or numpy array
        perr: (optional) positive errors on data: list or numpy array
        nerr: (optional) negative errors on data: list or numpy array

        If errors are returned for data implement serr or nerr and perr.
        nerr should be negative values if used.
        perr should be positive values if used.
        """
        self.name = name
        self.data = data
        self.serr = serr
        self.perr = perr
        self.nerr = nerr

class ImportDataset2D(object):
    """Return 2D dataset."""
    def __init__(self, name, data, rangex=None, rangey=None):
        """2D dataset.
        name: name of dataset
        data: 2D numpy array of values or list of lists of floats
        rangex: optional tuple with X range of data (min, max)
        rangey: optional tuple with Y range of data (min, max)
        """
        self.name = name
        self.data = data
        self.rangex = rangex
        self.rangey = rangey

class ImportDatasetText(object):
    """Return a text dataset."""
    def __init__(self, name, data):
        """A text dataset
        name: name of dataset
        data: data in dataset: list of strings
        """
        self.name = name
        self.data = data

class ImportPlugin(object):
    """Define a plugin to read data in a particular format.
    
    override doImport and optionally getPreview to define a new plugin
    register the class by adding to the importpluginregistry list
    """

    name = 'Import plugin'
    author = ''
    description = ''

    def __init__(self):
        """Override this to declare a list of input fields if required."""
        # a list of ImportField objects to display
        self.fields = []

    def getPreview(self, params):
        """Get data to show in a text box to show a preview.
        params is a ImportPluginParams object.
        Returns (text, okaytoimport)
        """
        f = params.openFileWithEncoding()
        return f.read(4096), True

    def doImport(self, params):
        """Actually import data
        params is a ImportPluginParams object.
        Return a list of ImportDataset1D, ImportDataset2D objects
        """
        return []

#################################################################

class ImportPluginExample(ImportPlugin):
    """An example plugin for reading a set of unformatted numbers
    from a file."""

    name = "Example plugin"
    author = "Jeremy Sanders"
    description = "Reads a list of numbers in a text file"

    def __init__(self):
        self.fields = [
            ImportFieldText("name", descr="Dataset name", default="name"),
            ImportFieldCheck("invert", descr="invert values"),
            ImportFieldFloat("mult", descr="Multiplication factor", default=1),
            ImportFieldInt("skip", descr="Skip N lines",
                           default=0, minval=0),
            ImportFieldCombo("subtract", items=("0", "1", "2"),
                             editable=False, default="0")
            ]

    def doImport(self, params):
        """Actually import data
        params is a ImportPluginParams object.
        Return a list of ImportDataset1D, ImportDataset2D objects
        """
        f = params.openFileWithEncoding()
        data = []
        mult = params.field_results["mult"]
        sub = float(params.field_results["subtract"])
        if params.field_results["invert"]:
            mult *= -1
        for i in xrange(params.field_results["skip"]):
            f.readline()
        for line in f:
            data += [float(x)*mult-sub for x in line.split()]

        return [ImportDataset1D(params.field_results["name"], data)]

importpluginregistry.append(ImportPluginExample())
