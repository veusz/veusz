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

# $Id: toolsplugin.py 1332 2010-07-17 21:07:17Z jeremysanders $

"""Plugins for creating datasets."""

import numpy as N

# add an instance of your class to this list to be registered
datasetpluginregistry = []

class DatasetPluginException(RuntimeError):
    """Raise this to report an error.
    """
    pass

class Dataset1D(object):
    """1D dataset for ImportPlugin or DatasetPlugin."""
    def __init__(self, name, data=None, serr=None, perr=None, nerr=None):
        """1D dataset
        name: name of dataset
        data: data in dataset: list of floats or numpy 1D array
        serr: (optional) symmetric errors on data: list or numpy array
        perr: (optional) positive errors on data: list or numpy array
        nerr: (optional) negative errors on data: list or numpy array

        If errors are returned for data give serr or nerr and perr.
        nerr should be negative values if used.
        perr should be positive values if used.
        """
        self.name = name
        self.data = data
        self.serr = serr
        self.perr = perr
        self.nerr = nerr

    def _null(self):
        """Empty data contents."""
        self.data = N.array([])
        self.serr = self.perr = self.nerr = None

class Dataset2D(object):
    """2D dataset for ImportPlugin or DatasetPlugin."""
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

    def _null(self):
        """Empty data contents."""
        self.data = N.array([[]])
        self.rangex = self.rangey = (0, 1)

class DatasetText(object):
    """Text dataset for ImportPlugin or DatasetPlugin."""
    def __init__(self, name, data):
        """A text dataset
        name: name of dataset
        data: data in dataset: list of strings
        """
        self.name = name
        self.data = data

    def _null(self):
        """Empty data contents."""
        self.data = []

# 1. plugin takes set of parameters
# 2. plugin returns set of plugindataset objects
# 3. create set of dataset objects which link to the plugin and parameters
# 4. on document change
#    - rerun plugin
#    - regenerate or update dataset objects

# issues
#  - how to synchronise - datasets share same set of parameters
#  - what happens if datasets change?
#     - delete old datasets?
#     - create new ones
#  - problem if evaluation starts from a disappearing dataset - what then?
#  - could have rule: always the same list of datasets
#  - how to save

class DatasetPluginParams(object):
    """Set of parameters to be passed to plugin, and helpers to get
    existing datasets."""
    
    def __init__(self, fields, doc):
        """Construct parameter object to pass to DatasetPlugins."""
        self.fields = fields
        self.doc = doc

    @property
    def datasets1d(self):
        """Return list of existing 1D numeric datasets"""
        return [name for name, ds in self.doc.datasets if
                (ds.dimensions == 1 and ds.datatype == 'numeric')]

    @property
    def datasets2d(self):
        """Return list of existing 2D numeric datasets"""
        return [name for name, ds in self.doc.datasets if
                (ds.dimensions == 2 and ds.datatype == 'numeric')]

    @property
    def datasetstext(self):
        """Return list of existing 1D text datasets"""
        return [name for name, ds in self.doc.datasets if
                (ds.dimensions == 1 and ds.datatype == 'text')]

    def getDataset(self, name):
        """Return dataset object for name given.
        Please don't modify these
        """
        try:
            ds = doc.datasets[name]
        except KeyError:
            raise ValueError, "Unknown dataset '%s'" % name
        if ds.dimensions == 1 and ds.datatype == 'numeric':
            return Dataset1D(name, data=ds.data, serr=ds.serr, 
                             perr=ds.perr, nerr=ds.nerr)
        elif ds.dimensions == 2 and ds.datatype == 'numeric':
            return Dataset2D(name, ds.data,
                             rangex=ds.xrange, rangey=ds.yrange)
        elif ds.dimensions == 1 and ds.datatype == 'text':
            return DatasetText(name, ds.data)
        else:
            raise RuntimeError, "Unknown dataset type found"

class _DatasetPluginSync(object):
    """Synchronise datasets generated by plugin."""

    def __init__(self, plugin, doc, fields):
        self.plugin = plugin
        self.document = doc
        self.fields = fields
        self.changeset = -1
        self.datasetnames = []

    def nullDatasets(self):
        for name in self.datasetnames:
            try:
                self.document.data[name].ds._null()
            except KeyError:
                pass

    def checkUpToDate(self):
        """Check that datasets are up to date."""
        if self.document.changeset != self.changeset:
            self.changeset = self.document.changeset

            params = DatasetPluginParams(self.fields, self.document)
            try:
                datasets = self.plugin.update(params)
            except DatasetPluginException, ex:
                # if there's an error, then log and null out outputs
                self.document.log( unicode(ex) )
                self.nullDatasets()
                return

class DatasetPlugin(object):

    # the plugin will get inserted into the menu in a hierarchy based on
    # the elements of this tuple
    menu = ('Base plugin',)
    name = 'Base plugin'

    author = ''
    description_short = ''
    description_full = ''

    def __init__(self):
        """Override this to declare a list of input fields if required."""
        self.fields = []

    def update(self, parameters):
        """Override this to return a dataset or list of datasets."""
        return []
