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

"""Plugins for creating datasets."""

import numpy as N
from itertools import izip
import field

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

    def _makeRealDataset(self, manager):
        """Make a Veusz dataset from the plugin dataset."""
        # need to do the import here as otherwise we get a loop
        import veusz.document as document
        return document.Dataset1DPlugin(manager, self)

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

    def _makeRealDataset(self, manager):
        """Make a Veusz dataset from the plugin dataset."""
        import veusz.document as document
        return document.Dataset2DPlugin(manager, self)

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

    def _makeRealDataset(self, manager):
        """Make a Veusz dataset from the plugin dataset."""
        import veusz.document as document
        return document.DatasetTextPlugin(manager, self)

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
#     - rename datasets
#  - problem if evaluation starts from a disappearing dataset - what then?
#  - could have rule: always the same list of datasets
#  - how to save

class DatasetPluginParams(object):
    """Set of parameters to be passed to plugin, and helpers to get
    existing datasets."""
    
    def __init__(self, fields, _doc):
        """Construct parameter object to pass to DatasetPlugins."""
        self.fields = fields
        self._doc = _doc

    @property
    def datasets1d(self):
        """Return list of existing 1D numeric datasets"""
        return [name for name, ds in self._doc.datasets if
                (ds.dimensions == 1 and ds.datatype == 'numeric')]

    @property
    def datasets2d(self):
        """Return list of existing 2D numeric datasets"""
        return [name for name, ds in self._doc.datasets if
                (ds.dimensions == 2 and ds.datatype == 'numeric')]

    @property
    def datasetstext(self):
        """Return list of existing 1D text datasets"""
        return [name for name, ds in self._doc.datasets if
                (ds.dimensions == 1 and ds.datatype == 'text')]

    def getDataset(self, name):
        """Return dataset object for name given.
        Please don't modify these
        """
        try:
            ds = self._doc.data[name]
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

class DatasetPluginManager(object):
    """Manage datasets generated by plugin."""

    def __init__(self, plugin, doc, fields):
        """Construct manager object.

        plugin - instance of plugin class
        doc - document instance
        fields - fields to pass to plugin
        """
        
        self.plugin = plugin
        self.document = doc
        self.params = DatasetPluginParams(fields, doc)

    def initialConstruct(self):
        """Do initial construction of datasets."""

        self.datasetnames = []
        self.datasets = []
        for ds in self.plugin.update(self.params):
            self.datasetnames.append(ds.name)
            realds = ds._makeRealDataset(self)
            realds.document = self.document
            self.datasets.append(realds)

        self.changeset = self.document.changeset        

    def nullDatasets(self):
        """Clear out contents of datasets."""
        for ds in self.datasets:
            ds.pluginds._null()

    def saveToFile(self, fileobj):
        """Save command to load in plugin and parameters."""

        args = [ repr(self.plugin.name), repr(self.params.fields) ]

        # look for renamed or deleted datasets
        names = {}
        for ds, dsname in izip( self.datasets, self.datasetnames ):
            try:
                currentname = self.document.datasetName(ds)
            except ValueError:
                # deleted
                currentname = None

            if currentname != dsname:
                names[dsname] = currentname

        if names:
            args.append( "datasetnames="+repr(names) )

        fileobj.write( 'DatasetPlugin(%s)\n' % (', '.join(args)) )

    def update(self):
        """Update created datasets."""

        if self.document.changeset == self.changeset:
            return
        self.changeset = self.document.changeset

        # run the plugin with its parameters
        try:
            datasets = self.plugin.update(self.params)
        except DatasetPluginException, ex:
            # if there's an error, then log and null out outputs
            self.document.log( unicode(ex) )
            self.nullDatasets()
            return

        if len(datasets) != len(self.datasets):
            self.document.log(
                "Dataset plugin error (%s) -" % self.plugin.name +
                " number of created datasets changed" )
            self.nullDatasets()
            return

        # update datasets
        for dataset, pluginds in izip(self.datasets, datasets):
            dataset.pluginds = pluginds

class DatasetPlugin(object):
    """Base class for defining dataset plugins."""

    # the plugin will get inserted into the menu in a hierarchy based on
    # the elements of this tuple
    menu = ('Base plugin',)
    name = 'Base plugin'

    author = ''
    description_short = ''
    description_full = ''

    # if the plugin takes no parameters, set this to False
    has_parameters = True

    def __init__(self):
        """Override this to declare a list of input fields if required."""
        self.fields = []

    def update(self, params):
        """Override this to return a list of datasets.

        params is a DatasetPluginParams object, containing a field
        attribute, and routines to get existing datasets in the document
        """
        return []

class DatasetDoublePlugin(DatasetPlugin):
    menu = ('Double',)
    name = 'Double plugin'
    author = 'Jeremy Sanders'
    description_short = 'Doubles a dataset'
    description_full = 'Doubles a dataset well'

    def __init__(self):
        """Override this to declare a list of input fields if required."""
        self.fields = [
            field.FieldDataset('dsin', 'Input dataset'),
            field.FieldDataset('dsout', 'Output dataset name'),
            ]

    def update(self, params):
        """Override this to return a list of datasets.

        params is a DatasetPluginParams object, containing a field
        attribute, and routines to get existing datasets in the document
        """

        try:
            dsin = params.getDataset(params.fields['dsin'])
        except ValueError:
            raise DatasetPluginException('Could not find dataset')

        return [ Dataset1D(params.fields['dsout'], data=dsin.data*2),
                 Dataset1D('foo', data=dsin.data*4),
                 ]

datasetpluginregistry.append(DatasetDoublePlugin)
