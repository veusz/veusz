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

# these classes are returned from dataset plugins
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

# class to pass to plugin to give parameters
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
        Make sure that dataset data are not modified.

        if name is not found, raise a DatasetPluginException
        """
        try:
            ds = self._doc.data[name]
        except KeyError:
            raise DatasetPluginException("Unknown dataset '%s'" % name)
        if ds.dimensions == 1 and ds.datatype == 'numeric':
            return Dataset1D(name, data=ds.data, serr=ds.serr, 
                             perr=ds.perr, nerr=ds.nerr)
        elif ds.dimensions == 2 and ds.datatype == 'numeric':
            return Dataset2D(name, ds.data,
                             rangex=ds.xrange, rangey=ds.yrange)
        elif ds.dimensions == 1 and ds.datatype == 'text':
            return DatasetText(name, ds.data)
        else:
            raise RuntimeError("Failed to work out dataset type")

# internal object to synchronise datasets created by a plugin
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

class ScaleDatasetPlugin(DatasetPlugin):
    """Dataset plugin to scale a dataset."""

    menu = ('Scale',)
    name = 'Scale'
    description_short = 'Scale a dataset'
    description_full = ('Scale a dataset by a factor. '
                        'Error bars are also scaled.')
    
    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', 'Input dataset'),
            field.FieldFloat('factor', 'Scale factor', default=1.),
            field.FieldDataset('ds_out', 'Output dataset name'),
            ]

    def update(self, params):
        """Do scaling of dataset."""

        ds_in = params.getDataset(params.fields['ds_in'])
        f = params.fields['factor']

        data, serr, perr, nerr = ds_in.data, ds_in.serr, ds_in.perr, ds_in.nerr
        data = data * f
        if serr is not None: serr = serr * f
        if perr is not None: perr = perr * f
        if nerr is not None: nerr = nerr * f

        if params.fields['ds_out'] == '':
            raise DatasetPluginException('Invalid output dataset name')

        return [ Dataset1D(params.fields['ds_out'],
                           data=data, serr=serr, perr=perr, nerr=nerr) ]

class ShiftDatasetPlugin(DatasetPlugin):
    """Dataset plugin to shift a dataset."""

    menu = ('Shift',)
    name = 'Shift'
    description_short = 'Shift a dataset'
    description_full = ('Shift a dataset by adding a value. '
                        'Error bars remain the same.')
    
    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', 'Input dataset'),
            field.FieldFloat('value', 'Shift value', default=0.),
            field.FieldDataset('ds_out', 'Output dataset name'),
            ]

    def update(self, params):
        """Do shifting of dataset."""

        ds_in = params.getDataset(params.fields['ds_in'])
        value = params.fields['value']

        if params.fields['ds_out'] == '':
            raise DatasetPluginException('Invalid output dataset name')

        return [ Dataset1D(params.fields['ds_out'],
                           data = ds_in.data + value,
                           serr=ds_in.serr, perr=ds_in.perr, nerr=ds_in.nerr) ]

class ConcatenateDatasetPlugin(DatasetPlugin):
    """Dataset plugin to concatenate datasets."""

    menu = ('Concatenate',)
    name = 'Concatenate'
    description_short = 'Concatenate datasets'
    description_filter = ('Concatenate datasets into single datasets. '
                          'Error bars are merged.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDatasetMulti('ds_in', 'Input datasets'),
            field.FieldDataset('ds_out', 'Output dataset name'),
            ]

    def doConcat(self, name, ds):
        """Concatentate datasets, returning parts."""

        # what sort of error bars do we need?
        symerr, asymerr = False, False
        for d in ds:
            if d.serr is not None:
                symerr = True
            if d.perr is not None or d.nerr is not None:
                asymerr = True
                
        # concatenate main data
        dstack = N.hstack([d.data for d in ds])
        sstack = pstack = nstack = None

        if not symerr and not asymerr:
            # no error bars
            pass
        elif symerr and not asymerr:
            # symmetric and not asymmetric error bars
            sstack = []
            for d in ds:
                if d.serr is not None:
                    sstack.append(d.serr)
                else:
                    sstack.append(N.zeros(d.data.shape, dtype=N.float64))
            sstack = N.hstack(sstack)
        else:
            # asymmetric error bars
            pstack = []
            nstack = []
            for d in ds:
                p = n = N.zeros(d.data.shape, dtype=N.float64)
                if d.serr is not None:
                    p, n = d.serr, -d.serr
                else:
                    if d.perr is not None: p = d.perr
                    if d.nerr is not None: n = d.nerr
                pstack.append(p)
                nstack.append(n)
            pstack = N.hstack(pstack)
            nstack = N.hstack(nstack)

        return Dataset1D(name, data=dstack, serr=sstack, perr=pstack, nerr=nstack)

    def update(self, params):
        """Do shifting of dataset."""

        dsin = params.fields['ds_in']
        dsout = params.fields['ds_out']

        if len(dsin) == 0:
            raise DatasetPluginException("Requires one or more input datasets")
        if dsout == '':
            raise DatasetPluginException('Invalid output dataset name')

        # get datasets
        ds = [params.getDataset(d) for d in dsin]

        # return concatentated datasets
        return [ self.doConcat(dsout, ds) ]

class SplitDatasetPlugin(DatasetPlugin):
    """Dataset plugin to split datasets."""

    menu = ('Split',)
    name = 'Split'
    description_short = 'Split datasets'
    description_filter = ('Split out a section of a dataset. Give starting '
                          'index of data and number of datapoints to take.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', 'Input dataset'),
            field.FieldInt('start', 'Starting index (from 1)', default=1),
            field.FieldInt('num', 'Maximum number of datapoints', default=1),
            field.FieldDataset('ds_out', 'Output dataset name'),
            ]

    def update(self, params):
        """Do scaling of dataset."""

        ds_in = params.getDataset(params.fields['ds_in'])
        ds_out = params.fields['ds_out']
        if ds_out == '':
            raise DatasetPluginException('Invalid output dataset name')
        start = params.fields['start']
        num = params.fields['num']

        data, serr, perr, nerr = ds_in.data, ds_in.serr, ds_in.perr, ds_in.nerr

        # chop the data
        data = data[start-1:start-1+num]
        if serr is not None: serr = serr[start-1:start-1+num]
        if perr is not None: perr = perr[start-1:start-1+num]
        if nerr is not None: nerr = nerr[start-1:start-1+num]

        return [ Dataset1D(ds_out, data=data, serr=serr, perr=perr, nerr=nerr) ]

datasetpluginregistry += [
    ScaleDatasetPlugin,
    ShiftDatasetPlugin,
    ConcatenateDatasetPlugin,
    SplitDatasetPlugin,
    ]
