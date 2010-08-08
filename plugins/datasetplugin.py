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

def numpyCopyOrNone(data):
    """If data is None return None
    Otherwise return a numpy array corresponding to data."""
    if data is None:
        return None
    return N.array(data, dtype=N.float64)

# these classes are returned from dataset plugins
class Dataset1D(object):
    """1D dataset for ImportPlugin or DatasetPlugin."""
    def __init__(self, name, data=[], serr=None, perr=None, nerr=None):
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
        self.update(data=data, serr=serr, perr=perr, nerr=nerr)

    def update(self, data=[], serr=None, perr=None, nerr=None):
        """Update values to those given."""
        self.data = numpyCopyOrNone(data)
        self.serr = numpyCopyOrNone(serr)
        self.perr = numpyCopyOrNone(perr)
        self.nerr = numpyCopyOrNone(nerr)

    def _null(self):
        """Empty data contents."""
        self.data = N.array([])
        self.serr = self.perr = self.nerr = None

    def _makeVeuszDataset(self, manager):
        """Make a Veusz dataset from the plugin dataset."""
        # need to do the import here as otherwise we get a loop
        import veusz.document as document
        return document.Dataset1DPlugin(manager, self)

class Dataset2D(object):
    """2D dataset for ImportPlugin or DatasetPlugin."""
    def __init__(self, name, data=[[]], rangex=None, rangey=None):
        """2D dataset.
        name: name of dataset
        data: 2D numpy array of values or list of lists of floats
        rangex: optional tuple with X range of data (min, max)
        rangey: optional tuple with Y range of data (min, max)
        """
        self.name = name
        self.update(data=data, rangex=rangex, rangey=rangey)

    def update(self, data=[[]], rangex=None, rangey=None):
        self.data = N.array(data, dtype=N.float64)
        self.rangex = rangex
        self.rangey = rangey

    def _null(self):
        """Empty data contents."""
        self.data = N.array([[]])
        self.rangex = self.rangey = (0, 1)

    def _makeVeuszDataset(self, manager):
        """Make a Veusz dataset from the plugin dataset."""
        import veusz.document as document
        return document.Dataset2DPlugin(manager, self)

class DatasetText(object):
    """Text dataset for ImportPlugin or DatasetPlugin."""
    def __init__(self, name, data=[]):
        """A text dataset
        name: name of dataset
        data: data in dataset: list of strings
        """
        self.name = name
        self.update(data=data)

    def update(self, data=[]):
        self.data = list(data)

    def _null(self):
        """Empty data contents."""
        self.data = []

    def _makeVeuszDataset(self, manager):
        """Make a Veusz dataset from the plugin dataset."""
        import veusz.document as document
        return document.DatasetTextPlugin(manager, self)

# class to pass to plugin to give parameters
class DatasetPluginHelper(object):
    """Helpers to get existing datasets for plugins."""
    
    def __init__(self, doc):
        """Construct helper object to pass to DatasetPlugins."""
        self._doc = doc

    @property
    def datasets1d(self):
        """Return list of existing 1D numeric datasets"""
        return [name for name, ds in self._doc.data.iteritems() if
                (ds.dimensions == 1 and ds.datatype == 'numeric')]

    @property
    def datasets2d(self):
        """Return list of existing 2D numeric datasets"""
        return [name for name, ds in self._doc.data.iteritems() if
                (ds.dimensions == 2 and ds.datatype == 'numeric')]

    @property
    def datasetstext(self):
        """Return list of existing 1D text datasets"""
        return [name for name, ds in self._doc.data.iteritems() if
                (ds.dimensions == 1 and ds.datatype == 'text')]

    def getDataset(self, name, dimensions=1):
        """Return dataset object for name given.
        Make sure that dataset data are not modified.

        name: name of dataset
        dimensions: number of dimensions dataset requires

        name not found: raise a DatasetPluginException
        dimensions not right: raise a DatasetPluginException
        """
        try:
            ds = self._doc.data[name]
        except KeyError:
            raise DatasetPluginException("Unknown dataset '%s'" % name)
        if ds.dimensions != dimensions:
            raise DatasetPluginException(
                "Dataset '%s' does not have %i dimensions" % (name, dimensions))

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

    def getDatasets(self, names, dimensions=1):
        """Get a list of datasets."""
        return [ self.getDataset(n, dimensions=dimensions) for n in names ]

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
        self.helper = DatasetPluginHelper(doc)
        self.fields = fields
        self.changeset = -1

        self.setupDatasets()

    def setupDatasets(self):
        """Do initial construction of datasets."""

        self.datasetnames = []
        self.datasets = []
        self.veuszdatasets = []
        self.datasets = self.plugin.getDatasets(self.fields)
        for ds in self.datasets:
            self.datasetnames.append(ds.name)
            veuszds = ds._makeVeuszDataset(self)
            veuszds.document = self.document
            self.veuszdatasets.append(veuszds)

    def nullDatasets(self):
        """Clear out contents of datasets."""
        for ds in self.datasets:
            ds._null()

    def saveToFile(self, fileobj):
        """Save command to load in plugin and parameters."""

        args = [ repr(self.plugin.name), repr(self.fields) ]

        # look for renamed or deleted datasets
        names = {}
        for ds, dsname in izip( self.veuszdatasets, self.datasetnames ):
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

    def update(self, raiseerrors=False):
        """Update created datasets.

        if raiseerrors is True, raise an exception if there is an exeception
        when updating the dataset
        """

        if self.document.changeset == self.changeset:
            return
        self.changeset = self.document.changeset

        # run the plugin with its parameters
        try:
            self.plugin.updateDatasets(self.fields, self.helper)
        except DatasetPluginException, ex:
            # this is for immediate notification
            if raiseerrors:
                raise

            # otherwise if there's an error, then log and null outputs
            self.document.log( unicode(ex) )
            self.nullDatasets()

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

    def getDatasets(self, fields):
        """Override this to return a list of (empty) Dataset1D,
        Dataset2D and DatasetText objects to provide the initial names
        and type of datasets.

        These should be saved for updating in updateDatasets.

        fields: dict of results to the field objects given in self.fields
        raise a DatasetPluginException if there is a problem with fields
        """
        return []

    def updateDatasets(self, fields, helper):
        """Override this to update the dataset objects provided by this plugin.

        fields: dict of field results (also provided to setup)
        helper: DatasetPluginHelper object, to get other datasets in document

        raise a DatasetPluginException if there is a problem
        """

class _OneOutputDatasetPlugin(DatasetPlugin):
    """Simplify plugins which create one output with field ds_out."""

    def getDatasets(self, fields):
        """Returns single output dataset (self.dsout)."""
        if fields['ds_out'] == '':
            raise DatasetPluginException('Invalid output dataset name')
        self.dsout = Dataset1D(fields['ds_out'])
        return [self.dsout]

def errorBarType(ds):
    """Return type of error bars in list of datasets.
    'none', 'symmetric', 'asymmetric'
    """

    symerr = False
    for d in ds:
        if d.serr is not None:
            symerr = True
        elif d.perr is not None or d.nerr is not None:
            return 'asymmetric'
    if symerr:
        return 'symmetric'
    return 'none'

###########################################################################
## Real plugins are below

class ScaleDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to scale a dataset."""

    menu = ('Scale',)
    name = 'Scale'
    description_short = 'Scale dataset by factor'
    description_full = ('Scale a dataset by a factor. '
                        'Error bars are also scaled.')
    
    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', 'Input dataset'),
            field.FieldFloat('factor', 'Scale factor', default=1.),
            field.FieldDataset('ds_out', 'Output dataset name'),
            ]

    def updateDatasets(self, fields, helper):
        """Do scaling of dataset."""

        ds_in = helper.getDataset(fields['ds_in'])
        f = fields['factor']

        data, serr, perr, nerr = ds_in.data, ds_in.serr, ds_in.perr, ds_in.nerr
        data = data * f
        if serr is not None: serr = serr * f
        if perr is not None: perr = perr * f
        if nerr is not None: nerr = nerr * f

        self.dsout.update(data=data, serr=serr, perr=perr, nerr=nerr)

class ShiftDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to shift a dataset."""

    menu = ('Shift',)
    name = 'Shift'
    description_short = 'Shift dataset by value'
    description_full = ('Shift a dataset by adding a value. '
                        'Error bars remain the same.')
    
    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', 'Input dataset'),
            field.FieldFloat('value', 'Shift value', default=0.),
            field.FieldDataset('ds_out', 'Output dataset name'),
            ]

    def updateDatasets(self, fields, helper):
        """Do shifting of dataset."""
        ds_in = helper.getDataset(fields['ds_in'])
        self.dsout.update(data = ds_in.data + fields['value'],
                          serr=ds_in.serr, perr=ds_in.perr, nerr=ds_in.nerr)

class ConcatenateDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to concatenate datasets."""

    menu = ('Concatenate',)
    name = 'Concatenate'
    description_short = 'Concatenate datasets'
    description_full = ('Concatenate datasets into single dataset.\n'
                        'Error bars are merged.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDatasetMulti('ds_in', 'Input datasets'),
            field.FieldDataset('ds_out', 'Output dataset name'),
            ]

    def updateDatasets(self, fields, helper):
        """Do concatenation of dataset."""

        dsin = helper.getDatasets(fields['ds_in'])
        if len(dsin) == 0:
            raise DatasetPluginException("Requires one or more input datasets")

        # concatenate main data
        dstack = N.hstack([d.data for d in dsin])
        sstack = pstack = nstack = None

        # what sort of error bars do we need?
        errortype = errorBarType(dsin)
        if errortype == 'symmetric':
            # symmetric and not asymmetric error bars
            sstack = []
            for d in dsin:
                if d.serr is not None:
                    sstack.append(d.serr)
                else:
                    sstack.append(N.zeros(d.data.shape, dtype=N.float64))
            sstack = N.hstack(sstack)
        elif errortype == 'asymmetric':
            # asymmetric error bars
            pstack = []
            nstack = []
            for d in dsin:
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

        self.dsout.update(data=dstack, serr=sstack, perr=pstack, nerr=nstack)

class ChopDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to chop datasets."""

    menu = ('Chop',)
    name = 'Chop'
    description_short = 'Chop dataset part into new dataset'
    description_full = ('Chop out a section of a dataset. Give starting '
                        'index of data and number of datapoints to take.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', 'Input dataset'),
            field.FieldInt('start', 'Starting index (from 1)', default=1),
            field.FieldInt('num', 'Maximum number of datapoints', default=1),
            field.FieldDataset('ds_out', 'Output dataset name'),
            ]

    def updateDatasets(self, fields, helper):
        """Do chopping of dataset."""

        ds_in = helper.getDataset(fields['ds_in'])
        start = fields['start']
        num = fields['num']

        data, serr, perr, nerr = ds_in.data, ds_in.serr, ds_in.perr, ds_in.nerr

        # chop the data
        data = data[start-1:start-1+num]
        if serr is not None: serr = serr[start-1:start-1+num]
        if perr is not None: perr = perr[start-1:start-1+num]
        if nerr is not None: nerr = nerr[start-1:start-1+num]

        self.dsout.update(data=data, serr=serr, perr=perr, nerr=nerr)

class MeanDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to mean datasets together."""

    menu = ('Compute mean',)
    name = 'Mean'
    description_short = 'Compute mean of datasets'
    description_full = ('Compute mean of multiple datasets to create '
                        'a single dataset.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDatasetMulti('ds_in', 'Input datasets'),
            field.FieldDataset('ds_out', 'Output dataset name'),
            ]

    def updateDatasets(self, fields, helper):
        """Compute means of dataset."""

        inds = helper.getDatasets(fields['ds_in'])
        if len(inds) == 0:
            raise DatasetPluginException("Requires one or more input datasets")
        maxlength = max( [len(d.data) for d in inds] )

        # mean data (only use finite values)
        tot = N.zeros(maxlength, dtype=N.float64)
        num = N.zeros(maxlength, dtype=N.int)
        for d in inds:
            f = N.isfinite(d.data)
            tot[f] += d.data[f]
            num[f] += 1
        data = tot / num

        def averageError(errtype, fallback=None):
            """Get average for an error value."""
            tot = N.zeros(maxlength, dtype=N.float64)
            num = N.zeros(maxlength, dtype=N.int)
            for d in inds:
                vals = getattr(d, errtype)
                if vals is None and fallback:
                    vals = getattr(d, fallback)

                # add values if not missing
                if vals is not None:
                    f = N.isfinite(vals)
                    tot[f] += (vals[f]) ** 2
                    num[f] += 1
                else:
                    # treat as zero errors if missing errors
                    num[:len(d.data)] += 1
            return N.sqrt(tot) / num

        # do error bar handling
        serr = perr = nerr = None
        errortype = errorBarType(inds)
        if errortype == 'symmetric':
            serr = averageError('serr')
        elif errortype == 'asymmetric':
            perr = averageError('perr', fallback='serr')
            nerr = -averageError('nerr', fallback='serr')

        self.dsout.update(data=data, serr=serr, perr=perr, nerr=nerr)

class AddDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to mean datasets together."""

    menu = ('Add datasets',)
    name = 'Add'
    description_short = 'Add datasets together'
    description_full = ('Add datasets together to make a single dataset. '
                        'Error bars are combined.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDatasetMulti('ds_in', 'Input datasets'),
            field.FieldDataset('ds_out', 'Output dataset name'),
            ]

    def updateDatasets(self, fields, helper):
        """Compute means of dataset."""

        inds = helper.getDatasets(fields['ds_in'])
        if len(inds) == 0:
            raise DatasetPluginException("Requires one or more input datasets")
        maxlength = max( [len(d.data) for d in inds] )

        # add data where finite
        data = N.zeros(maxlength, dtype=N.float64)
        for d in inds:
            f = N.isfinite(d.data)
            data[f] += d.data[f]

        # do error bar handling
        def computeError(errortype, fallback=None):
            """Add together error bars in quadrature."""
            toterrsqd = N.zeros(maxlength, dtype=N.float64)
            for d in inds:
                v = getattr(d, errortype)
                if v is None and fallback:
                    v = getattr(d, fallback)
                if v is not None:
                    f = N.isfinite(v)
                    toterrsqd[f] += (v[f])**2
            return N.sqrt(toterrsqd)

        serr = perr = nerr = None
        errortype = errorBarType(inds)
        if errortype == 'symmetric':
            serr = computeError('serr')
        elif errortype == 'asymmetric':
            perr = computeError('perr', fallback='serr')
            nerr = -computeError('nerr', fallback='serr')

        # update output dataset
        self.dsout.update(data=data, serr=serr, perr=perr, nerr=nerr)

class ExtremesDatasetPlugin(DatasetPlugin):
    """Dataset plugin to get extremes of dataset."""

    menu = ('Data extremes',)
    name = 'DataExtremes'
    description_short = 'Compute extreme values of input datasets'
    description_long = ('Compute extreme values of input datasets. Creates '
                        'minimum and maximum datasets.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDatasetMulti('ds_in', 'Input datasets'),
            field.FieldBool('errorbars', 'Include error bars'),
            field.FieldDataset('ds_min', 'Output minimum dataset (optional)'),
            field.FieldDataset('ds_max', 'Output maximum dataset (optional)'),
            field.FieldDataset('ds_errorbar', 'Output range as error bars '
                               'in dataset (optional)'),
            ]

    def getDatasets(self, fields):
        """Returns output dataset."""
        dsout = []
        self.dsmin = self.dsmax = self.dserror = None
        if fields['ds_min'] != '':
            self.dsmin = Dataset1D(fields['ds_min'])
            dsout.append(self.dsmin)
        if fields['ds_max'] != '':
            self.dsmax = Dataset1D(fields['ds_max'])
            dsout.append(self.dsmax)
        if fields['ds_errorbar'] != '':
            self.dserror = Dataset1D(fields['ds_errorbar'])
            dsout.append(self.dserror)
        if not dsout:
            raise DatasetPluginException('Provide at least one output dataset')
        return dsout

    def updateDatasets(self, fields, helper):
        """Compute extremes of datasets."""

        names = fields['ds_in']
        inds = [ helper.getDataset(d) for d in names ]
        maxlength = max( [d.data.shape[0] for d in inds] )

        minvals = N.zeros(maxlength, dtype=N.float64) + 1e100
        maxvals = N.zeros(maxlength, dtype=N.float64) - 1e100
        for d in inds:
            f = N.isfinite(d.data)

            v = d.data
            if fields['errorbars']:
                if d.serr is not None:
                    v = v - d.serr
                elif d.nerr is not None:
                    v = v + d.nerr
            minvals[f] = N.min( (minvals[f], v[f]), axis=0 )

            v = d.data
            if fields['errorbars']:
                if d.serr is not None:
                    v = v + d.serr
                elif d.perr is not None:
                    v = v + d.perr
            maxvals[f] = N.max( (maxvals[f], v[f]), axis=0 )

        if self.dsmin is not None:
            self.dsmin.update(data=minvals)
        if self.dsmax is not None:
            self.dsmax.update(data=maxvals)
        if self.dserror is not None:
            # compute mean and look at differences from it
            tot = N.zeros(maxlength, dtype=N.float64)
            num = N.zeros(maxlength, dtype=N.int)
            for d in inds:
                f = N.isfinite(d.data)
                tot[f] += d.data[f]
                num[f] += 1
            mean = tot / num
            self.dserror.update(data=mean, nerr=minvals-mean, perr=maxvals-mean)

datasetpluginregistry += [
    ScaleDatasetPlugin,
    ShiftDatasetPlugin,
    ConcatenateDatasetPlugin,
    ChopDatasetPlugin,
    MeanDatasetPlugin,
    AddDatasetPlugin,
    ExtremesDatasetPlugin,
    ]
