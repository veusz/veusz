# -*- coding: utf-8 -*-

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

"""Plugins for creating datasets."""

import numpy as N
from itertools import izip
import field

import veusz.utils as utils
try:
    import veusz.helpers.qtloops as qtloops
except ImportError:
    pass
import veusz.qtall as qt4

def _(text, disambiguation=None, context='DatasetPlugin'):
    """Translate text."""
    return unicode(
        qt4.QCoreApplication.translate(context, text, disambiguation))

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

class DatasetDateTime(object):
    """Date-time dataset for ImportPlugin or DatasetPlugin."""

    def __init__(self, name, data=[]):
        """A date dataset
        name: name of dataset
        data: list of datetime objects
        """
        self.name = name
        self.update(data=data)

    def update(self, data=[]):
        self.data = N.array(data)

    @staticmethod
    def datetimeToFloat(datetimeval):
        """Return a python datetime object to the required float type."""
        return utils.datetimeToFloat(datetimeval)

    @staticmethod
    def dateStringToFloat(text):
        """Try to convert an iso or local date time to the float type."""
        return utils.dateStringToDate(text)

    @staticmethod
    def floatToDateTime(val):
        """Convert float format datetime to Python datetime."""
        return utils.floatToDateTime(val)

    def _null(self):
        """Empty data contents."""
        self.data = N.array([])

    def _makeVeuszDataset(self, manager):
        """Make a Veusz dataset from the plugin dataset."""
        import veusz.document as document
        return document.DatasetDatePlugin(manager, self)

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

class Constant(object):
    """Dataset to return to set a Veusz constant after import.
    This is only useful in an ImportPlugin, not a DatasetPlugin
    """
    def __init__(self, name, val):
        """Map string value val to name.
        Convert float vals to strings first!"""
        self.name = name
        self.val = val

class Function(object):
    """Dataset to return to set a Veusz function after import."""
    def __init__(self, name, val):
        """Map string value val to name.
        name is "funcname(param,...)", val is a text expression of param.
        This is only useful in an ImportPlugin, not a DatasetPlugin
        """
        self.name = name
        self.val = val

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

    @property
    def datasetsdatetime(self):
        """Return list of existing date-time datesets"""
        import veusz.document as document
        return [name for name, ds in self._doc.data.iteritems() if
                isinstance(ds, document.DatasetDateTime)]

    def evaluateExpression(self, expr, part='data'):
        """Return results of evaluating a 1D dataset expression.
        part is 'data', 'serr', 'perr' or 'nerr' - these are the
        dataset parts which are evaluated by the expression
        """
        return self._doc.evalDatasetExpression(expr, part=part)

    def getDataset(self, name, dimensions=1):
        """Return numerical dataset object for name given.
        Please make sure that dataset data are not modified.

        name: name of dataset
        dimensions: number of dimensions dataset requires

        name not found: raise a DatasetPluginException
        dimensions not right: raise a DatasetPluginException
        """
        import veusz.document as document
        try:
            ds = self._doc.data[name]
        except KeyError:
            raise DatasetPluginException("Unknown dataset '%s'" % name)

        if ds.dimensions != dimensions:
            raise DatasetPluginException(
                "Dataset '%s' does not have %i dimensions" % (name, dimensions))
        if ds.datatype != 'numeric':
            raise DatasetPluginException(
                "Dataset '%s' is not a numerical dataset" % name)

        if isinstance(ds, document.DatasetDateTime):
            return DatasetDateTime(name, data=ds.data)
        elif ds.dimensions == 1:
            return Dataset1D(name, data=ds.data, serr=ds.serr, 
                             perr=ds.perr, nerr=ds.nerr)
        elif ds.dimensions == 2:
            return Dataset2D(name, ds.data,
                             rangex=ds.xrange, rangey=ds.yrange)
        else:
            raise RuntimeError("Invalid number of dimensions in dataset")

    def getDatasets(self, names, dimensions=1):
        """Get a list of numerical datasets (of the dimension given)."""
        return [ self.getDataset(n, dimensions=dimensions) for n in names ]

    def getTextDataset(self, name):
        """Return a text dataset with name given.
        Do not modify this dataset.

        name not found: raise a DatasetPluginException
        """

        try:
            ds = self._doc.data[name]
        except KeyError:
            raise DatasetPluginException("Unknown dataset '%s'" % name)
        if ds.datatype == 'text':
            return DatasetText(name, ds.data)
        raise DatasetPluginException("Dataset '%s' is not a text datset" % name)

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

def combineAddedErrors(inds, length):
    """Combine error bars from list of input dataset, adding
    errors squared (suitable for adding/subtracting)."""

    errortype = errorBarType(inds)
    serr = perr = nerr = None
    if errortype == 'symmetric':
        serr = N.zeros(length, dtype=N.float64)
    elif errortype == 'asymmetric':
        perr = N.zeros(length, dtype=N.float64)
        nerr = N.zeros(length, dtype=N.float64)

    for d in inds:
        f = N.isfinite(d.data)

        if errortype == 'symmetric' and d.serr is not None:
            serr[f] += d.serr[f]**2 
        elif errortype == 'asymmetric':
            if d.serr is not None:
                v = (d.serr[f])**2
                perr[f] += v
                nerr[f] += v
            if d.perr is not None:
                perr[f] += (d.perr[f])**2
            if d.nerr is not None:
                nerr[f] += (d.nerr[f])**2

    if serr is not None: serr = N.sqrt(serr)
    if perr is not None: perr = N.sqrt(perr)
    if nerr is not None: nerr = -N.sqrt(nerr)
    return serr, perr, nerr

def combineMultipliedErrors(inds, length, data):
    """Combine error bars from list of input dataset, adding
    fractional errors squared (suitable for multipling/dividing)."""

    errortype = errorBarType(inds)
    serr = perr = nerr = None
    if errortype == 'symmetric':
        serr = N.zeros(length, dtype=N.float64)
    elif errortype == 'asymmetric':
        perr = N.zeros(length, dtype=N.float64)
        nerr = N.zeros(length, dtype=N.float64)

    for d in inds:
        f = N.isfinite(d.data)
        if len(f) > length:
            f = f[:length]

        if errortype == 'symmetric' and d.serr is not None:
            serr[f] += (d.serr[f]/d.data[f])**2 
        elif errortype == 'asymmetric':
            if d.serr is not None:
                v = (d.serr[f]/d.data[f])**2
                perr[f] += v
                nerr[f] += v
            if d.perr is not None:
                perr[f] += (d.perr[f]/d.data[f])**2
            if d.nerr is not None:
                nerr[f] += (d.nerr[f]/d.data[f])**2

    if serr is not None: serr = N.abs(N.sqrt(serr) * data)
    if perr is not None: perr = N.abs(N.sqrt(perr) * data)
    if nerr is not None: nerr = -N.abs(N.sqrt(nerr) * data)
    return serr, perr, nerr

###########################################################################
## Real plugins are below

class MultiplyDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to scale a dataset."""

    menu = (_('Multiply'), _('By constant'),)
    name = 'Multiply'
    description_short = _('Multiply dataset by a constant')
    description_full = _('Multiply a dataset by a factor. '
                         'Error bars are also scaled.')
    
    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', _('Input dataset')),
            field.FieldFloat('factor', _('Factor'), default=1.),
            field.FieldDataset('ds_out', _('Output dataset name')),
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

class AddDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to add a constant to a dataset."""

    menu = (_('Add'), _('Constant'),)
    name = 'Add'
    description_short = _('Add a constant to a dataset')
    description_full = _('Add a dataset by adding a value. '
                         'Error bars remain the same.')
    
    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', _('Input dataset')),
            field.FieldFloat('value', _('Add value'), default=0.),
            field.FieldDataset('ds_out', _('Output dataset name')),
            ]

    def updateDatasets(self, fields, helper):
        """Do shifting of dataset."""
        ds_in = helper.getDataset(fields['ds_in'])
        self.dsout.update(data = ds_in.data + fields['value'],
                          serr=ds_in.serr, perr=ds_in.perr, nerr=ds_in.nerr)

class ConcatenateDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to concatenate datasets."""

    menu = (_('Join'), _('Concatenate'),)
    name = 'Concatenate'
    description_short = _('Concatenate datasets')
    description_full = _('Concatenate datasets into single dataset.\n'
                         'Error bars are merged.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDatasetMulti('ds_in', _('Input datasets')),
            field.FieldDataset('ds_out', _('Output dataset name')),
            ]

    def updateDatasets(self, fields, helper):
        """Do concatenation of dataset."""

        dsin = helper.getDatasets(fields['ds_in'])
        if len(dsin) == 0:
            raise DatasetPluginException(_('Requires one or more input datasets'))

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

class InterleaveDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to interleave datasets."""

    menu = (_('Join'), _('Element by element'),)
    name = 'Interleave'
    description_short = _('Join datasets, interleaving element by element')
    description_full = _('Join datasets, interleaving element by element.\n'
                         'Error bars are merged.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDatasetMulti('ds_in', _('Input datasets')),
            field.FieldDataset('ds_out', _('Output dataset name')),
            ]

    def updateDatasets(self, fields, helper):
        """Do concatenation of dataset."""

        dsin = helper.getDatasets(fields['ds_in'])
        if len(dsin) == 0:
            raise DatasetPluginException(_('Requires one or more input datasets'))

        maxlength = max( [len(d.data) for d in dsin] )

        def interleave(datasets):
            """This is complex to account for different length datasets."""
            # stick in columns
            ds = [ N.hstack( (d, N.zeros(maxlength-len(d))) )
                   for d in datasets ]
            # which elements are valid
            good = [ N.hstack( (N.ones(len(d), dtype=N.bool),
                                N.zeros(maxlength-len(d), dtype=N.bool)) )
                     for d in datasets ]

            intl = N.column_stack(ds).reshape(maxlength*len(datasets))
            goodintl = N.column_stack(good).reshape(maxlength*len(datasets))
            return intl[goodintl]

        # do interleaving
        data = interleave([d.data for d in dsin])

        # interleave error bars
        errortype = errorBarType(dsin)
        serr = perr = nerr = None
        if errortype == 'symmetric':
            slist = []
            for ds in dsin:
                if ds.serr is None:
                    slist.append(N.zeros_like(ds.data))
                else:
                    slist.append(ds.serr)
            serr = interleave(slist)
        elif errortype == 'asymmetric':
            plist = []
            nlist = []
            for ds in dsin:
                if ds.serr is not None:
                    plist.append(ds.serr)
                    nlist.append(-ds.serr)
                else:
                    if ds.perr is not None:
                        plist.append(ds.perr)
                    else:
                        plist.append(N.zeros_like(ds.data))
                    if ds.nerr is not None:
                        nlist.append(ds.nerr)
                    else:
                        nlist.append(N.zeros_like(ds.data))
            perr = interleave(plist)
            nerr = interleave(nlist)

        # finally update
        self.dsout.update(data=data, serr=serr, nerr=nerr, perr=perr)

class ChopDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to chop datasets."""

    menu = (_('Split'), _('Chop'),)
    name = 'Chop'
    description_short = _('Chop dataset part into new dataset')
    description_full = _('Chop out a section of a dataset. Give starting '
                         'index of data and number of datapoints to take.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', _('Input dataset')),
            field.FieldInt('start', _('Starting index (from 1)'), default=1),
            field.FieldInt('num', _('Maximum number of datapoints'), default=1),
            field.FieldDataset('ds_out', _('Output dataset name')),
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

class PartsDatasetPlugin(DatasetPlugin):
    """Dataset plugin to split datasets into parts."""

    menu = (_('Split'), _('Parts'),)
    name = 'Parts'
    description_short = _('Split dataset into equal-size parts')
    description_full = _('Split dataset into equal-size parts. '
                         'The parts will differ in size if the dataset '
                         'cannot be split equally.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', _('Input dataset')),
            field.FieldDatasetMulti('ds_out', _('Output datasets')),
            ]

    def getDatasets(self, fields):
        """Get output datasets."""
        self.dsout = []
        for d in fields['ds_out']:
            if d.strip() != '':
                self.dsout.append( Dataset1D(d.strip()) )
        if len(self.dsout) == 0:
            raise DatasetPluginException(_('Needs at least one output dataset'))

        return self.dsout

    def updateDatasets(self, fields, helper):
        """Do chopping of dataset."""

        ds_in = helper.getDataset(fields['ds_in'])
        data, serr, perr, nerr = ds_in.data, ds_in.serr, ds_in.perr, ds_in.nerr

        plen = float(len(data)) / len(self.dsout)
        for i, ds in enumerate(self.dsout):
            minv, maxv = int(plen*i), int(plen*(i+1))
            pserr = pperr = pnerr = None
            pdata = data[minv:maxv]
            if serr is not None: pserr = serr[minv:maxv]
            if perr is not None: pperr = perr[minv:maxv]
            if nerr is not None: pnerr = nerr[minv:maxv]
            ds.update(data=pdata, serr=pserr, perr=pperr, nerr=pnerr)

class ThinDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to thin datasets."""

    menu = (_('Split'), _('Thin'),)
    name = 'Thin'
    description_short = _('Select data points at intervals from dataset')
    description_full = _('Select data points at intervals from dataset '
                         'to create new dataset')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', _('Input dataset')),
            field.FieldInt('start', _('Starting index (from 1)'), default=1),
            field.FieldInt('interval', _('Interval between data points'), default=1),
            field.FieldDataset('ds_out', _('Output dataset name')),
            ]

    def updateDatasets(self, fields, helper):
        """Do thinning of dataset."""

        ds_in = helper.getDataset(fields['ds_in'])
        start = fields['start']
        interval = fields['interval']

        data, serr, perr, nerr = ds_in.data, ds_in.serr, ds_in.perr, ds_in.nerr

        data = data[start-1::interval]
        if serr is not None: serr = serr[start-1::interval]
        if perr is not None: perr = perr[start-1::interval]
        if nerr is not None: nerr = nerr[start-1::interval]

        self.dsout.update(data=data, serr=serr, perr=perr, nerr=nerr)

class MeanDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to mean datasets together."""

    menu = (_('Compute'), _('Mean of datasets'),)
    name = 'Mean'
    description_short = _('Compute mean of datasets')
    description_full = _('Compute mean of multiple datasets to create '
                         'a single dataset.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDatasetMulti('ds_in', _('Input datasets')),
            field.FieldDataset('ds_out', _('Output dataset name')),
            ]

    def updateDatasets(self, fields, helper):
        """Compute means of dataset."""

        inds = helper.getDatasets(fields['ds_in'])
        if len(inds) == 0:
            raise DatasetPluginException(_('Requires one or more input datasets'))
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

class AddDatasetsPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to mean datasets together."""

    menu = (_('Add'), _('Datasets'),)
    name = 'Add Datasets'
    description_short = _('Add two or more datasets together')
    description_full = _('Add datasets together to make a single dataset. '
                         'Error bars are combined.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDatasetMulti('ds_in', _('Input datasets')),
            field.FieldDataset('ds_out', _('Output dataset name')),
            ]

    def updateDatasets(self, fields, helper):
        """Compute means of dataset."""

        inds = helper.getDatasets(fields['ds_in'])
        if len(inds) == 0:
            raise DatasetPluginException(_('Requires one or more input datasets'))
        maxlength = max( [len(d.data) for d in inds] )

        # add data where finite
        data = N.zeros(maxlength, dtype=N.float64)
        anyfinite = N.zeros(maxlength, dtype=N.bool)
        for d in inds:
            f = N.isfinite(d.data)
            data[f] += d.data[f]
            anyfinite[f] = True
        data[N.logical_not(anyfinite)] = N.nan

        # handle error bars
        serr, perr, nerr = combineAddedErrors(inds, maxlength)

        # update output dataset
        self.dsout.update(data=data, serr=serr, perr=perr, nerr=nerr)

class SubtractDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to subtract two datasets."""

    menu = (_('Subtract'), _('Datasets'),)
    name = 'Subtract Datasets'
    description_short = _('Subtract two datasets')
    description_full = _('Subtract two datasets. '
                         'Combined error bars are also calculated.')
    
    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in1', _('Input dataset 1')),
            field.FieldDataset('ds_in2', _('Input dataset 2')),
            field.FieldDataset('ds_out', _('Output dataset name')),
            ]

    def updateDatasets(self, fields, helper):
        """Do scaling of dataset."""

        dsin1 = helper.getDataset(fields['ds_in1'])
        dsin2 = helper.getDataset(fields['ds_in2'])

        minlength = min( len(dsin1.data), len(dsin2.data) )
        data = dsin1.data[:minlength] - dsin2.data[:minlength]

        # computing error bars is non trivial!
        serr = perr = nerr = None
        errortype = errorBarType([dsin1, dsin2])
        if errortype == 'symmetric':
            serr1 = serr2 = 0
            if dsin1.serr is not None:
                serr1 = dsin1.serr[:minlength]
            if dsin2.serr is not None:
                serr2 = dsin2.serr[:minlength]
            serr = N.sqrt(serr1**2 + serr2**2)
        elif errortype == 'asymmetric':
            perr1 = perr2 = nerr1 = nerr2 = 0
            if dsin1.serr is not None:
                perr1 = nerr1 = dsin1.serr[:minlength]
            else:
                if dsin1.perr is not None: perr1 = dsin1.perr[:minlength]
                if dsin1.nerr is not None: nerr1 = dsin1.nerr[:minlength]
            if dsin2.serr is not None:
                perr2 = nerr2 = dsin2.serr[:minlength]
            else:
                if dsin2.perr is not None: perr2 = dsin2.perr[:minlength]
                if dsin2.nerr is not None: nerr2 = dsin2.nerr[:minlength]
            perr = N.sqrt(perr1**2 + nerr2**2)
            nerr = -N.sqrt(nerr1**2 + perr2**2)

        self.dsout.update(data=data, serr=serr, perr=perr, nerr=nerr)

class SubtractMeanDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to subtract mean from dataset."""

    menu = (_('Subtract'), _('Mean'),)
    name = 'Subtract Mean'
    description_short = _('Subtract mean from dataset')
    description_full = _('Subtract mean from dataset,'
                         ' optionally dividing by standard deviation.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', _('Input dataset 1')),
            field.FieldBool('divstddev', _('Divide by standard deviation')),
            field.FieldDataset('ds_out', _('Output dataset name')),
            ]

    def updateDatasets(self, fields, helper):
        """Do scaling of dataset."""

        dsin = helper.getDataset(fields['ds_in'])

        vals = dsin.data
        mean = vals[N.isfinite(vals)].mean()
        vals = vals - mean

        if fields['divstddev']:
            vals /= vals[N.isfinite(vals)].std()

        self.dsout.update(
            data=vals, serr=dsin.serr, perr=dsin.perr, nerr=dsin.nerr)

class SubtractMinimumDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to subtract minimum from dataset."""

    menu = (_('Subtract'), _('Minimum'),)
    name = 'Subtract Minimum'
    description_short = _('Subtract minimum from dataset')
    description_full = _('Subtract the minimum value from a dataset')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', _('Input dataset 1')),
            field.FieldDataset('ds_out', _('Output dataset name')),
            ]

    def updateDatasets(self, fields, helper):
        """Do scaling of dataset."""

        dsin = helper.getDataset(fields['ds_in'])

        vals = dsin.data
        minval = vals[N.isfinite(vals)].min()
        vals = vals - minval

        self.dsout.update(
            data=vals, serr=dsin.serr, perr=dsin.perr, nerr=dsin.nerr)

class MultiplyDatasetsPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to multiply two or more datasets."""

    menu = (_('Multiply'), _('Datasets'),)
    name = 'Multiply Datasets'
    description_short = _('Multiply two or more datasets')
    description_full = _('Multiply two or more datasets. '
                         'Combined error bars are also calculated.')
    
    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDatasetMulti('ds_in', _('Input datasets')),
            field.FieldDataset('ds_out', _('Output dataset name')),
            ]

    def updateDatasets(self, fields, helper):
        """Multiply the datasets."""

        names = fields['ds_in']
        inds = [ helper.getDataset(d) for d in names ]
        maxlength = max( [d.data.shape[0] for d in inds] )

        # output data and where data is finite
        data = N.ones(maxlength, dtype=N.float64)
        anyfinite = N.zeros(maxlength, dtype=N.bool)
        for d in inds:
            f = N.isfinite(d.data)
            anyfinite[f] = True
            data[f] *= d.data[f]

        # where always NaN, make NaN
        data[N.logical_not(anyfinite)] = N.nan

        # get error bars
        serr, perr, nerr = combineMultipliedErrors(inds, maxlength, data)

        self.dsout.update(data=data, serr=serr, perr=perr, nerr=nerr)

class DivideDatasetsPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to divide two datasets."""

    menu = (_('Divide'), _('Datasets'),)
    name = 'Divide Datasets'
    description_short = _('Compute ratio or fractional difference'
                          ' between two datasets')
    description_full = _('Divide or compute fractional difference'
                         ' between two datasets')
    
    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in1', _('Input dataset 1')),
            field.FieldDataset('ds_in2', _('Input dataset 2')),
            field.FieldBool('frac', _('Compute fractional difference'),
                            default=False),
            field.FieldDataset('ds_out', _('Output dataset name')),
            ]

    def updateDatasets(self, fields, helper):
        """Compute ratio."""

        inds1 = helper.getDataset( fields['ds_in1'] )
        inds2 = helper.getDataset( fields['ds_in2'] )
        length = min( len(inds1.data), len(inds2.data) )

        # compute ratio
        data = inds1.data[:length] / inds2.data[:length]

        # get error bars
        serr, perr, nerr = combineMultipliedErrors([inds1, inds2], length, data)

        # convert to fractional difference (if reqd)
        if fields['frac']:
            data -= 1

        self.dsout.update(data=data, serr=serr, perr=perr, nerr=nerr)

class ExtremesDatasetPlugin(DatasetPlugin):
    """Dataset plugin to get extremes of dataset."""

    menu = (_('Compute'), _('Dataset extremes'),)
    name = 'Extremes'
    description_short = _('Compute extreme values of input datasets')
    description_full = _('Compute extreme values of input datasets. Creates '
                         'minimum and maximum datasets.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDatasetMulti('ds_in', _('Input datasets')),
            field.FieldBool('errorbars', _('Include error bars')),
            field.FieldDataset('ds_min', _('Output minimum dataset (optional)')),
            field.FieldDataset('ds_max', _('Output maximum dataset (optional)')),
            field.FieldDataset('ds_errorbar', _('Output range as error bars '
                                                'in dataset (optional)')),
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
            raise DatasetPluginException(_('Provide at least one output dataset'))
        return dsout

    def updateDatasets(self, fields, helper):
        """Compute extremes of datasets."""

        names = fields['ds_in']
        inds = [ helper.getDataset(d) for d in names ]
        maxlength = max( [d.data.shape[0] for d in inds] )

        minvals = N.zeros(maxlength, dtype=N.float64) + 1e100
        maxvals = N.zeros(maxlength, dtype=N.float64) - 1e100
        anyfinite = N.zeros(maxlength, dtype=N.bool)
        for d in inds:
            f = N.isfinite(d.data)
            anyfinite[f] = True

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

        minvals[N.logical_not(anyfinite)] = N.nan
        maxvals[N.logical_not(anyfinite)] = N.nan

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

class CumulativePlugin(_OneOutputDatasetPlugin):
    """Compute cumulative values."""

    menu = (_('Compute'), _('Cumulative value'),)
    name = 'Cumulative'
    description_short = _('Compute the cumulative value of a dataset')
    description_full = _('Compute the cumulative value of a dataset. '
                         ' Error bars are combined.\n'
                         'Default behaviour is to accumulate from start.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', _('Input dataset')),
            field.FieldBool('fromend', _('Compute cumulative value from end')),
            field.FieldDataset('ds_out', _('Output dataset')),
            ]

    def updateDatasets(self, fields, helper):
        """Do accumulation."""

        ds_in = helper.getDataset(fields['ds_in'])
        fromend = fields['fromend']

        def cumsum(v):
            """Compute cumulative, handing nans and reverse."""
            v = N.array(v)
            if fromend: v = v[::-1]
            v[ N.logical_not(N.isfinite(v)) ] = 0.
            c = N.cumsum(v)
            if fromend: c = c[::-1]
            return c

        # compute cumulative values
        data, serr, perr, nerr = ds_in.data, ds_in.serr, ds_in.perr, ds_in.nerr
        data = cumsum(data)
        if serr is not None: serr = N.sqrt( cumsum(serr**2) )
        if perr is not None: perr = N.sqrt( cumsum(perr**2) )
        if nerr is not None: nerr = -N.sqrt( cumsum(nerr**2) )
        self.dsout.update(data=data, serr=serr, perr=perr, nerr=nerr)

class DemultiplexPlugin(DatasetPlugin):
    """Dataset plugin to split a dataset into multiple datasets, element-by-element."""

    menu = (_('Split'), _('Element by element'),)
    name = 'Demultiplex'
    description_short = _('Split dataset into multiple datasets element-by-element')
    description_full = _('Split dataset into multiple datasets on an '
                         'element-by-element basis.\n'
                         'e.g. 1, 2, 3, 4, 5, 6 could be converted to '
                         '1, 3, 5 and 2, 4, 6.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', _('Input dataset')),
            field.FieldDatasetMulti('ds_out', _('Output datasets')),
            ]

    def getDatasets(self, fields):
        """Returns demuxed output datasets."""
        names = [n.strip() for n in fields['ds_out'] if n.strip() != '']
        if len(names) == 0:
            raise DatasetPluginException(_('Requires at least one output dataset'))

        self.ds_out = [ Dataset1D(n) for n in names ]
        return self.ds_out

    def updateDatasets(self, fields, helper):
        """Compute means of dataset."""

        ds_in = helper.getDataset( fields['ds_in'] )

        num = len(self.ds_out)
        for i, ds in enumerate(self.ds_out):
            data = ds_in.data[i::num]
            serr = nerr = perr = None
            if ds_in.serr is not None:
                serr = ds_in.serr[i::num]
            if ds_in.perr is not None:
                perr = ds_in.perr[i::num]
            if ds_in.nerr is not None:
                nerr = ds_in.nerr[i::num]
            ds.update(data=data, serr=serr, perr=perr, nerr=nerr)

class PolarToCartesianPlugin(DatasetPlugin):
    """Convert from r,theta to x,y coordinates."""

    menu = (_('Convert'), _('Polar to Cartesian'),)
    name = 'PolarToCartesian'
    description_short = _('Convert r,theta coordinates to x,y coordinates')
    description_full = _('Convert r,theta coordinates to x,y coordinates.\n'
                         'Error bars are ignored.')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('r_in', _('Input dataset (r)')),
            field.FieldDataset('theta_in', _('Input dataset (theta)')),
            field.FieldCombo('units', _('Angular units'),
                             items=('radians', 'degrees'),
                             editable=False),
            field.FieldDataset('x_out', _('Output dataset (x)')),
            field.FieldDataset('y_out', _('Output dataset (y)')),
            ]

    def getDatasets(self, fields):
        """Returns x and y output datasets."""
        if fields['x_out'] == '':
            raise DatasetPluginException(_('Invalid output x dataset name'))
        if fields['y_out'] == '':
            raise DatasetPluginException(_('Invalid output y dataset name'))
        self.x_out = Dataset1D(fields['x_out'])
        self.y_out = Dataset1D(fields['y_out'])
        return [self.x_out, self.y_out]

    def updateDatasets(self, fields, helper):
        """Compute means of dataset."""

        ds_r = helper.getDataset( fields['r_in'] ).data
        ds_theta = helper.getDataset( fields['theta_in'] ).data
        if fields['units'] == 'degrees':
            # convert to radians
            ds_theta = ds_theta * (N.pi / 180.)

        x = ds_r * N.cos(ds_theta)
        y = ds_r * N.sin(ds_theta)
        self.x_out.update(data=x)
        self.y_out.update(data=y)

class FilterDatasetPlugin(_OneOutputDatasetPlugin):
    """Dataset plugin to filter a dataset using an expression."""

    menu = (_('Filter'), _('Expression'),)
    name = 'FilterExpression'
    description_short = _('Filter a dataset using an expression')
    description_full = _('Filter a dataset using an expression, '
                         'e.g. "x>10" or "(x>1) & (y<2)"')
    
    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', _('Input dataset')),
            field.FieldText('filter', _('Filter expression')),
            field.FieldBool('replacenan', _('Replace excluded points by NaN\n'
                                            '(indicate missing points)'),
                            default=False),
            field.FieldDataset('ds_out', _('Output dataset')),
            ]

    def updateDatasets(self, fields, helper):
        """Do shifting of dataset."""
        ds_in = helper.getDataset(fields['ds_in'])
        filt = helper.evaluateExpression(fields['filter'])
        data, serr, perr, nerr = ds_in.data, ds_in.serr, ds_in.perr, ds_in.nerr

        try:
            if fields['replacenan']:
                # replace bad points with nan
                data = data.copy()
                data[N.logical_not(filt)] = N.nan
            else:
                # just select good points
                data = data[filt]
                if serr is not None: serr = serr[filt]
                if perr is not None: perr = perr[filt]
                if nerr is not None: nerr = nerr[filt]
        except:
            raise DatasetPluginException(_('Error filtering dataset'))

        self.dsout.update(data=data, serr=serr, perr=perr, nerr=nerr)

class MovingAveragePlugin(_OneOutputDatasetPlugin):
    """Compute moving average for dataset."""

    menu = (_('Filtering'), _('Moving Average'),)
    name = 'MovingAverage'
    description_short = _('Compute moving average for regularly spaced data')
    description_full = _('Compute moving average for regularly spaced data.'
                         'Average is computed either\nside of each data point '
                         'by number of points given.')
    
    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', _('Input dataset')),
            field.FieldInt('width', _('Points either side of point to average'),
                           default=1, minval=0),
            field.FieldBool('weighterrors', _('Weight by error bars'),
                            default=True),
            field.FieldDataset('ds_out', _('Output dataset')),
            ]
    
    def updateDatasets(self, fields, helper):
        """Do shifting of dataset."""
        ds_in = helper.getDataset(fields['ds_in'])
        weights = None
        if fields['weighterrors']:
            if ds_in.serr is not None:
                weights = 1. / ds_in.serr**2
            elif ds_in.perr is not None and ds_in.nerr is not None:
                weights = 1. / ( (ds_in.perr**2+ds_in.nerr**2)/2. )
        width = fields['width']
        data = qtloops.rollingAverage(ds_in.data, weights, width)
        self.dsout.update(data=data)

class LinearInterpolatePlugin(_OneOutputDatasetPlugin):
    """Do linear interpolation of data."""

    menu = (_('Filtering'), _('Linear interpolation'),)
    name = 'LinearInterpolation'
    description_short = _('Linear interpolation of x,y data')
    description_full = _("Compute linear interpolation of x,y data.\n"
                         "Given datasets for y = f(x), compute y' = f(x'), "
                         "using linear interpolation.\n"
                         "Assumes x dataset increases in value.")

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_x', _('Input dataset x')),
            field.FieldDataset('ds_y', _('Input dataset y')),
            field.FieldDataset('ds_xprime', _("Input dataset x'")),
            field.FieldBool('edgenan', _('Use nan for values outside x range')),
            field.FieldDataset('ds_out', _("Output dataset y'")),
            ]

    def updateDatasets(self, fields, helper):
        """Do shifting of dataset."""
        ds_x = helper.getDataset(fields['ds_x']).data
        ds_y = helper.getDataset(fields['ds_y']).data
        ds_xprime = helper.getDataset(fields['ds_xprime']).data

        minlenin = min( len(ds_x), len(ds_y) )
        pad = None
        if fields['edgenan']:
            pad = N.nan

        interpol = N.interp(ds_xprime,
                            ds_x[:minlenin], ds_y[:minlenin],
                            left=pad, right=pad)

        self.dsout.update(data=interpol)

class SortPlugin(_OneOutputDatasetPlugin):
    """Sort a dataset."""

    menu = (_('Compute'), _('Sorted'),)
    name = 'Sort'
    description_short = description_full = _('Sort a dataset')

    def __init__(self):
        """Define fields."""
        self.fields = [
            field.FieldDataset('ds_in', _('Input dataset')),
            field.FieldBool('reverse', _('Reverse')),
            field.FieldDataset('ds_out', _('Output dataset')),
            ]

    def updateDatasets(self, fields, helper):
        """Do sorting of dataset."""
        ds = helper.getDataset(fields['ds_in'])

        idxs = N.argsort(ds.data)
        if fields['reverse']:
            idxs = idxs[::-1]

        out = { 'data': ds.data[idxs] }
        if ds.serr is not None: out['serr'] = ds.serr[idxs]
        if ds.perr is not None: out['perr'] = ds.perr[idxs]
        if ds.nerr is not None: out['nerr'] = ds.nerr[idxs]

        self.dsout.update(**out)

datasetpluginregistry += [
    AddDatasetPlugin,
    AddDatasetsPlugin,
    SubtractDatasetPlugin,
    SubtractMeanDatasetPlugin,
    SubtractMinimumDatasetPlugin,
    MultiplyDatasetPlugin,
    MultiplyDatasetsPlugin,
    DivideDatasetsPlugin,
    MeanDatasetPlugin,
    ExtremesDatasetPlugin,
    CumulativePlugin,

    ConcatenateDatasetPlugin,
    InterleaveDatasetPlugin,

    ChopDatasetPlugin,
    PartsDatasetPlugin,
    DemultiplexPlugin,
    ThinDatasetPlugin,

    PolarToCartesianPlugin,

    FilterDatasetPlugin,

    MovingAveragePlugin,
    LinearInterpolatePlugin,
    SortPlugin,
    ]
