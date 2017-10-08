#    Copyright (C) 2013 Jeremy S. Sanders
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

from __future__ import division, print_function

import collections
import re

import numpy as N
from .. import qtall as qt4
from ..compat import citems, cvalues, cbytes, cunicode, cpy3
from .. import document
from .. import datasets
from .. import utils
from . import base
from . import fits_hdf5_helpers

def _(text, disambiguation=None, context="Import_HDF5"):
    return qt4.QCoreApplication.translate(context, text, disambiguation)

h5py = None
def inith5py():
    global h5py
    try:
        import h5py
    except ImportError:
        raise RuntimeError(
            "Cannot load Python h5py module. "
            "Please install before loading documents using HDF5 data.")

def bconv(s):
    """Hack for h5py byte problem with python3.
    https://github.com/h5py/h5py/issues/379

    Byte string attributes are not converted to normal strings."""

    if isinstance(s, cbytes):
        return s.decode('utf-8', 'replace')
    return s

def auto_deref_attr(attr, attrs, grp):
    """Automatic dereference any attributes which are references."""
    val = attrs[attr]
    # convert reference to a dataset
    if isinstance(val, h5py.Reference):
        # have to find root to dereference reference
        root = grp
        while root.name != '/':
            root = root.parent
        val = root[val]
    # convert dataset to an array
    if isinstance(val, h5py.Dataset):
        val = N.array(val)
    return bconv(val)

class ImportParamsHDF5(base.ImportParamsBase):
    """HDF5 file import parameters.

    Additional parameters:
     items: list of datasets and items to import
     namemap: map hdf datasets to veusz names
     slices: dict to map hdf names to slices
     twodranges: map hdf names to 2d range (minx, miny, maxx, maxy)
     twod_as_oned: set of hdf names to read 2d dataset as 1d dataset
     convert_datetime: map float or strings to datetime
    """

    defaults = {
        'items': None,
        'namemap': None,
        'slices': None,
        'twodranges': None,
        'twod_as_oned': None,
        'convert_datetime': None,
        }
    defaults.update(base.ImportParamsBase.defaults)

class LinkedFileHDF5(base.LinkedFileBase):
    """Links a HDF5 file to the data."""

    def createOperation(self):
        """Return operation to recreate self."""
        return OperationDataImportHDF5

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""
        self._saveHelper(
            fileobj,
            'ImportFileHDF5',
            ('filename', 'items'),
            relpath=relpath)

class _DataRead:
    """Data read from file during import.

    This is so we can store the original name and options stored in
    attributes from the file.
    """
    def __init__(self, origname, data, options):
        self.origname = origname
        self.data = data
        self.options = options

class OperationDataImportHDF5(base.OperationDataImportBase):
    """Import 1d, 2d, text or nd data from a HDF5 file."""

    descr = _("import HDF5 file")

    def readDataset(self, dataset, dsattrs, dsname, dsread):
        """Given hdf5 dataset, its attributes and name, get data and
        set it in dict dsread.

        dsread maps names to _DataRead object
        """

        # store options associated with dataset
        options = {}
        for a in dsattrs:
            if a[:4] == "vsz_":
                options[a] = auto_deref_attr(a, dsattrs, dataset)

        # find name for dataset
        if (self.params.namemap is not None and
            dsname in self.params.namemap ):
            name = self.params.namemap[dsname]
        else:
            if "vsz_name" in options:
                # override name using attribute
                name = options["vsz_name"]
            else:
                name = dsname.split("/")[-1].strip()

        # use full path if dataset already exists
        if name in dsread:
            name = dsname.strip()

        try:
            # implement slicing
            aslice = None
            if "vsz_slice" in options:
                s = fits_hdf5_helpers.convertTextToSlice(
                    options["vsz_slice"], len(dataset.shape))
                if s != -1:
                    aslice = s
            if self.params.slices and dsname in self.params.slices:
                aslice = self.params.slices[dsname]

            # finally return data
            objdata = fits_hdf5_helpers.convertDatasetToObject(
                dataset, aslice)
            dsread[name] = _DataRead(dsname, objdata, options)

        except fits_hdf5_helpers.ConvertError:
            pass

    def walkFile(self, item, dsread, names=None):
        """Walk an hdf file, adding datasets to dsread.

        If names is set to a list, only read names from list given
        """

        if isinstance(item, h5py.Dataset):
            try:
                dtype = item.dtype
            except TypeError:
                # not supported by h5py
                return

            if dtype.kind == 'V':
                # compound dataset - walk columns
                if not names:
                    names = item.dtype.names 

                for name in names:
                    attrs = fits_hdf5_helpers.filterAttrsByName(item.attrs, name)
                    self.readDataset(item[name], attrs, item.name+"/"+name,
                                     dsread)
            else:
                self.readDataset(item, item.attrs, item.name, dsread)

        elif isinstance(item, h5py.Group):
            if not names:
                names = sorted(item.keys())

            for dsname in names:
                try:
                    child = item[dsname]
                except KeyError:
                    # this does happen!
                    continue
                self.walkFile(child, dsread)

    def readDataFromFile(self):
        """Read data from hdf5 file and return a dict of names to data."""

        dsread = {}
        with h5py.File(self.params.filename, "r") as hdff:
            for hi in self.params.items:
                # workaround for h5py bug
                # using unicode names for groups/datasets does not work
                if not cpy3 and isinstance(hi, cunicode):
                    hi = hi.encode("utf-8")

                # lookup group/dataset in file
                names = [x for x in hi.split("/") if x != ""]
                node = hdff

                # Repeat until we get a dataset. Note: if we get a
                # dataset which names is not empty, this is a table
                # column, so we pass the remainder of names to
                # walkFile
                while names and not isinstance(node, h5py.Dataset):
                    node = node[names[0]]
                    names.pop(0)

                self.walkFile(node, dsread, names=names)
        return dsread

    def collectErrorBarDatasets(self, dsread):
        """Identify error bar datasets and separate out.
        Returns error bar datasets."""

        # separate out datasets with error bars
        # this a defaultdict of defaultdict with None as default
        errordatasets = collections.defaultdict(
            lambda: collections.defaultdict(lambda: None))
        for name in list(dsread):
            dr = dsread[name]
            ds = dr.data
            if not isinstance(ds, N.ndarray) or len(ds.shape) != 1:
                # skip non-numeric or 2d datasets
                continue

            for err in ('+', '-', '+-'):
                ln = len(err)+3
                if name[-ln:] == (' (%s)' % err):
                    refname = name[:-ln].strip()
                    if refname in dsread:
                        errordatasets[refname][err] = ds
                        del dsread[name]
                        break

        return errordatasets

    def numericDataToDataset(self, name, dread, errordatasets):
        """Convert numeric data to a veusz dataset."""

        data = dread.data

        ds = None
        if data.ndim == 1:
            if ( (self.params.convert_datetime and
                  dread.origname in self.params.convert_datetime) or
                 "vsz_convert_datetime" in dread.options ):

                try:
                    mode = self.params.convert_datetime[dread.origname]
                except (TypeError, KeyError):
                    mode = dread.options["vsz_convert_datetime"]

                if mode == 'unix':
                    data = utils.floatUnixToVeusz(data)
                ds = datasets.DatasetDateTime(data)

            else:
                # Standard 1D Import
                # handle any possible error bars
                args = { 'data': data,
                         'serr': errordatasets[name]['+-'],
                         'nerr': errordatasets[name]['-'],
                         'perr': errordatasets[name]['+'] }

                # find minimum length and cut down if necessary
                minlen = min([len(d) for d in cvalues(args)
                              if d is not None])
                for a in list(args):
                    if args[a] is not None and len(args[a]) > minlen:
                        args[a] = args[a][:minlen]

                ds = datasets.Dataset(**args)

        elif data.ndim == 2:
            # 2D dataset
            if ( ((self.params.twod_as_oned and
                   dread.origname in self.params.twod_as_oned) or
                  dread.options.get("vsz_twod_as_oned") ) and
                 data.shape[1] in (2,3) ):
                # actually a 1D dataset in disguise
                if data.shape[1] == 2:
                    ds = datasets.Dataset(data=data[:,0], serr=data[:,1])
                else:
                    ds = datasets.Dataset(
                        data=data[:,0], perr=data[:,1], nerr=data[:,2])
            else:
                # this really is a 2D dataset

                attrs = {}
                # find any ranges
                if "vsz_range" in dread.options:
                    r = dread.options["vsz_range"]
                    attrs["xrange"] = (r[0], r[2])
                    attrs["yrange"] = (r[1], r[3])
                for attr in ("xrange", "yrange", "xcent", "ycent",
                             "xedge", "yedge"):
                    if "vsz_"+attr in dread.options:
                        attrs[attr] = dread.options.get("vsz_"+attr)

                if ( self.params.twodranges and
                     dread.origname in self.params.twodranges ):
                    r = self.params.twodranges[dread.origname]
                    attrs["xrange"] = (r[0], r[2])
                    attrs["yrange"] = (r[1], r[3])

                # create the object
                ds = datasets.Dataset2D(data, **attrs)

        else:
            # N-dimensional dataset
            ds = datasets.DatasetND(data)

        return ds

    def textDataToDataset(self, name, dread):
        """Convert textual data to a veusz dataset."""

        data = dread.data

        if ( (self.params.convert_datetime and
              dread.origname in self.params.convert_datetime) or
             "vsz_convert_datetime" in dread.options ):

            try:
                fmt = self.params.convert_datetime[dread.origname]
            except (TypeError, KeyError):
                fmt = dread.options["vsz_convert_datetime"]

            if fmt.strip() == 'iso':
                fmt = 'YYYY-MM-DD|T|hh:mm:ss'

            try:
                datere = re.compile(utils.dateStrToRegularExpression(fmt))
            except Exception:
                raise base.ImportingError(
                    _("Could not interpret date-time syntax '%s'") % fmt)

            dout = N.empty(len(data), dtype=N.float64)
            for i, ditem in enumerate(data):
                ditem = bconv(ditem)
                try:
                    match = datere.match(ditem)
                    val = utils.dateREMatchToDate(match)
                except ValueError:
                    val = N.nan
                dout[i] = val

            ds = datasets.DatasetDateTime(dout)

        else:
            # unfortunately byte strings are returned in py3
            tdata = [bconv(d) for d in dread.data]

            # standard text dataset
            ds = datasets.DatasetText(tdata)

        return ds

    def doImport(self):
        """Do the import."""

        inith5py()
        par = self.params

        dsread = self.readDataFromFile()

        # find datasets which are error bars
        errordatasets = self.collectErrorBarDatasets(dsread)

        if par.linked:
            linkedfile = LinkedFileHDF5(par)
        else:
            linkedfile = None

        # create the veusz output datasets
        for name, dread in citems(dsread):
            if isinstance(dread.data, N.ndarray):
                # numeric
                ds = self.numericDataToDataset(name, dread, errordatasets)
            else:
                # text
                ds = self.textDataToDataset(name, dread)

            if ds is None:
                # error above
                continue

            ds.linked = linkedfile

            # finally set dataset in document
            fullname = par.prefix + name + par.suffix
            self.outdatasets[fullname] = ds

def ImportFileHDF5(comm, filename,
                   items,
                   namemap=None,
                   slices=None,
                   twodranges=None,
                   twod_as_oned=None,
                   convert_datetime=None,
                   prefix='', suffix='',
                   renames=None,
                   linked=False):
    """Import data from a HDF5 file

    items is a list of groups and datasets which can be imported.
    If a group is imported, all child datasets are imported.

    namemap maps an input dataset to a veusz dataset name. Special
    suffixes can be used on the veusz dataset name to indicate that
    the dataset should be imported specially.

    'foo (+)': import as +ve error for dataset foo
    'foo (-)': import as -ve error for dataset foo
    'foo (+-)': import as symmetric error for dataset foo

    slices is an optional dict specifying slices to be selected when
    importing. For each dataset to be sliced, provide a tuple of
    values, one for each dimension. The values should be a single
    integer to select that index, or a tuple (start, stop, step),
    where the entries are integers or None.

    twodranges is an optional dict giving data ranges for 2d
    datasets. It maps names to (minx, miny, maxx, maxy).

    twod_as_oned: optional set containing 2d datasets to attempt to
    read as 1d

    convert_datetime should be a dict mapping hdf name to specify
    date/time importing
      for a 1d numeric dataset
        if this is set to 'veusz', this is the number of seconds since
          2009-01-01
        if this is set to 'unix', this is the number of seconds since
          1970-01-01
       for a text dataset, this should give the format of the date/time,
          e.g. 'YYYY-MM-DD|T|hh:mm:ss' or 'iso' for iso format
 
    renames is a dict mapping old to new dataset names, to be renamed
    after importing

    linked specifies that the dataset is linked to the file.

    Attributes can be used in datasets to override defaults:
     'vsz_name': set to override name for dataset in veusz
     'vsz_slice': slice on importing (use format "start:stop:step,...")
     'vsz_range': should be 4 item array to specify x and y ranges:
                  [minx, miny, maxx, maxy]
     'vsz_xrange' / 'vsz_yrange': individual ranges for x and y
     'vsz_xcent' / 'vsz_ycent': arrays giving the centres of pixels
     'vsz_xedge' / 'vsz_yedge': arrays giving the edges of pixels
     'vsz_twod_as_oned': treat 2d dataset as 1d dataset with errors
     'vsz_convert_datetime': treat as date/time, set to one of the values
                             above.
    References to other datasets can be provided in thes attributes.
 
    For compound datasets these attributes can be given on a
    per-column basis using attribute names
    vsz_attributename_columnname.

    Returns: list of imported datasets
    """

    # lookup filename
    realfilename = comm.findFileOnImportPath(filename)

    params = ImportParamsHDF5(
        filename=realfilename,
        items=items,
        namemap=namemap,
        slices=slices,
        twodranges=twodranges,
        twod_as_oned=twod_as_oned,
        convert_datetime=convert_datetime,
        prefix=prefix, suffix=suffix,
        renames=renames,
        linked=linked)
    op = OperationDataImportHDF5(params)
    comm.document.applyOperation(op)

    if comm.verbose:
        print("Imported datasets %s" % ', '.join(op.outnames))
    return op.outnames

document.registerImportCommand("ImportFileHDF5", ImportFileHDF5)
