#    Copyright (C) 2017 Jeremy S. Sanders
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

from __future__ import division, print_function, absolute_import

import collections

import numpy as N
from .. import qtall as qt
from ..compat import citems, cvalues
from .. import document
from .. import datasets
from . import base
from . import fits_hdf5_helpers

def _(text, disambiguation=None, context="Import_FITS"):
    return qt.QCoreApplication.translate(context, text, disambiguation)

# lazily imported
fits = None

def loadFITSModule():
    global fits
    try:
        from astropy.io import fits
    except ImportError:
        try:
            import pyfits as fits
        except ImportError:
            raise RuntimeError(
                "Cannot load astropy.io.fits or pyfits module. "
                "Please install before loading documents with FITS data.")

class ImportParamsFITS(base.ImportParamsBase):
    """HDF5 file import parameters.

    Additional parameters:
     items: list of datasets and items to import
     namemap: map hdf datasets to veusz names
     slices: dict to map hdf names to slices
     twodranges: map hdf names to 2d range (minx, miny, maxx, maxy)
     twod_as_oned: set of hdf names to read 2d dataset as 1d dataset
     wcsmodes: how to treat wcs when importing
    """

    defaults = {
        'items': None,
        'namemap': None,
        'slices': None,
        'twodranges': None,
        'twod_as_oned': None,
        'wcsmodes': None,
        }
    defaults.update(base.ImportParamsBase.defaults)

class LinkedFileFITS(base.LinkedFileBase):
    """Links a HDF5 file to the data."""

    def createOperation(self):
        """Return operation to recreate self."""
        return OperationDataImportFITS

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""
        self._saveHelper(
            fileobj,
            'ImportFileFITS',
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

class OperationDataImportFITS(base.OperationDataImportBase):
    """Import 1d, 2d, text or nd data from a fits file."""

    descr = _("import FITS file")

    def convertDataset(self, data, options, dsname, dsread):
        """Given some data read from a file, its attributes and name, get data
        and set it in dict dsread.

        dsread maps names to _DataRead object

        """

        # find name for dataset
        if (self.params.namemap is not None and
            dsname in self.params.namemap ):
            name = self.params.namemap[dsname]
        else:
            if "name" in options:
                # override name using attribute
                name = options["name"]
            else:
                name = dsname.split("/")[-1].strip()

        # use full path if dataset already exists
        if name in dsread:
            name = dsname.strip()

        try:
            # implement slicing
            aslice = None
            if "slice" in options:
                s = fits_hdf5_helpers.convertTextToSlice(
                    options["slice"], len(data.shape))
                if s != -1:
                    aslice = s
            if self.params.slices and dsname in self.params.slices:
                aslice = self.params.slices[dsname]

            # finally return data
            objdata = fits_hdf5_helpers.convertDatasetToObject(
                data, aslice)
            dsread[name] = _DataRead(dsname, objdata, options)

        except fits_hdf5_helpers.ConvertError:
            pass

    def getImageWCS(self, hdu, dsname, attr):
        """Get WCS values for rangex, rangey, applying appropriate mode."""

        # WCS only supported for 2D datasets
        if len(hdu.shape) != 2:
            return

        # mode is None for default
        mode = "linear_wcs"
        if self.params.wcsmodes:
            mode = self.params.wcsmodes.get(dsname, "linear_wcs")
        if 'wcsmode' in attr:
            mode = attr['wcsmode']

        # standard linear wcs keywords
        wcs = [hdu.header.get(x, None) for x in (
            'CRVAL1', 'CRPIX1', 'CDELT1',
            'CRVAL2', 'CRPIX2', 'CDELT2')]

        if mode == "pixel" or (None in wcs and "wcs" in mode):
            rangex = rangey = None
        elif mode == "fraction":
            rangex = rangey = (0., 1.)
        elif mode == "pixel_wcs":
            rangex = (hdu.shape[1]-wcs[1], 0-wcs[1])
            rangey = (0-wcs[4], hdu.shape[0]-wcs[4])
        elif mode == "linear_wcs":
            rangex = (
                (0.5-wcs[1])*wcs[2] + wcs[0],
                (hdu.shape[1]+0.5-wcs[1])*wcs[2] + wcs[0])
            rangey = (
                (0.5-wcs[4])*wcs[5] + wcs[3],
                (hdu.shape[0]+0.5-wcs[4])*wcs[5] + wcs[3])
        else:
            raise RuntimeError("Invalid WCS mode")

        if rangex and "xrange" not in attr:
            attr["xrange"] = rangex
        if rangey and "yrange" not in attr:
            attr["yrange"] = rangey

    def readHduImage(self, hdu, dsname, dsread):
        """Read an image in a HDU."""

        attr, colattr = fits_hdf5_helpers.hduVeuszAttrs(hdu)
        self.getImageWCS(hdu, dsname, attr)

        self.convertDataset(hdu.data, attr, dsname, dsread)

    def readTableColumn(self, hdu, dsname, dsread):
        """Read a specific column from a FITS file."""

        # dsname is /hduname/colname
        colname = dsname.split('/')[-1].strip().lower()

        # get attributes for column
        attr, colattr = fits_hdf5_helpers.hduVeuszAttrs(hdu)
        if colname in colattr:
            attr.update(colattr[colname])

        data = hdu.data.field(colname)
        self.convertDataset(data, attr, dsname, dsread)

    def walkHdu(self, hdu, dsname, dsread):
        """Import everything from a table HDU."""

        if hdu.data is None:
            # ignore empty HDU
            pass
        elif hdu.is_image:
            # Primary or Image HDU
            self.readHduImage(hdu, dsname, dsread)
        else:
            # Table HDU
            for col in hdu.data.columns:
                self.readTableColumn(
                    hdu, '%s/%s' % (dsname, col.name.lower()), dsread)

    def walkFile(self, fitsf, hdunames, dsread):
        """Import everything from a fits file."""

        for hdu, name in zip(fitsf, hdunames):
            self.walkHdu(hdu, '/%s' % name, dsread)

    def readDataFromFile(self):
        """Read data from fits file and return a dict of names to data."""

        dsread = {}
        with fits.open(self.params.filename, 'readonly') as fitsf:
            hdunames = fits_hdf5_helpers.getFITSHduNames(fitsf)

            for item in self.params.items:
                parts = [p.strip() for p in item.split('/') if p.strip()]

                if not parts:
                    # / or empty
                    self.walkFile(fitsf, hdunames, dsread)
                elif len(parts) >= 1:
                    try:
                        idx = hdunames.index(parts[0])
                    except ValueError:
                        raise RuntimeError(
                            "Cannot find HDU '%s' in FITS file" % parts[0])
                    hdu = fitsf[idx]
                    if len(parts) == 1:
                        # read whole HDU
                        self.walkHdu(hdu, '/%s' % parts[0], dsread)
                    elif len(parts) == 2:
                        # column of table
                        self.readTableColumn(
                            hdu, '/%s/%s' % (parts[0], parts[1]), dsread)
                    else:
                        raise RuntimeError(
                            'Too many parts in FITS dataset name')

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
                  dread.options.get("twod_as_oned") ) and
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
                if "range" in dread.options:
                    r = dread.options["range"]
                    attrs["xrange"] = (r[0], r[2])
                    attrs["yrange"] = (r[1], r[3])
                for attr in ("xrange", "yrange", "xcent", "ycent",
                             "xedge", "yedge"):
                    if attr in dread.options:
                        attrs[attr] = dread.options.get(attr)

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

        tdata = list(dread.data)
        return datasets.DatasetText(tdata)

    def doImport(self):
        """Do the import."""

        loadFITSModule()
        par = self.params

        dsread = self.readDataFromFile()

        # find datasets which are error bars
        errordatasets = self.collectErrorBarDatasets(dsread)

        if par.linked:
            linkedfile = LinkedFileFITS(par)
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

def ImportFileFITS(
        comm,
        filename,
        items,
        namemap=None,
        slices=None,
        twodranges=None,
        twod_as_oned=None,
        wcsmodes=None,
        prefix='', suffix='',
        renames=None,
        linked=False):
    """Import data from a FITS file

    items is a list of datasets to be imported.
    items are formatted like the following:
     '/': import whole file
     '/hduname': import whole HDU (image or table)
     '/hduname/column': import column from table HDU
    all values in items should be lower case.

    HDU names have to follow a Veusz-specific naming. If the HDU has a
    standard name (e.g. primary or events), then this is used.  If the
    HDU has a EXTVER keyword then this number is appended to this
    name.  An extra number is appended if this name is not unique.  If
    the HDU has no name, then the name used should be 'hduX', where X
    is the HDU number (0 is the primary HDU).

    namemap maps an input dataset (using the scheme above for items)
    to a Veusz dataset name. Special suffixes can be used on the Veusz
    dataset name to indicate that the dataset should be imported
    specially.

    'foo (+)': import as +ve error for dataset foo
    'foo (-)': import as -ve error for dataset foo
    'foo (+-)': import as symmetric error for dataset foo

    slices is an optional dict specifying slices to be selected when
    importing. For each dataset to be sliced, provide a tuple of
    values, one for each dimension. The values should be a single
    integer to select that index, or a tuple (start, stop, step),
    where the entries are integers or None.

    twodranges is an optional dict giving data ranges for 2D
    datasets. It maps names to (minx, miny, maxx, maxy).

    twod_as_oned: optional set containing 2D datasets to attempt to
    read as 1D, treating extra columns as error bars

    wcsmodes is an optional dict specfying the WCS import mode for 2D
    datasets in HDUs. The keys are '/hduname' and the values can be
      'pixel':      number pixel range from 0 to maximum (default)
      'pixel_wcs':  pixel number relative to WCS reference pixel
      'linear_wcs': linear coordinate system from the WCS keywords
      'fraction':   fractional values from 0 to 1.

    renames is an optional dict mapping old to new dataset names, to
    be renamed after importing

    linked specifies that the dataset is linked to the file.

    Values under the VEUSZ header keyword can be used to override defaults:
     'name': override name for dataset
     'slice': slice on importing (use format "start:stop:step,...")
     'range': should be 4 item array to specify x and y ranges:
       [minx, miny, maxx, maxy]
     'xrange' / 'yrange': individual ranges for x and y
     'xcent' / 'ycent': arrays giving the centres of pixels
     'xedge' / 'yedge': arrays giving the edges of pixels
     'twod_as_oned': treat 2d dataset as 1d dataset with errors
     'wcsmode': use specific WCS mode for dataset (see values above)
    These are specified under the VEUSZ header keyword in the form
      KEY=VALUE
    or for column-specific values
      COLUMNNAME: KEY=VALUE
 
    Returns: list of imported datasets

    """

    # lookup filename
    realfilename = comm.findFileOnImportPath(filename)

    params = ImportParamsFITS(
        filename=realfilename,
        items=items,
        namemap=namemap,
        slices=slices,
        twodranges=twodranges,
        twod_as_oned=twod_as_oned,
        wcsmodes=wcsmodes,
        prefix=prefix, suffix=suffix,
        renames=renames,
        linked=linked)
    op = OperationDataImportFITS(params)
    comm.document.applyOperation(op)

    if comm.verbose:
        print("Imported datasets %s" % ', '.join(op.outnames))
    return op.outnames

def ImportFITSFile(comm, dsname, filename, hdu,
                   datacol = None, symerrcol = None,
                   poserrcol = None, negerrcol = None,
                   wcsmode = None,
                   renames = None,
                   linked = False):
    """Compatibility wrapper for ImportFileFITS.
    Do not use this in new code.

    Import data from a FITS file

    dsname is the name of the dataset
    filename is name of the fits file to open
    hdu is the number/name of the hdu to access

    if the hdu is a table, datacol, symerrcol, poserrcol and negerrcol
    specify the columns containing the data, symmetric error,
    positive and negative errors.

    wcsmode is one of ('pixel', 'pixel_wcs' or 'linear_wcs'). None
    gives 'linear_wcs'. 'pixel' mode just gives pixel values from
    0 to maximum. 'pixel_wcs' is the pixel number relative to the
    wcs reference pixel. 'linear_wcs' takes the wcs coordinate,
    assuming a linear coordinate system. 'fraction' assumes
    fractional values from 0 to 1.

    renames: dict mapping old to new names if datasets are to
      be renamed after import

    linked specfies that the dataset is linked to the file

    Returns: list of imported datasets

    """

    # lookup filename
    realfilename = comm.findFileOnImportPath(filename)

    # work out new HDU name by looking up what would have been chosen
    # before
    loadFITSModule()
    with fits.open(realfilename, 'readonly') as fitsf:
        hdunames = fits_hdf5_helpers.getFITSHduNames(fitsf)
        hdu = fitsf[hdu]
        idx = list(fitsf).index(hdu)
        hduname = hdunames[idx]

    if datacol is None:
        # default is pixel here
        if wcsmode is None:
            wcsmode = 'pixel'

        # image mode
        fullname = '/'+hduname
        return ImportFileFITS(
            comm,
            filename,
            [fullname],
            namemap={fullname: dsname},
            renames=renames,
            wcsmodes={fullname: wcsmode},
            linked=linked,
            )

    else:
        # handle tables
        dsnames = []
        namemap = {}

        name = '/%s/%s' % (hduname, datacol)
        namemap[name] = dsname
        dsnames.append(name)

        # handle conversion of errors
        if symerrcol:
            name = '/%s/%s' % (hduname, symerrcol)
            namemap[name] = '%s (+-)' % dsname
            dsnames.append(name)
        if poserrcol:
            name = '/%s/%s' % (hduname, poserrcol)
            namemap[name] = '%s (+)' % dsname
            dsnames.append(name)
        if negerrcol:
            name = '/%s/%s' % (hduname, negerrcol)
            namemap[name] = '%s (-)' % dsname
            dsnames.append(name)

        return ImportFileFITS(
            comm,
            filename,
            dsnames,
            namemap=namemap,
            renames=renames,
            linked=linked,
            )

# new import command
document.registerImportCommand("ImportFileFITS", ImportFileFITS)
# compatibility with old fits import
document.registerImportCommand("ImportFITSFile", ImportFITSFile, filenamearg=1)
