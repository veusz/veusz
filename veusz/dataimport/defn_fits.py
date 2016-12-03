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

from .. import qtall as qt4
from ..compat import cbasestr, cbytes
from .. import document
from .. import datasets
from . import base

def _(text, disambiguation=None, context="Import_FITS"):
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class ImportParamsFITS(base.ImportParamsBase):
    """FITS file import parameters.

    Additional parameters:
     dsname: name of dataset
     hdu: name/number of hdu
     datacol: name of column
     symerrcol: symmetric error column
     poserrcol: positive error column
     negerrcol: negative error column
    """

    defaults = {
        'dsname': None,
        'hdu': None,
        'datacol': None,
        'symerrcol': None,
        'poserrcol': None,
        'negerrcol': None,
        'wcsmode': None,
        }
    defaults.update(base.ImportParamsBase.defaults)

class LinkedFileFITS(base.LinkedFileBase):
    """Links a FITS file to the data."""

    def createOperation(self):
        """Return operation to recreate self."""
        return OperationDataImportFITS

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""
        self._saveHelper(
            fileobj,
            'ImportFITSFile',
            ('dsname', 'filename', 'hdu'),
            relpath=relpath)

class OperationDataImportFITS(base.OperationDataImportBase):
    """Import 1d or 2d data from a fits file."""

    descr = _('import FITS file')

    def _import1d(self, hdu):
        """Import 1d data from hdu."""

        data = hdu.data
        datav = None
        symv = None
        posv = None
        negv = None

        # read the columns required
        p = self.params
        if p.datacol is not None:
            datav = data.field(p.datacol)
        if p.symerrcol is not None:
            symv = data.field(p.symerrcol)
        if p.poserrcol is not None:
            posv = data.field(p.poserrcol)
        if p.negerrcol is not None:
            negv = data.field(p.negerrcol)

        if len(datav) > 1 and isinstance(datav[0], cbasestr):
            # text dataset
            return datasets.DatasetText(list(datav))
        elif len(datav) > 1 and isinstance(datav[0], cbytes):
            return datasets.DatasetText(
                [x.decode('ascii') for x in datav])

        # numeric dataset
        return datasets.Dataset(data=datav, serr=symv, perr=posv, nerr=negv)

    def _import1dimage(self, hdu):
        """Import 1d image data form hdu."""
        return datasets.Dataset(data=hdu.data)

    def _import2dimage(self, hdu):
        """Import 2d image data from hdu."""

        p = self.params
        if ( p.datacol is not None or p.symerrcol is not None
             or p.poserrcol is not None
             or p.negerrcol is not None ):
            print("Warning: ignoring columns as import 2D dataset")

        header = hdu.header
        data = hdu.data

        try:
            # try to read WCS for image, and work out x/yrange
            wcs = [header[i] for i in ('CRVAL1', 'CRPIX1', 'CDELT1',
                                       'CRVAL2', 'CRPIX2', 'CDELT2')]

            if p.wcsmode == 'pixel':
                # no coordinate system - just pixel values
                rangex = rangey = None
            elif p.wcsmode == 'pixel_wcs':
                rangex = (data.shape[1]-wcs[1], 0-wcs[1])
                rangey = (0-wcs[4], data.shape[0]-wcs[4])
            elif p.wcsmode == 'fraction':
                rangex = rangey = (0., 1.)
            else:
                # linear wcs mode (linear_wcs)
                rangex = ( (data.shape[1]-wcs[1])*wcs[2] + wcs[0],
                           (0-wcs[1])*wcs[2] + wcs[0])
                rangey = ( (0-wcs[4])*wcs[5] + wcs[3],
                           (data.shape[0]-wcs[4])*wcs[5] + wcs[3] )

                rangex = (rangex[1], rangex[0])

        except KeyError:
            # no / broken wcs
            rangex = rangey = None

        return datasets.Dataset2D(data, xrange=rangex, yrange=rangey)

    def _importnd(self, hdu):
        """Import 2d image data from hdu."""

        p = self.params
        if ( p.datacol is not None or p.symerrcol is not None
             or p.poserrcol is not None
             or p.negerrcol is not None ):
            print("Warning: ignoring columns as import nD dataset")

        return datasets.DatasetND(hdu.data)

    def doImport(self):
        """Do the import."""

        try:
            from astropy.io import fits as pyfits
        except ImportError:
            try:
                import pyfits
            except ImportError:
                raise base.ImportingError(
                    'Either Astropy or PyFITS is required '
                    'to import data from FITS files')

        p = self.params

        with pyfits.open(p.filename, 'readonly') as f:
            hdu = f[p.hdu]

            if ( isinstance(hdu, pyfits.TableHDU) or
                 isinstance(hdu, pyfits.BinTableHDU) ):
                # import table
                ds = self._import1d(hdu)

            else:
                # import image
                naxis = hdu.header.get('NAXIS')
                if naxis == 1:
                    ds = self._import1dimage(hdu)
                elif naxis == 2:
                    ds = self._import2dimage(hdu)
                else:
                    ds = self._importnd(hdu)

        if p.linked:
            ds.linked = LinkedFileFITS(p)
        outname = p.dsname.strip()
        self.outdatasets[outname] = ds

def ImportFITSFile(comm, dsname, filename, hdu,
                   datacol = None, symerrcol = None,
                   poserrcol = None, negerrcol = None,
                   wcsmode = None,
                   renames = None,
                   linked = False):
    """Import data from a FITS file

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

    params = ImportParamsFITS(
        dsname=dsname, filename=realfilename, hdu=hdu,
        datacol=datacol, symerrcol=symerrcol,
        poserrcol=poserrcol, negerrcol=negerrcol,
        wcsmode=wcsmode,
        linked=linked)
    op = OperationDataImportFITS(params)
    comm.document.applyOperation(op)

    if comm.verbose:
        print("Imported datasets %s" % ', '.join(op.outnames))
    return op.outnames

document.registerImportCommand('ImportFITSFile', ImportFITSFile)
