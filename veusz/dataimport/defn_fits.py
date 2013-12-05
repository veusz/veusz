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
from ..compat import citems, cstr, crepr
from .. import document
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

        p = self.params
        args = [p.dsname, self._getSaveFilename(relpath), p.hdu]
        args = [crepr(i) for i in args]
        for param, val in ( ("datacol", p.datacol),
                            ("symerrcol", p.symerrcol),
                            ("poserrcol", p.poserrcol),
                            ("negerrcol", p.negerrcol),
                            ("wcsmode", p.wcsmode),
                            ):
            if val is not None:
                args.append("%s=%s" % (param, crepr(val)))
        args.append("linked=True")

        fileobj.write("ImportFITSFile(%s)\n" % ", ".join(args))

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

        # actually create the dataset
        return document.Dataset(data=datav, serr=symv, perr=posv, nerr=negv)

    def _import1dimage(self, hdu):
        """Import 1d image data form hdu."""
        return document.Dataset(data=hdu.data)

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

        return document.Dataset2D(data, xrange=rangex, yrange=rangey)

    def doImport(self, document):
        """Do the import."""

        try:
            from astropy.io import fits as pyfits
        except ImportError:
            try:
                import pyfits
            except ImportError:
                raise RuntimeError('Either Astropy or PyFITS is required '
                                   'to import data from FITS files')

        p = self.params
        f = pyfits.open( str(p.filename), 'readonly')
        hdu = f[p.hdu]

        try:
            # raise an exception if this isn't a table therefore is an image
            hdu.get_coldefs()
            ds = self._import1d(hdu)

        except AttributeError:
            naxis = hdu.header.get('NAXIS')
            if naxis == 1:
                ds = self._import1dimage(hdu)
            elif naxis == 2:
                ds = self._import2dimage(hdu)
            else:
                raise RuntimeError("Cannot import images with %i dimensions" % naxis)
        f.close()

        if p.linked:
            ds.linked = LinkedFileFITS(self.params)
        if p.dsname in document.data:
            self.olddataset = document.data[p.dsname]
        else:
            self.olddataset = None
        document.setData(p.dsname.strip(), ds)
        self.outdatasets.append(p.dsname)

def ImportFITSFile(comm, dsname, filename, hdu,
                   datacol = None, symerrcol = None,
                   poserrcol = None, negerrcol = None,
                   wcsmode = None,
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

    linked specfies that the dataset is linked to the file
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

document.registerImportCommand('ImportFITSFile', ImportFITSFile)
