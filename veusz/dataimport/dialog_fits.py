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
from ..dialogs import importdialog
from ..compat import crange
from . import defn_fits

def _(text, disambiguation=None, context="Import_FITS"):
    return qt4.QCoreApplication.translate(context, text, disambiguation)

pyfits = None
class ImportTabFITS(importdialog.ImportTab):
    """Tab for importing from a FITS file."""

    resource = 'import_fits.ui'
    filetypes = ('.fits', '.fit')
    filefilter = _('FITS files')

    def loadUi(self):
        importdialog.ImportTab.loadUi(self)
        # if different items are selected in fits tab
        self.fitshdulist.itemSelectionChanged.connect(
            self.slotFitsUpdateCombos)
        self.fitsdatasetname.editTextChanged.connect(
            self.dialog.enableDisableImport)
        self.fitsdatacolumn.currentIndexChanged.connect(
            self.dialog.enableDisableImport)
        self.dialog.prefixcombo.editTextChanged.connect(
            self.dialog.enableDisableImport)
        self.dialog.suffixcombo.editTextChanged.connect(
            self.dialog.enableDisableImport)

        self.fitswcsmode.defaultlist = [
            _('Pixel (simple)'), _('Pixel (WCS)'), _('Fractional'),
            _('Linear (WCS)')
            ]

    def reset(self):
        """Reset controls."""
        self.fitsdatasetname.setEditText("")
        for c in ('data', 'sym', 'pos', 'neg'):
            cntrl = getattr(self, 'fits%scolumn' % c)
            cntrl.setCurrentIndex(0)
        self.fitswcsmode.setCurrentIndex(0)

    def doPreview(self, filename, encoding):
        """Set up controls for FITS file."""

        # load pyfits if available
        global pyfits
        if pyfits is None:
            try:
                from astropy.io import fits as PF
                pyfits = PF
            except ImportError:
                try:
                    import pyfits as PF
                    pyfits = PF
                except ImportError:
                    pyfits = None

        # if it isn't
        if pyfits is None:
            self.fitslabel.setText(
                _('FITS file support requires that astropy or PyFITS '
                  'are installed'))
            return False

        # try to identify fits file
        try:
            f = pyfits.open(filename)
            f[0].header
            f.close()
        except Exception:
            self.clearFITSView()
            return False

        self.updateFITSView(filename)
        return True

    def clearFITSView(self):
        """If invalid filename, clear fits preview."""
        self.fitshdulist.clear()
        for c in ('data', 'sym', 'pos', 'neg'):
            cntrl = getattr(self, 'fits%scolumn' % c)
            cntrl.clear()
            cntrl.setEnabled(False)

    def updateFITSView(self, filename):
        """Update the fits file details in the import dialog."""
        f = pyfits.open(str(filename), 'readonly')
        l = self.fitshdulist
        l.clear()

        # this is so we can lookup item attributes later
        self.fitsitemdata = []
        items = []
        for hdunum, hdu in enumerate(f):
            header = hdu.header
            hduitem = qt4.QTreeWidgetItem([str(hdunum), hdu.name])
            data = []
            try:
                # if this fails, show an image
                cols = hdu.columns

                # it's a table
                data = ['table', cols]
                rows = header['NAXIS2']
                descr = _('Table (%i rows)') % rows

            except AttributeError:
                # this is an image
                naxis = header['NAXIS']
                data = ['image']
                dims = [ str(header['NAXIS%i' % (i+1)])
                         for i in crange(naxis) ]
                dims = '*'.join(dims)
                if dims:
                    dims = '(%s)' % dims
                descr = _('%iD image %s') % (naxis, dims)

            hduitem = qt4.QTreeWidgetItem([str(hdunum), hdu.name, descr])
            items.append(hduitem)
            self.fitsitemdata.append(data)

        if items:
            l.addTopLevelItems(items)
            l.setCurrentItem(items[0])

    def slotFitsUpdateCombos(self):
        """Update list of fits columns when new item is selected."""

        items = self.fitshdulist.selectedItems()
        if len(items) != 0:
            item = items[0]
            hdunum = int( str(item.text(0)) )
        else:
            item = None
            hdunum = -1

        cols = ['N/A']
        enablecolumns = False
        if hdunum >= 0:
            data = self.fitsitemdata[hdunum]
            if data[0] == 'table':
                enablecolumns = True
                cols = ['None']
                cols += ['%s (%s)' %
                         (i.name, i.format) for i in data[1]]

        for c in ('data', 'sym', 'pos', 'neg'):
            cntrl = getattr(self, 'fits%scolumn' % c)
            cntrl.setEnabled(enablecolumns)
            cntrl.clear()
            cntrl.addItems(cols)

        self.dialog.enableDisableImport()

    def okToImport(self):
        """Check validity of Fits import."""

        items = self.fitshdulist.selectedItems()
        if len(items) != 0:
            item = items[0]
            hdunum = int( str(item.text(0)) )

            # any name for the dataset?
            filename = self.dialog.filenameedit.text()
            prefix, suffix = self.dialog.getPrefixSuffix(filename)

            if not ( prefix + self.fitsdatasetname.text() +
                     suffix ).strip():
                return False

            # if a table, need selected item
            data = self.fitsitemdata[hdunum]
            if data[0] != 'image' and self.fitsdatacolumn.currentIndex() == 0:
                return False

            return True
        return False

    def doImport(self, doc, filename, linked, encoding, prefix, suffix, tags):
        """Import fits file."""

        item = self.fitshdulist.selectedItems()[0]
        hdunum = int( str(item.text(0)) )
        data = self.fitsitemdata[hdunum]

        name = (prefix + self.fitsdatasetname.text() + suffix).strip()

        wcsmode = None
        if data[0] == 'table':
            # get list of appropriate columns
            cols = []

            # get data from controls
            for c in ('data', 'sym', 'pos', 'neg'):
                cntrl = getattr(self, 'fits%scolumn' % c)

                i = cntrl.currentIndex()
                if i == 0:
                    cols.append(None)
                else:
                    cols.append(data[1][i-1].name)

        else:
            # item is an image, so no columns
            cols = [None]*4
            wcsmode = ('pixel', 'pixel_wcs', 'fraction', 'linear_wcs')[
                self.fitswcsmode.currentIndex()]

        # construct operation to import fits
        params = defn_fits.ImportParamsFITS(
            dsname=name,
            filename=filename,
            hdu=hdunum,
            datacol=cols[0],
            symerrcol=cols[1],
            poserrcol=cols[2],
            negerrcol=cols[3],
            wcsmode=wcsmode,
            tags=tags,
            linked=linked,
            )

        op = defn_fits.OperationDataImportFITS(params)

        # actually do the import
        doc.applyOperation(op)

        # inform user
        self.fitsimportstatus.setText(_("Imported dataset '%s'") % name)
        qt4.QTimer.singleShot(2000, self.fitsimportstatus.clear)

importdialog.registerImportTab(_('FI&TS'), ImportTabFITS)
