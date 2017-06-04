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

from .. import qtall as qt
from .. import setting
from ..dialogs import importdialog
from ..compat import cstr

from . import base
from . import defn_fits

from . import fits_hdf5_tree
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
            pass

def makeimagenode(parent, hdu, idx, name, dispname, datanodes):
    """Node for image-like HDUs."""

    if hdu.data is None:
        return fits_hdf5_tree.EmptyDataNode(parent, name, dispname)

    attrs, colattrs = fits_hdf5_helpers.hduVeuszAttrs(hdu)
    datatype = 'numeric'
    shape = hdu.shape

    node = fits_hdf5_tree.FileDataNode(
        parent,
        name,
        attrs,
        datatype,
        str(hdu.header.get('BITPIX', '')),
        shape,
        dispname)
    datanodes.append(node)
    parent.children.append(node)

def constructTree(fitsfile):
    """Turn fits file into a tree of nodes.

    Returns root and list of nodes showing datasets
    """

    hdunames = fits_hdf5_helpers.getFITSHduNames(fitsfile)

    root = fits_hdf5_tree.FileGroupNode(None, '/', '/')

    # now iterate over file
    datanodes = []
    for idx, hdu in enumerate(fitsfile):
        hduname = hdunames[idx]
        dispname = '%s [%i]' % (hduname, idx)

        if hdu.is_image:
            # image hdu
            makeimagenode(root, hdu, idx, '/%s' % hduname, dispname, datanodes)

        elif hasattr(hdu, 'columns'):
            # parent for table
            tabshape = hdu.data.shape
            childnode = fits_hdf5_tree.FileCompoundNode(
                root, '/%s' % hduname, dispname, tabshape)
            root.children.append(childnode)

            attrs, colattrs = fits_hdf5_helpers.hduVeuszAttrs(hdu)

            # create new nodes for each column in table
            for col in hdu.columns:
                cname = col.name.lower()
                cdatatype, clen = fits_hdf5_helpers.convertFITSDataFormat(
                    col.format)
                cshape = tabshape if clen==1 else tuple(list(tabshape)+[clen])
                # attributes specific to column
                cattrs = colattrs.get(cname, {})

                cnode = fits_hdf5_tree.FileDataNode(
                    childnode,
                    '/%s/%s' % (hduname, cname),
                    cattrs,
                    cdatatype,
                    col.format,
                    cshape,
                    cname)
                childnode.children.append(cnode)
                datanodes.append(cnode)

    return root, datanodes

class ImportTabFITS(importdialog.ImportTab):
    """Tab for importing FITS file."""

    resource = "import_fits.ui"
    filetypes = ('.fits', '.fit', '.FITS', '.FIT')
    filefilter = _('FITS files')

    def showError(self, err):
        node = fits_hdf5_tree.ErrorNode(None, err)
        model = fits_hdf5_tree.GenericTreeModel(self, node, [''])
        self.fitstreeview.setModel(model)
        self.oldselection = (None, None)
        self.newCurrentSel(None, None)

    def loadUi(self):
        importdialog.ImportTab.loadUi(self)
        self.datanodes = []

        valid = qt.QDoubleValidator(self)
        valid.setNotation(qt.QDoubleValidator.ScientificNotation)
        for w in (self.fitstwodminx, self.fitstwodminy,
                  self.fitstwodmaxx, self.fitstwodmaxy):
            w.setValidator(valid)

    def doPreview(self, filename, encoding):
        """Show file as tree."""

        loadFITSModule()
        if fits is None:
            self.showError(_("Cannot load fits module"))
            return False

        if not filename:
            self.showError(_("Cannot open file"))
            return False

        try:
            # check can be opened first
            with open(filename, "r") as f:
                pass
            with fits.open(filename, "readonly") as f:
                self.rootnode, self.datanodes = constructTree(f)
        except IOError:
            self.showError(_("Cannot open file"))
            return False

        fits_hdf5_tree.setupTreeView(
            self.fitstreeview, self.rootnode, self.datanodes)
        self.fitstreeview.selectionModel().currentChanged.connect(
            self.newCurrentSel)

        # update widgets for options at bottom
        self.oldselection = (None, None)
        self.newCurrentSel(None, None)

        return True

    def showOptionsTwoD(self, node):
        """Update options for 2d datasets on dialog."""
        ranges = node.options.get('twodranges')
        if ranges is None:
            ranges = [None]*4

        for w, v in zip((self.fitstwodminx, self.fitstwodminy,
                         self.fitstwodmaxx, self.fitstwodmaxy), ranges):
            if v is None:
                w.clear()
            else:
                w.setText(setting.uilocale.toString(v))

        readas1d = node.options.get('twod_as_oned')
        self.fitstwodimport1d.setChecked(bool(readas1d))

        wcsmode = node.options.get('wcsmode', 'linear_wcs')
        idx = {
            'linear_wcs': 0,
            'pixel': 1,
            'pixel_wcs': 2,
            'fraction': 3,
            }[wcsmode]
        self.fitswcsmode.setCurrentIndex(idx)

    def updateOptionsTwoD(self, node):
        """Read options for 2d datasets on dialog."""

        rangeout = []
        for w in (self.fitstwodminx, self.fitstwodminy,
                  self.fitstwodmaxx, self.fitstwodmaxy):
            txt = w.text()
            val, ok = setting.uilocale.toDouble(txt)
            if not ok: val = None
            rangeout.append(val)

        if rangeout == [None, None, None, None]:
            try:
                del node.options['twodranges']
            except KeyError:
                pass

        elif None not in rangeout:
            # update
            node.options['twodranges'] = tuple(rangeout)

        readas1d = self.fitstwodimport1d.isChecked()
        if readas1d:
            node.options['twod_as_oned'] = True
        else:
            try:
                del node.options['twod_as_oned']
            except KeyError:
                pass

        wcsmode = ['linear_wcs', 'pixel', 'pixel_wcs', 'fraction'][
            self.fitswcsmode.currentIndex()]
        node.options['wcsmode'] = wcsmode

    def updateOptions(self):
        """Update options for nodes from dialog."""
        if self.oldselection[0] is not None:
            node, name = self.oldselection
            # update node options
            if name == 'twod':
                self.updateOptionsTwoD(node)

    def newCurrentSel(self, new, old):
        """New item selected in the tree."""

        self.updateOptions()

        # show appropriate widgets at bottom for editing options
        toshow = node = None
        if new is not None and new.isValid():
            node = new.internalPointer()
            if isinstance(node, fits_hdf5_tree.FileDataNode):
                if node.getDims() == 2 and node.numeric:
                    toshow = 'twod'
                    self.showOptionsTwoD(node)

        # so we know which options to update next
        self.oldselection = (node, toshow)

        for widget, name in (
            (self.fitstwodgrp, 'twod'),
            ):
            if name == toshow:
                widget.show()
            else:
                widget.hide()

    def doImport(self, doc, filename, linked, encoding, prefix, suffix, tags):
        """Import file."""

        self.updateOptions()

        namemap = {}
        slices = {}
        twodranges = {}
        twod_as_oned = set()
        wcsmodes = {}

        for node in self.datanodes:
            inname = node.importname.strip()
            if inname:
                namemap[node.fullname] = inname
            if node.slice:
                slices[node.fullname] = node.slice
            if 'twodranges' in node.options:
                twodranges[node.fullname]= node.options['twodranges']
            if 'twod_as_oned' in node.options:
                twod_as_oned.add(node.fullname)
            if ('wcsmode' in node.options and
                node.options['wcsmode'] != 'linear_wcs'):
                wcsmodes[node.fullname] = node.options['wcsmode']

        items = []
        def recursiveitems(node):
            if isinstance(node, fits_hdf5_tree.FileGroupNode):
                if node.grpimport:
                    items.append(node.fullname)
                else:
                    for c in node.children:
                        recursiveitems(c)
            else:
                if node.toimport:
                    items.append(node.fullname)

        recursiveitems(self.rootnode)

        prefix, suffix = self.dialog.getPrefixSuffix(filename)
        params = defn_fits.ImportParamsFITS(
            filename=filename,
            items=items,
            namemap=namemap,
            slices=slices,
            twodranges=twodranges,
            twod_as_oned=twod_as_oned,
            wcsmodes=wcsmodes,
            tags=tags,
            prefix=prefix, suffix=suffix,
            linked=linked,
            )

        op = defn_fits.OperationDataImportFITS(params)

        try:
            # actually do the import
            doc.applyOperation(op)

            # inform user
            self.fitsimportstatus.setText(
                _("Import complete (%i datasets)") % len(op.outnames))

        except base.ImportingError as e:
            self.fitsimportstatus.setText(_("Error: %s") % cstr(e))

        qt.QTimer.singleShot(4000, self.fitsimportstatus.clear)

importdialog.registerImportTab(_('FI&TS'), ImportTabFITS)
