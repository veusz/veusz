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


from __future__ import division, print_function

from .. import qtall as qt4
from .. import setting
from ..dialogs import importdialog
from ..compat import cstr

from . import base
from . import defn_hdf5

from . import fits_hdf5_tree
from . import fits_hdf5_helpers

def _(text, disambiguation=None, context="Import_FITS"):
    return qt4.QCoreApplication.translate(context, text, disambiguation)

# lazily imported
fits = None

def loadFitsModule():
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

    vszattrs = {}
    for n in hdu.header:
        if n[:4].lower() == 'vsz_':
            vszattrs[n.lower()] = hdu.header[n]
    dtype = hdu.data.dtype
    datatype = computedatatype(dtype)
    shape = hdu.data.shape

    node = fits_hdf5_tree.FileDataNode(
        parent,
        name,
        vszattrs,
        datatype,
        dtype,
        shape,
        dispname)
    datanodes.append(node)
    parent.children.append(node)

def constructTree(fitsfile):
    """Turn fits file into a tree of nodes.

    Returns root and list of nodes showing datasets
    """

    hdunames = fits_hdf5_helpers.getFitsHduNames(fitsfile)

    root = fits_hdf5_tree.FileGroupNode(None, '', '/')

    # now iterate over file
    datanodes = []
    for idx, hdu in enumerate(fitsfile):
        name = hdunames[idx]
        dispname = '%s (%i)' % (name, idx+1)

        if hdu.is_image:
            # image hdu
            makeimagenode(root, hdu, idx, name, dispname, datanodes)

        elif hasattr(hdu, 'columns'):
            # parent for table
            tabshape = hdu.data.shape
            childnode = fits_hdf5_tree.FileCompoundNode(
                root, name, dispname, tabshape)
            root.children.append(childnode)

            # for per-column attributes
            attrs = {}
            for k in hdu.header:
                attrs[k.lower()] = hdu.header[k]

            # create new nodes for each column in table
            for col in hdu.columns:
                cdtype = col.dtype
                cname = col.name.lower()

                cdatatype, clen = fits_hdf5_helpers.convertFitsDataFormat(
                    col.format)
                cshape = tabshape if clen==1 else tuple(list(tabshape)+[clen])
                cattrs = fits_hdf5_helpers.filterAttrsByName(attrs, cname)

                cnode = fits_hdf5_tree.FileDataNode(
                    childnode,
                    name+'/'+cname,
                    cattrs,
                    cdatatype,
                    col.format,
                    cshape,
                    cname)
                childnode.children.append(cnode)
                datanodes.append(cnode)

    return root, datanodes

class ImportTabFits(importdialog.ImportTab):
    """Tab for importing Fits file."""

    resource = "import_fits2.ui"
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

        valid = qt4.QDoubleValidator(self)
        valid.setNotation(qt4.QDoubleValidator.ScientificNotation)
        for w in (self.fitstwodminx, self.fitstwodminy,
                  self.fitstwodmaxx, self.fitstwodmaxy):
            w.setValidator(valid)

    def doPreview(self, filename, encoding):
        """Show file as tree."""

        loadFitsModule()
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
        params = defn_hdf5.ImportParamsFits(
            filename=filename,
            items=items,
            namemap=namemap,
            slices=slices,
            twodranges=twodranges,
            twod_as_oned=twod_as_oned,
            tags=tags,
            prefix=prefix, suffix=suffix,
            linked=linked,
            )

        op = defn_hdf5.OperationDataImportFits(params)

        try:
            # actually do the import
            doc.applyOperation(op)

            # inform user
            self.fitsimportstatus.setText(
                _("Import complete (%i datasets)") % len(op.outnames))

        except base.ImportingError as e:
            self.fitsimportstatus.setText(_("Error: %s") % cstr(e))

        qt4.QTimer.singleShot(4000, self.fitsimportstatus.clear)

importdialog.registerImportTab(_('FI&TS'), ImportTabFits)
