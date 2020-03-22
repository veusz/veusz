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

from .. import qtall as qt
from .. import setting
from .. import utils
from ..dialogs import importdialog
from ..compat import cstr

from . import base
from . import defn_hdf5

from . import fits_hdf5_tree
from . import fits_hdf5_helpers

def _(text, disambiguation=None, context="Import_HDF5"):
    return qt.QCoreApplication.translate(context, text, disambiguation)

# lazily imported
h5py = None

def dispname(child):
    """Get display name for HDF5 group/dataset."""
    return child.name.split('/')[-1]

def computedatatype(dtype):
    """Compute 'simple' datatype for tree widget from dtype."""

    datatype = 'invalid'
    k = dtype.kind
    if k in ('b', 'i', 'u', 'f'):
        datatype = 'numeric'
    elif k in ('S', 'a'):
        datatype = 'text'
    elif k == 'O':
        # FIXME: only supporting variable length strings so far
        typ = h5py.check_dtype(vlen=dtype)
        if typ is str:
            datatype = 'text'
    return datatype

def makedatanode(parent, ds):
    """Make a node in the tree for importable data."""

    # combine shape from dataset and column (if any)
    shape = tuple(list(ds.shape)+list(ds.dtype.shape))
    dtype = computedatatype(ds.dtype)

    vszattrs = {}
    for attr in ds.attrs:
        if attr[:4] == 'vsz_':
            vszattrs[attr] = defn_hdf5.bconv(ds.attrs[attr])

    return fits_hdf5_tree.FileDataNode(
        parent, ds.name, vszattrs, dtype, ds.dtype, shape, dispname(ds))

def addsub(parent, grp, datanodes):
    """Recursively descend through groups in the hdf5 file."""

    for child in sorted(grp.keys()):
        try:
            hchild = grp[child]
        except KeyError:
            continue
        if isinstance(hchild, h5py.Group):
            childnode = fits_hdf5_tree.FileGroupNode(
                parent, hchild.name, dispname(hchild))
            addsub(childnode, hchild, datanodes)
        elif isinstance(hchild, h5py.Dataset):
            try:
                dtype = hchild.dtype
            except TypeError:
                # raised if datatype not supported by h5py
                continue

            if dtype.kind == 'V':
                # compound data type - add a special group for
                # the compound, then its children
                childnode = fits_hdf5_tree.FileCompoundNode(
                    parent, hchild.name, dispname(hchild), hchild.shape)

                for field in sorted(hchild.dtype.fields.keys()):
                    # get types and shape for individual sub-parts
                    fdtype = hchild.dtype[field]
                    fdatatype = computedatatype(fdtype)
                    fshape = tuple(
                        list(hchild[field].shape)+list(fdtype.shape))

                    fattrs = fits_hdf5_helpers.filterAttrsByName(
                        hchild.attrs, field)
                    fnode = fits_hdf5_tree.FileDataNode(
                        childnode,
                        hchild.name+'/'+field,
                        fattrs,
                        fdatatype,
                        fdtype,
                        fshape,
                        field)

                    childnode.children.append(fnode)
                    datanodes.append(fnode)

            else:
                # normal dataset
                childnode = makedatanode(parent, hchild)
                datanodes.append(childnode)

        parent.children.append(childnode)

def constructTree(hdf5file):
    """Turn hdf5 file into a tree of nodes.

    Returns root and list of nodes showing datasets
    """

    datanodes = []
    root = fits_hdf5_tree.FileGroupNode(None, '', '/')
    addsub(root, hdf5file, datanodes)
    return root, datanodes

class ImportTabHDF5(importdialog.ImportTab):
    """Tab for importing HDF5 file."""

    resource = "import_hdf5.ui"
    filetypes = ('.hdf', '.hdf5', '.h5', '.he5')
    filefilter = _('HDF5 files')

    def __init__(self, *args):
        importdialog.ImportTab.__init__(self, *args)
        self.oldselection = (None, None)

    def showError(self, err):
        node = fits_hdf5_tree.ErrorNode(None, err)
        model = fits_hdf5_tree.GenericTreeModel(self, node, [''])
        self.hdftreeview.setModel(model)
        self.oldselection = (None, None)
        self.newCurrentSel(None, None)

    def loadUi(self):
        importdialog.ImportTab.loadUi(self)
        self.datanodes = []

        valid = qt.QDoubleValidator(self)
        valid.setNotation(qt.QDoubleValidator.ScientificNotation)
        for w in (self.hdftwodminx, self.hdftwodminy,
                  self.hdftwodmaxx, self.hdftwodmaxy):
            w.setValidator(valid)

        self.hdftextdate.addItems([
            _('No'),
            'YYYY-MM-DD|T|hh:mm:ss',
            'DD/MM/YY| |hh:mm:ss',
            'M/D/YY| |hh:mm:ss',
            ])

    def doPreview(self, filename, encoding):
        """Show file as tree."""

        global h5py
        if h5py is None:
            try:
                import h5py
            except ImportError:
                self.showError(_("Cannot load h5py module"))
                return False

        if not filename:
            self.showError(_("Cannot open file"))
            return False

        try:
            # check can be opened first
            with open(filename, "r") as f:
                pass
            with h5py.File(filename, "r") as f:
                self.rootnode, self.datanodes = constructTree(f)
        except IOError:
            self.showError(_("Cannot open file"))
            return False

        fits_hdf5_tree.setupTreeView(
            self.hdftreeview, self.rootnode, self.datanodes)
        self.hdftreeview.selectionModel().currentChanged.connect(
            self.newCurrentSel)

        # update widgets for options at bottom
        self.oldselection = (None, None)
        self.newCurrentSel(None, None)

        return True

    def showOptionsOneD(self, node):
        """Show options for 1d datasets on dialog."""

        dt = node.options.get('convert_datetime')
        self.hdfoneddate.setCurrentIndex({
            None: 0,
            'veusz': 1,
            'unix': 2}[dt])

    def showOptionsTwoD(self, node):
        """Update options for 2d datasets on dialog."""
        ranges = node.options.get('twodranges')
        if ranges is None:
            ranges = [None]*4

        for w, v in zip((self.hdftwodminx, self.hdftwodminy,
                         self.hdftwodmaxx, self.hdftwodmaxy), ranges):
            if v is None:
                w.clear()
            else:
                w.setText(setting.uilocale.toString(v))

        readas1d = node.options.get('twod_as_oned')
        self.hdftwodimport1d.setChecked(bool(readas1d))

    def showOptionsText(self, node):
        """Update options for text datasets on dialog."""

        text = node.options.get('convert_datetime')
        if not text:
            self.hdftextdate.setCurrentIndex(0)
        else:
            if self.hdftextdate.findText(text) == -1:
                self.hdftextdate.addItem(text)
            self.hdftextdate.lineEdit().setText(text)

    def updateOptionsOneD(self, node):
        """Read options for 1d datasets on dialog."""

        idx = self.hdfoneddate.currentIndex()
        if idx == 0:
            try:
                del node.options['convert_datetime']
            except KeyError:
                pass
        else:
            node.options['convert_datetime'] = {
                1: 'veusz',
                2: 'unix'}[idx]

    def updateOptionsTwoD(self, node):
        """Read options for 2d datasets on dialog."""

        rangeout = []
        for w in (self.hdftwodminx, self.hdftwodminy,
                  self.hdftwodmaxx, self.hdftwodmaxy):
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

        readas1d = self.hdftwodimport1d.isChecked()
        if readas1d:
            node.options['twod_as_oned'] = True
        else:
            try:
                del node.options['twod_as_oned']
            except KeyError:
                pass

    def updateOptionsText(self, node):
        """Read options for text datasets on dialog."""

        dtext = self.hdftextdate.currentText().strip()
        if self.hdftextdate.currentIndex() == 0 or dtext == '':
            try:
                del node.options['convert_datetime']
            except KeyError:
                pass
        else:
            node.options['convert_datetime'] = dtext

    def updateOptions(self):
        """Update options for nodes from dialog."""
        if self.oldselection[0] is not None:
            node, name = self.oldselection
            # update node options
            if name == 'oned':
                self.updateOptionsOneD(node)
            elif name == 'twod':
                self.updateOptionsTwoD(node)
            elif name == 'text':
                self.updateOptionsText(node)

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
                elif node.getDims() == 1 and node.numeric:
                    toshow = 'oned'
                    self.showOptionsOneD(node)
                elif node.text:
                    toshow = 'text'
                    self.showOptionsText(node)

        # so we know which options to update next
        self.oldselection = (node, toshow)

        for widget, name in (
            (self.hdfonedgrp, 'oned'),
            (self.hdftwodgrp, 'twod'),
            (self.hdftextgrp, 'text'),
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
        convert_datetime = {}

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
            if 'convert_datetime' in node.options:
                convert_datetime[node.fullname] = node.options['convert_datetime']

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
        params = defn_hdf5.ImportParamsHDF5(
            filename=filename,
            items=items,
            namemap=namemap,
            slices=slices,
            twodranges=twodranges,
            twod_as_oned=twod_as_oned,
            convert_datetime=convert_datetime,
            tags=tags,
            prefix=prefix, suffix=suffix,
            linked=linked,
            )

        op = defn_hdf5.OperationDataImportHDF5(params)

        try:
            # actually do the import
            doc.applyOperation(op)

            # inform user
            self.hdfimportstatus.setText(_("Import complete (%i datasets)") %
                                         len(op.outnames))

            # feature feedback
            utils.feedback.importcts['hdf5'] += 1

        except base.ImportingError as e:
            self.hdfimportstatus.setText(_("Error: %s") % cstr(e))

        qt.QTimer.singleShot(4000, self.hdfimportstatus.clear)

importdialog.registerImportTab(_('HDF&5'), ImportTabHDF5)
