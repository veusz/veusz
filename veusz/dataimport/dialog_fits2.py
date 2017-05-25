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

def constructTree(fitsfile):
    """Turn fits file into a tree of nodes.

    Returns root and list of nodes showing datasets
    """

    datanodes = []

    def computedatatype(dtype):
        datatype = 'invalid'
        k = dtype.kind
        if k in ('b', 'i', 'u', 'f'):
            datatype = 'numeric'
        elif k in ('S', 'a'):
            datatype = 'text'
        return datatype

    def makedatanode(parent, ds):
        # combine shape from dataset and column (if any)
        shape = tuple(list(ds.shape)+list(ds.dtype.shape))
        dtype = computedatatype(ds.dtype)

        vszattrs = {}
        for attr in ds.attrs:
            if attr[:4] == 'vsz_':
                vszattrs[attr] = defn_hdf5.bconv(ds.attrs[attr])

        return fits_hdf5_tree.FileDataNode(
            parent, ds.name, vszattrs, dtype, ds.dtype, shape)

    def addsub(parent, grp):
        """To recursively iterate over each parent."""
        for child in sorted(grp.keys()):
            try:
                hchild = grp[child]
            except KeyError:
                continue
            if isinstance(hchild, h5py.Group):
                childnode = fits_hdf5_tree.FileGroupNode(parent, hchild)
                addsub(childnode, hchild)
            elif isinstance(hchild, h5py.Dataset):
                try:
                    dtype = hchild.dtype
                except TypeError:
                    # raised if datatype not supported by h5py
                    continue

                if dtype.kind == 'V':
                    # compound data type - add a special group for
                    # the compound, then its children
                    childnode = fits_hdf5_tree.FileCompoundNode(parent, hchild)

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
                            fshape)

                        childnode.children.append(fnode)
                        datanodes.append(fnode)

                else:
                    # normal dataset
                    childnode = makedatanode(parent, hchild)
                    datanodes.append(childnode)

            parent.children.append(childnode)

    root = fits_hdf5_tree.FileGroupNode(None, hdf5file)
    addsub(root, hdf5file)
    return root, datanodes

