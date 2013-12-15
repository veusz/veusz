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
import numpy as N
from .. import qtall as qt4
from ..compat import citems, ckeys, cvalues, cstr, crepr
from .. import document
from . import base

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

class ImportParamsHDF5(base.ImportParamsBase):
    """HDF5 file import parameters.

    Additional parameters:
     items: list of datasets and items to import
     namemap: map hdf datasets to veusz names
     none: dict to map objects to slices
    """

    defaults = {
        'items': None,
        'namemap': None,
        'slices': None,
        }
    defaults.update(base.ImportParamsBase.defaults)

class LinkedFileHDF5(base.LinkedFileBase):
    """Links a HDF5 file to the data."""

    def createOperation(self):
        """Return operation to recreate self."""
        return OperationDataImportHDF5

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""

        def reprdict(d):
            """Reproducable repr for dict (alphabetical)."""
            v = [ "%s: %s" % (crepr(k), crepr(d[k]))
                  for k in sorted(d) ]
            return "{%s}" % " ".join(v)

        p = self.params
        args = [ crepr(self._getSaveFilename(relpath)),
                 crepr(p.items) ]
        if p.namemap:
            args.append("namemap=%s" % reprdict(p.namemap))
        if p.slices:
            args.append("slices=%s" % reprdict(p.slices))
        if p.prefix:
            args.append("prefix=%s" % crepr(p.prefix))
        if p.suffix:
            args.append("suffix=%s" % crepr(p.suffix))
        args.append("linked=True")
        fileobj.write("ImportFileHDF5(%s)\n" % ", ".join(args))

class _ConvertError(RuntimeError):
    pass

def convertDataset(data, slices):
    """Convert dataset to suitable data for veusz.
    Raise RuntimeError if cannot."""

    if slices:
        # build up list of slice objects and apply to data
        slist = []
        for s in slices:
            if isinstance(s, int):
                slist.append(s)
            else:
                slist.append(slice(*s))
                if s[2] < 0:
                    # negative slicing doesn't work in h5py, so we
                    # make a copy
                    data = N.array(data)
        try:
            data = data[tuple(slist)]
        except (ValueError, IndexError):
            data = N.array([], dtype=N.float64)

    kind = data.dtype.kind
    if kind in ('b', 'i', 'u', 'f'):
        data = N.array(data, dtype=N.float64)
        if len(data.shape) > 2:
            raise _ConvertError(_("HDF5 dataset %s has more than"
                                  " 2 dimensions" % data.name))
        return data

    elif kind in ('S', 'a') or (
        kind == 'O' and h5py.check_dtype(vlen=data.dtype)):
        if len(data.shape) != 1:
            raise _ConvertError(_("HDF5 dataset %s has more than"
                                  " 1 dimension" % data.name))

        strcnv = list(data)
        return strcnv

    raise _ConvertError(_("HDF5 dataset %s has an invalid type" %
                          data.name))

class OperationDataImportHDF5(base.OperationDataImportBase):
    """Import 1d or 2d data from a fits file."""

    descr = _("import HDF5 file")

    def walkFile(self, item, dsread):
        """Walk an hdf file, adding datasets to dsread."""

        if isinstance(item, h5py.Dataset):
            if (self.params.namemap is not None and
                item.name in self.params.namemap ):
                name = self.params.namemap[item.name]
            else:
                name = item.name.split("/")[-1].strip()
            if name in dsread:
                name = item.name.strip()
            try:
                dsread[name] = convertDataset(
                    item, self.params.slices.get(item.name))
            except _ConvertError:
                pass
        elif isinstance(item, h5py.Group):
            for dsname in sorted(item.keys()):
                try:
                    child = item[dsname]
                except KeyError:
                    # this does happen!
                    continue
                self.walkFile(child, dsread)

    def collectErrorDatasets(self, dsread):
        """Identify error bar datasets and separate out.
        Returns error bar datasets."""

        # separate out datsets with error bars
        # this a defaultdict of defaultdict with None as default
        errordatasets = collections.defaultdict(
            lambda: collections.defaultdict(lambda: None))
        for name in list(ckeys(dsread)):
            ds = dsread[name]
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

    def doImport(self, doc):
        """Do the import."""

        inith5py()
        p = self.params

        # read the data
        dsread = {}
        with h5py.File(p.filename) as hdff:
            for hi in p.items:
                self.walkFile(hdff[hi], dsread)

        # find datasets which are error datasets
        errordatasets = self.collectErrorDatasets(dsread)

        if p.linked:
            linkedfile = LinkedFileHDF5(p)
        else:
            linkedfile = None

        # create the veusz output datasets
        for name, data in citems(dsread):
            ds = None
            if isinstance(data, N.ndarray):
                # numeric
                if len(data.shape) == 1:
                    # handle any possible error bars
                    args = { 'data': data,
                             'serr': errordatasets[name]['+-'],
                             'nerr': errordatasets[name]['-'],
                             'perr': errordatasets[name]['+'] }

                    # find minimum length and cut down if necessary
                    minlen = min([len(d) for d in cvalues(args)
                                  if d is not None])
                    for a in list(ckeys(args)):
                        if args[a] is not None and len(args[a]) > minlen:
                            args[a] = args[a][:minlen]

                    args['linked'] = linkedfile
                    ds = document.Dataset(**args)

                elif len(data.shape) == 2:
                    # 2D dataset
                    if name[-5:] == ' (1D)' and data.shape[1] in (2,3):
                        # actually a 1D dataset in disguise
                        name = name[:-5].strip()
                        if data.shape[1] == 2:
                            ds = document.Dataset(
                                data=data[:,0], serr=data[:,1],
                                linked=linkedfile)
                        else:
                            ds = document.Dataset(
                                data=data[:,0], perr=data[:,1], nerr=data[:,2],
                                linked=linkedfile)
                    else:
                        # this really is a 2D dataset
                        ds = document.Dataset2D(data)
                        ds.linked = linkedfile

            else:
                # text dataset
                ds = document.DatasetText(data, linked=linkedfile)

            # finally set dataset in document
            fullname = p.prefix + name + p.suffix
            doc.setData(fullname, ds)
            self.outdatasets.append(fullname)

def ImportFileHDF5(comm, filename,
                   items,
                   namemap=None,
                   slices=None,
                   prefix='', suffix='',
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
    'foo (1D)': import 2D dataset as 1D dataset with errors
      (for Nx2 or Nx3 datasets)

    slices is an optional dict specifying slices to be selected when
    importing. For each dataset to be sliced, provide a tuple of
    values, one for each dimension. The values should be a single
    integer to select that index, or a tuple (start, stop, step),
    where the entries are integers or None.

    linked specfies that the dataset is linked to the file
    """

    # lookup filename
    realfilename = comm.findFileOnImportPath(filename)
    params = ImportParamsHDF5(
        filename=realfilename,
        items=items,
        namemap=namemap,
        slices=slices,
        prefix=prefix, suffix=suffix,
        linked=linked)
    op = OperationDataImportHDF5(params)
    comm.document.applyOperation(op)

document.registerImportCommand("ImportFileHDF5", ImportFileHDF5)
