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
    return h5py

class ImportParamsHDF5(base.ImportParamsBase):
    """HDF5 file import parameters.

    Additional parameters:
     singledatasets: map of dataset names to veusz names
    """

    defaults = {
        'singledatasets': None,
        'singledatasets_extras': None,
        'groups': None,
        }
    defaults.update(base.ImportParamsBase.defaults)

class LinkedFileHDF5(base.LinkedFileBase):
    """Links a HDF5 file to the data."""

    def createOperation(self):
        """Return operation to recreate self."""
        return OperationDataImportHDF5

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""

        p = self.params
        args = [ crepr(self._getSaveFilename(relpath)) ]
        if p.singledatasets:
            # sorted to ensure ordering in dict here
            singles = [ "%s: %s" % (crepr(k), crepr(p.singledatasets[k]))
                        for k in sorted(p.singledatasets) ]
            args.append("singledatasets={%s}" % ", ".join(singles))
        if p.singledatasets_extras:
            # sorted to ensure ordering in dict here
            extras = [ "%s: %s" % (crepr(k), crepr(p.singledatasets_extras[k]))
                        for k in sorted(p.singledatasets_extras) ]
            args.append("singledatasets_extras={%s}" % ", ".join(extras))        
        if p.groups:
            args.append("groups=%s" % crepr(p.groups))
        if p.prefix:
            args.append("prefix=%s" % crepr(p.prefix))
        if p.suffix:
            args.append("suffix=%s" % crepr(p.suffix))
        args.append("linked=True")
        fileobj.write("ImportFileHDF5(%s)\n" % ", ".join(args))

class _ConvertError(RuntimeError):
    pass

def convertDataset(data):
    """Convert dataset to suitable data for veusz.
    Raise RuntimeError if cannot."""

    kind = data.dtype.kind
    if kind in ('b', 'i', 'u', 'f'):
        data = N.array(data, dtype=N.float64)
        if len(data.shape) > 2:
            raise _ConvertError("HDF5 dataset %s has more than"
                                " 2 dimensions" % data.name)
        return data

    elif kind in ('S', 'a') or (
        kind == 'O' and h5py.check_dtype(vlen=data.dtype)):
        if len(data.shape) != 1:
            raise _ConvertError("HDF5 dataset %s has more than"
                                " 1 dimension" % data.name)

        strcnv = list(data)
        return strcnv

    raise _ConvertError("HDF5 dataset %s has an invalid type" %
                        data.name)

class OperationDataImportHDF5(base.OperationDataImportBase):
    """Import 1d or 2d data from an HDF5 file."""

    descr = _("import HDF5 file")

    def readSingleDatasets(self, hdff, singledatasets, singledatasets_extras, dsread):
        for hdfname, vzname in citems(singledatasets):
            vzname = vzname.strip()
            cs = None
            if 'custom_slice' in singledatasets_extras[vzname]:
                cs = singledatasets_extras[vzname]['custom_slice']
            if cs:
                # check that there are the correct number of slice params given the shape
                if len(cs) == len(hdff[hdfname].shape):
                    # Thanks to hpaulj on stackoverflow for the tip that using `tuple`
                    # here makes this work.
                    data = hdff[hdfname][tuple(cs)]
                else:
                    data = hdff[hdfname]
            else:
                data = hdff[hdfname]
            dsread[vzname] = convertDataset(data)

    def walkGroup(self, grp, dsread):
        """Walk a group in the hdf file, adding datasets to dsread."""

        for dsname in sorted(grp.keys()):
            try:
                child = grp[dsname]
            except KeyError:
                # this does happen!
                continue
            if isinstance(child, h5py.Group):
                self.walkGroup(child, dsread)
            elif isinstance(child, h5py.Dataset):
                name = dsname.split("/")[-1].strip()
                if name in dsread:
                    name = dsname.strip()
                try:
                    dsread[name] = convertDataset(child)
                except _ConvertError:
                    pass

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
            if p.singledatasets is not None:
                self.readSingleDatasets(hdff, p.singledatasets, p.singledatasets_extras, dsread)
            if p.groups is not None:
                for grp in p.groups:
                    self.walkGroup(hdff[grp], dsread)

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
                        if a is not None and len(a) > minlen:
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
                        # Add xrange, yrange here if custom params are specified
                        xrange = None
                        yrange = None
                        if 'xrange' in p.singledatasets_extras[name]:
                            xrange = p.singledatasets_extras[name]['xrange']
                        if 'yrange' in p.singledatasets_extras[name]:
                            yrange = p.singledatasets_extras[name]['yrange']
                        ds = document.Dataset2D(data,xrange,yrange) 
                        ds.linked = linkedfile

            else:
                # text dataset
                ds = document.DatasetText(data, linked=linkedfile)

            # finally set dataset in document
            fullname = p.prefix + name + p.suffix
            doc.setData(fullname, ds)
            self.outdatasets.append(fullname)

def ImportFileHDF5(comm, filename,
                   singledatasets=None,
                   singledatasets_extras=None,
                   groups=None,
                   prefix='', suffix='',
                   linked=False):
    """Import data from a HDF5 file

    singledatasets is a dict, mapping of HDF5 dataset names to Veusz dataset
    names. Veusz dataset names can be given special suffixes to denote
    how they are imported:

    'foo (+)': import as +ve error for dataset foo
    'foo (-)': import as -ve error for dataset foo
    'foo (+-)': import as symmetric error for dataset foo
    'foo (1D)': import 2D dataset as 1D dataset with errors
      (for Nx2 or Nx3 datasets)

    groups can also be set. This is a list of groups to import with
    all their child datasets and child groups. Child datasets retain
    the same names, although the special suffixes are used.

    Do not duplicate datasets between singledatasets and groups.

    linked specfies that the dataset is linked to the file
    """

    # lookup filename
    realfilename = comm.findFileOnImportPath(filename)
    params = ImportParamsHDF5(
        filename=realfilename,
        groups=groups,
        singledatasets=singledatasets,
        singledatasets_extras=singledatasets_extras,
        prefix=prefix, suffix=suffix,
        linked=linked)
    op = OperationDataImportHDF5(params)
    comm.document.applyOperation(op)

document.registerImportCommand("ImportFileHDF5", ImportFileHDF5)
