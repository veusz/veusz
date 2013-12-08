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
     toimport: map of dataset names to veusz names
    """

    defaults = {
        'toimport': None,
        'readall': False,
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
        if p.toimport:
            # sorted to ensure ordering in dict here
            toimport = [ "%s: %s" % (crepr(k), crepr(p.toimport[k]))
                         for k in sorted(p.toimport) ]
            args.append("toimport={%s}" % ", ".join(toimport))
        if p.readall:
            args.append("readall=True")
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
    """Import 1d or 2d data from a fits file."""

    descr = _("import HDF5 file")

    def readSpecificData(self):
        """Return read datasets."""

        p = self.params
        dsread = {}
        with h5py.File(p.filename) as hdff:
            for hdfname, vzname in citems(p.toimport):
                data = hdff[hdfname]
                vzname = vzname.strip()
                dsread[vzname] = convertDataset(data)
        return dsread

    def readAllData(self):
        """Read data with original names in file."""
        dsread = {}

        def walkgrp(grp):
            for dsname in sorted(grp.keys()):
                try:
                    child = grp[dsname]
                except KeyError:
                    # this does happen!
                    continue
                if isinstance(child, h5py.Group):
                    walkgrp(child)
                elif isinstance(child, h5py.Dataset):
                    name = dsname.split("/")[-1].strip()
                    if name in dsread:
                        name = dsname.strip()
                    try:
                        dsread[name] = convertDataset(child)
                    except _ConvertError:
                        pass

        p = self.params
        with h5py.File(p.filename) as hdff:
            walkgrp(hdff)

        return dsread

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

        if p.toimport:
            dsread = self.readSpecificData()
        elif p.readall:
            dsread = self.readAllData()
        else:
            dsread = {}
        errordatasets = self.collectErrorDatasets(dsread)

        if p.linked:
            linkedfile = LinkedFileHDF5(p)
        else:
            linkedfile = None

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
                        ds = document.Dataset2D(data)
                        ds.linked = linkedfile

            else:
                # text dataset
                ds = document.DatasetText(data, linked=linkedfile)

            # finally set dataset in document
            fullname = p.prefix + name + p.suffix
            doc.setData(fullname, ds)
            self.outdatasets.append(fullname)

def ImportFileHDF5(comm, filename, toimport={}, readall=False,
                   prefix='', suffix='',
                   linked=False):
    """Import data from a HDF5 file

    toimport is a dict, mapping of HDF5 dataset names to Veusz dataset
    names. Veusz dataset names can be given special suffixes to denote
    how they are imported:

    'foo (+)': import as +ve error for dataset foo
    'foo (-)': import as -ve error for dataset foo
    'foo (+-)': import as symmetric error for dataset foo
    'foo (1D)': import 2D dataset as 1D dataset with errors
      (for Nx2 or Nx3 datasets)

    readall can be set instead. This reads all datasets with their
    original names.

    linked specfies that the dataset is linked to the file
    """

    # lookup filename
    realfilename = comm.findFileOnImportPath(filename)
    params = ImportParamsHDF5(
        filename=realfilename,
        readall=readall,
        toimport=toimport,
        prefix=prefix, suffix=suffix,
        linked=linked)
    op = OperationDataImportHDF5(params)
    comm.document.applyOperation(op)

document.registerImportCommand("ImportFileHDF5", ImportFileHDF5)
