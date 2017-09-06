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

import sys
import ast
import re
import numpy as N

from .. import qtall as qt

def _(text, disambiguation=None, context="Import_FITS_HDF5"):
    return qt.QCoreApplication.translate(context, text, disambiguation)

def filterAttrsByName(attrs, name):
    """For compound datasets, attributes can be given on a per-column basis.
    This filters the attributes by the column name."""

    name = name.strip()
    attrsout = {}
    for a in attrs:
        # attributes with _dsname suffixes are copied
        if a[:4] == "vsz_" and a[-len(name)-1:] == "_"+name:
            attrsout[a[:-len(name)-1]] = attrs[a]
    return attrsout

def convertTextToSlice(slicetxt, numdims):
    """Convert a value like 0:1:3,:,::-1 to a tuple slice
    ((0,1,3), (None, None, None), (None, None, -1))
    or reduce dimensions such as :,3 -> ((None,None,None),3)

    Also checks number of dimensions (including reduced) is numdims.

    Return -1 on error
    """

    if slicetxt.strip() == '':
        return None

    slicearray = slicetxt.split(',')
    if len(slicearray) != numdims:
        # slice needs same dimensions as data
        return -1

    allsliceout = []
    for sliceap_idx, sliceap in enumerate(slicearray):
        sliceparts = sliceap.strip().split(':')

        if len(sliceparts) == 1:
            # reduce dimensions with single index
            try:
                allsliceout.append(int(sliceparts[0]))
            except ValueError:
                # invalid index
                return -1
        elif len(sliceparts) not in (2, 3):
            return -1
        else:
            sliceout = []
            for p in sliceparts:
                p = p.strip()
                if not p:
                    sliceout.append(None)
                else:
                    try:
                        sliceout.append(int(p))
                    except ValueError:
                        return -1
            if len(sliceout) == 2:
                sliceout.append(None)
            allsliceout.append(tuple(sliceout))

    allempty = True
    for s in allsliceout:
        if s != (None, None, None):
            allempty = False
    if allempty:
        return None

    return tuple(allsliceout)

def convertSliceToText(slice):
    """Convert tuple slice into text."""
    if slice is None:
        return ''
    out = []
    for spart in slice:
        if isinstance(spart, int):
            # single index
            out.append(str(spart))
            continue

        sparttxt = []
        for p in spart:
            if p is not None:
                sparttxt.append(str(p))
            else:
                sparttxt.append('')
        if sparttxt[-1] == '':
            del sparttxt[-1]
        out.append(':'.join(sparttxt))
    return ', '.join(out)

def applySlices(data, slices):
    """Given hdf/numpy dataset, apply slicing tuple to it and return data."""
    slist = []
    for s in slices:
        if isinstance(s, int):
            slist.append(s)
        else:
            slist.append(slice(*s))
            if s[2] is not None and s[2] < 0:
                # negative slicing doesn't work in h5py, so we
                # make a copy
                data = N.array(data)
    try:
        data = data[tuple(slist)]
    except (ValueError, IndexError):
        data = N.array([], dtype=N.float64)
    return data

class ConvertError(RuntimeError):
    pass

def convertFromBytes(s):
    """h5py often returns bytes instead of unicode.
    This decodes if in bytes
    """
    if sys.version_info[0] == 3:
        if isinstance(s, bytes):
            return s.decode('utf-8')
    return s

def convertDatasetToObject(data, slices):
    """Convert numpy/hdf dataset to suitable data for veusz.
    Raise ConvertError if cannot."""

    # lazily-loaded h5py
    try:
        from h5py import check_dtype
    except ImportError:
        # fallback if no h5py, e.g. only installed fits
        def check_dtype(vlen=None):
            return False

    if slices:
        data = applySlices(data, slices)

    try:
        kind = data.dtype.kind
    except TypeError:
        raise ConvertError(_("Could not get data type of dataset"))

    if kind in ('b', 'i', 'u', 'f'):
        data = N.array(data, dtype=N.float64)
        if data.ndim == 0:
            raise ConvertError(_("Dataset has no dimensions"))
        return data

    elif kind in ('S', 'a', 'U') or (
        kind == 'O' and check_dtype(vlen=data.dtype) is str):
        if hasattr(data, 'ndim') and data.ndim != 1:
            raise ConvertError(_("Text datasets must have 1 dimension"))

        strcnv = list(data)
        return strcnv

    raise ConvertError(_("Dataset has an invalid type"))

def getFITSHduNames(fitsfile):
    """Return list of names to give HDUs given a FITS file."""

    nameset = set()
    names = []

    for i, hdu in enumerate(fitsfile):
        name = hdu.name

        if not name:
            name = 'hdu%i' % i
            # just in case people start naming HDUs hduX...
            while name in nameset:
                name += '~'
        else:
            name = name.lower()

            # EXTVER distinguishes identical names
            if 'EXTVER' in hdu.header:
                name = '%s%i' % (name, hdu.header['EXTVER'])

            # prevent duplicates
            if name in nameset:
                i = 2
                while name+str(i) in nameset:
                    i += 1
                name += str(i)

        nameset.add(name)
        names.append(name)

    return names

def convertFITSDataFormat(fmt):
    """Convert FITS TFORM codes into:

    (code, nlen)

    code is 'invalid', 'numeric' or 'text'
    nlen is number of entries for column
    """

    # match the fits format text code [r]F[w[.d]]
    m = re.match(r'^([0-9]*)([A-Za-z])([0-9]*)(\.[0-9]+)?$', fmt)
    if not m:
        return (None, 'invalid', ())
    grps = m.groups()

    # length of array
    try:
        nlen = int(grps[0])
    except ValueError:
        nlen = 1
    fcode = grps[1].upper()
    width = grps[2]

    if fcode == 'A' and not width:
        # note: we can't handle 2d text arrays, so we handle as invalid
        code = 'text'
        # even though strings are N characters, they are handled as singles
        nlen = 1

    elif fcode in 'LXBIJKED':
        code = 'numeric'

    else:
        code = 'invalid'

    return code, nlen

def hduVeuszAttrs(hdu):
    """Get veusz-specific attributes from a HDU header.

    These use the VEUSZ keyword and have the format of
    KEY=VALUE
     or
    COLUMN: KEY=VALUE

    Returns (attrs, colattrs)

    Where attrs is HDU-specific and colattrs is column specific (dict
    of dicts)

    """

    attrs = {}
    colattrs = {}

    for k, v in hdu.header.items():
        if k.lower() == 'veusz':

            # match syntax [OPTIONAL COLUMN:] KEY=VALUE
            match = re.match(
                r'^(?:([a-zA-Z0-9_]+)[ ]*:)?[ ]*([a-zA-Z0-9_]+)[ ]*=[ ]*(.*)$',
                v)
            if not match:
                continue

            col, key, tval = match.groups()
            key = key.lower()
            tval = tval.strip()

            # convert to python type if possible
            try:
                val = ast.literal_eval(tval)
            except Exception:
                val = tval

            if col:
                col = col.lower()

                # column-specific key value
                if col not in colattrs:
                    colattrs[col] = {key: val}
                else:
                    colattrs[col][key] = val
            else:
                # hdu-specific key value
                attrs[key] = val

    return attrs, colattrs
