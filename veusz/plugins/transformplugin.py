#    Copyright (C) 2016 Jeremy S. Sanders
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
###############################################################################

from __future__ import division, print_function
import numpy as N

from .. import qtall as qt4
from .datasetplugin import Dataset1D

def _(text, disambiguation=None, context='TransformPlugin'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

transformpluginregistry = {}

def registerTransformPlugin(
        name, username, category='Base', description=None):

    def decorator(f):
        transformpluginregistry[name] = (
            f, username, category, description)
        return f

    return decorator

def isDataset1D(v):
    return isinstance(v, Dataset1D)

def dsCodeToIdx(code):
    """Convert code for dataset to internal index."""
    try:
        return {
            'x': 0,
            'y': 1,
            'l': 2,
            'label': 2,
            's': 3,
            'size': 3,
            'c': 4,
            'color': 4,
        }[code.lower()]
    except (KeyError, AttributeError):
        raise ValueError('Unknown dataset code %s' % code)


###############################################################################
# Add

def _addSubDataset(d1, d2, sub=False):
    minlen = min(len(d1.data), len(d2.data))

    # add/subtract data points
    if sub:
        d1.data = d1.data[:minlen] - d2.data[:minlen]
    else:
        d1.data = d1.data[:minlen] + d2.data[:minlen]

    # below, combine errors
    # note: if d1err and not d2err, keep existing errors
    # if no errors, do nothing

    d1err = d1.hasErrors()
    d2err = d2.hasErrors()
    if d2err and not d1err:
        # copy errors
        if d2.serr is not None:
            d1.serr = N.array(d2.serr[:minlen])
        if d2.perr is not None:
            d1.perr = N.array(d2.perr[:minlen])
        if d2.nerr is not None:
            d1.nerr = N.array(d2.nerr[:minlen])
    elif d1err and d2err:
        # combine errors square errors, add, sqrt
        # symmetrise errors
        if d1.serr is not None:
            d1err2 = d1.serr**2
        else:
            d1err2 = 0.5*(getattr(d1, 'perr', 0)**2 + getattr(d1, 'nerr', 0)**2)
        if d2.serr is not None:
            d2err2 = d2.serr**2
        else:
            d2err2 = 0.5*(getattr(d2, 'perr', 0)**2 + getattr(d2, 'nerr', 0)**2)
        d1.serr = N.sqrt(d1err2[:minlen] + d2err2[:minlen])
        d1.perr = d1.nerr = None

@registerTransformPlugin(
    'AddX', _('Add to X'), category=_('Maths'),
    description=_('Add value or dataset to X dataset'))
def mathsAddX(dss):
    def AddX(val):
        if isDataset1D(val):
            _addSubDataset(dss[0], val, sub=False)
        else:
            dss[0].data += val
    return AddX

@registerTransformPlugin(
    'AddY', _('Add to Y'), category=_('Maths'),
    description=_('Add value or dataset to Y dataset'))
def mathsAddY(dss):
    def AddY(val):
        if isDataset1D(val):
            _addSubDataset(dss[1], val, sub=False)
        else:
            dss[1].data += val
    return AddY

@registerTransformPlugin(
    'Add', _('Add to dataset'), category=_('Maths'),
    description=_('Add value or dataset to dataset'))
def mathsAdd(dss):
    def Add(ds, val):
        idx = dsCodeToIdx(ds)
        if isDataset1D(val):
            _addSubDataset(dss[idx], val, sub=False)
        else:
            dss[idx].data += val
    return Add

###############################################################################
# Subtract

@registerTransformPlugin(
    'SubX', _('Subtract from X'), category=_('Maths'),
    description=_('Subtract value or dataset from X dataset'))
def mathsSubX(dss):
    def SubX(val):
        if isDataset1D(val):
            _addSubDataset(dss[0], val, sub=True)
        else:
            dss[0].data -= val
    return SubX

@registerTransformPlugin(
    'SubY', _('Subtract from Y'), category=_('Maths'),
    description=_('Subtract value or dataset from Y dataset'))
def mathsSubY(dss):
    def SubY(val):
        if isDataset1D(val):
            _addSubDataset(dss[1], val, sub=True)
        else:
            dss[1].data -= val
    return SubY

@registerTransformPlugin(
    'Sub', _('Subtract from dataset'), category=_('Maths'),
    description=_('Subtract value or dataset from dataset'))
def mathsSub(dss):
    def Sub(ds, val):
        idx = dsCodeToIdx(ds)
        if isDataset1D(val):
            _addSubDataset(dss[idx], val, sub=True)
        else:
            dss[idx].data -= val
    return Sub

###############################################################################
# Multiply

def _multiplyDatasetScalar(ds, val):
    ds.data *= val
    if ds.serr is not None:
        ds.serr *= val
    if ds.perr is not None:
        ds.perr *= val
    if ds.nerr is not None:
        ds.nerr *= val

def _multiplyDatasetDataset(d1, d2):
    minlen = min(len(d1.data), len(d2.data))

    # below, combine errors
    # if no errors, do nothing

    d1err = d1.hasErrors()
    d2err = d2.hasErrors()
    if d2err and not d1err:
        d1av = N.abs(d1.data[:minlen])
        if d2.serr is not None:
            d1.serr = d2.serr[:minlen]*d1av
        if d2.perr is not None:
            d1.perr = d2.perr[:minlen]*d1av
        if d2.nerr is not None:
            d1.nerr = d2.nerr[:minlen]*d1av
    elif d1err and not d2err:
        d2av = N.abs(d2.data[:minlen])
        if d2.serr is not None:
            d1.serr = d1.serr[:minlen]*d2av
        if d2.perr is not None:
            d1.perr = d1.perr[:minlen]*d2av
        if d2.nerr is not None:
            d1.nerr = d1.nerr[:minlen]*d2av
    elif d1err and d2err:
        # combine errors square fractional errors, add, sqrt
        # (symmetrising)
        if d1.serr is not None:
            d1ferr2 = (d1.serr/d1.data)**2
        else:
            d1err2 = 0.5*(getattr(d1, 'perr', 0)**2 + getattr(d1, 'nerr', 0)**2)
            d1ferr2 = d1err2 / d1.data**2
        if d2.serr is not None:
            d2ferr2 = (d2.serr/d2.data)**2
        else:
            d2err2 = 0.5*(getattr(d2, 'perr', 0)**2 + getattr(d2, 'nerr', 0)**2)
            d2ferr2 = d2err2 / d2.data**2

        d1.serr = N.sqrt(d1ferr2[:minlen] + d2ferr2[:minlen]) * N.abs(
            d1.data[:minlen] * d2.data[:minlen])
        d1.perr = d1.nerr = None

    # multiply data points
    d1.data = d1.data[:minlen] * d2.data[:minlen]

@registerTransformPlugin(
    'MulX', _('Multiply X'), category=_('Maths'),
    description=_('Multiply X dataset by value or dataset'))
def mathsMulX(dss):
    def MulX(val):
        if isDataset1D(val):
            _multiplyDatasetDataset(dss[0], val)
        else:
            _multiplyDatasetScalar(dss[0], val)
    return MulX

@registerTransformPlugin(
    'MulY', _('Multiply Y'), category=_('Maths'),
    description=_('Multiply Y dataset by value or dataset'))
def mathsMulY(dss):
    def MulY(val):
        if isDataset1D(val):
            _multiplyDatasetDataset(dss[1], val)
        else:
            _multiplyDatasetScalar(dss[1], val)
    return MulY

@registerTransformPlugin(
    'Mul', _('Multiply dataset'), category=_('Maths'),
    description=_('Multiply dataset by value or dataset'))
def mathsMul(dss):
    def Mul(ds, val):
        idx = dsCodeToIdx(ds)
        if isDataset1D(val):
            _multiplyDatasetDataset(dss[idx], val)
        else:
            _multiplyDatasetScalar(dss[idx], val)
    return Mul

###############################################################################
# Divide

def _divideDatasetDataset(d1, d2):
    minlen = min(len(d1.data), len(d2.data))

    # below, combine errors
    # if no errors, do nothing

    ratio = d1.data[:minlen] / d2.data[:minlen]

    d1err = d1.hasErrors()
    d2err = d2.hasErrors()
    if d2err and not d1err:
        # copy fractional error
        aratiodiv = N.abs(ratio / d2.data[:minlen])
        if d2.serr is not None:
            d1.serr = aratiodiv * d2.serr[:minlen]
        if d2.perr is not None:
            d1.serr = aratiodiv * d2.perr[:minlen]
        if d2.nerr is not None:
            d1.serr = aratiodiv * d2.nerr[:minlen]

    elif d1err and not d2err:
        # divide error by scalar
        if d2.serr is not None:
            d1.serr = d1.serr[:minlen] / N.abs(d2.data[:minlen])
        if d2.perr is not None:
            d1.perr = d1.perr[:minlen] / N.abs(d2.data[:minlen])
        if d2.nerr is not None:
            d1.nerr = d1.nerr[:minlen] / N.abs(d2.data[:minlen])

    elif d1err and d2err:
        # combine errors square fractional errors, add, sqrt
        # (symmetrising)
        if d1.serr is not None:
            d1ferr2 = (d1.serr/d1.data)**2
        else:
            d1err2 = 0.5*(getattr(d1, 'perr', 0)**2 + getattr(d1, 'nerr', 0)**2)
            d1ferr2 = d1err2 / d1.data**2
        if d2.serr is not None:
            d2ferr2 = (d2.serr/d2.data)**2
        else:
            d2err2 = 0.5*(getattr(d2, 'perr', 0)**2 + getattr(d2, 'nerr', 0)**2)
            d2ferr2 = d2err2 / d2.data**2

        d1.serr = N.sqrt(d1ferr2[:minlen] + d2ferr2[:minlen]) * N.abs(ratio)
        d1.perr = d1.nerr = None

    # the divided points
    d1.data = ratio

@registerTransformPlugin(
    'DivX', _('Divide X'), category=_('Maths'),
    description=_('Divide X dataset by value or dataset'))
def mathsDivX(dss):
    def DivX(val):
        if isDataset1D(val):
            _divideDatasetDataset(dss[0], val)
        else:
            _multiplyDatasetScalar(dss[0], 1./val)
    return DivX

@registerTransformPlugin(
    'DivY', _('Divide Y'), category=_('Maths'),
    description=_('Divide Y dataset by value or dataset'))
def mathsDivY(dss):
    def DivY(val):
        if isDataset1D(val):
            _divideDatasetDataset(dss[1], val)
        else:
            _multiplyDatasetScalar(dss[1], 1./val)
    return DivY

@registerTransformPlugin(
    'Div', _('Divide dataset'), category=_('Maths'),
    description=_('Divide dataset by value or dataset'))
def mathsDiv(dss):
    def Div(ds, val):
        idx = dsCodeToIdx(ds)
        if isDataset1D(val):
            _divideDatasetDataset(dss[idx], val)
        else:
            _multiplyDatasetScalar(dss[idx], 1./val)
    return Div

###############################################################################
# Geometry

@registerTransformPlugin(
    'Rotate', _('Rotate coordinates'), category=_('Geometry'),
    description=_('Rotate coordinates by angle in radians, with optional centre'))
def geometryRotate(dss):
    def Rotate(angle_rad, cx=0, cy=0):
        xvals = dss[0].data - cx
        yvals = dss[1].data - cy

        nx = N.cos(angle_rad)*xvals - N.sin(angle_rad)*yvals + cx
        ny = N.sin(angle_rad)*xvals + N.cos(angle_rad)*yvals + cy

        dss[0].data = nx
        dss[1].data = ny
        dss[0].serr = dss[0].perr = dss[0].nerr = None
        dss[1].serr = dss[1].perr = dss[1].nerr = None
    return Rotate

@registerTransformPlugin(
    'Translate', _('Translate coordinates'), category=_('Geometry'),
    description=_('Translate coordinates'))
def geometryTranslate(dss):
    def Translate(dx, dy):
        dss[0].data += dx
        dss[1].data += dy
    return Translate

###############################################################################
# Filter

@registerTransformPlugin(
    'Thin', _('Thin values'), category=_('Filtering'),
    description=_('Thin values by step and optional starting index (from 0)'))
def filteringThin(dss):
    def Thin(num, start=0):
        for ds in dss:
            if ds is None:
                continue
            for attr in 'data', 'serr', 'perr', 'nerr':
                if getattr(ds, attr, None) is not None:
                    setattr(ds, attr, getattr(ds, attr)[start::num])
    return Thin

@registerTransformPlugin(
    'Range', _('Select range'), category=_('Filtering'),
    description=_(
        'Select values from start, with optional end index and step '
        '(Python-style indexing from 0)'))
def filteringRange(dss):
    def Range(start, end=None, step=None):
        for ds in dss:
            if ds is None:
                continue
            for attr in 'data', 'serr', 'perr', 'nerr':
                if getattr(ds, attr, None) is not None:
                    setattr(ds, attr, getattr(ds, attr)[start:end:step])
    return Range
