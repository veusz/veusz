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

# TODO
# log, exp
# cumulative
# normalise
# filter
# moving average
# rebin

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

catarith=_('Arithmetic')

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
    'AddX', _('Add to X'), category=catarith,
    description=_('Add value or dataset [val] to X dataset'))
def mathsAddX(dss):
    def AddX(val):
        if isDataset1D(val):
            _addSubDataset(dss[0], val, sub=False)
        else:
            dss[0].data += val
    return AddX

@registerTransformPlugin(
    'AddY', _('Add to Y'), category=catarith,
    description=_('Add value or dataset [val] to Y dataset'))
def mathsAddY(dss):
    def AddY(val):
        if isDataset1D(val):
            _addSubDataset(dss[1], val, sub=False)
        else:
            dss[1].data += val
    return AddY

@registerTransformPlugin(
    'Add', _('Add to dataset'), category=catarith,
    description=_('Add value or dataset [val] to specified output dataset [outds]'))
def mathsAdd(dss):
    def Add(outds, val):
        idx = dsCodeToIdx(outds)
        if isDataset1D(val):
            _addSubDataset(dss[idx], val, sub=False)
        else:
            dss[idx].data += val
    return Add

###############################################################################
# Subtract

@registerTransformPlugin(
    'SubX', _('Subtract from X'), category=catarith,
    description=_('Subtract value or dataset [val] from X dataset'))
def mathsSubX(dss):
    def SubX(val):
        if isDataset1D(val):
            _addSubDataset(dss[0], val, sub=True)
        else:
            dss[0].data -= val
    return SubX

@registerTransformPlugin(
    'SubY', _('Subtract from Y'), category=catarith,
    description=_('Subtract value or dataset [val] from Y dataset'))
def mathsSubY(dss):
    def SubY(val):
        if isDataset1D(val):
            _addSubDataset(dss[1], val, sub=True)
        else:
            dss[1].data -= val
    return SubY

@registerTransformPlugin(
    'Sub', _('Subtract from dataset'), category=catarith,
    description=_('Subtract value or dataset [val] from specified output dataset [outds]'))
def mathsSub(dss):
    def Sub(outds, val):
        idx = dsCodeToIdx(outds)
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
    'MulX', _('Multiply X'), category=catarith,
    description=_('Multiply X dataset by value or dataset [val]'))
def mathsMulX(dss):
    def MulX(val):
        if isDataset1D(val):
            _multiplyDatasetDataset(dss[0], val)
        else:
            _multiplyDatasetScalar(dss[0], val)
    return MulX

@registerTransformPlugin(
    'MulY', _('Multiply Y'), category=catarith,
    description=_('Multiply Y dataset by value or dataset [val]'))
def mathsMulY(dss):
    def MulY(val):
        if isDataset1D(val):
            _multiplyDatasetDataset(dss[1], val)
        else:
            _multiplyDatasetScalar(dss[1], val)
    return MulY

@registerTransformPlugin(
    'Mul', _('Multiply dataset'), category=catarith,
    description=_('Multiply output dataset [outds] by value or dataset [val]'))
def mathsMul(dss):
    def Mul(outds, val):
        idx = dsCodeToIdx(outds)
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
    'DivX', _('Divide X'), category=catarith,
    description=_('Divide X dataset by value or dataset [val]'))
def mathsDivX(dss):
    def DivX(val):
        if isDataset1D(val):
            _divideDatasetDataset(dss[0], val)
        else:
            _multiplyDatasetScalar(dss[0], 1./val)
    return DivX

@registerTransformPlugin(
    'DivY', _('Divide Y'), category=catarith,
    description=_('Divide Y dataset by value or dataset [val]'))
def mathsDivY(dss):
    def DivY(val):
        if isDataset1D(val):
            _divideDatasetDataset(dss[1], val)
        else:
            _multiplyDatasetScalar(dss[1], 1./val)
    return DivY

@registerTransformPlugin(
    'Div', _('Divide dataset'), category=catarith,
    description=_('Divide output dataset [outds] by value or dataset [val]'))
def mathsDiv(dss):
    def Div(outds, val):
        idx = dsCodeToIdx(outds)
        if isDataset1D(val):
            _divideDatasetDataset(dss[idx], val)
        else:
            _multiplyDatasetScalar(dss[idx], 1./val)
    return Div

###############################################################################
## Log10, Log, Exp, Pow

def _applyFn(ds, fun):
    prange = nrange = None
    if ds.serr is not None:
        prange = ds.data+ds.serr
        nrange = ds.data-ds.serr
    if ds.nerr is not None:
        nrange = ds.data+ds.nerr
    if ds.perr is not None:
        prange = ds.data+ds.perr

    ds.data = fun(ds.data)
    if prange is not None:
        ds.perr = fun(prange) - ds.data
    if nrange is not None:
        ds.nerr = fun(nrange) - ds.data
    ds.serr = None

catlog=_('Exponential / Log')

@registerTransformPlugin(
    'Log10X', _('Log10 of X'), category=catlog,
    description=_('Set X dataset to be log10 of input X'))
def mathsLog10X(dss):
    def Log10X():
        _applyFn(dss[0], N.log10)
    return Log10X

@registerTransformPlugin(
    'Log10Y', _('Log10 of Y'), category=catlog,
    description=_('Set Y dataset to be log10 of input Y'))
def mathsLog10Y(dss):
    def Log10Y():
        _applyFn(dss[1], N.log10)
    return Log10Y

@registerTransformPlugin(
    'Log10', _('Log10 of dataset'), category=catlog,
    description=_('Set output dataset [outds] to be log10 of input'))
def mathsLog10(dss):
    def Log10(outds):
        _applyFn(dss[dsCodeToIdx(outds)], N.log10)
    return Log10

@registerTransformPlugin(
    'LogX', _('Natural log of X'), category=catlog,
    description=_('Set X dataset to be natural log of input X'))
def mathsLogX(dss):
    def LogX():
        _applyFn(dss[0], N.log)
    return LogX

@registerTransformPlugin(
    'LogY', _('Natural log of Y'), category=catlog,
    description=_('Set Y dataset to be natural log of input Y'))
def mathsLogY(dss):
    def LogY():
        _applyFn(dss[1], N.log)
    return LogY

@registerTransformPlugin(
    'Log', _('Natural log of dataset'), category=catlog,
    description=_('Set output dataset [outds] to be natural log of input'))
def mathsLog(dss):
    def Log(outds):
        _applyFn(dss[dsCodeToIdx(outds)], N.log)
    return Log

@registerTransformPlugin(
    'ExpX', _('Calculate exponential of X'), category=catlog,
    description=_('Set X dataset to be e^X'))
def mathsExpX(dss):
    def ExpX():
        _applyFn(dss[0], N.exp)
    return ExpX

@registerTransformPlugin(
    'ExpY', _('Calculate exponential of Y'), category=catlog,
    description=_('Set Y dataset to be e^Y'))
def mathsExpY(dss):
    def ExpY():
        _applyFn(dss[1], N.exp)
    return ExpY

@registerTransformPlugin(
    'Exp', _('Calculate exponential of dataset'), category=catlog,
    description=_('Calculate exponential of output dataset [outds]'))
def mathsExp(dss):
    def Exp(outds):
        _applyFn(dss[dsCodeToIdx(outds)], N.exp)
    return Exp

@registerTransformPlugin(
    'Exp10X', _('Raise 10 to the power of X'), category=catlog,
    description=_('Set X dataset to be 10^X'))
def mathsExp10X(dss):
    def Exp10X():
        _applyFn(dss[0], lambda x: 10**x)
    return Exp10X

@registerTransformPlugin(
    'Exp10Y', _('Raise 10 to the power of Y'), category=catlog,
    description=_('Set Y dataset to be 10^Y'))
def mathsExp10Y(dss):
    def Exp10Y():
        _applyFn(dss[1], lambda x: 10**x)
    return Exp10Y

@registerTransformPlugin(
    'Exp10', _('Raise 10 to the power of dataset'), category=catlog,
    description=_('Raise 10 to the power of output dataset [outds]'))
def mathsExp10(dss):
    def Exp10(outds):
        _applyFn(dss[dsCodeToIdx(outds)], lambda x: 10**x)
    return Exp10

@registerTransformPlugin(
    'ExpVX', _('Raise value to the power of X dataset'), category=catlog,
    description=_('Raise value [val] to the power of X dataset'))
def mathsExpVX(dss):
    def ExpVX(val):
        _applyFn(dss[0], lambda x: val**x)
    return ExpVX

@registerTransformPlugin(
    'ExpVY', _('Raise value to the power of Y dataset'), category=catlog,
    description=_('Raise value [val] to the power of Y dataset'))
def mathsExpVY(dss):
    def ExpVY(val):
        _applyFn(dss[1], lambda x: val**x)
    return ExpVY

@registerTransformPlugin(
    'ExpV', _('Raise value to the power of dataset'), category=catlog,
    description=_('Raise value [val] to the power of output dataset [outds]'))
def mathsExpV(dss):
    def ExpV(val, outds):
        _applyFn(dss[dsCodeToIdx(outds)], lambda x: val**x)
    return ExpV

###############################################################################
## Clip

def _clip_dataset(d, minv, maxv):
    """Internal dataset clip range."""
    data = d.data
    if d.serr is not None:
        prange = d.data+d.serr
        nrange = d.data-d.serr
    else:
        prange = d.perr+d.data if d.perr is not None else None
        nrange = d.nerr+d.data if d.nerr is not None else None

    d.data = N.clip(data, minv, maxv)
    if prange is not None:
        d.perr = N.clip(prange, minv, maxv) - d.data
    if nrange is not None:
        d.nerr = N.clip(nrange, minv, maxv) - d.data
    d.serr = None

@registerTransformPlugin(
    'Clip', _('Clip dataset'), category=_('Maths'),
    description=_('Clip output dataset [outds] to lie within range [minv to maxv]'))
def mathsClip(dss):
    def Clip(outds, minv=-N.inf, maxv=N.inf):
        idx = dsCodeToIdx(outds)
        _clip_dataset(dss[idx], minv, maxv)
    return Clip

@registerTransformPlugin(
    'ClipX', _('Clip X dataset'), category=_('Maths'),
    description=_('Clip X dataset values to lie within range [minv to maxv]'))
def mathsClip(dss):
    def ClipX(minv=-N.inf, maxv=N.inf):
        _clip_dataset(dss[0], minv, maxv)
    return ClipX

@registerTransformPlugin(
    'ClipY', _('Clip Y dataset'), category=_('Maths'),
    description=_('Clip Y dataset values to lie within range [minv to maxv]'))
def mathsClip(dss):
    def ClipY(minv=-N.inf, maxv=N.inf):
        _clip_dataset(dss[1], minv, maxv)
    return ClipY

###############################################################################
# Geometry

@registerTransformPlugin(
    'Rotate', _('Rotate coordinates'), category=_('Geometry'),
    description=_('Rotate coordinates by angle in radians [angle_rad], with '
                  'optional centre [cx,cy]'))
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
    description=_('Translate coordinates by given shifts [dx,dy]'))
def geometryTranslate(dss):
    def Translate(dx, dy):
        dss[0].data += dx
        dss[1].data += dy
    return Translate

###############################################################################
# Filter

@registerTransformPlugin(
    'Thin', _('Thin values'), category=_('Filtering'),
    description=_('Thin values by step [step] and optional starting index ([start] from 0)'))
def filteringThin(dss):
    def Thin(step, start=0):
        for ds in dss:
            if ds is None:
                continue
            for attr in 'data', 'serr', 'perr', 'nerr':
                if getattr(ds, attr, None) is not None:
                    setattr(ds, attr, getattr(ds, attr)[start::step])
    return Thin

@registerTransformPlugin(
    'Range', _('Select range'), category=_('Filtering'),
    description=_(
        'Select values between index ranges from start [start], '
        'with optional end index [end] and step '
        '[step] (Python-style indexing from 0)'))
def filteringRange(dss):
    def Range(start, end=None, step=None):
        for ds in dss:
            if ds is None:
                continue
            for attr in 'data', 'serr', 'perr', 'nerr':
                if getattr(ds, attr, None) is not None:
                    setattr(ds, attr, getattr(ds, attr)[start:end:step])
    return Range
