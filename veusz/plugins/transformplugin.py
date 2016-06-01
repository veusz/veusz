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
    try:
        return {
            'x': 0,
            'y': 1,
            'l': 2,
            'label': 2,
            'c': 3,
            'color': 3,
            's': 4,
            'size': 4
        }[code.lower()]
    except KeyError:
        raise ValueError('Unknown dataset code %s' % code)

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
def transformAddX(dss):
    def AddX(val):
        if isDataset1D(val):
            _addSubDataset(dss[0], val, sub=False)
        else:
            dss[0].data += val
    return AddX

@registerTransformPlugin(
    'AddY', _('Add to Y'), category=_('Maths'),
    description=_('Add value or dataset to Y dataset'))
def transformAddY(dss):
    def AddY(val):
        if isDataset1D(val):
            _addSubDataset(dss[1], val, sub=False)
        else:
            dss[1].data += val
    return AddY

@registerTransformPlugin(
    'Add', _('Add to dataset'), category=_('Maths'),
    description=_('Add value or dataset to dataset'))
def transformAdd(dss):
    def Add(ds, val):
        idx = dsCodeToIdx(ds)
        if isDataset1D(val):
            _addSubDataset(dss[idx], val, sub=False)
        else:
            dss[idx].data += val
    return Add

@registerTransformPlugin(
    'SubX', _('Subtract from X'), category=_('Maths'),
    description=_('Subtract value or dataset from X dataset'))
def transformSubX(dss):
    def SubX(val):
        if isDataset1D(val):
            _addSubDataset(dss[0], val, sub=True)
        else:
            dss[0].data -= val
    return SubX

@registerTransformPlugin(
    'SubY', _('Subtract from Y'), category=_('Maths'),
    description=_('Subtract value or dataset from Y dataset'))
def transformSubY(dss):
    def SubY(val):
        if isDataset1D(val):
            _addSubDataset(dss[1], val, sub=True)
        else:
            dss[1].data -= val
    return SubY

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
            d2err2 = d2.serr**2
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
def transformMulX(dss):
    def MulX(val):
        if isDataset1D(val):
            _multiplyDatasetDataset(dss[0], val)
        else:
            _multiplyDatasetScalar(dss[0], val)
    return MulX

@registerTransformPlugin(
    'MulY', _('Multiply Y'), category=_('Maths'),
    description=_('Multiply Y dataset by value or dataset'))
def transformMulY(dss):
    def MulY(val):
        if isDataset1D(val):
            _multiplyDatasetDataset(dss[1], val)
        else:
            _multiplyDatasetScalar(dss[1], val)
    return MulY
