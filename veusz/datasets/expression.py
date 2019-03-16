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

"""For evaluating dataset expressions and dataset classes using expressions."""

from __future__ import division
import re
import numpy as N

from .commonfn import _
from .base import DatasetExpressionException
from .oned import Dataset1DBase, Dataset
from .twod import Dataset2DBase, Dataset2D
from .text import DatasetText

from ..compat import czip, crange, cstr, crepr
from .. import utils

# split expression on python operators or quoted `DATASET`
dataexpr_split_re = re.compile(r'(`.*?`|[\.+\-*/\(\)\[\],<>=!|%^~& ])')
# identify whether string is a quoted identifier
dataexpr_quote_re = re.compile(r'^`.*`$')
dataexpr_columns = {'data':True, 'serr':True, 'perr':True, 'nerr':True}

def substituteDatasets(datasets, expression, thispart):
    """Substitute the names of datasets with calls to a function which will
    evaluate them.

    Returns (new expression, list of substituted datasets)
    """

    # split apart the expression to look for dataset names
    bits = dataexpr_split_re.split(expression)

    dslist = []
    for i, bit in enumerate(bits):
        # test whether there's an _data, _serr or such at the end of the name
        part = thispart

        if dataexpr_quote_re.match(bit):
            # quoted text, so remove backtick-"quotes"
            bit = bit[1:-1]
            # replace name with a function to call
            bits[i] = "_DS_(%s, %s)" % (crepr(bit), crepr(part))
            dslist.append(bit)

        else:
            bitbits = bit.split('_')
            if len(bitbits) > 1:
                if bitbits[-1] in dataexpr_columns:
                    part = bitbits.pop(-1)
                bit = '_'.join(bitbits)
            if bit in datasets:
                # replace name with a function to call
                bits[i] = "_DS_(%s, %s)" % (crepr(bit), crepr(part))
                dslist.append(bit)
            
    return ''.join(bits), dslist

def _evaluateDataset(datasets, dsname, dspart):
    """Return the dataset given.

    dsname is the name of the dataset
    dspart is the part to get (e.g. data, serr)
    """
    if dspart not in dataexpr_columns:
        raise DatasetExpressionException(
            'Internal error - invalid dataset part')
    if dsname not in datasets:
        raise DatasetExpressionException(
            _("Dataset '%s' is not defined") % (dsname, ))
    val = getattr(datasets[dsname], dspart)
    if val is None:
        raise DatasetExpressionException(
            _("Dataset '%s' does not have part '%s'") % (dsname, dspart))
    return val

def _returnNumericDataset(doc, vals, dimensions, subdatasets):
    """Used internally to convert a set of values (which needs to be
    numeric) into a Dataset.

    subdatasets is list of datasets substituted into expression
    """

    err = None

    # try to convert array to a numpy array
    try:
        vals = N.array(vals, dtype=N.float64)
    except (ValueError, TypeError) as e:
        err = _('Could not convert to array')

    # if error on first time, try to sanitize input arrays
    if err and dimensions == 1:
        try:
            vals = list(vals)
            vals[0] = N.array(vals[0])
            minlen = len(vals[0])
            if len(vals) in (2,3):
                # expand/convert error bars
                for i in crange(1, len(vals)):
                    if N.isscalar(vals[i]):
                        # convert to scalar
                        vals[i] = N.zeros(minlen) + vals[i]
                    else:
                        # convert to array
                        vals[i] = N.array(vals[i])
                        if vals[i].ndim != 1:
                            raise ValueError
                    minlen = min(minlen, len(vals[i]))

                # chop to minimum length
                for i in crange(len(vals)):
                    vals[i] = vals[i][:minlen]
            vals = N.array(vals, dtype=N.float64)
            err = None
        except (ValueError, IndexError, TypeError):
            pass

    if not err:
        if dimensions == 1:
            # see whether data values suitable for a 1d dataset
            if vals.ndim == 1:
                # 1d, so ok
                return Dataset(data=vals)
            elif vals.ndim == 0:
                # single value
                return Dataset(data=[vals])
            elif vals.ndim == 2:
                # 2d, see whether data are error bars
                if vals.shape[0] == 2:
                    return Dataset(
                        data=vals[0,:], serr=vals[1,:])
                elif vals.shape[0] == 3:
                    return Dataset(
                        data=vals[0,:], perr=vals[1,:],
                        nerr=vals[2,:])
                else:
                    err = _('Expression has wrong dimensions')
        elif dimensions == 2 and vals.ndim == 2:
            # try to use dimensions of first-substituted dataset
            dsrange = {}
            for ds in subdatasets:
                d = doc.data[ds]
                if d.dimensions == 2:
                    for p in ('xrange', 'yrange', 'xedge', 'yedge',
                              'xcent', 'ycent'):
                        dsrange[p] = getattr(d, p)
                    break

            return Dataset2D(vals, **dsrange)
        else:
            err = _('Expression has wrong dimensions')

    raise DatasetExpressionException(err)

def evalDatasetExpression(doc, origexpr, datatype='numeric',
                          dimensions=1, part='data'):
    """Evaluate expression and return an appropriate Dataset.

    part is 'data', 'serr', 'perr' or 'nerr' - these are the
    dataset parts which are evaluated by the expression

    Returns None if error
    """

    if not origexpr:
        # ignore blank names
        return None

    d = doc.data.get(origexpr)

    if ( d is not None and
         d.datatype == datatype and
         d.dimensions == dimensions ):
        return d

    # support nD datasets by converting to requested shape
    if d is not None and d.dimensions == -1 and d.data.ndim == dimensions:
        if dimensions == 1:
            return Dataset(d.data)
        elif dimensions == 2:
            return Dataset2D(d.data)

    if utils.id_re.match(origexpr):
        # if name is a python identifier, then it has to be a dataset
        # name. As it wasn't there, just return with nothing rather
        # than print error message.
        return None

    # replace dataset names by calls to _DS_(name,part)
    expr, subdatasets = substituteDatasets(doc.data, origexpr, part)

    comp = doc.evaluate.compileCheckedExpression(expr, origexpr=origexpr)
    if comp is None:
        return

    # set up environment for evaluation
    env = doc.evaluate.context.copy()
    def doeval(dsname, dspart):
        return _evaluateDataset(doc.data, dsname, dspart)
    env['_DS_'] = doeval

    # do evaluation
    try:
        evalout = eval(comp, env)
    except Exception as ex:
        doc.log(_("Error evaluating '%s': '%s'" % (origexpr, cstr(ex))))
        return None

    # return correct dataset for data type
    try:
        if datatype == 'numeric':
            return _returnNumericDataset(doc, evalout, dimensions, subdatasets)
        elif datatype == 'text':
            return DatasetText([cstr(x) for x in evalout])
        else:
            raise RuntimeError('Invalid data type')
    except DatasetExpressionException as ex:
        doc.log(_("Error evaluating '%s': %s\n") % (origexpr, cstr(ex)))

    return None

class DatasetExpression(Dataset1DBase):
    """A dataset which is linked to another dataset by an expression."""

    dstype = _('Expression')

    def __init__(self, data=None, serr=None, nerr=None, perr=None,
                 parametric=None):
        """Initialise the dataset with the expressions given.

        parametric is option and can be (minval, maxval, steps) or None
        """

        Dataset1DBase.__init__(self)

        # store the expressions to use to generate the dataset
        self.expr = {}
        self.expr['data'] = data
        self.expr['serr'] = serr
        self.expr['nerr'] = nerr
        self.expr['perr'] = perr
        self.parametric = parametric

        self.docchangeset = -1
        self.evaluated = {}

    def evaluateDataset(self, dsname, dspart):
        """Return the dataset given.

        dsname is the name of the dataset
        dspart is the part to get (e.g. data, serr)
        """
        return _evaluateDataset(self.document.data, dsname, dspart)

    def _evaluatePart(self, expr, part):
        """Evaluate expression expr for part part.

        Returns True if succeeded
        """
        # replace dataset names with calls
        newexpr = substituteDatasets(self.document.data, expr, part)[0]

        comp = self.document.evaluate.compileCheckedExpression(
            newexpr, origexpr=expr)
        if comp is None:
            return False

        # set up environment to evaluate expressions in
        environment = self.document.evaluate.context.copy()

        # create dataset using parametric expression
        if self.parametric:
            p = self.parametric
            if p[2] >= 2:
                deltat = (p[1]-p[0]) / (p[2]-1)
                t = N.arange(p[2])*deltat + p[0]
            else:
                t = N.array([p[0]])
            environment['t'] = t

        # this fn gets called to return the value of a dataset
        environment['_DS_'] = self.evaluateDataset

        # actually evaluate the expression
        try:
            result = eval(comp, environment)
            evalout = N.array(result, N.float64)

            if len(evalout.shape) > 1:
                raise RuntimeError("Number of dimensions is not 1")
        except Exception as ex:
            self.document.log(
                _("Error evaluating expression: %s\n"
                  "Error: %s") % (self.expr[part], cstr(ex)) )
            return False

        # make evaluated error expression have same shape as data
        if part != 'data':
            data = self.evaluated['data']
            if evalout.shape == ():
                # zero dimensional - expand to data shape
                evalout = N.resize(evalout, data.shape)
            else:
                # 1-dimensional - make it right size and trim
                oldsize = evalout.shape[0]
                evalout = N.resize(evalout, data.shape)
                evalout[oldsize:] = N.nan
        else:
            if evalout.shape == ():
                # zero dimensional - make a single point
                evalout = N.resize(evalout, 1)

        self.evaluated[part] = evalout
        return True

    def updateEvaluation(self):
        """Update evaluation of parts of dataset.

        Returns False if problem with any evaluation
        """
        ok = True
        if self.docchangeset != self.document.changeset:
            # avoid infinite recursion!
            self.docchangeset = self.document.changeset

            # zero out previous values
            for part in self.columns:
                self.evaluated[part] = None

            # update all parts
            for part in self.columns:
                expr = self.expr[part]
                if expr is not None and expr.strip() != '':
                    ok = ok and self._evaluatePart(expr, part)

        return ok

    def _propValues(self, part):
        """Check whether expressions need reevaluating,
        and recalculate if necessary."""

        self.updateEvaluation()

        # catch case where error in setting data, need to return "real" data
        if self.evaluated['data'] is None:
            self.evaluated['data'] = N.array([])
        return self.evaluated[part]

    # expose evaluated data as properties
    # this allows us to recalculate the expressions on the fly
    data = property(lambda self: self._propValues('data'))
    serr = property(lambda self: self._propValues('serr'))
    perr = property(lambda self: self._propValues('perr'))
    nerr = property(lambda self: self._propValues('nerr'))

    def saveDataRelationToText(self, fileobj, name):
        '''Save data to file.
        '''

        parts = [crepr(name), crepr(self.expr['data'])]
        if self.expr['serr']:
            parts.append('symerr=%s' % crepr(self.expr['serr']))
        if self.expr['nerr']:
            parts.append('negerr=%s' % crepr(self.expr['nerr']))
        if self.expr['perr']:
            parts.append('poserr=%s' % crepr(self.expr['perr']))
        if self.parametric is not None:
            parts.append('parametric=%s' % crepr(self.parametric))

        parts.append('linked=True')

        s = 'SetDataExpression(%s)\n' % ', '.join(parts)
        fileobj.write(s)

    def __getitem__(self, key):
        """Return a dataset based on this dataset

        We override this from DatasetConcreteBase as it would return a
        DatsetExpression otherwise, not chopped sets of data.
        """
        return Dataset(**self._getItemHelper(key))

    def canUnlink(self):
        """Whether dataset can be unlinked."""
        return True

    def linkedInformation(self):
        """Return information about linking."""
        text = []
        if self.parametric:
            text.append(_('Linked parametric dataset'))
        else:
            text.append(_('Linked expression dataset'))
        for label, part in czip(self.column_descriptions,
                                self.columns):
            if self.expr[part]:
                text.append('%s: %s' % (label, self.expr[part]))

        if self.parametric:
            text.append(_("where t goes from %g:%g in %i steps") % self.parametric)

        return '\n'.join(text)

def getSpacing(data):
    """Given a set of values, get minimum, maximum, step size
    and number of steps.

    Function allows that values may be missing

    Function assumes that at least one of the steps is the minimum step size
    (i.e. steps are not all multiples of some mininimum)
    """

    try:
        data = N.array(data) + 0
    except ValueError:
        raise DatasetExpressionException('Expression is not an array')

    if len(data.shape) != 1:
        raise DatasetExpressionException('Array is not 1D')
    if len(data) < 2:
        raise DatasetExpressionException('Two values required to convert to 2D')

    uniquesorted = N.unique(data)

    sigfactor = (uniquesorted[-1]-uniquesorted[0])*1e-13

    # differences between elements
    deltas = N.unique( N.ediff1d(uniquesorted) )

    mindelta = None
    for delta in deltas:
        if delta > sigfactor:
            if mindelta is None:
                # first delta
                mindelta = delta
            elif N.fabs(mindelta-delta) > sigfactor:
                # new delta - check is multiple of old delta
                ratio = delta/mindelta
                if N.fabs(int(ratio)-ratio) > 1e-3:
                    raise DatasetExpressionException(
                        'Variable spacings not yet supported '
                        'in constructing 2D datasets')

    if mindelta is None or mindelta == 0:
        raise DatasetExpressionException('Could not identify delta')

    return (uniquesorted[0], uniquesorted[-1], mindelta,
            int((uniquesorted[-1]-uniquesorted[0])/mindelta)+1)

class Dataset2DXYZExpression(Dataset2DBase):
    '''A 2d dataset with expressions for x, y and z.'''

    dstype = _('2D XYZ')

    def __init__(self, exprx, expry, exprz):
        """Initialise dataset.

        Parameters are mathematical expressions based on datasets."""
        Dataset2DBase.__init__(self)

        self.lastchangeset = -1
        self.cacheddata = None
        self.xedge = self.yedge = self.xcent = self.ycent = None

        # copy parameters
        self.exprx = exprx
        self.expry = expry
        self.exprz = exprz

    def evaluateDataset(self, dsname, dspart):
        """Return the dataset given.

        dsname is the name of the dataset
        dspart is the part to get (e.g. data, serr)
        """
        return _evaluateDataset(self.document.data, dsname, dspart)

    def evalDataset(self):
        """Return the evaluated dataset."""

        # FIXME: handle irregular grids
        # return cached data if document unchanged
        if self.document.changeset == self.lastchangeset:
            return self.cacheddata
        self.lastchangeset = self.document.changeset
        self.cacheddata = None

        evaluated = {}

        environment = self.document.evaluate.context.copy()
        environment['_DS_'] = self.evaluateDataset

        # evaluate the x, y and z expressions
        for name in ('exprx', 'expry', 'exprz'):
            origexpr = getattr(self, name)
            expr = substituteDatasets(self.document.data, origexpr, 'data')[0]

            comp = self.document.evaluate.compileCheckedExpression(
                expr, origexpr=origexpr)
            if comp is None:
                return None

            try:
                evaluated[name] = eval(comp, environment)
            except Exception as e:
                self.document.log(_("Error evaluating expression: %s\n"
                                    "Error: %s") % (expr, cstr(e)) )
                return None

        minx, maxx, stepx, stepsx = getSpacing(evaluated['exprx'])
        miny, maxy, stepy, stepsy = getSpacing(evaluated['expry'])

        # update cached x and y ranges
        self._xrange = (minx-stepx*0.5, maxx+stepx*0.5)
        self._yrange = (miny-stepy*0.5, maxy+stepy*0.5)

        self.cacheddata = N.empty( (stepsy, stepsx) )
        self.cacheddata[:,:] = N.nan
        xpts = ((1./stepx)*(evaluated['exprx']-minx)).astype('int32')
        ypts = ((1./stepy)*(evaluated['expry']-miny)).astype('int32')

        # this is ugly - is this really the way to do it?
        try:
            self.cacheddata.flat [ xpts + ypts*stepsx ] = evaluated['exprz']
        except Exception as e:
            self.document.log(_("Shape mismatch when constructing dataset\n"
                                "Error: %s") % cstr(e) )
            return None

        return self.cacheddata

    @property
    def xrange(self):
        """Get x range of data as a tuple (min, max)."""
        return self.getDataRanges()[0]

    @property
    def yrange(self):
        """Get y range of data as a tuple (min, max)."""
        return self.getDataRanges()[1]

    def getDataRanges(self):
        """Get both ranges of axis."""
        ds = self.evalDataset()
        if ds is None:
            return ( (0., 1.), (0., 1.) )
        return (self._xrange, self._yrange)

    @property
    def data(self):
        """Get data, or none if error."""
        ds = self.evalDataset()
        if ds is None:
            return N.array( [[]] )
        return ds

    def saveDataRelationToText(self, fileobj, name):
        '''Save expressions to file.
        '''

        s = 'SetData2DExpressionXYZ(%s, %s, %s, %s, linked=True)\n' % (
            crepr(name), crepr(self.exprx), crepr(self.expry), crepr(self.exprz) )
        fileobj.write(s)

    def canUnlink(self):
        """Can relationship be unlinked?"""
        return True

    def linkedInformation(self):
        """Return linking information."""
        return _('Linked 2D function: x=%s, y=%s, z=%s') % (
            self.exprx, self.expry, self.exprz)

class Dataset2DExpression(Dataset2DBase):
    """Evaluate an expression of 2d datasets."""

    dstype = _('2D Expr')

    def __init__(self, expr):
        """Create 2d expression dataset."""

        Dataset2DBase.__init__(self)

        self.expr = expr
        self.lastchangeset = -1

    @property
    def data(self):
        """Return data, or empty array if error."""
        ds = self.evalDataset()
        return ds.data if ds is not None else N.array([[]])

    @property
    def xrange(self):
        """Return x range."""
        ds = self.evalDataset()
        return ds.xrange if ds is not None else [0., 1.]

    @property
    def yrange(self):
        """Return y range."""
        ds = self.evalDataset()
        return ds.yrange if ds is not None else [0., 1.]

    @property
    def xedge(self):
        """Return x grid points."""
        ds = self.evalDataset()
        return ds.xedge if ds is not None else None

    @property
    def yedge(self):
        """Return y grid points."""
        ds = self.evalDataset()
        return ds.yedge if ds is not None else None

    @property
    def xcent(self):
        """Return x cent points."""
        ds = self.evalDataset()
        return ds.xcent if ds is not None else None

    @property
    def ycent(self):
        """Return y cent points."""
        ds = self.evalDataset()
        return ds.ycent if ds is not None else None

    def evalDataset(self):
        """Do actual evaluation."""
        return evalDatasetExpression(self.document, self.expr, dimensions=2)

    def saveDataRelationToText(self, fileobj, name):
        '''Save expression to file.'''
        s = 'SetData2DExpression(%s, %s, linked=True)\n' % (
            crepr(name), crepr(self.expr) )
        fileobj.write(s)

    def canUnlink(self):
        """Can relationship be unlinked?"""
        return True

    def linkedInformation(self):
        """Return linking information."""
        return _('Linked 2D expression: %s') % self.expr
