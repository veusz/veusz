# python numerical fitting routine
# Based on fit_lm.m Octave Code

from numarray import matrixmultiply, zeros, transpose, identity, sqrt, array, \
     fabs, arange, Float64
from numarray.linear_algebra import inverse
from numarray.random_array import random

# FIXME diagonal should make matrix with vector on diagonal

def diag(vec):
    assert len(vec.shape) == 1
    return identity( vec.size() ) * vec

def fitLM(func, params, xvals, thedata, weights = 1.):
    """Fit the function with the parameters

    func is passed params, xvals
    params is a numarray of parameters to pass to the function
    xvals are the data points the data are defined at
    thedata are the data points
    weights are proportional to 1/sigma
    """

    eps = 1e-5
    minstep = eps
    minover = 1. - sqrt(eps)
    maxeval = 1000
    dstep = sqrt(eps)

    alp = 10.

    val = func(params, xvals)*weights
    data = thedata * weights
    neval = 1
    new = matrixmultiply( transpose(val-data).conjugate(), val-data )

    best = new+1
    dpar = array( (2*minstep,) )

    while ( ((fabs(dpar)).mean() > minstep or
             new / best < minover) and
            neval < maxeval ):

        if new <= best:
            print params, new, alp
            best = new

            deri = zeros( (val.size(), params.size()), Float64 )
            for j in range(params.size()):
                # fix by jss to get variable stepping to calculate
                # derivative
                step = params[j] * 0.001
                if step < 1e-10:
                    step = 1e-10

                dpar = zeros( params.size(), Float64 )
                dpar[j] = step
                params = params - dpar
                val1 = func(params, xvals)
                params = params + 2*dpar
                val2 = func(params, xvals)
                params = params - dpar
                deri[:,j] = (val2-val1) * weights / (2*step)

            neval += 2*params.size()

            korr = matrixmultiply( transpose(deri).conjugate(), deri )
            korrdiag = diag( 1. / sqrt(korr.diagonal()) )
            korr = matrixmultiply(matrixmultiply(korrdiag, korr), korrdiag)
            alp *= 0.1

        else:
            alp *= 10.
            params = parbak

        inv = matrixmultiply(korrdiag, inverse(korr + alp *
                                               diag(korr.diagonal())) )
        inv = matrixmultiply(inv, korrdiag)
        dpar = matrixmultiply(inv,
                              (matrixmultiply(transpose(deri).conjugate(),
                                              val-data)).getreal())
        parbak = params.copy()
        params = params - dpar
        val = func(params, xvals)*weights
        neval += 1
        new = matrixmultiply(transpose(val-data).conjugate(), val-data)

    print params
    #return (fabs(dpar)).mean() > minstep or new/best < minover
    return params

## def testfunc(params, xvals):
##     return params[0] + params[1]*xvals + params[2]*xvals**2

## xvals = arange(0.,10.,0.1)
## yvals = (xvals**2)*0.01 + xvals*2 + (random(xvals.shape)-0.5)*3. + 100.
## errors = 3.

## inparams = array((0.0, 1.0, 0.0))

## fitLM(testfunc, inparams, xvals, yvals)
## fitLM(testfunc, inparams, xvals, yvals)

### JSS version

def computeChi2(func, params, xvals, yvals, errors):
    """Compute the chi2 for the given data."""
    
    ymodel = func(params, xvals)
    return ((ymodel-yvals)**2) / (errors**2)

def fit(func, params, xvals, yvals, errors):

    lamda = 1e-4
    
