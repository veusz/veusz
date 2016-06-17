#    Copyright (C) 2005 Jeremy S. Sanders
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

"""
Numerical fitting of functions to data.
"""

from __future__ import division, print_function
import sys

import numpy as N
try:
    import numpy.linalg as NLA
except:
    import scipy.linalg as NLA

from ..compat import crange

def fitLM(func, params, xvals, yvals, errors,
          stopdeltalambda = 1e-5,
          deltaderiv = 1e-5, maxiters = 20, Lambda = 1e-4):

    """
    Use Marquardt method as described in Bevington & Robinson to fit data

    func is a python function to evaluate. It takes two parameters, the
    parameters to fit as a numpy, and the values of x to evaluate the
    function at

    params is a numpy of parameters to fit for. These are passed to
    the function.

    xvals are x data points (numpy), yvals are the y data points (numarray)
    errors are a numpy of errors on the y data points.
    Set all to 1 if not important.

    stopdeltalambda: minimum change in chi2 to carry on fitting
    deltaderiv: change to make in parameters to calculate derivative
    maxiters: maximum number of better fitting solutions before stopping
    Lambda: starting lambda value (as described in Bevington)
    """

    # optimisation to avoid computing this all the time
    inve2 = 1. / errors**2

    # work out fit using current parameters
    oldfunc = func(params, xvals)
    chi2 = ( (oldfunc - yvals)**2 * inve2 ).sum()

    # initialise temporary space
    beta = N.zeros( len(params), dtype='float64' )
    alpha = N.zeros( (len(params), len(params)), dtype='float64' )
    derivs = N.zeros( (len(params), len(xvals)), dtype='float64' )

    done = False
    iters = 0
    while iters < maxiters and not done:
        # iterate over each of the parameters and calculate the derivatives
        # of chi2 to populate the beta vector

        # also calculate the derivative of the function at each of the points
        # wrt the parameters

        for i in crange( len(params) ):
            params[i] += deltaderiv
            new_func = func(params, xvals)
            chi2_new = ((new_func - yvals)**2 * inve2).sum()
            params[i] -= deltaderiv

            beta[i] = chi2_new - chi2
            derivs[i] = new_func - oldfunc

        # beta is now dchi2 / dparam
        beta *= (-0.5 / deltaderiv)
        derivs *= (1. / deltaderiv)

        # calculate alpha matrix
        # FIXME: stupid - must be a better way to avoid this iteration
        for j in crange( len(params) ):
            for k in crange(j+1):
                v = (derivs[j]*derivs[k] * inve2).sum()
                alpha[j][k] = v
                alpha[k][j] = v

        # twiddle alpha using lambda
        alpha *= 1. + N.identity(len(params), dtype='float64')*Lambda

        # now work out deltas on parameters to get better fit
        deltas = NLA.solve(alpha, beta)

        # new solution
        new_params = params+deltas
        new_func = func(new_params, xvals)
        new_chi2 = ( (new_func - yvals)**2 * inve2 ).sum()

        if N.isnan(new_chi2):
            sys.stderr.write('Chi2 is NaN. Aborting fit.\n')
            break

        if new_chi2 > chi2:
            # if solution is worse, increase lambda
            Lambda *= 10.
        else:
            # better fit, so we accept this solution

            # if the change is small
            done = chi2 - new_chi2 < stopdeltalambda

            chi2 = new_chi2
            params = new_params
            oldfunc = new_func
            Lambda *= 0.1

            # format new parameters
            iters += 1
            p = [iters, chi2] + params.tolist()
            str = ("%5i " + "%8g " * (len(params)+1)) % tuple(p)
            print(str)

    if not done:
        sys.stderr.write("Warning: maximum number of iterations reached\n")

    # print out fit statistics at end
    dof = len(yvals) - len(params)
    redchi2 = chi2 / dof
    print("chi^2 = %g, dof = %i, reduced-chi^2 = %g" % (chi2, dof, redchi2))

    return (params, chi2, dof)

