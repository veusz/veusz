#!/usr/bin/env python

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Library General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
############################################################################

# $Id$

####################################################
# code modified from the Gnumeric regression code  #
# by Daniel Carrera <dcarrera@math.toronto.edu>    #
# which was released under the GPL                 #
####################################################

from math import *
from random import random
from numarray import *

# Constants
OK    = 0
ERROR = 1
NULL  = 0
DELTA = 1e-5


"""
HOW TO DEFINE A FUNCTION FOR REGRESSION
	
The regression code here can find best-fit parameters for any
smooth user-defined function.  This function might be of any
dimension and it may have any number of parameters.


However, there is a certain form that your function is expected to take:

The calling form of the function must be  'y = f(x,par)'

Where:
x  ==	The point where the function is evaluated.
        This point might well be an n-vector, a matrix
        or whatever.

par ==	Parameters that the function takes.
        This must be an array.

y ==	The value of the function at that point.  That is,
        the function must return the result.
	

	EXAMPLE:

	This function implements a generic sine function in two dimensions.

	def f(X,par)
		x,y = X
		a,b,c,d,e = par

		return a*sin(b*x + c*y + d) + e
	
	It could be called with:
	z = f( [1,2],  [1,2,3,4,5])
	or
	x = [1,2]
	s = [1,2,3,4,5]
	z = f(x,s)
"""

####################
#
#  Generic matrix functions
#
####################

def _alloc_matrix(m,p):
    return zeros([m,p],Float64)

def _alloc_vector(m):
    return zeros(m,Float64)

def _add_vector(a,b):
    numarray = array((2,2))
    if type(a) == type(b) == type(numarray):
        return add(a,b)
    
    # Otherwise return a regular Python array.
    if len(b) != len(a):
        print "ERROR:  Vectors to add are not the same size"
        exit(0)
	
    m = len(a)
    ans = range(m)
    for i in range(m):
        ans[i] = a[i] + b[i]
	
    return ans
	
def linear_solve(A,b):
    '''
    x = linear_solve(A,b)

	Returns the solution to the system A*x = b.
	
	A  == A 2-dimensional square matrix.  It must be a numarray.  A python
	      array *will not work*.
	b  == A vector (1-dim array).  It must be a numarray.  A python array 
		  *will not work*
	
	The x returned is a numarray.
        '''
	
    if len(A.shape) != 2 or len(b.shape) != 1:
        return ERROR
    
    rows,cols = A.shape
    b_rows    = len(b)
    
    if rows != cols or rows !=  b_rows:
        return ERROR
    
    n = rows  # A more intuitive name.
    
    # Now create the augmented matrix [A|b]
    Ab = _alloc_matrix(n,n+1)  # One more column to fit 'b'.
    for row in range(n):
        for col in range(n):
            Ab[row,col] = A[row,col]
        Ab[row,n] = b[row]
                
    # Turn [A|b] into reduced echelon form.
    for index in range(n):
        Ab = gauss_jordan_iterate(Ab,index)
		
    # Now return only the last column of the matrix.
    x = Ab[:,-1]
    return x

def gauss_jordan_iterate(A,index):
    """
    A = gauss_jordan_iterate(A,index)
          
    Performs the Gauss-Jordan reduction for the given 'index'.
    It assumes that all previous indices are already reduced.
    
    A     == A 2-dimensional numarray.  A python array *will not work*.
    index == an integer.
	
    """
    rows,cols = A.shape
        
    # If A[index][index] == 0 we cannot divide by it.
    if A[index,index] == 0:
                
        singular_matrix = True # Matrix might be singular.
                
        # Look for a non-zero entry below this row.
        for r in range(index,rows):
            if A[r,index] != 0:
                A[index],A[r] = A[r].copy(),A[index].copy() # Switch rows.
                singular_matrix = False       # Crisis adverted.
                break   # Exit the for loop.
            
        if singular_matrix == True:
            return SINGULAR_MATRIX_ERROR
        
    # Normalise the row A[index] so that A[index][index] == 1.
    factor = A[index,index]  # No need for .copy()
    f1 = 1.0 / factor
    for col in range(cols):
        A[index,col] *= f1
        
    # Now go through all the other rows and do the subtractions.
    for row in range(rows):
        if row == index:
            continue # Skip this iteration.
        
        factor = A[row,index] # No need for .copy()
        for col in range(cols):
            A[row,col] -= factor*A[index,col]
            
    # We have finished reducing for this pivot.
    return A

#####################
#
#  Supporting Functions
#
#####################

def derivative(f,x,par_orig,index):
    '''
    df = derivative(f,x,par,i)

    f   ->  any (smooth) user-defined function.
    x   ->  point where to take the derivative.
    par ->  values of the parameters.
    i   ->  differentiate with respect to the ith parameter,
            numbered from 0 to (number_of_parameters - 1)

    See the beginning of this document for details on
    how f must be defined.
    '''
    par = list( par_orig )
    par[index] -= DELTA
    y1 = f( *([x] + par) )
	
    par[index] += 2*DELTA
    y2 = f( *([x] + par) )
	
    df = 1.0*(y2-y1)/(2*DELTA)
    return df

def chi_squared(f,x,par,y,sigma):
    """
    chi_sqrd = def chi_squared(f,xvals,par,yvals,sigmas)
    
    xvals,yvals,par,sigma_vals may all be any either numarrays or regular python arrays.
	   
    xvals  == array whose elements are the input for f.
    yvals  == array whose elements are the y values.
    par    == array whose elements are the parameters.
    sigmas == array whose elements are the the errors.
	   
    f      ==   User-defined function which can be called with the form:
               'f(xvals[k],par)' for an integer k.
	   
	   
	                      /  y  - f(x ; par) \ 2
	      2              |    i      i        |
	   Chi   ==   Sum (  | ------------------ |   )
	                      \     sigma        /
	                                 i
	
    The Chi Squared measures the 'goodness of fit'.  To interpret it you need to 
    compute the corresponding p-value (which also depends on your degrees of freedom).
    
    This value is MEANINGLESS if you don't provide errors.  The only reason why you can
    call this function without giving 'sigma' is that we can still use the number to get
    a set of best fit parameters (but they won't be as good).
    """
    
    chi_sqrd = 0.
    n = len(x)   # This works for both numarrays and regular arrays.
    
    lp = list(par)
    for i in range(n):
        parms = [ x[i] ] + lp
        chi_sqrd += ( ( y[i] - f(*parms) ) / sigma[i]  )**2
        
    return chi_sqrd

def chi_derivative(f,x,par_orig,index,y,sigma):
    '''
    dchi = chi_derivative(f,x,par,index,y,sigma)
    
    This is a simple adaptation of the derivative() function
    specific to the Chi squared.
    '''
    
    par = par_orig.copy()
    par[index] -= DELTA
    y1 = chi_squared(f,x,par,y,sigma)
    
    par[index] += 2*DELTA
    y2 = chi_squared(f,x,par,y,sigma)
    
    return 1.0*(y2-y1)/(2*DELTA)

def coeff_matrix(f,x,par,y,sigma,r):
    """
		This matrix is defined by
		
		       N        1      df  df
		A   = Sum  ( -------   --  --  ( i == j? 1 + r : 1 )  )
		  ij  k=1    sigma^2   dp  dp
		                  k      i   j
		
		In the event that the errors are not known, we are forced
		to assume that they are all equal and factor them out.
		
		r is a positive constant used to give a particular weight to
		the diagonal elements.  It's value is altered during the
		procedure.
    """
    p = len(par) #  len() works for both numarrays and regular arrays.
    N = len(y)   #
    
    A = _alloc_matrix(p,p)
    
    # Compute the entries of this *symmetric* matrix.
    for i in range(p):
        # for j == 0..i-1
        for j in range(i):
            tmp = 0
            for k in range(N):
                df_i = derivative(f,x[k],par,i)
                df_j = derivative(f,x[k],par,j)
                tmp += (1/sigma[k]**2)*df_i*df_j
                
            A[j][i] = A[i][j] = tmp
            
        # now for j == i
        tmp = 0
        for k in range(N):
            df_ii = derivative(f,x[k],par,i)
            tmp += (1/sigma[k]**2)*df_ii*df_ii*(1+r)
        A[i][i] = tmp
        
    return A


####################
#
#  Levenberg-Marquardt
#
#####################


def non_linear_regression(f,x,par_orig,y,sigma=None):
    """
    err,chi,par = non_linear_regression(f,x,par,y,sigma)
		
    Performs the non-linear regression on the function f.
    The optimum parameters are placed back into 'par'.
		
    Parameters:
    ===========
    
    x     == array of x values (independent variables).
    par   == initial guess as to the parameters.
    y     == array of corresponding y values (dependent variable).
    sigma == array of corresponding y errors (optional) 
    
    Notice that providing errors 'sigma' not only gives a better
    fit, but without them, the chi value is meaningless.
		
		
    Error
    =====
		
    This is the Standard Deviation of the sample.  That is, 
    it is the square root of the sum of the residuals divided by
    N-1.  Where N is the number of data points.
		
    IMPORTANT:
		
    The error is not the be all	of your analysis.  The Chi Squared
    is just as important, as that is what tells you if you are doing 
    the experiment wrong.
		
		
    Chi Squared  ('goodness of fit'  - lower values are better).
    =========== 
		
    This is a very IMPORTANT value.  It tells you if you are doing the experiment 
    wrong. To interpret it you need to find the corresponding p-value.  There is 
    a p-value calculator at:
		
    http://ergo.ucsd.edu/unixstats/probcalc/index.shtml
			
    degrees of freedom = number of data points  -  number of parameters
		
    The p-value is the probability that you'd get that a Chi Squared as high or higher
    if your experiment is right. In general, a p-value as low as 0.001 can be accepted
    on occasion, but not frequently.  Lower values are too low.
		
    What if the p-value is too low?
    -------------------------------
    There are two possible reasons:
    
    i)  The model is wrong.
    ii) The errors have been underestimated.
			
    What is the p-value is too high (near 1)?
    -----------------------------------------
    Believe it or not, this is possible.  In real life there are errors.  You should be 
    weary of results that are 'too good to be true'.
    
    i)  Your errors are probably too high.
    """
    
    m = len(x)         # It important that x,y,par be regular arrays.
    p = len(par_orig)  # This gives the user more flexibility.
	
    # Python is a pass-by-reference language (which I really hate).  Here I
    # turn the parameters into a numarray so I can use the .copy() method.
    par = _alloc_vector(p)
    for i in range(p):
        par[i] = par_orig[i]

    # we have no sigmas
    if sigma == None:
        print "No sigmas specified"
        sigma = ones(m)
	
    # Initial values of r and Chi squared.
    r = 0.001
    chi_pre = chi_squared(f,x,par,y,sigma)
    chi_pos = chi_pre*10 # All we need is chi_pos/chi_pre > 1.01
	
    #         d Chi
    # b   ==  -----
    #  k       d p
    #             k
    b = _alloc_vector(p)
    for k in range(p):
        b[k] = - chi_derivative(f,x,par,k,y,sigma)/2
	
    #print "Chi Squared : ",chi_pre
    while 1:
        A = coeff_matrix(f,x,par,y,sigma,r)
        
        dpar = linear_solve(A,b)
        if type(dpar) == type(ERROR):
            print "ERROR:  Could solve the linear system"
            exit(0)
            
        tmp_par = _add_vector(par, dpar)
        
        if type(tmp_par) == type(ERROR):
            print "ERROR: Could not add vectors"
            exit(0)
            
        chi_pos = chi_squared(f,x,tmp_par,y,sigma)
        
        if chi_pos > chi_pre:  # worse
            r *= 2.

        else:                  # better
            print "Chi Squared : ",chi_pre
            print "Chi Squared : ",chi_pos
            print "r    == ",r
            print ""
            
            if (chi_pre - chi_pos) < 1e-3:
                break
            r /= 2.
            par = tmp_par
            chi_pre = chi_pos

        
    residuals = 0
    lp = list(par)
    for i in range(m):
        residuals += (y[i] - f( *([x[i]] + lp) ) )**2
	
    StdDev = sqrt(residuals)/(m-1)
    
    return StdDev,chi_pos,par

''' Artificial Separation. '''
#####################
#
#	 M A I N 
#
#####################

"""
#	FUNCTIONS TESTED INDEPENDENTLY
#
#	- linear_solve() and gauss_jordan()
#	- derivative()
#	- chi_squared()

# Used to test linear_solve() and gauss_jordan_iterate() with numarrays
A = array((  \
	[1,2,3], \
	[1,1,2], \
	[3,2,1]  \
	))
# x == [1,2,1]
b = array((8,5,8))
x = linear_solve(A,b)
print x

# Used to test derivative()
par = [1,1,1]
x   = 1
df  = derivative(f,x,par,1)
print df

#  You are free to use either numarrays or regular python arrays.
#  The only conditions are (assume you chose the names 'xvals' and 'yvals'):
#	-	yvals[k]  be the y-value corresponding to xvals[k]
#	-	Your function must accept an 'xvals[k]' in its imput, because that's how
#	        it will be called.
#
#   NOTE:
#	You will find that numarrays will give you higher precision.  However, in some cases
#	they will not be flexible enough.  For instance, suppose that your 'x-values' are
#	actually text strings which your function processes.  You would need regular python
#	arrays to do that.
"""

def f (x,a,b,c):
    return a*sin(b*x) + c

N = 30

#
# Here I'm using numarrays.
#
#xvals = arange(N)
#yvals = _alloc_vector(N)
#sigma = _alloc_vector(N)
#for x in xvals:
#    sigma[x] = random()*0.01
#    yvals[x] = 5*sin(2*x) + 1

#
# Here I'm using regular python arrays.
#
xvals = range(N)
yvals = []
sigma = []
for x in xvals:
    sigma.append(0.1)
    yvals.append(1*sin(2*x) + 1 )

par = [3.,2.001,3.]

err,chi,par = non_linear_regression(f,xvals,par,yvals,sigma)

print "Error:       ",err
print "Chi Squared: ",chi
print "Parameters:  ",par

