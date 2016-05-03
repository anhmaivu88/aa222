# Standard libraries
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.optimize import fmin_cg
# Custom libraries
from grad import grad
from line_search import backtrack, quad_fit

# TODO -- implement evalMax stopping criterion
def cg(f,x0,evalMax,eps=1e-3,lin=0,nIter=100,h=1e-2):
    """Conjugate Gradient solver
    Usage
    (x0, f0, ct, X, it) = cg(f,x0,evalMax)

    Arguments
    f       = function to minimize
    x0      = initial guess
    evalMax = maximum function evaluations

    Keyword arguments
    eps     = convergence criterion
    lin     = linsearch type (0=backtracking,1=quad fit)

    Returns
    xs      = minimal point
    fs      = minimal value
    ct      = number of function evaluations
    X       = sequence of points
    it      = number of iterations used
    """
    if (lin!=0) and (lin!=1):
        raise ValueError("Unrecognized linsearch")
    # Setup
    ct  = 0              # Function calls
    it  = 0              # Iteration count
    x0  = np.array(x0)   # initial guess
    X   = np.array([x0]) # point history
    n   = np.size(x0)    # dim of problem
    f0  = f(x0);                ct += 1
    err = eps * 2        # initial error
    ### Initial direction: steepest descent
    dF0 = grad(x0,f,f0);      ct += n
    d0  = -dF0

    ### Main loop
    while (err>eps) and (ct<evalMax) and (it<nIter):
        # Perform line search
        p  = d0 / norm(d0)
        if (lin==0):
            m = np.dot(dF0,p)
            alp, f0, k = backtrack(x0,f,m,p,f0,em=evalMax-ct)
            ct += k
        elif (lin==1):
            alp, f0, k = quad_fit(x0,f,p,f0)
            ct += k
        x0 = x0 + alp*p
        X = np.append(X,[x0],axis=0)
        
        # Compute conjugate direction
        if (ct+n<evalMax):
            dF1 = grad(x0,f,f0); ct += n
        else:
            return x0, f0, ct, X, it
        beta = max(np.dot(dF1,dF1-dF0)/np.dot(dF0,dF0),0)
        d1 = -dF1 + beta*d0
        # Swap old directions
        d0  = d1
        dF0 = dF1
        # Compute error (norm of grad)
        err = norm(dF0)
        # Iterate counter
        it += 1

    # Complete CG solve
    return x0, f0, ct, X, it
