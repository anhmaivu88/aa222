# Standard libraries
from math import log, exp
import numpy as np
from numpy.linalg import norm
# Custom libraries
from grad import grad
from line_search import backtrack

def inv_barrier(g):
    """Inverse barrier function
    Takes a constraint evaluation, and builds a scalar inverse barrier
    Usage
        res = inv_barrier(g)
    Inputs
        g = iterable of scalar values, leq vector constraint R^n->R^k
    Outputs
        res = value of inverse barrier for given values
    """
    res = 0
    for val in g:
        res = res - 1/val
    return res


def log_barrier(g):
    """Log barrier function
    Takes a constraint evaluation, and builds a scalar log barrier
    Usage
        res = inv_barrier(g)
    Inputs
        g = iterable of scalar values, leq vector constraint R^n->R^k
    Outputs
        res = value of log barrier for given values
    """
    res = 0
    for val in g:
        if val < -1:
            pass
        else:
            if val >= 0:
                res = float('inf')
                # res = res + exp(val)
            else:
                res = res - log(-val)
    return res

def ext_obj(g,e):
    """Exterior Point Objective
    Creates an objective function for the exterior point method
    Usage
        res = ext_obj(g,e)
    Inputs
        g = iterable of scalar values, leq vector constraint R^n->R^k
        e = scalar value, represents interior distance from boundary sought
    Outputs
        res = value of objective function
    """
    res = 0
    for val in g:
        res = res + max(0,val+e)**2
    return res

def feasible(g):
    """Checks if a constraint value is feasible
    Usage
        res = feasible(g)
    Inputs
        g = iterable of scalar values, leq vector constraint R^n->R^k
    Outputs
        res = boolean feasibility, True/False
    """
    slack = 1e-1
    res = True
    for val in g:
        # res = res and (val<=0)
        res = res and (val<0)
    return res

def feasibility_problem(g,x0,evalMax,slack=1e-3,eps=1e-3,nIter=100,h=1e-2):
    """Feasibility problem via CG solver
    Usage
    xf, gf, ct, X, it = feasibility_problem(g,x0,evalMax)

    Arguments
    g       = leq constraint function R^n->R^k
    x0      = initial guess
    evalMax = maximum function evaluations

    Keyword arguments
    slack   = slackness on constraints
    eps     = convergence criterion

    Returns
    xf      = feasible point
    gf      = constraint values
    ct      = number of function evaluations
    X       = sequence of points
    it      = number of iterations used
    """
    # Setup
    f   = lambda x: ext_obj(g(x),slack)

    ct  = 0              # Function calls
    it  = 0              # Iteration count
    x0  = np.array(x0)   # initial guess
    X   = np.array([x0]) # point history
    n   = np.size(x0)    # dim of problem
    g0  = g(x0);         ct += 1
    f0  = ext_obj(g0,slack)
    err = eps * 2        # initial error
    # Check for feasibility
    if feasible(g0):
        return x0, g0, ct, X, it

    # Initial direction: steepest descent
    dF0 = grad(x0,f,f0);      ct += n
    d0  = -dF0

    ### Main loop
    while (not feasible(g0)) and (ct<evalMax) and (it<nIter):
        # Perform line search
        p  = d0 / norm(d0)
        m = np.dot(dF0,p)
        alp, f0, k = backtrack(x0,f,m,p,f0,em=evalMax-ct,alp=5)
        ct += k
        x0 = x0 + alp*p
        g0 = g(x0); ct += 1
        X = np.append(X,[x0],axis=0)
        
        # Compute conjugate direction
        if (ct+n<evalMax):
            dF1 = grad(x0,f,f0); ct += n
        else:
            return x0, g0, ct, X, it
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
    return x0, g0, ct, X, it