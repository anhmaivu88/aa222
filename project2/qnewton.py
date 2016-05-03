# Standard libraries
import numpy as np
from numpy.linalg import norm
from numpy import dot, outer, eye, shape
import matplotlib.pyplot as plt

# Custom libraries
from grad import grad
from line_search import backtrack, quad_fit

def dfp(H,delta,gamma):
    H1 = outer(dot(H,gamma),dot(gamma,H))/\
         (gamma.dot(H)*gamma).sum()
    H2 = outer(delta,delta)/dot(delta,delta)
    return H - H1 + H2

def bfgs(H,delta,gamma):
    I  = eye(shape(H)[0])
    # Limit the 1/delta^T*gamma value
    # - I learned this by studying the SciPy
    # - implementation of BFGS
    val = dot(delta,gamma)
    if val < 1e-3:
        k = 1000.0
    else:
        k = 1.0 / val
    # Compute update
    T1 = (I-outer(delta,gamma)*k)
    T2 = (I-outer(gamma,delta)*k)
    return dot(T1,dot(H,T2)) + outer(delta,delta)*k

def qnewton(fcn,x0,evalMax,eps=np.finfo(float).eps,lin=0,nIter=1e10,\
            qtype=1,fk=None,dF0=None,Hk=None):
    """Quasi-Newton Method
    Usage
    (x0, f0, ct, X, it) = qnewton(f,x0,evalMax)

    Arguments
    f       = function to minimize
    x0      = initial guess
    evalMax = maximum function evaluations

    Keyword Arguments
    eps     = convergence criterion
    lin     = linsearch type (0=backtracking,1=quad fit)
    nIter   = maximum iterations
    qtype   = hessian update type (0=DFP,1=BFGS)
    fk      = initial function value
    dF0     = initial function gradient
    Hk      = initial inverse Hessian approximation
    
    Returns
    xs      = minimal point
    fs      = minimal value
    dFs     = minimal point gradient
    Hs      = minimal point inv Hessian approx
    ct      = number of function evaluations
    X       = sequence of points
    it      = number of iterations used
    """
    if (lin!=0) and (lin!=1):
        raise ValueError("Unrecognized linsearch")
    if qtype==0:
        update = dfp
    elif qtype==1:
        update = bfgs

    # Setup
    ct  = 0              # Function calls
    it  = 0              # Iteration count
    x0  = np.array(x0)   # initial guess
    X   = np.array([x0]) # point history
    n   = np.size(x0)    # dim of problem
    if fk == None:
        fk  = fcn(x0);  ct += 1
    err = eps * 2        # initial error
    ### Initial direction: steepest descent
    if dF0 == None:
        dF0 = grad(x0,fcn,fk);      ct += n
    d0  = -dF0
    if Hk == None:
        Hk  = np.eye(n)

    ### Main Loop
    while (err>eps) and (ct<evalMax) and (it<nIter):
        # Compute new step direction
        d0 = -Hk.dot(dF0)
        # Perform line search
        p  = d0 / norm(d0)
        if (lin==0):
            m = np.dot(dF0,p)
            alp, fk, k = backtrack(x0,fcn,m,p,fk,em=evalMax-ct)
            ct += k
        elif (lin==1):
            alp, fk, k = quad_fit(x0,fcn,p,fk)
            ct += k
        x1 = x0 + alp*p
        X = np.append(X,[x1],axis=0)

        # Update inverse hessian
        if (ct+n<evalMax):
            dF1 = grad(x1,fcn,fk);      ct += n
        else:
            return x0, fk, dF0, Hk, ct, X, it
        delta = x1 - x0
        gamma = dF1-dF0
        Hk = update(Hk,delta,gamma)
        # Swap values
        x0  = x1
        dF0 = dF1
        # Compute error (norm of grad)
        err = norm(dF0)
        # Iterate counter
        it += 1

    # Complete qNewton solve
    return x0, fk, dF0, Hk, ct, X, it
