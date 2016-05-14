# Standard libraries
import numpy as np
from scipy.optimize import fmin_bfgs

# Custom libraries
from barrier import log_barrier, inv_barrier, ext_obj

# BFGS Call
# res = fmin_bfgs(F,x0,fprime=dF,retall=True)
# res = (x*,Xs)

def constrained_opt(F,G,x0,tol=1e-8):
    """ Constrained Optimization via interior point method
    Usage
        xs, Fs, X, it = constrained_opt(F,G,x0)
    Arguments
        F       = objective function, R^n->R
        G       = leq constraints, R^n->R^k
        x0      = initial guess
    Keyword Arguments
        tol     = convergence tolerance (L2 gradient)
    Outputs
        xs      = optimal point
        Fs      = optimal value
        X       = sequence of iterates
        it      = iterations
    """
    ### Setup
    r   = 1e2       # Initial relaxation
    r_max = 1e3
    fac = 2         # Relaxation factor
    eps = 1/r       # Initial gradient tolerance
    err = tol*2     # Initial error

    it  = 0     # iteration count
    s   = 1e-1  # interior slack
    x0  = np.array(x0)  # initial guess
    n   = np.size(x0)   # dim of problem

    ### Feasibility problem
    # G0  = ext_obj(G,s)  # exterior objective
    G0     = lambda x: ext_obj(G(x),s)
    # Minimize G0
    xs, X = fmin_bfgs(G0,x0,retall=True)
    it = it + 1

    ### Interior point problem sequence
    while (err > tol):      # Not converged
        # Relax the barrier
        fcn = lambda x: F(x) + log_barrier(G(x))/r
        # Enforce a tighter convergence criterion
        xn, Xn = fmin_bfgs(fcn,xs,retall=True,gtol=eps,epsilon=1e-8)
        it = it + 1 # TODO -- grab iter count from bfgs
        X = np.append(X,Xn,axis=0)
        # Increment to next problem
        if r < r_max:
            r   = r * fac
            eps = 1 / r
        else:
            r   = r_max
            eps = eps=np.finfo(float).eps
        # Compute error
        err = np.linalg.norm(xn-xs)
        # Set new start guess
        xs = xn

    Fs = F(xs)
    ### Terminate
    return xs, Fs, X, it
