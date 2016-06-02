import numpy as np
from scipy.linalg import qr
from numpy.random import random
from numpy.linalg import norm

import util
from interior import constrained_opt

def seek_am(M,reset_max=50,res_thresh=1e-3,verbose=False,full=False,\
            m_des=None):
    """Seeks the Active Manifolds by solving
    sequential constrained minimization problems
    Usage
        W,Res = seek_am(M)
        W,Res,resets = seek_am(M,full=True)
    Arguments
        M = matrix defining problem
    Keyword Arguments
        reset_max  = maximum number of resets
        res_thresh = absolute residual tolerance
        verbose    = verbose (terminal) output
        full       = full function return
        m_des      = desired number of manifolds
    Outputs
        W       = orthogonal matrix
        Res     = list of residual values
        resets  = number of resets
    """
    m = M.shape[1]
    if m_des == None:
        m_des = m
    ##################################################
    ## Optimization Problem
    ##################################################

    beta = 1.0      # tunable parameter
    # Residual + L1
    f = lambda x: norm(np.dot(M,np.array(x))) + beta * norm(x,ord=1)
    # L2 >= 11
    g = lambda x: (-1.0*norm(x,ord=2) + 1,)

    ##################################################
    ## Solver
    ##################################################
    # Setup
    W = np.array([]); Res = []    # Reserve space
    Res_full = []; Obj_full = []  # Optional outputs
    F = f; G = g        # Unmodified for first run
    Qc = np.eye(m)      # First parameterization
    # Initial guess for solver
    x0 = random([1,Qc.shape[1]])     # random guess

    # Main loop
    resets = 0
    i = 0

    # for i in range(m):
    while i < m_des:
        # Solve optimization problem
        xs, Fs, Gs, X, Ft = constrained_opt(F,G,x0)
        w = util.col(Qc.dot(xs)) # map to physical space
        res = norm(np.dot(M,w))
        # Continue if solution meets residual tolerance,
        # or we're out of reset budget
        if (res < res_thresh) or (resets >= reset_max):
            # Store results
            if W.shape==(0,):
                W = w
            else:
                W = np.append(W,w,axis=1)
            Res.append( res )
            # Store residual sequence for full output
            if full==True:
                wf = [util.col(Qc.dot(v)) for v in X]
                Res_full.append([norm(np.dot(M,v)) for v in wf])
                Obj_full.append(Ft)
            # Reparameterize
            if (i < m-1): # Check if last iteration
                A = np.append(np.array(W),np.eye(m),axis=1)
                Q,R = qr(A,mode='economic')
                Qc = util.col(Q[:,i+1:])
                F = lambda alpha: f(Qc.dot(util.col(np.array(alpha))))
                G = lambda alpha: g(Qc.dot(util.col(np.array(alpha))))
            # Iterate
            i += 1
        # Iterate the reset budget
        else:
            resets += 1
            if verbose:
                print("Residual tolerance not reached, solver stage reset...")
        # New initial guess
        x0 = random([1,Qc.shape[1]])     # random guess

    if full:
        return util.norm_col(W), Res, resets, Res_full, Obj_full
    else:
        return util.norm_col(W), Res