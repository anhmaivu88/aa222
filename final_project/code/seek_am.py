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
        W,Res,Resets_total, Res_full, Obj_full = seek_am(M,full=True)
    Arguments
        M = matrix defining problem
    Keyword Arguments
        reset_max  = maximum number of resets
        res_thresh = absolute residual tolerance
        verbose    = verbose (terminal) output
        full       = full function return
        m_des      = desired number of manifolds
    Outputs
        W        = orthogonal matrix
        Res      = list of residual values
        Resets_total  = number of resets, organized by stage
        Res_full = Residual sequence, organized by stage
        Obj_full = Objective sequence, organized by stage
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

    # Temporary values: best in stage
    wb = None; Wb = None; Rb = None; Ob = None; 
    res_b = float('inf');

    # Main loop
    resets = 0
    Resets_total = []
    i = 0

    # for i in range(m):
    while i < m_des:
        ##################################################
        ### Solve optimization problem
        ##################################################
        xs, Fs, Gs, X, Ft = constrained_opt(F,G,x0)
        w = util.col(Qc.dot(xs)) # map to physical space
        res = norm(np.dot(M,w))

        ##################################################
        ### Stage Logic
        ##################################################
        # Met residual tolerance: store, reparameterize, and iterate
        if (res < res_thresh):
            # Store results
            if W.shape==(0,):
                W = w
            else:
                W = np.append(W,w,axis=1)
            Res.append( res )
            # Store residual and objective sequences for full output
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
            # Move to next stage
            i += 1
            # Reset the reset budget
            Resets_total.append(resets)
            resets = 0
            # Reset the best residual
            res_b = float('inf');

        # Out of resets: recall best run, reparameterize, and iterate
        elif (resets >= reset_max):
            # Verbose output
            if verbose:
                print("res_tol not met in Stage {}, advancing with best stage result...".format(i))
            # Move forward with our best attempt
            if W.shape==(0,):
                W = wb
            else:
                W = np.append(W,wb,axis=1)
            # Store residual and objective sequences for full output
            if full==True:
                Res_full.append(Rb)
                Obj_full.append(Ob)
            # Reparameterize
            if (i < m-1): # Check if last iteration
                A = np.append(np.array(W),np.eye(m),axis=1)
                Q,R = qr(A,mode='economic')
                Qc = util.col(Q[:,i+1:])
                F = lambda alpha: f(Qc.dot(util.col(np.array(alpha))))
                G = lambda alpha: g(Qc.dot(util.col(np.array(alpha))))
            # Move to next stage
            i += 1 
            # Reset the reset budget
            Resets_total.append(resets)
            resets = 0
            # Reset the best residual
            res_b = float('inf');

        # Reset stage
        else:
            # Verbose output
            if verbose:
                print("res_tol not met in Stage {}, solver stage reset...".format(i))
            # Store if we did better than before
            if res < res_b:
                wb = w
                if full==True:
                    wf = [util.col(Qc.dot(v)) for v in X]
                    Rb = [norm(np.dot(M,v)) for v in wf]
                    Ob = Ft
                res_b = res
            # Iterate the reset counter
            resets += 1

        # New initial guess
        x0 = random([1,Qc.shape[1]])     # random guess

    if full:
        return util.norm_col(W), Res, Resets_total, Res_full, Obj_full
    else:
        return util.norm_col(W), Res