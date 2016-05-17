import numpy as np
from scipy.linalg import qr
from numpy.random import random
from numpy.linalg import norm

import util
from interior import constrained_opt

def seek_am(M):
    """Seeks the Active Manifolds by solving
    sequential constrained minimization problems
    Usage
        W,Res = seek_am(M)
    Arguments
        M = matrix defining problem
    Outputs
        W   = orthogonal matrix
        Res = list of residual values
    """
    m = M.shape[1]
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
    F = f; G = g        # Unmodified for first run
    Qc = np.eye(m)      # First parameterization
    # Initial guess for solver
    # x0 = [1.0] * m         # first orthant
    x0 = random([1,m])     # random guess

    # Main loop
    for i in range(m):
        # Solve optimization problem
        xs, Fs, Xs, it = constrained_opt(F,G,x0)
        w = util.col(Qc.dot(xs)) # map to physical space
        # Store results
        if W.shape==(0,):
            W = w
        else:
            W = np.append(W,w,axis=1)
        Res.append( norm(np.dot(M,w)) )
        # Reparameterize if continuing
        if (i < m-1):
            A = np.append(np.array(W),np.eye(m),axis=1)
            Q,R = qr(A,mode='economic')
            Qc = util.col(Q[:,i+1:])
            F = lambda alpha: f(Qc.dot(util.col(np.array(alpha))))
            G = lambda alpha: g(Qc.dot(util.col(np.array(alpha))))
            x0 = random([1,Qc.shape[1]])     # random guess

    return W, Res