import numpy as np

def backtrack(x0,f,m,p,f0,alp=1,tau=0.5,c=0.5,em=100):
    """Backtracking line search
    Usage
    (alp,f1,cnt) = backtrack(x0,f,m,p,f0)
    Arguments
    x0       = initial guess
    f        = function for line search
    m        = slope along search direction
    p        = search direction
    f0       = f(x0)
    Keyword arguments
    alp      = initial search step
    tau      = step reduction ratio
    c        = control parameter
    em       = max function evaluations
    Return values
    alp      = line search distance
    f1       = function value at new point
    cnt      = function evaluations used
    """
    # Setup
    cnt = 0
    t   = -c*m
    # Evaluate first test point
    f1 = f(x0+alp*p)
    # Iterate backtracking
    while ((f0-f1) < alp*t) and (cnt<em):
        # Iterate
        cnt += 1
        alp = tau * alp
        # Evaluate next point
        f1 = f(x0+alp*p)
    # Return
    return (alp,f1,cnt)

def quad_fit(x0,f,p,f0,d=0.5):
    """Quadratic fit line search
    """
    # Query forward and back
    x1 = x0 - d*p; f1 = f(x1)
    x2 = x0 + d*p; f2 = f(x2)
    # Solve the fit quadratic
    alp = (f1-f2)*d / (2*f1+2*f2-4*f0)
    return (alp, f(x0+alp*p), 3)
