import numpy as np
from grad import grad
from line_search import backtrack, quad_fit

# TODO -- implement evalMax stopping criterion
def cg(f,x0,evalMax,eps=1e-6,lin=0):
    """Conjugate Gradient solver
    Usage
    (xs,fs,ct) = cg(f,x0,evalMax)
    Arguments
    f       = function to minimize
    x0      = initial guess
    evalMax = maximum function evaluations
    Keyword arguments
    eps     = convergence criterion
    lin     = linsearch type (0=backtracking,)
    Returns
    xs      = minimal point
    fs      = minimal value
    ct      = number of function evaluations
    """
    if (lin!=0) and (lin!=1):
        raise ValueError("Unsupported linsearch")
    # Setup
    ct  = 0
    x0  = np.array(x0)
    n   = np.size(x0)
    f0  = f(x0);                ct += 1
    err = eps * 2
    # Main solver loop
    while (err>eps) and (ct < evalMax):
        # Initial step: Steepest Descent
        if (ct+n<evalMax):
            dF0 = grad(x0,fcn,f0);      ct += n
        else:
            return x0, f0, ct
        d0 = -dF0 / np.linalg.norm(dF0)
        m  = np.dot(d0,dF0)
        # Perform line search
        if (lin==0):
            alp, f1, k = backtrack(x0,fcn,m,d0,f0,\
                                   em=evalMax-ct);
        elif (lin==1):
            alp, f1, k = quad_fit(x0,fcn,d0,f0)
        ct += k
        x1 = x0 + alp*d0
        # n-1 Conjugate Gradient steps
        j = 1
        while (j < n) and (err>eps) and \
                (ct < evalMax):
            # Calculate new gradient
            if (ct+n<evalMax):
                dF1 = grad(x1,fcn,f1);  ct += n
            else:
                return x1, f1, ct
            # Conjugate direction
            beta = np.linalg.norm(dF1)**2/\
                   np.linalg.norm(dF0)**2
            d1 = -dF1 + beta*d0
            # Line search
            m = np.dot(d1,dF1)
            if (lin==0):
                alp, f1, k = backtrack(x1,fcn,m,d1,f1,\
                                       em=evalMax-ct);
            elif (lin==1):
                alp, f1, k = quad_fit(x1,fcn,d1,f1)
            ct += k
            x1 = x0 + alp*d1
            # Store previous values
            d0  = d1
            dF0 = dF1
            x0  = x1
            # Complete the step
            err = np.linalg.norm(dF0)
            j += 1
    # Complete CG solve
    return x1, f1, ct

if __name__ == "__main__":
    from rosenbrock import fcn
    x0 = [0,0]
    print "f(x0)=%f" % fcn(x0)
    x0,fs,ct = cg(fcn,x0,2e4)
    print "f(xs)=%f" % fs
    print "niter=%d" % ct
