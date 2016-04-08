# Standard libraries
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
# Custom libraries

from grad import grad
from line_search import backtrack, quad_fit

# TODO -- implement evalMax stopping criterion
def cg(f,x0,evalMax,eps=1e-6,lin=0,nIter=100,h=1e-2):
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
    X       = sequence of points
    """
    if (lin!=0) and (lin!=1):
        raise ValueError("Unsupported linsearch")
    # Setup
    ct  = 0
    it  = 0
    x0  = np.array(x0)
    X   = np.array([x0])
    n   = np.size(x0)
    f0  = f(x0);                ct += 1
    err = eps * 2
    ### Initial step: steepest descent
    dF0 = grad(x0,fcn,f0,h=h);      ct += n
    d0 = -dF0
    # Perform line search
    p  = d0 / norm(d0)
    if (lin==0):
        m   = np.dot(dF0,p)
        alp, f1, k = backtrack(x0,fcn,m,p,f0,em=evalMax-ct)
        ct += k
    elif (lin==1):
        alp, f1, k = quad_fit(x0,fcn,p,f0)
    x0 = x0 + alp*p
    f0 = fcn(x0)
    X = np.append(X,[x0],axis=0)
    it += 1

    ### Main loop
    while (err>eps) and (ct<evalMax) and (it<nIter):
        # Compute conjugate direction
        if (ct+n<evalMax):
            dF1 = grad(x0,fcn,f0,h=h); ct += n
            print(dF1)
        else:
            return x0, f0, ct, X, it
        beta = max(np.dot(dF1,dF1-dF0)/np.dot(dF0,dF0),0)
        # beta = 0
        d1 = -dF1 + beta*dF0
        # Perform line search
        p  = d1 / norm(d1)
        if (lin==0):
            m = np.dot(dF1,p)
            alp, f1, k = backtrack(x0,fcn,m,p,f0,em=evalMax-ct)
            ct += k
        elif (lin==1):
            alp, f1, k = quad_fit(x0,fcn,p,f0)
            ct += k
        x0 = x0 + alp*p
        X = np.append(X,[x0],axis=0)
        # Swap old directions
        d0  = d1
        dF0 = dF1
        # Iterate counter
        it += 1

    # Complete CG solve
    return x0, f0, ct, X, it

if __name__ == "__main__":
    ### Setup
    # from rosenbrock import fcn
    from simple_quad import fcn; xstar = [0,0]
    # Set initial guess
    x0 = [1,0.5]
    print "f(x0)=%f" % fcn(x0)

    ### Solver call
    xs,fs,ct,Xs,it = cg(fcn,x0,2e4,lin=1,nIter=10)
    print(xs)
    print "f(xs)=%f" % fs
    print "calls=%d" % ct
    print "iter=%d" % it

    ### Plotting
    # Define meshgrid
    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    dim = np.shape(X)
    # Compute function values
    Xv = X.flatten(); Yv=Y.flatten()
    Input = zip(Xv,Yv)
    Zv = []
    for x in Input:
        Zv.append(fcn(x))
    # Restore arrays to proper dimensions
    Z = np.array(Zv).reshape(dim)
    
    # Open figure
    plt.figure()

    # Plot contour
    CS = plt.contour(X, Y, Z)
    manual_locations = [(-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)]
    plt.clabel(CS, inline=1, fontsize=10, manual=manual_locations)
    plt.title('Sequence of iterates')
    plt.plot(xstar[0],xstar[1],'o')

    # Overlay point sequence
    for i in range(np.shape(Xs)[0]-1):
        plt.plot([Xs[i][0],Xs[i+1][0]],[Xs[i][1],Xs[i+1][1]])

    plt.show()