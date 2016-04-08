# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
# Custom libraries

from grad import grad
from line_search import backtrack, quad_fit

# TODO -- implement evalMax stopping criterion
def cg(f,x0,evalMax,eps=1e-6,lin=0,nIter=100):
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
    # Main solver loop
    while (err>eps) and (ct < evalMax) and \
            (it<nIter):
        # Initial step: Steepest Descent
        if (ct+n<evalMax):
            dF0 = grad(x0,fcn,f0);      ct += n
        else:
            return x0, f0, ct
        d0 = -dF0
        # Perform line search
        p = d0 / np.linalg.norm(d0)
        if (lin==0):
            m  = np.dot(p,dF0)
            alp, f1, k = backtrack(x0,fcn,m,p,f0,\
                                   em=evalMax-ct);
        elif (lin==1):
            alp, f1, k = quad_fit(x0,fcn,p,f0)
        ct += k
        x1 = x0 + alp*p
        X = np.append(X,[x1],axis=0)
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
            # beta = 0
            # Polack-Rebiere
            beta = np.dot(dF1,dF1-dF0)/np.dot(dF0,dF0)
            print(beta)
            # Fletcher-Reeves
            # beta = np.linalg.norm(dF1)**2/\
            #        np.linalg.norm(dF0)**2
            d1 = -dF1 + beta*d0
            # Line search
            p = d1 / np.linalg.norm(d1)
            if (lin==0):
                m = np.dot(p,dF1)
                alp, f1, k = backtrack(x1,fcn,m,p,f1,\
                                       em=evalMax-ct);
            elif (lin==1):
                alp, f1, k = quad_fit(x1,fcn,p,f1)
            ct += k
            x1 = x0 + alp*p
            # Store previous values
            d0  = d1
            dF0 = dF1
            x0  = x1
            # Complete the step
            err = np.linalg.norm(dF0)
            j += 1
            X = np.append(X,[x0],axis=0)
        # Count full iterations
        it += 1
    # Complete CG solve
    return x1, f1, ct, X

if __name__ == "__main__":
    ### Setup
    # from rosenbrock import fcn
    from simple_quad import fcn
    # Set initial guess
    x0 = [1,0.5]
    print "f(x0)=%f" % fcn(x0)

    ### Solver call
    xs,fs,ct,Xs = cg(fcn,x0,2e4,lin=1,nIter=2)
    print "f(xs)=%f" % fs
    print "calls=%d" % ct

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

    # Overlay point sequence
    for i in range(np.shape(Xs)[0]-1):
        plt.plot([Xs[i][0],Xs[i+1][0]],[Xs[i][1],Xs[i+1][1]])

    plt.show()