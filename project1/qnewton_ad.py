# Standard libraries
import numpy as np
from numpy.linalg import norm
from numpy import dot, outer, eye, shape
import matplotlib.pyplot as plt
from ad import gh

# Custom libraries
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
    try:
        k = 1.0 / dot(delta,gamma)
    except ZeroDivisionError:
        rhok = 1000.0
    # Compute update
    T1 = (I-outer(delta,gamma)*k)
    T2 = (I-outer(gamma,delta)*k)
    return dot(T1,dot(H,T2)) + outer(delta,delta)*k

def qnewton(f,x0,evalMax,eps=1e-3,lin=0,nIter=100,\
            qtype=1):
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
    
    Returns
    xs      = minimal point
    fs      = minimal value
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
    fk  = f(x0);                ct += 1
    err = eps * 2        # initial error
    # Use automatic differentiation
    grad, hess = gh(fcn)
    ### Initial direction: steepest descent
    dF0 = grad(x0); ct += 1
    d0  = -dF0
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
            dF1 = grad(x1); ct += 1
        else:
            return x0, fk, ct, X, it
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
    return x0, fk, ct, X, it

if __name__ == "__main__":
    ### Setup
    # Choose objective function
    ## Rosenbrock function
    # from rosenbrock import fcn 
    # xstar = [1,1]; x0 = [1,1.5]
    ## Simple quadratic function
    # from simple_quad import fcn
    # xstar = [0,0]; x0 = [1,1.5]
    ## Wood function
    # from wood import fcn
    # xstar = [1,1,1,1]; x0 = [0,0,0,0]
    ## Powell function
    from powell import fcn
    xstar = [0,0,0,0]; x0 = [1,1,1,1]

    # Set parameters
    nIter = 100
    nCall = 1e4
    comsci = False      # compare vs SciPy
    plotting = False    # plot the result

    ### Solver call
    xs,fs,ct,Xs,it = qnewton(fcn,x0,nCall,\
                             lin=0,nIter=nIter,\
                             qtype=1)
    
    print "f(xs)=%f" % fs
    print "calls=%d" % ct
    print "iter=%d" % it

    # Scipy call
    if comsci==True:
        res = fmin_cg(fcn,x0,retall=True)

    ### Plotting
    if plotting == True:
        # Define meshgrid
        delta = 0.025
        x = np.arange(min(x0[0],xstar[0])-0.5, \
                      max(x0[0],xstar[0])+0.5, delta)
        y = np.arange(min(x0[1],xstar[1])-0.5, \
                      max(x0[1],xstar[1])+0.5, delta)
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
        plt.clabel(CS, inline=1, fontsize=10)
        plt.title('Sequence of iterates')
        plt.plot(xstar[0],xstar[1],'ok') # optimum
        plt.plot(x0[0],x0[1],'or') # starting point

        # Overlay point sequence
        for i in range(np.shape(Xs)[0]-1):
            plt.plot([Xs[i][0],Xs[i+1][0]],[Xs[i][1],Xs[i+1][1]],'b')

        ### Compare against SciPy
        if (comsci==True):
            # SciPy point sequence
            for i in range(np.shape(res[1])[0]-1):
                plt.plot([res[1][i][0],res[1][i+1][0]],\
                         [res[1][i][1],res[1][i+1][1]],'r--')

        plt.show()