# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

# Custom libraries
from unconstrained_optimization import opt_full

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
comsci = True      # compare vs SciPy
plotting = False    # plot the result

### Solver call
xs,fs,ct,Xs,it = opt_full(fcn,x0,nCall)

print "f(xs)=%f" % fs
print "calls=%d" % ct
print "iter=%d" % it

# Scipy call
if comsci==True:
    res = fmin_bfgs(fcn,x0,retall=True)

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