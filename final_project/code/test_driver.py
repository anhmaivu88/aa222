# Standard libraries
import numpy as np
import matplotlib.pyplot as plt

# Custom libraries
from interior import constrained_opt

### Setup

# Choose problem

## Example 1
from example1 import f,g,x_star,f_star, constraints, xc
x0 = [1.5,1]

## Example 2
# from example2 import f,g,x_star,f_star, constraints, xc
# x0 = [0,0.5]

## Example 3
# from example3 import f,g,x_star,f_star
# x0 = [0,1,0]

# Set parameters
plotting = True    # plot the result

s = 1e-1
g_ext = lambda x: ext_obj(g(x),s)

### Solver call
xs, Fs, Xs, it = constrained_opt(f,g,x0)

print(x_star)

print "Fs(xs)=%f" % Fs
print "abs(Fs-F*)={}".format(abs(Fs-f_star))
print "iter=%d" % it

### Plotting
if plotting == True:
    # Define meshgrid
    delta = 0.025
    x = np.arange(min(x0[0],x_star[0])-0.5, \
                  max(x0[0],x_star[0])+0.5, delta)
    y = np.arange(min(x0[1],x_star[1])-0.5, \
                  max(x0[1],x_star[1])+0.5, delta)
    X, Y = np.meshgrid(x, y)
    dim = np.shape(X)
    # Compute function values
    Xv = X.flatten(); Yv=Y.flatten()
    Input = zip(Xv,Yv)
    Zv = []
    for x in Input:
        Zv.append(f(x))
    # Restore arrays to proper dimensions
    Z = np.array(Zv).reshape(dim)
    
    # Open figure
    plt.figure()

    # Plot objective contours
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Sequence of iterates')
    plt.plot(x_star[0],x_star[1],'xk') # optimum
    plt.xlim([np.min(X),np.max(X)])
    plt.ylim([np.min(Y),np.max(Y)])

    # Plot constraints
    for con in constraints:
        plt.plot(xc,con(xc),'k-')

    # Overlay point sequence
    for i in range(np.shape(Xs)[0]-1):
        plt.plot([Xs[i][0],Xs[i+1][0]],[Xs[i][1],Xs[i+1][1]],'b-o')
    plt.plot(x0[0],x0[1],'or') # starting point

    plt.show()