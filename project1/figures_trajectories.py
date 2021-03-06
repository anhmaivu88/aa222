# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

# Custom libraries
from project1 import opt_full

### ----- TRAJECTORIES ----- ###
### Setup
from rosenbrock import fcn 
xstar = [1,1]; 
x0 = [0,1.5]; x1 = [1.5,1.5]; x2 = [-0.5,-0.5]

# Set parameters
nIter = 100
nCall = 100
comsci = True      # compare vs SciPy
plotting = False    # plot the result

### Solver call
_,_,_,Xs0,_ = opt_full(fcn,x0,nCall)
_,_,_,Xs1,_ = opt_full(fcn,x1,nCall)
_,_,_,Xs2,_ = opt_full(fcn,x2,nCall)

### Plotting
# Define meshgrid
delta = 0.025
x = np.arange(min([x0[0],x1[0],x2[0],xstar[0]])-0.5, \
              max([x0[0],x1[0],x1[0],xstar[0]])+0.5, \
              delta)
y = np.arange(min([x0[1],x1[1],x2[1],xstar[1]])-0.5, \
              max([x0[1],x1[1],x2[1],xstar[1]])+0.5, \
              delta)
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
# plt.title('Sequence of Iterates: Finite Differences')
plt.title('Sequence of Iterates: Automatic Differentiation')
plt.plot(xstar[0],xstar[1],'ok') # optimum
plt.plot(x0[0],x0[1],'ob') # starting point
plt.plot(x1[0],x1[1],'or')
plt.plot(x2[0],x2[1],'og')

# Trajectory 0
for i in range(np.shape(Xs0)[0]-1):
    plt.plot([Xs0[i][0],Xs0[i+1][0]],\
             [Xs0[i][1],Xs0[i+1][1]],'b')
# Trajectory 1
for i in range(np.shape(Xs1)[0]-1):
    plt.plot([Xs1[i][0],Xs1[i+1][0]],\
             [Xs1[i][1],Xs1[i+1][1]],'r')
# Trajectory 1
for i in range(np.shape(Xs2)[0]-1):
    plt.plot([Xs2[i][0],Xs2[i+1][0]],\
             [Xs2[i][1],Xs2[i+1][1]],'g')

plt.show()
