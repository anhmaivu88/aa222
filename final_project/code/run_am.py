##################################################
## Standard libraries
##################################################
import numpy as np
from numpy.linalg import norm
from ad.admath import log
from numpy.random import random
from ad import gh
import sys

import matplotlib
# choose the backend to handle window geometry
matplotlib.use("Qt4Agg")
# Import pyplot
import matplotlib.pyplot as plt

##################################################
## Custom libraries
##################################################
from interior import constrained_opt
from barrier import feasible
import util

from seek_am import seek_am

##################################################
## Setup
##################################################
# Parameters
n       = int(1e2)  # monte carlo samples
m       = 3         # Dimension of problem
# Plot settings
offset = [(0,500),(700,500)] # Plot window locations

##################################################
# Command Line Inputs
##################################################
# if len(sys.argv) < 2:
#     print('Usage:')
#     print('    python {} [grid file] [solution file]'.format(sys.argv[0]))
#     exit()

# Problem selection
if len(sys.argv) > 1:
    my_case = int(sys.argv[1])
else:
    print("Default case selected...")
    my_case = 0
# Basis selection
if len(sys.argv) > 2:
    my_base = int(sys.argv[2])
else:
    print("Default basis selected...")
    my_base = 0

# Define basis functions
if my_base == 1:
    # Second Order
    Phi = [ lambda x: x[0],
            lambda x: x[1],
            lambda x: x[2],
            lambda x: x[0]**2,
            lambda x: x[1]**2,
            lambda x: x[2]**2,
            lambda x: log(abs(x[0])),
            lambda x: log(abs(x[1])),
            lambda x: log(abs(x[2]))]
    Labels = ["x_1",
              "x_2",
              "x_3",
              "x_1^2",
              "x_2^2",
              "x_3^2",
              "log(|x_1|)",
              "log(|x_2|)",
              "log(|x_3|)"]
elif my_base == 2:
    # Third Order
    Phi = [ lambda x: x[0],
            lambda x: x[1],
            lambda x: x[2],
            lambda x: x[0]**2,
            lambda x: x[1]**2,
            lambda x: x[2]**2,
            lambda x: x[0]**3,
            lambda x: x[1]**3,
            lambda x: x[2]**3,
            lambda x: log(abs(x[0])),
            lambda x: log(abs(x[1])),
            lambda x: log(abs(x[2])),
            lambda x: x[0]**(-1),
            lambda x: x[1]**(-1),
            lambda x: x[2]**(-1),]
    Labels = ["x_1",
              "x_2",
              "x_3",
              "x_1^2",
              "x_2^2",
              "x_3^2",
              "x_1^3",
              "x_2^3",
              "x_3^3",
              "log(|x_1|)",
              "log(|x_2|)",
              "log(|x_3|)",
              "x_1^(-1)",
              "x_2^(-1)",
              "x_3^(-1)"]
else:
    # Active Subspace
    Phi = [ lambda x: x[0],
            lambda x: x[1],
            lambda x: x[2]]
    Labels = ["x_1",
              "x_2",
              "x_3"]
# Gradients 
dPhi = [gh(f)[0] for f in Phi]

#################################################
## Experiment Cases
#################################################
if my_case == 1:
    # Quadratic
    fcn = lambda x: x[0]**2 + x[1]**2 - 2.0*x[2]**2
    grad, _ = gh(fcn)
elif my_case == 2:
    # Mixed Terms
    fcn = lambda x: x[0] + x[1] - 2.0*x[2]**2
    grad, _ = gh(fcn)
else:
    # Ridge Function
    fcn = lambda x: 0.5*(0.3*x[0]+0.3*x[1]+0.7*x[2])**2
    grad, _ = gh(fcn)

#################################################
## Monte Carlo Method
#################################################
# Draw samples
X = 2.0*(random([n,m])-0.5) # Uniform on [-1,1]
# Build matrix
M = []
for i in range(int(n)):
    M.append( [np.dot( dPhi[j](X[i]),grad(X[i]) ) for j in range(len(dPhi))] )
M = np.array(M)

##################################################
## Active Manifold Pursuit
##################################################
W, Res = seek_am(M)

##################################################
## Results
##################################################

# Command line printback
print "AM Results:"
print "Residuals  = \n{}".format(Res)
print "Leading Vectors:"
print "W[:,0] = \n{}".format(W[:,0])
print "W[:,1] = \n{}".format(W[:,1])
print "W[:,2] = \n{}".format(W[:,2])

##################################################
# Plotting
##################################################

### Residuals
fig = plt.figure()
plt.plot(Res,'k*')
plt.yscale('log')
plt.xlim([-0.5,len(Res)-0.5])
plt.xticks(range(len(Res)))
# Annotation
plt.title("Residuals")
plt.xlabel("Index")
plt.ylabel("Residual")
# Set plot location on screen
manager = plt.get_current_fig_manager()
x,y,dx,dy = manager.window.geometry().getRect()
manager.window.setGeometry(offset[0][0],offset[0][1],dx,dy)

### Vectors
N   = len(W[:,0])
ind = np.arange(N)
wid = 0.35

fig = plt.figure()
plt.bar(ind    ,W[:,0],wid,color='b')
plt.bar(ind+wid,W[:,1],wid,color='r')
plt.xlim([-0.5,N+0.5])
plt.xticks(ind+wid,Labels)
# Annotation
plt.title("Vectors")
plt.xlabel("Index")
plt.ylabel("Entry")
# Set plot location on screen
manager = plt.get_current_fig_manager()
x,y,dx,dy = manager.window.geometry().getRect()
manager.window.setGeometry(offset[1][0],offset[1][1],dx,dy)

# Show all plots
plt.show()
