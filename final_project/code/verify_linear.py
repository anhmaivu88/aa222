##################################################
## Standard libraries
##################################################
import numpy as np
from numpy.linalg import norm
from ad.admath import log, sqrt
from numpy.random import random
from ad import gh
import sys
from scipy.linalg import svd

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
runs    = 10        # Number of runs to try
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
# Compute Active Subspace
##################################################
C = M.T.dot(M)
U,L,V = svd(C)

##################################################
## Run multiple pursuits
##################################################
Err = []
Dist = []
W_hist = []
for i in range(runs):
    x0 = random([1,m])
    W, Res = seek_am(M,x0=x0)
    Err.append(Res[0]+Res[1])
    Dist.append( util.subspace_distance(util.col(U[:,1:]),util.col(W[:,0:2])) )
    W_hist.append(W)

##################################################
# Plot Results
##################################################

fig = plt.figure()
plt.plot(Err,'*')
plt.yscale('log')
# Annotation
plt.title("Residual Values")
plt.xlabel("Run index")
plt.ylabel("Residual")

fig = plt.figure()
plt.plot(Dist,'*')
plt.yscale('log')
# Annotation
plt.title("Subspace Distances")
plt.xlabel("Run index")
plt.ylabel("Subspace Distance")

plt.show()