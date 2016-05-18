##################################################
## Standard libraries
##################################################
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from ad.admath import log
from numpy.random import random
from ad import gh

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
my_case = 0         # Problem selection
my_base = 1         # Basis selection

# Define basis functions
if my_base == 1:
    Phi = [ lambda x: x[0],
            lambda x: x[1],
            lambda x: x[2],
            lambda x: x[0]**2,
            lambda x: x[1]**2,
            lambda x: x[2]**2,
            lambda x: log(abs(x[0])),
            lambda x: log(abs(x[1])),
            lambda x: log(abs(x[2]))]
else:
    Phi = [ lambda x: x[0],
            lambda x: x[1],
            lambda x: x[2]]
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

print "AM Results:"
print "Residuals  = \n{}".format(Res)
