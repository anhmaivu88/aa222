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
# From util/ directory
import util

##################################################
## Setup
##################################################
# Parameters
n = int(1e2) # monte carlo samples
my_case = 0     # Problem selection
# Define basis functions
phi0 = lambda x: x[0];      d_phi0, _ = gh(phi0)
phi1 = lambda x: x[1];      d_phi1, _ = gh(phi1)
phi2 = lambda x: x[0]**2;   d_phi2, _ = gh(phi2)
phi3 = lambda x: x[1]**2;   d_phi3, _ = gh(phi3)
phi4 = lambda x: log(abs(x[0])); d_phi4, _ = gh(phi4)
phi5 = lambda x: log(abs(x[1])); d_phi5, _ = gh(phi5)
Phi  = (phi0,phi1,phi2,phi3,phi4,phi5)
dPhi = (d_phi0,d_phi1,d_phi2,d_phi3,d_phi4,d_phi5)

#################################################
## Experiment Cases
#################################################
if my_case == 1:
    # Quadratic
    m = 2
    fcn = lambda x: x[0]**2 - 2.0*x[1]**2
    grad, _ = gh(fcn)
    # Expected basis
    B = np.array([[0.],[0.],[0.],[0.],[4.],[1.]])
elif my_case == 2:
    # Mixed Terms
    m = 2
    fcn = lambda x: x[0] - 2.0*x[1]**2
    grad, _ = gh(fcn)
    # Expected basis
    B = np.array([[4.],[0.],[0.],[0.],[0.],[1.]])
elif my_case == 3:
    # Composed Mixed Terms
    m = 2
    fcn = lambda x: 0.5*(x[0] - 2.0*x[1]**2)**2
    grad, _ = gh(fcn)
    # Expected basis
    B = np.array([[4.],[0.],[0.],[0.],[0.],[1.]])
elif my_case == 4:
    # Log terms
    m = 2
    fcn = lambda x: log(abs(x[0])) + log(abs(x[1]))
    grad, _ = gh(fcn)
    # Expected basis
    B = np.array([[0.],[0.],[1.],[-1.],[0.],[0.]])
elif my_case == 5:
    # Log terms
    m = 2
    fcn = lambda x: x[0]**2 + log(abs(x[1]))
    grad, _ = gh(fcn)
    # Expected basis
    B = np.array([[0.],[0.],[0.],[-1.],[1.],[0.]])
else:
    # Ridge Function
    m = 2
    fcn = lambda x: 0.5*(0.3*x[0]+0.7*x[1])**2
    grad, _ = gh(fcn)
    # Expected basis
    B = np.array([[0.7],[-0.3],[0.],[0.],[0.],[0.]])

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
## Optimization Problem
##################################################

beta = 1.0      # tunable parameter
def f(x):
    # L1 maximization
    return norm(np.dot(M,np.array(x))) - beta * norm(x,ord=1)
def g(x):
    # L2 constraint
    return (norm(x,ord=2) - 1,)
# Initial guess
x0 = [1.0] * M.shape[1]

##################################################
## Solver
##################################################
xs, Fs, Xs, it = constrained_opt(f,g,x0)
xs = util.col(xs) # make column vector
##################################################
## Results
##################################################

residual = norm(np.dot(M,np.array(xs)))
print "Solver Results:"
print "xs         = \n{}".format(xs)
print "|M*xs|     = {}".format(residual)
print "Feasible   = {}".format(feasible(g(xs)))
print "iter       = {}".format(it)
print "B^T * xs   = {}".format(np.dot(B,xs))
