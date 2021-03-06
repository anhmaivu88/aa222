##################################################
# Standard libraries
##################################################
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from ad.admath import log
from numpy.random import random
from ad import gh

##################################################
# Custom libraries
##################################################
from interior import constrained_opt
from barrier import feasible

##################################################
# Setup
##################################################
# Parameters
m = 8        # problem dimensionality
n = int(1e2) # monte carlo samples

# Problem definition
beta = 1.0      # tunable parameter
my_case = 1     # Problem selection

### Problem: L1 Maximization, maximum distance
if my_case == 1:
    title = "L1 Maximization"
    def f(x):
        # L1 maximization
        return -1.0 * norm(x,ord=1)
    def g(x):
        # L1 constraint
        return (norm(x,ord=2) - 1,)
    x0 = [0] * m
    x0[0] = 1.5
else:
### Problem: L1 Minimization, minimum distance
    title = "L1 Minimization"
    def f(x):
        # L1 minimization
        return 1.0 * norm(x,ord=1)
    def g(x):
        # L1 constraint
        return (-1.0*norm(x,ord=2) + 1.0,)
    x0 = [0] * m
    x0[0] = 1e-2

##################################################
# Solver
##################################################
xs, Fs, Xs, it = constrained_opt(f,g,x0)


##################################################
# Results
##################################################
print "Case: {}".format(title)
print "xs         = {}".format(xs)
print "Fs(xs)     = {}".format(Fs)
print "Feasible   = {}".format(feasible(g(xs)))
print "iter       = {}".format(it)
