##################################################
## Standard libraries
##################################################
import numpy as np
from numpy.linalg import norm
from ad.admath import log
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
from get_basis import get_basis
from get_objective import get_objective

##################################################
## Setup
##################################################
# Parameters
n         = int(1e2)  # monte carlo samples
res_t     = 1e-3      # Absolute tolerance for residual
reset_max = 10        # Maximum restarts for solver
# Plot settings
offset = [(0,700),(700,700),(1400,700),(2100,700)] # Plot window locations

eps = np.finfo(float).eps # Machine epsilon

##################################################
# Command Line Inputs
##################################################
if len(sys.argv) < 2:
    print('Usage:')
    print('    python {} [input file]'.format(sys.argv[0]))
    exit()

# Problem selection
if len(sys.argv) > 1:
    filename = sys.argv[1] # name of input file
    f = open(filename)
    for line in f:
        l = line.split()
        # Switch based on flag
        if l[0].lower() == 'm':
            m = int(l[1])
        if l[0].lower() == 'm_des':
            m_des = int(l[1])
        elif l[0].lower() == 'basis':
            my_base = int(l[1])
        elif l[0].lower() == 'case':
            my_case = int(l[1])
        elif l[0].lower() == 'title':
            title_case = l[1]
        elif l[0].lower() == 'plot':
            plotting = int(l[1])

# Disable plotting if dimensionality too high
if m > 6:
    plotting = False

# Choose basis
Phi, dPhi, basis_name, Labels = get_basis(my_base,m)

#################################################
## Experiment Cases
#################################################
fcn, grad, case_name, opt = get_objective(my_case,m,full=True)

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
W, Res, resets, Res_full, Obj_full = \
            seek_am(M, res_thresh = res_t, m_des = m_des, reset_max = reset_max, \
                    verbose=True, full=True)

sequence = [len(l) for l in Res_full]

##################################################
## Results
##################################################

# Command line printback
print "Problem Setup:"
print "Dimensionality = {}".format(m)
print "Basis type     = {}".format(basis_name)
print "Objective type = {}".format(case_name)
print "AM Results:"
print "Solver resets  = {}".format(resets)
print "Residuals      = \n{}".format(Res)
print "Iter. counts   = \n{}".format(sequence)
print "Function param = \n{}".format(opt)
print "Leading Vectors:"
for i in range(m_des):
    print "W[:,"+str(i)+"] = \n{}".format(W[:,i])

##################################################
# Plotting
##################################################

if plotting:
    ### Residuals
    # Floor the residuals on machine epsilon
    Res = [max(Res[i],eps) for i in range(len(Res))]

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

    ### Residual sequences
    longest = 0
    res_colors  = ['b-*','r-*','g-*','k-*','c-*','m-*','y-*']

    fig = plt.figure()
    for i in range(len(Res_full)):
        plt.plot(Res_full[i],res_colors[i])
        longest = max(longest,len(Res_full[i]))

    plt.yscale('log')
    plt.xlim([-0.5,longest+0.5])
    # Annotation
    plt.title("Residual Sequences")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    # Set plot location on screen
    manager = plt.get_current_fig_manager()
    x,y,dx,dy = manager.window.geometry().getRect()
    manager.window.setGeometry(offset[2][0],offset[2][1],dx,dy)

    ### Residual sequences
    longest = 0
    obj_colors  = ['b-*','r-*','g-*','k-*','c-*','m-*','y-*']

    fig = plt.figure()
    for i in range(len(Obj_full)):
        plt.plot(Obj_full[i],obj_colors[i])
        longest = max(longest,len(Obj_full[i]))

    plt.yscale('log')
    plt.xlim([-0.5,longest+0.5])
    # Annotation
    plt.title("Objective Sequences")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    # Set plot location on screen
    manager = plt.get_current_fig_manager()
    x,y,dx,dy = manager.window.geometry().getRect()
    manager.window.setGeometry(offset[3][0],offset[3][1],dx,dy)

    # Show all plots
    plt.show()
