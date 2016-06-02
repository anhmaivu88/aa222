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
import pyutil.numeric as util
from pyutil.plotting import linspecer

from seek_am import seek_am
from get_basis import get_basis
from get_objective import get_objective

##################################################
## Setup
##################################################
# Parameters
n         = int(1e2)  # monte carlo samples
# Default parameters
m         = 3
m_des     = 2
my_base   = 0
my_case   = 0
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
    print('Input File Format:')
    print('    m     3 # 3-dimensional objective function')
    print('    m_des 2 # number of desired Inactive Manifolds')
    print('    basis 0 # choice of basis: 0 Linear, 1 Quadratic, 3 Cubic')
    print('    case  0 # objective function: 0 Ridge, 1 Double Ridge, 2 Mixed')
    print('    title Title_String # case title for plots,')
    print('                       # underscores replaced with spaces')
    print('    plot  0 # boolean flag for plotting')
    print('    reset_max 20 # maximum solver resets')
    print('    res_threshold 1e-3 # residual threshold for restart')
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
        elif l[0].lower() == 'reset_max':
            reset_max = int(l[1])
        elif l[0].lower() == 'res_threshold':
            res_t = float(l[1])
        elif l[0].lower() == 'title':
            title_case = l[1].replace("_"," ")
        elif l[0].lower() == 'plot':
            plotting = int(l[1])

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
W, Res, Resets_total, Res_full, Obj_full = \
            seek_am(M, res_thresh = res_t, m_des = m_des, reset_max = reset_max, \
                    verbose=True, full=True)

sequence = [len(l) for l in Res_full]

##################################################
# Compare the SVD
##################################################
if my_base == 0:
    u,l,v = svd(M)
    W_svd = v.T
    Res_svd = l
    # Subspace distance
    W_svd = W_svd[:,1:]
    dist = util.subspace_distance(W_svd,W)
else:
    dist = None

##################################################
## Results
##################################################

# Command line printback
print "Problem Setup:"
print "Dimensionality = {}".format(m)
print "Basis type     = {}".format(basis_name)
print "Objective type = {}".format(case_name)
print "AM Results:"
print "Iter. counts   = \n{}".format(sequence)
print "Function param = \n{}".format(opt)
print "Solver resets  = \n{}".format(Resets_total)
print "Residuals      = \n{}".format(Res)
if dist != None:
    print "Subspace Dist  = \n{}".format(dist)
print "Leading Vectors:"
for i in range(m_des):
    print "W[:,"+str(i)+"] = \n{}".format(W[:,i])

##################################################
# Plotting
##################################################

if plotting:
    ### Residual sequences
    length = len(Res_full)
    longest = 0
    colors  = linspecer(length)
    sty = '-'
    mkr = 'o'

    fig = plt.figure()
    for i in range(length):
        plt.plot(Res_full[i],color=colors[i],linestyle=sty,marker=mkr)
        longest = max(longest,len(Res_full[i]))

    plt.yscale('log')
    plt.xlim([-0.5,longest+0.5])
    # Annotation
    plt.title("Residual Sequences: "+title_case)
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.legend(['Stage '+str(i+1) for i in range(length)])
    # Set plot location on screen
    manager = plt.get_current_fig_manager()
    x,y,dx,dy = manager.window.geometry().getRect()
    manager.window.setGeometry(offset[0][0],offset[0][1],dx,dy)

    ### Objective sequences
    longest = 0

    fig = plt.figure()
    for i in range(length):
        plt.plot(Obj_full[i],color=colors[i],linestyle=sty,marker=mkr)
        longest = max(longest,len(Obj_full[i]))

    plt.yscale('log')
    plt.xlim([-0.5,longest+0.5])
    # Annotation
    plt.title("Objective Sequences: "+title_case)
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.legend(['Stage '+str(i+1) for i in range(length)])
    # Set plot location on screen
    manager = plt.get_current_fig_manager()
    x,y,dx,dy = manager.window.geometry().getRect()
    manager.window.setGeometry(offset[1][0],offset[1][1],dx,dy)

    # Show all plots
    plt.show()
