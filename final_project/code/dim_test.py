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

##################################################
## Setup
##################################################
# Parameters
n       = int(1e2)  # monte carlo samples
m_des   = 2         # Desired manifolds
res_t   = 1e-3      # Absolute tolerance for residual
iter_max= 10        # Maximum restarts for solver
# Plot settings
offset = [(0,500),(700,500),(1400,500)] # Plot window locations

##################################################
# Command Line Inputs
##################################################
# if len(sys.argv) < 2:
#     print('Usage:')
#     print('    python {} [case choice] [basis choice]'.format(sys.argv[0]))
#     exit()

# Problem selection
if len(sys.argv) > 1:
    m       = int(sys.argv[1])
else:
    print("Default dimension m=3 selected...")
    m       = 3         # Dimension of problem
if len(sys.argv) > 2:
    my_case = int(sys.argv[2])
else:
    print("Default case selected...")
    my_case = 0
# Basis selection
if len(sys.argv) > 3:
    my_base = int(sys.argv[3])
else:
    print("Default basis selected...")
    my_base = 0

# Choose basis
Phi,dPhi,Labels = get_basis(my_base,m)

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
W, Res, it = seek_am(M, res_thresh = res_t, m_des = m_des, \
                     verbose=True, full=True)

##################################################
## Results
##################################################

# Command line printback
print "AM Results:"
print "Solver resets = {}".format(it)
print "Residuals     = \n{}".format(Res)
print "Leading Vectors:"
for i in range(m_des):
    print "W[:,"+str(i)+"] = \n{}".format(W[:,i])

##################################################
# Compare the SVD
##################################################
u,l,v = svd(M)
W_svd = v.T
Res_svd = l

##################################################
# Plotting
##################################################

### Residuals
fig = plt.figure()
plt.plot(Res,'k*')
# plt.plot(Res_svd,'bo') # Eigenvalues from SVD
plt.yscale('log')
plt.xlim([-0.5,len(Res_svd)-0.5])
plt.xticks(range(len(Res_svd)))
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

# ### Vectors via SVD
# N   = len(W_svd[:,0])
# ind = np.arange(N)
# wid = 1./3. * 0.9

# fig = plt.figure()
# plt.bar(ind    ,W_svd[:,0],wid,color='k')
# plt.bar(ind+wid,W_svd[:,1],wid,color='b')
# plt.bar(ind+wid*2,W_svd[:,2],wid,color='r')
# plt.xlim([-0.5,N+0.5])
# plt.xticks(ind+wid,Labels)
# # Annotation
# plt.title("Vectors via SVD")
# plt.xlabel("Index")
# plt.ylabel("Entry")
# # Set plot location on screen
# manager = plt.get_current_fig_manager()
# x,y,dx,dy = manager.window.geometry().getRect()
# manager.window.setGeometry(offset[2][0],offset[2][1],dx,dy)

# Show all plots
plt.show()
