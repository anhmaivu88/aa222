# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from numpy.linalg import norm

# Version
from qnewton_ad import qnewton; title_ext = "Automatic Differentiation"
# from qnewton import qnewton; title_ext = "Finite Differences"

### ----- CONVERGENCE ----- ###
### Setup
# Rosenbrock function
from rosenbrock import fcn as fcn0
xstar0 = [1,1]; x0 = [-0.5,-0.5]; title0="Rosenbrock"
# Wood function
from wood import fcn as fcn1
xstar1 = [1,1,1,1]; x1 = [0,0,0,0]; title1="Wood"
# Powell function
from powell import fcn as fcn2
xstar2 = [0,0,0,0]; x2 = [1,1,1,1]; title2="Powell"

Xstar = [xstar0,xstar1,xstar2]
X0    = [x0,x1,x2]
Fcn   = [fcn0,fcn1,fcn2]
Label = [title0,title1,title2]

# Set parameters
Calls = [1,2,5,1e1,2e1,5e1,1e2,2e2,5e2,1e3,2e3,5e3,1e4,2e4,5e4,1e5]
Error = [[],[],[]]

# Generate convergence data
for i in range(len(Xstar)):

    for c in Calls:
        ### Solver call
        xs0,_,_,_,_ = qnewton(Fcn[i],X0[i],c)
        # Error calculation
        err = norm(xs0-Xstar[i])
        Error[i].append(err)

### Plotting
# Open figure
plt.figure()

# Plot convergence history
plt.title('Error vs Evaluation Limit: '+title_ext)
plt.loglog(Calls,Error[0],'.-b',label=Label[0])
plt.loglog(Calls,Error[1],'.-r',label=Label[1])
plt.loglog(Calls,Error[2],'.-g',label=Label[2])
plt.xlabel('Evaluation Limit')
plt.ylabel('Error')
plt.legend()

plt.show()
