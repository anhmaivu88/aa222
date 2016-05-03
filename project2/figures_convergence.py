# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from numpy.linalg import norm

# Version
from project2 import opt_full

### ----- CONVERGENCE ----- ###
### Setup
# Example 1
title0='Example 1'
from example1 import f as f0
from example1 import g as g0
from example1 import x_star as xstar0
x0 = [0,1]
# Example 2
title1='Example 2'
from example2 import f as f1
from example2 import g as g1
from example2 import x_star as xstar1
x1 = [0,1]
# Example 3
title2='Example 3'
from example3 import f as f2
from example3 import g as g2
from example3 import x_star as xstar2
x2 = [0,1,0]

Xstar = [xstar0,xstar1,xstar2]
X0    = [x0,x1,x2]
Fcn   = [f0,f1,f2]
Con   = [g0,g1,g2]
Label = [title0,title1,title2]

# Set parameters
Calls = [1,2,5,1e1,2e1,5e1,1e2,2e2,5e2,1e3,2e3,5e3,1e4,2e4,5e4,1e5]
Error = [[],[],[]]

# Generate convergence data
for i in range(len(Xstar)):

    for c in Calls:
        ### Solver call
        # xs0,_,_,_,_ = qnewton(Fcn[i],X0[i],c)
        xs0, _, _, _, _ = opt_full(Fcn[i],Con[i],X0[i],c)
        # Error calculation
        err = norm(xs0-Xstar[i])
        Error[i].append(err)

### Plotting
# Open figure
plt.figure()

# Plot convergence history
plt.title('Error vs Evaluation Limit')
plt.loglog(Calls,Error[0],'.-b',label=Label[0])
plt.loglog(Calls,Error[1],'.-r',label=Label[1])
plt.loglog(Calls,Error[2],'.-g',label=Label[2])
plt.xlabel('Evaluation Limit')
plt.ylabel('Error')
plt.legend()

plt.show()
